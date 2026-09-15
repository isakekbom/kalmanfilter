"""Raw-vector likelihood objective and independent numerical derivative tools.

Close over the layout and problem builder before JIT/autodiff. Numerical checks
use checkify; throw its error outside JIT and discard all outputs after failure.
Finite differences use function values only, never automatic differentiation.
These small diagnostic loops are separate from the EKF and from optimization.
"""

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.typing import ArrayLike

from .ekf import EKFInputs, FilterState
from .likelihood import run_likelihood
from .params import ModelParameters, ParameterLayout, unpack_parameters


ProblemBuilder = Callable[[ModelParameters], tuple[FilterState, Sequence[EKFInputs]]]
ScalarObjective = Callable[[Array], ArrayLike]


def raw_negative_log_likelihood(
    raw_vector: ArrayLike, layout: ParameterLayout, build_problem: ProblemBuilder,
) -> Array:
    """NLL(r) = -ell(theta(r)), using the existing complete EKF/likelihood.

    The caller supplies a pure, fixed-structure builder with explicit coordinate
    maps and observations. No sharing/extraction across dates is inferred here.
    Returns a float64 scalar (). Failures propagate without catch or penalty.
    """
    parameters = unpack_parameters(raw_vector, layout)
    initial, inputs = build_problem(parameters)
    result = run_likelihood(initial, inputs, return_trace=False)
    return -result.total_log_likelihood


class CentralDifference(NamedTuple):
    """Gradient and actual coordinate steps, both float64 vectors of shape (n,)."""

    gradient: Array
    steps: Array


class DirectionalDifference(NamedTuple):
    """Scalar derivative, the unit direction actually used, and scalar step."""

    derivative: Array
    direction: Array
    step: Array


class GradientErrors(NamedTuple):
    """Per-component absolute and genuine relative errors, without a floor.

    Both-zero derivatives have relative error zero. Worst indices are None for
    empty vectors; otherwise ties select the first index. Maxima are zero when
    empty. Index the original AD/FD vectors to inspect the offending values.
    """

    absolute: Array
    relative: Array

    @property
    def max_absolute(self) -> Array:
        return jnp.max(self.absolute, initial=0.0)

    @property
    def max_relative(self) -> Array:
        return jnp.max(self.relative, initial=0.0)

    @property
    def worst_absolute_index(self) -> Array | None:
        return jnp.argmax(self.absolute) if self.absolute.size else None

    @property
    def worst_relative_index(self) -> Array | None:
        return jnp.argmax(self.relative) if self.relative.size else None


def _finite(value: Array, name: str) -> Array:
    checkify.check(jnp.all(jnp.isfinite(value)), f"{name} must be finite")
    return value


def _real64(value: ArrayLike, name: str, ndim: int) -> Array:
    array = jnp.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; got {array.shape}")
    if not (jnp.issubdtype(array.dtype, jnp.floating)
            or jnp.issubdtype(array.dtype, jnp.integer)):
        raise TypeError(f"{name} must contain real numbers")
    return _finite(array.astype(jnp.float64), name)


def _step(base_step: ArrayLike) -> Array:
    step = _real64(base_step, "finite-difference step", 0)
    checkify.check(step > 0, "finite-difference step must be positive")
    return step


def _value(objective: ScalarObjective, point: Array) -> Array:
    value = jnp.asarray(objective(point))
    if value.shape != ():
        raise ValueError("finite-difference objective must return a scalar ()")
    # Casting a float32 objective result cannot recover the lost precision.
    if value.dtype != jnp.float64:
        raise TypeError("finite-difference objective must return float64")
    return _finite(value, "finite-difference objective value")


def central_difference_gradient(
    objective: ScalarObjective, raw_vector: ArrayLike, base_step: ArrayLike,
) -> CentralDifference:
    """[f(r+h_i e_i)-f(r-h_i e_i)]/(2 h_i), h_i=h*max(1,abs(r_i)).

    Evaluates the objective exactly twice per coordinate in a Python loop. Step
    selection is independent of AD. The objective must return a checked finite
    float64 scalar; when using a compiled callable, throw its error on each call.
    Empty vectors return empty results without evaluating the objective.
    """
    raw = _real64(raw_vector, "raw vector", 1)
    steps = _finite(_step(base_step) * jnp.maximum(1.0, jnp.abs(raw)), "coordinate steps")
    derivatives = []
    for index in range(raw.size):
        offset = jnp.zeros_like(raw).at[index].set(steps[index])
        plus = _finite(raw + offset, "positive perturbation")
        minus = _finite(raw - offset, "negative perturbation")
        checkify.check((plus[index] != raw[index]) & (minus[index] != raw[index]),
                       "coordinate step is too small to change the raw value")
        derivatives.append((_value(objective, plus) - _value(objective, minus)) / (2 * steps[index]))
    gradient = jnp.stack(derivatives) if derivatives else jnp.empty((0,), dtype=jnp.float64)
    return CentralDifference(_finite(gradient, "finite-difference gradient"), steps)


def directional_central_difference(
    objective: ScalarObjective, raw_vector: ArrayLike, direction: ArrayLike, base_step: ArrayLike,
) -> DirectionalDifference:
    """[f(r+h*d)-f(r-h*d)]/(2h), normalizing the supplied nonzero direction.

    h is an absolute step along the returned unit direction, without coordinate
    scaling. Compare derivative to dot(AD_gradient, result.direction).
    """
    raw = _real64(raw_vector, "raw vector", 1)
    direction = _real64(direction, "direction", 1)
    if direction.shape != raw.shape:
        raise ValueError("direction shape must match the raw vector")
    scale = jnp.max(jnp.abs(direction), initial=0.0)
    checkify.check(scale > 0, "direction must be nonzero")
    scaled = direction / scale
    unit = scaled / jnp.linalg.norm(scaled)
    step = _step(base_step)
    plus = _finite(raw + step * unit, "positive perturbation")
    minus = _finite(raw - step * unit, "negative perturbation")
    checkify.check(jnp.all((direction == 0) | ((plus != raw) & (minus != raw))),
                   "directional step is too small to change a nonzero component")
    derivative = (_value(objective, plus) - _value(objective, minus)) / (2 * step)
    return DirectionalDifference(_finite(derivative, "directional finite difference"), unit, step)


def gradient_errors(automatic: ArrayLike, numerical: ArrayLike) -> GradientErrors:
    """abs(AD-FD) and abs(AD-FD)/max(abs(AD),abs(FD)); both zero gives zero.

    No acceptance tolerance is built in. Apply abs(AD-FD) <= atol+rtol*abs(AD)
    externally, so a near-zero derivative is assessed using absolute error too.
    """
    automatic = _real64(automatic, "automatic gradient", 1)
    numerical = _real64(numerical, "numerical gradient", 1)
    if automatic.shape != numerical.shape:
        raise ValueError("gradient shapes must match")
    absolute = _finite(jnp.abs(automatic - numerical), "absolute gradient error")
    scale = jnp.maximum(jnp.abs(automatic), jnp.abs(numerical))
    relative = absolute / jnp.where(scale == 0, 1.0, scale)
    return GradientErrors(absolute, relative)
