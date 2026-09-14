"""Explicit flat coordinates for the PDF's six mathematical parameter blocks.

theta_f is unconstrained by default: Q9 does not establish a persistence domain.
Covariance transforms select a positive interior, without floors or repair.
For fixed layouts use jax.jit(checkify.checkify(...)); after any checked failure,
discard all numerical outputs. See docs/params.md for ordering and float64 limits.
"""

from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.typing import ArrayLike

from .transition import DiagonalMatrix


class ParameterSlices(NamedTuple):
    """Flat raw-coordinate slices, explicitly in PDF tuple order.

    sigma_w/sigma_v slices hold raw variances; sigma_0 holds raw Cholesky
    entries, not covariance entries. These slices are static Python metadata.
    """

    theta_f: slice
    sigma_w: slice
    sigma_v: slice
    a_x: slice
    sigma_0: slice
    theta_g: slice


@partial(jax.tree_util.register_dataclass, data_fields=[],
         meta_fields=["n_f", "n_w", "n_v", "n_x0", "n_g", "theta_f_transform"])
@dataclass(frozen=True, slots=True, kw_only=True)
class ParameterLayout:
    """Static dimensions/order for one parameter vector, not a master state.

    n_x0 describes only the initial distribution. Identity is the source-faithful
    theta_f default; unit_interval is an explicit optional convention, not a
    constraint inferred from the PDF. All counts are nonnegative Python ints.
    """

    n_f: int
    n_w: int
    n_v: int
    n_x0: int
    n_g: int
    theta_f_transform: Literal["identity", "unit_interval"] = "identity"

    def __post_init__(self):
        for name in ("n_f", "n_w", "n_v", "n_x0", "n_g"):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{name} must be a Python integer")
            if count < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not isinstance(self.theta_f_transform, str):
            raise TypeError("theta_f_transform must be a string")
        if self.theta_f_transform not in ("identity", "unit_interval"):
            raise ValueError("theta_f_transform must be 'identity' or 'unit_interval'")

    @property
    def n_cholesky(self) -> int:
        return self.n_x0 * (self.n_x0 + 1) // 2

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return self.n_f, self.n_w, self.n_v, self.n_x0, self.n_cholesky, self.n_g

    @property
    def n_parameters(self) -> int:
        return sum(self.block_sizes)

    @property
    def slices(self) -> ParameterSlices:
        start = 0
        slices = []
        for size in self.block_sizes:
            slices.append(slice(start, start + size))
            start += size
        return ParameterSlices(*slices)

    @property
    def lower_triangle_order(self) -> tuple[tuple[int, int], ...]:
        """Row-major lower triangle: (0,0), (1,0), (1,1), (2,0), ... ."""
        return tuple((i, j) for i in range(self.n_x0) for j in range(i + 1))


class ModelParameters(NamedTuple):
    """Mathematical parameters, with immutable JAX arrays as pytree leaves.

    Fields follow PDF theta=(theta_F, Sigma_w, Sigma_v, a_x, Sigma_0, theta_g).
    sigma_w/sigma_v store variances in DiagonalMatrix; sigma_0 is dense.
    Container construction does not validate: pack/unpack are the boundaries.
    """

    theta_f: Array
    sigma_w: DiagonalMatrix
    sigma_v: DiagonalMatrix
    a_x: Array
    sigma_0: Array
    theta_g: Array


def _finite(value: Array, name: str) -> Array:
    checkify.check(jnp.all(jnp.isfinite(value)), f"{name} must be finite")
    return value


def _real64(value: ArrayLike, name: str, ndim: int | None = None) -> Array:
    array = jnp.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; got {array.shape}")
    if not (jnp.issubdtype(array.dtype, jnp.floating)
            or jnp.issubdtype(array.dtype, jnp.integer)):
        raise TypeError(f"{name} must contain real numbers")
    return _finite(array.astype(jnp.float64, copy=False), name)


def _vector(value: ArrayLike, size: int, name: str) -> Array:
    vector = _real64(value, name, 1)
    if vector.shape != (size,):
        raise ValueError(f"{name} shape must be {(size,)}; got {vector.shape}")
    return vector


def _positive(raw: Array, name: str) -> Array:
    positive = _finite(jax.nn.softplus(raw), name)
    checkify.check(jnp.all(positive > 0), f"{name} must be positive (softplus underflow)")
    return positive


def inverse_softplus(value: ArrayLike) -> Array:
    """Stable inverse for finite y>0: y + log(-expm1(-y)); no epsilon floor.

    Elementwise on scalar/array inputs. Exact zero is a boundary, not a finite
    raw coordinate. Floating-point underflow/saturation is never repaired.
    """
    positive = _real64(value, "inverse softplus input")
    checkify.check(jnp.all(positive > 0), "inverse softplus input must be positive")
    return _finite(positive + jnp.log(-jnp.expm1(-positive)), "inverse softplus result")


def inverse_logit(value: ArrayLike) -> Array:
    """Stable log(p)-log1p(-p), requiring finite p strictly inside (0,1)."""
    probability = _real64(value, "inverse logit input")
    checkify.check(jnp.all((probability > 0) & (probability < 1)),
                   "inverse logit input must be strictly inside (0,1)")
    return _finite(jnp.log(probability) - jnp.log1p(-probability), "inverse logit result")


def _triangle_indices(layout: ParameterLayout) -> tuple[Array, Array]:
    order = layout.lower_triangle_order
    return (jnp.array([i for i, _ in order], dtype=jnp.int64),
            jnp.array([j for _, j in order], dtype=jnp.int64))


def _checked_cholesky(covariance: Array, name: str) -> Array:
    """Validate an actual factorization; never modify a failed covariance."""
    factor = jnp.linalg.cholesky(covariance, symmetrize_input=False)
    checkify.check(jnp.all(jnp.isfinite(factor)),
                   f"{name} Cholesky failed: covariance must be positive definite")
    checkify.check(jnp.all(jnp.diag(factor) > 0),
                   f"{name} Cholesky diagonal must be positive")
    return factor


def _initial_covariance(raw: Array, layout: ParameterLayout) -> Array:
    rows, columns = _triangle_indices(layout)
    diagonal = jnp.arange(layout.n_x0)
    factor = jnp.zeros((layout.n_x0, layout.n_x0), dtype=jnp.float64).at[rows, columns].set(raw)
    factor = factor.at[diagonal, diagonal].set(_positive(jnp.diag(factor), "Sigma_0 factor diagonal"))
    covariance = _finite(factor @ factor.T, "Sigma_0")
    # The Gram product is mathematically symmetric. Average floating-point drift
    # for the existing EKF's exact-symmetry boundary; this is not a PSD repair.
    covariance = 0.5 * covariance + 0.5 * covariance.T
    return covariance


def unpack_parameters(raw_vector: ArrayLike, layout: ParameterLayout) -> ModelParameters:
    """Transform flat raw coordinates in explicit PDF order to model parameters.

    Variance blocks use softplus directly (not squared). The initial covariance
    uses a lower factor with softplus diagonals and raw off-diagonals. Softplus
    underflow and covariance overflow are checked. Positive definiteness follows
    from the factor construction in exact arithmetic, without refactorization.
    """
    if not isinstance(layout, ParameterLayout):
        raise TypeError("layout must be ParameterLayout")
    raw = _vector(raw_vector, layout.n_parameters, "raw parameter vector")
    slices = layout.slices
    theta_f = raw[slices.theta_f]
    if layout.theta_f_transform == "unit_interval":
        theta_f = jax.nn.sigmoid(theta_f)
        checkify.check(jnp.all((theta_f > 0) & (theta_f < 1)),
                       "theta_f sigmoid must be strictly inside (0,1) (floating-point saturation)")
    return ModelParameters(
        theta_f=theta_f,
        sigma_w=DiagonalMatrix(_positive(raw[slices.sigma_w], "process variances")),
        sigma_v=DiagonalMatrix(_positive(raw[slices.sigma_v], "observation variances")),
        a_x=raw[slices.a_x],
        sigma_0=_initial_covariance(raw[slices.sigma_0], layout),
        theta_g=raw[slices.theta_g],
    )


def _pack_variances(covariance: DiagonalMatrix, size: int, name: str) -> Array:
    if not isinstance(covariance, DiagonalMatrix):
        raise TypeError(f"{name} must be DiagonalMatrix containing variances")
    return inverse_softplus(_vector(covariance.diagonal, size, name))


def _pack_initial_covariance(value: ArrayLike, layout: ParameterLayout) -> Array:
    covariance = _real64(value, "Sigma_0", 2)
    if covariance.shape != (layout.n_x0, layout.n_x0):
        raise ValueError("Sigma_0 shape must match layout.n_x0")
    checkify.check(jnp.all(covariance == covariance.T), "Sigma_0 must be symmetric")
    if layout.n_x0 == 0:
        return jnp.empty((0,), dtype=jnp.float64)
    factor = _checked_cholesky(covariance, "supplied Sigma_0")
    diagonal = jnp.arange(layout.n_x0)
    raw_factor = factor.at[diagonal, diagonal].set(inverse_softplus(jnp.diag(factor)))
    rows, columns = _triangle_indices(layout)
    return raw_factor[rows, columns]


def pack_parameters(parameters: ModelParameters, layout: ParameterLayout) -> Array:
    """Inverse/debugging map for interior parameters; returns a flat raw vector.

    Diagonal variances must be strictly positive; Sigma_0 must be SPD and exactly
    symmetric, consistent with existing supplied-covariance validation. No inverse
    exists at zero variance or at the optional logistic endpoints. No clipping,
    jitter, parameter tying, or covariance repair is performed.
    """
    if not isinstance(layout, ParameterLayout):
        raise TypeError("layout must be ParameterLayout")
    if not isinstance(parameters, ModelParameters):
        raise TypeError("parameters must be ModelParameters")
    theta_f = _vector(parameters.theta_f, layout.n_f, "theta_f")
    if layout.theta_f_transform == "unit_interval":
        theta_f = inverse_logit(theta_f)
    # The optimizer-vector contract is this explicit concatenation, not pytree
    # flattening or dictionary iteration order.
    return jnp.concatenate((
        theta_f,
        _pack_variances(parameters.sigma_w, layout.n_w, "Sigma_w variances"),
        _pack_variances(parameters.sigma_v, layout.n_v, "Sigma_v variances"),
        _vector(parameters.a_x, layout.n_x0, "a_x"),
        _pack_initial_covariance(parameters.sigma_0, layout),
        _vector(parameters.theta_g, layout.n_g, "theta_g"),
    ))
