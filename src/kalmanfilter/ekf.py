"""Forward EKF, PDF (38)-(56); no likelihood or parameter estimation.

Numerical kernels use float64 JAX and explicit checkify checks. Compile fixed
shapes with ``jax.jit(checkify.checkify(function))``. Check the returned error
outside JIT before using results: after any reported failure, all returned
numerical values are invalid and must be discarded. See docs/ekf.md.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from .ois import OISInstrument, observation_quotes, quote_state_jacobian
from .transition import (
    DiagonalMatrix,
    LinearMap,
    StateCoordinates,
    StructuralStep,
    apply_map,
    materialize_map,
    observation_covariance,
    process_covariance,
    split_state,
    transition_matrix,
    validate_step,
)

Covariance = DiagonalMatrix | Array


@partial(jax.tree_util.register_dataclass, data_fields=["state", "covariance"],
         meta_fields=["coordinates"])
@dataclass(frozen=True, slots=True)
class FilterState:
    """Mean (n_x,) and dense covariance (n_x,n_x), with exact coordinate order.

    Use initialize_filter to validate inputs. Supply immutable JAX arrays;
    validation occurs in kernels, allowing JAX to reconstruct tracer pytrees.
    """

    coordinates: StateCoordinates
    state: Array
    covariance: Array


class CovarianceAsymmetry(NamedTuple):
    """Before averaging: max(abs(P-P.T)) and that value / max(abs(P)).

    The relative diagnostic is zero for an empty or identically zero matrix.
    No floor or acceptance threshold is used; both fields are float64 scalars.
    """

    max_absolute: Array
    relative: Array


class Prediction(NamedTuple):
    transition: LinearMap
    process_covariance: Covariance
    predicted_state: Array
    predicted_covariance: Array
    covariance_asymmetry: CovarianceAsymmetry


class ObservationLinearization(NamedTuple):
    systematic_state: Array
    unsystematic_state: Array
    modeled_quotes: Array
    quote_jacobian: Array
    observation_jacobian: Array
    linearization_offset: Array
    predicted_observation: Array


class MeasurementUpdate(NamedTuple):
    innovation: Array
    observation_covariance: Covariance
    innovation_covariance: Array
    innovation_cholesky: Array | None
    kalman_gain: Array
    filtered_state: Array
    filtered_covariance: Array
    innovation_covariance_asymmetry: CovarianceAsymmetry
    filtered_covariance_asymmetry: CovarianceAsymmetry


class EKFInputs(NamedTuple):
    """Explicit inputs for one date; observations/instruments follow active IDs.

    theta_f, sigma_w, sigma_v follow A.columns, D.columns, G.columns respectively.
    OIS loadings follow current PCA/step coordinates. No parameter extraction or
    observation selection is inferred here; use transition.select_observations
    and assemble the instrument tuple in step.active_observations order.
    """

    step: StructuralStep
    theta_f: Array
    sigma_w: Covariance
    theta_g: Array
    sigma_v: Covariance
    observations: Array
    instruments: tuple[OISInstrument, ...]


class EKFStepResult(NamedTuple):
    """Immutable equation-level trace; F/Q/R retain compact storage when possible."""

    structural_step: StructuralStep
    observations: Array
    prediction: Prediction
    linearization: ObservationLinearization
    update: MeasurementUpdate
    observation_noise_asymmetry: CovarianceAsymmetry

    @property
    def filtered(self) -> FilterState:
        return FilterState(
            self.structural_step.current,
            self.update.filtered_state,
            self.update.filtered_covariance,
        )


class FilterResult(NamedTuple):
    """Initial values, ragged filtered outputs, and optional full per-date trace."""

    initial: FilterState
    filtered: tuple[FilterState, ...]
    trace: tuple[EKFStepResult, ...] | None

    @property
    def final(self) -> FilterState:
        return self.filtered[-1] if self.filtered else self.initial


def _finite(array: Array, name: str) -> Array:
    checkify.check(jnp.all(jnp.isfinite(array)), f"{name} must be finite")
    return array


def _real64(value: ArrayLike, name: str, ndim: int) -> Array:
    array = jnp.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; got {array.shape}")
    if not (jnp.issubdtype(array.dtype, jnp.floating)
            or jnp.issubdtype(array.dtype, jnp.integer)):
        raise TypeError(f"{name} must contain real numbers")
    return _finite(array.astype(jnp.float64), name)


def _covariance(value: ArrayLike, size: int, name: str) -> Array:
    covariance = _real64(value, name, 2)
    if covariance.shape != (size, size):
        raise ValueError(f"{name} shape must be {(size, size)}; got {covariance.shape}")
    # Match the structural layer's exact symmetry contract for supplied inputs.
    checkify.check(jnp.all(covariance == covariance.T), f"{name} must be symmetric")
    return covariance


def _noise_covariance(value: Covariance, size: int, name: str) -> Covariance:
    if isinstance(value, DiagonalMatrix):
        diagonal = _real64(value.diagonal, name, 1)
        if diagonal.shape != (size,):
            raise ValueError(f"{name} diagonal shape must be {(size,)}")
        checkify.check(jnp.all(diagonal >= 0), f"{name} diagonal must be nonnegative")
        return DiagonalMatrix(diagonal)
    return _covariance(value, size, name)


def _add_covariance(matrix: Array, covariance: Covariance) -> Array:
    if isinstance(covariance, DiagonalMatrix):
        indices = jnp.arange(matrix.shape[0])
        return matrix.at[indices, indices].add(covariance.diagonal)
    return matrix + covariance


def _symmetrize(matrix: Array, name: str) -> tuple[Array, CovarianceAsymmetry]:
    """Record roundoff before averaging; this neither checks nor repairs PSD."""
    matrix = _finite(matrix, name)
    absolute = _finite(jnp.max(jnp.abs(matrix - matrix.T), initial=0.0),
                       f"{name} asymmetry")
    scale = jnp.max(jnp.abs(matrix), initial=0.0)
    relative = absolute / jnp.where(scale == 0, 1.0, scale)
    # Algebraically 0.5*(P+P.T), without overflowing the sum of large finite P.
    symmetric = 0.5 * matrix + 0.5 * matrix.T
    return symmetric, CovarianceAsymmetry(absolute, relative)


def initialize_filter(
    coordinates: StateCoordinates, a_x: ArrayLike, sigma_0: ArrayLike
) -> FilterState:
    """PDF (39)-(40): supplied x_0=a_x, P_0=Sigma_0; no stationary default.

    Checks real float64 promotion, finite values, shape agreement, and exact
    symmetry. Dense Sigma_0 must be PSD (caller precondition, no factorization).
    """
    if not isinstance(coordinates, StateCoordinates):
        raise TypeError("coordinates must be StateCoordinates")
    state = _real64(a_x, "initial state", 1)
    if state.shape != (len(coordinates.all),):
        raise ValueError("initial state shape must match coordinates")
    covariance = _covariance(sigma_0, state.shape[0], "initial covariance")
    return FilterState(coordinates, state, covariance)


def predict(
    previous: FilterState, step: StructuralStep,
    theta_f: ArrayLike, sigma_w: Covariance,
) -> Prediction:
    """PDF (41)-(43): F x and F P F.T + Q, also for rectangular F.

    Two structured left applications transform the dense covariance. F is not
    materialized by this consumer; diagonal Q is added directly to the diagonal.
    """
    validate_step(step)
    if previous.coordinates != step.previous:
        raise ValueError("previous filtered coordinates must equal step.previous in order")
    state = _real64(previous.state, "previous state", 1)
    if state.shape != (len(step.previous.all),):
        raise ValueError("previous state shape must match step.previous")
    covariance = _covariance(previous.covariance, state.shape[0], "previous covariance")
    transition = transition_matrix(theta_f, step)
    q = process_covariance(step, sigma_w)
    predicted_state = apply_map(transition, state)
    left_product = apply_map(transition, covariance)  # (n_x_t,n_x_prev)
    propagated = apply_map(transition, left_product.T).T  # F P F.T
    predicted_covariance, asymmetry = _symmetrize(
        _add_covariance(propagated, q), "predicted covariance"
    )
    return Prediction(transition, q, predicted_state, predicted_covariance, asymmetry)


def build_observation_linearization(
    step: StructuralStep, theta_g: ArrayLike, predicted_state: ArrayLike,
    instruments: tuple[OISInstrument, ...],
) -> ObservationLinearization:
    """PDF (38), (44)-(45): g, J, H=[J I^z], u=g-J x_s, z_hat=g+I^z x_u.

    Instrument tuple order is exactly step.active_observations. Its loadings
    must describe step.current's PCA and step blocks, including their order.
    """
    x_s, x_u = split_state(step.current, predicted_state)
    if not isinstance(instruments, tuple):
        raise TypeError("instruments must be an ordered tuple")
    if len(instruments) != len(step.active_observations):
        raise ValueError("instruments must have one entry per active observation")
    # Let OIS validate ranks first; then check the separate named block sizes.
    g = observation_quotes(theta_g, x_s, instruments)
    for instrument in instruments:
        if (jnp.shape(instrument.pca_loading_map)[1] != step.dimensions.n_p_t
                or jnp.shape(instrument.step_loading)[1] != step.dimensions.n_c_t):
            raise ValueError("instrument PCA/step block dimensions must match step.current")
    jacobian = quote_state_jacobian(theta_g, x_s, instruments)
    h = jnp.concatenate((jacobian, materialize_map(step.observation_selector)), axis=1)
    offset = _finite(g - jacobian @ x_s, "linearization offset")
    predicted_observation = _finite(
        g + apply_map(step.observation_selector, x_u), "predicted observation"
    )
    return ObservationLinearization(x_s, x_u, g, jacobian, h, offset, predicted_observation)


def measurement_update(
    predicted_state: ArrayLike,
    predicted_covariance: ArrayLike,
    observations: ArrayLike,
    predicted_observation: ArrayLike,
    observation_jacobian: ArrayLike,
    noise_covariance: Covariance,
) -> MeasurementUpdate:
    """PDF (46)-(56), independent of OIS: supply z_hat and H explicitly.

    Shapes: x (n_x,), P (n_x,n_x), z/z_hat (n_z,), H (n_z,n_x),
    R (n_z,n_z), K (n_x,n_z). Supplied dense P/R must be symmetric PSD;
    symmetry is checked exactly, PSD is a caller precondition. Nonempty S must
    be positive definite. A failed check invalidates all checkified outputs.

    Empty observations give prediction only, empty arrays, and L=None, without
    attempting Cholesky. This is an implementation convention, not a PDF rule.
    """
    x = _real64(predicted_state, "predicted state", 1)
    p = _covariance(predicted_covariance, x.shape[0], "predicted covariance")
    z = _real64(observations, "observations", 1)
    z_hat = _real64(predicted_observation, "predicted observation", 1)
    h = _real64(observation_jacobian, "observation Jacobian", 2)
    if z_hat.shape != z.shape or h.shape != (z.shape[0], x.shape[0]):
        raise ValueError("observation shapes must agree: z/z_hat (n_z,), H (n_z,n_x)")
    r = _noise_covariance(noise_covariance, z.shape[0], "observation covariance")
    innovation = _finite(z - z_hat, "innovation")  # (49), equivalent to (46)-(48)
    hp = h @ p
    s, s_asymmetry = _symmetrize(_add_covariance(hp @ h.T, r), "innovation covariance")
    if z.shape[0] == 0:
        zero = jnp.array(0.0, dtype=jnp.float64)
        return MeasurementUpdate(
            innovation, r, s, None, jnp.empty((x.shape[0], 0), dtype=jnp.float64),
            x, p, s_asymmetry, CovarianceAsymmetry(zero, zero),
        )

    cholesky = jnp.linalg.cholesky(s, symmetrize_input=False)
    checkify.check(jnp.all(jnp.isfinite(cholesky)),
                   "innovation Cholesky failed: factor must be finite; S must be positive definite")
    diagonal = jnp.diag(cholesky)
    checkify.check(jnp.all(jnp.isfinite(diagonal) & (diagonal > 0)),
                   "innovation Cholesky diagonal must be finite and positive")
    # (51)-(52): S K.T = H P, reusing L for both triangular solves.
    intermediate = solve_triangular(cholesky, hp, lower=True)
    gain = _finite(solve_triangular(cholesky.T, intermediate, lower=False).T, "Kalman gain")
    filtered_state = _finite(x + gain @ innovation, "filtered state")  # (53)
    filtered_covariance, p_asymmetry = _symmetrize(
        p - gain @ hp, "filtered covariance"  # (55), equivalent to (54)/(56)
    )
    return MeasurementUpdate(
        innovation, r, s, cholesky, gain, filtered_state, filtered_covariance,
        s_asymmetry, p_asymmetry,
    )


def ekf_step(previous: FilterState, inputs: EKFInputs) -> EKFStepResult:
    """One OIS/structural EKF step, with the full immutable equation trace."""
    step = inputs.step
    prediction = predict(previous, step, inputs.theta_f, inputs.sigma_w)
    linearization = build_observation_linearization(
        step, inputs.theta_g, prediction.predicted_state, inputs.instruments
    )
    r = observation_covariance(step, inputs.sigma_v)
    if isinstance(r, DiagonalMatrix):
        zero = jnp.array(0.0, dtype=jnp.float64)
        r_asymmetry = CovarianceAsymmetry(zero, zero)
    else:
        # G Sigma_v G.T may have roundoff drift despite symmetric base inputs.
        # Record it before the generic update's exact input-symmetry validation.
        r, r_asymmetry = _symmetrize(r, "mapped observation covariance")
    update = measurement_update(
        prediction.predicted_state, prediction.predicted_covariance,
        inputs.observations, linearization.predicted_observation,
        linearization.observation_jacobian, r,
    )
    return EKFStepResult(
        step, _real64(inputs.observations, "observations", 1), prediction,
        linearization, update, r_asymmetry,
    )


def run_filter(
    initial: FilterState, inputs: Sequence[EKFInputs], *, return_trace: bool = False,
) -> FilterResult:
    """Python forward loop over explicit, possibly changing per-date spaces.

    Adjacent coordinates must agree exactly, including order. Each input owns
    its declared parameter/noise axes; the driver neither extracts nor ties
    parameters. Results remain a tuple of per-date shapes (no padding/scan).
    An empty sequence returns the validated initial state as result.final.
    """
    initial = initialize_filter(initial.coordinates, initial.state, initial.covariance)
    inputs = tuple(inputs)
    coordinates = initial.coordinates
    for item in inputs:
        if item.step.previous != coordinates:
            raise ValueError("filter sequence coordinate mismatch: current must equal next previous in order")
        coordinates = item.step.current
    previous = initial
    filtered = []
    trace = [] if return_trace else None
    for item in inputs:
        result = ekf_step(previous, item)
        previous = result.filtered
        filtered.append(previous)
        if trace is not None:
            trace.append(result)
    return FilterResult(initial, tuple(filtered), None if trace is None else tuple(trace))
