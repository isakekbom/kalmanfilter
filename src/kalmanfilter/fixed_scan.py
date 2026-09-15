"""Explicit fixed-shape scan execution of the existing EKF and likelihood.

Stack once during setup. The general ragged Python drivers remain independent.
All coordinates, map topology/weights, and corresponding instrument shapes are
constant within a segment; numerical parameters/instrument data may vary by date.
Use jax.jit(checkify.checkify(...)) for checked compiled calls, and discard all
outputs on failure. Direct eager calls also raise. See docs/fixed_scan_scaling.md.
"""

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify

from .ekf import EKFInputs, FilterState, ekf_step, initialize_filter
from .likelihood import likelihood_contribution
from .ois import OISInstrument
from .transition import DiagonalMatrix, StructuralStep, validate_step


class FixedScanInputs(NamedTuple):
    """One shared step plus numerical leaves with a leading time axis.

    Field order matches EKFInputs. DiagonalMatrix wraps (T,n) variance arrays
    here; its ordinary matrix .shape property is not used on batched containers.
    Dense covariances are (T,n,n). Instruments retain a fixed tuple structure,
    each with batched accrual/loading leaves. Use immutable JAX arrays.
    """

    step: StructuralStep
    theta_f: Array
    sigma_w: DiagonalMatrix | Array
    theta_g: Array
    sigma_v: DiagonalMatrix | Array
    observations: Array
    instruments: tuple[OISInstrument, ...]

    @property
    def n_steps(self) -> int:
        return self.observations.shape[0]


class FixedScanTrace(NamedTuple):
    """Numerical arrays with leading time axis, without repeated static objects."""

    predicted_states: Array
    predicted_covariances: Array
    filtered_states: Array
    filtered_covariances: Array
    innovations: Array
    innovation_covariances: Array
    per_step_contributions: Array


class FixedScanLikelihoodResult(NamedTuple):
    total_log_likelihood: Array
    per_step_contributions: Array
    final: FilterState
    trace: FixedScanTrace | None


def _real64(value) -> Array:
    array = jnp.asarray(value)
    if not (jnp.issubdtype(array.dtype, jnp.floating)
            or jnp.issubdtype(array.dtype, jnp.integer)):
        raise TypeError("fixed scan numerical inputs must contain real numbers")
    array = array.astype(jnp.float64, copy=False)
    checkify.check(jnp.all(jnp.isfinite(array)), "fixed scan inputs must be finite")
    return array


def _shape(value: Array, expected: tuple[int, ...], name: str):
    if value.shape != expected:
        raise ValueError(f"{name} shape must be {expected}; got {value.shape}")


def _validate_inputs(inputs: FixedScanInputs) -> FixedScanInputs:
    """Fixed-size tree/shape validation; this never iterates over the time axis."""
    if not isinstance(inputs, FixedScanInputs):
        raise TypeError("inputs must be FixedScanInputs; stack outside the objective")
    step = inputs.step
    validate_step(step)
    if step.previous != step.current:
        raise ValueError("fixed scan requires identical previous/current state coordinates")
    if not isinstance(inputs.instruments, tuple) or any(
        not isinstance(instrument, OISInstrument) for instrument in inputs.instruments
    ):
        raise TypeError("instruments must be an ordered tuple of OISInstrument inputs")
    inputs = FixedScanInputs(step, *jax.tree.map(_real64, inputs[1:]))
    if inputs.observations.ndim != 2 or inputs.theta_g.ndim != 2:
        raise ValueError("observations and theta_g must be rank two with a leading time axis")
    t = inputs.n_steps
    _shape(inputs.observations, (t, len(step.active_observations)), "observations")
    _shape(inputs.theta_f, (t, len(step.transition_post_map.columns)), "theta_f")
    _shape(inputs.theta_g, (t, inputs.theta_g.shape[1]), "theta_g")
    for covariance, size, name in (
        (inputs.sigma_w, len(step.process_noise_map.columns), "sigma_w"),
        (inputs.sigma_v, len(step.observation_noise_map.columns), "sigma_v"),
    ):
        if isinstance(covariance, DiagonalMatrix):
            _shape(covariance.diagonal, (t, size), f"{name} diagonal")
        else:
            _shape(covariance, (t, size, size), name)
    if len(inputs.instruments) != len(step.active_observations):
        raise ValueError("instrument count must match active observations")
    for instrument in inputs.instruments:
        accrual = instrument.accrual_factors
        if accrual.ndim != 2 or accrual.shape[1] == 0:
            raise ValueError("batched accrual_factors must have shape (T,K) with K >= 1")
        k = accrual.shape[1]
        _shape(accrual, (t, k), "accrual_factors")
        _shape(instrument.pca_loading_map,
               (t, k + 1, len(step.current.pca), inputs.theta_g.shape[1]), "pca_loading_map")
        _shape(instrument.step_loading, (t, k + 1, len(step.current.steps)), "step_loading")
    return inputs


def stack_fixed_inputs(inputs: Sequence[EKFInputs]) -> FixedScanInputs:
    """Validate and stack a nonempty segment once, outside JIT/differentiation.

    Require exact equality of structural metadata and numerical map weights,
    not just compatible dimensions. Corresponding covariance representations
    and instrument leaf shapes must agree. Active identities cannot change.
    This setup routine intentionally uses a Python time loop; the scan objective
    consumes only its batched output. It does not infer grouping or parameter ties.
    """
    inputs = tuple(inputs)
    if not inputs:
        raise ValueError("stack_fixed_inputs needs a nonempty segment to infer structure")
    if any(not isinstance(item, EKFInputs) for item in inputs):
        raise TypeError("segment entries must be EKFInputs")
    reference = inputs[0].step
    validate_step(reference)
    if reference.previous != reference.current:
        raise ValueError("fixed scan requires identical previous/current state coordinates")
    map_leaves, map_tree = jax.tree.flatten(reference)
    numerical_tree = jax.tree.structure(inputs[0][1:])
    for item in inputs:
        step = item.step
        if step.previous != reference.previous or step.current != reference.current:
            raise ValueError("state coordinate identity/order must stay fixed throughout the segment")
        if step.active_observations != reference.active_observations:
            raise ValueError("active observation identity/order must stay fixed throughout the segment")
        if step is not reference:
            validate_step(step)
            leaves, structure = jax.tree.flatten(step)
            if structure != map_tree:
                raise ValueError("structural map topology must stay fixed throughout the segment")
            if any(not bool(jnp.array_equal(a, b)) for a, b in zip(map_leaves, leaves, strict=True)):
                raise ValueError("structural map weights must stay identical throughout the segment")
        if jax.tree.structure(item[1:]) != numerical_tree:
            raise ValueError("instrument/covariance container structure must stay fixed")

    def stack_leaf(*leaves):
        shape = jnp.shape(leaves[0])
        if any(jnp.shape(leaf) != shape for leaf in leaves):
            raise ValueError("corresponding numerical/instrument shapes must stay fixed")
        return jnp.stack(leaves, axis=0)

    stacked = jax.tree.map(stack_leaf, *(item[1:] for item in inputs))
    return _validate_inputs(FixedScanInputs(reference, *stacked))


def _scan_likelihood(
    initial: FilterState, inputs: FixedScanInputs, return_trace: bool,
) -> FixedScanLikelihoodResult:
    inputs = _validate_inputs(inputs)
    initial = initialize_filter(initial.coordinates, initial.state, initial.covariance)
    step = inputs.step
    if initial.coordinates != step.previous:
        raise ValueError("initial filter coordinates must match the fixed segment")

    def body(carry, numerical):
        previous = FilterState(step.previous, *carry)
        result = ekf_step(previous, EKFInputs(step, *numerical))
        contribution = likelihood_contribution(result.update).contribution
        filtered = result.filtered
        trace = None
        if return_trace:
            trace = FixedScanTrace(
                result.prediction.predicted_state, result.prediction.predicted_covariance,
                filtered.state, filtered.covariance, result.update.innovation,
                result.update.innovation_covariance, contribution,
            )
        return (filtered.state, filtered.covariance), (contribution, trace)

    final, (contributions, trace) = jax.lax.scan(
        body, (initial.state, initial.covariance), inputs[1:], unroll=1,
    )
    total = jnp.sum(contributions)
    checkify.check(jnp.isfinite(total), "total log-likelihood must be finite")
    return FixedScanLikelihoodResult(total, contributions, FilterState(step.current, *final), trace)


def run_fixed_scan_likelihood(
    initial: FilterState, inputs: FixedScanInputs, *, return_trace: bool = False,
) -> FixedScanLikelihoodResult:
    """Run equations (38)-(57) through a fixed-shape scan, using existing kernels.

    The carry contains only filtered mean/covariance. Without trace retention,
    emitted arrays contain only per-date likelihood contributions; autodiff may
    retain residuals for the backward pass. All-empty observation segments are
    supported; mixed observation structures require separate caller-built segments.

    Checkify functionalizes the scan's checks before staging its body. Re-inject
    the resulting error with check_error so direct calls raise and an outer
    jax.jit(checkify.checkify(...)) can retain the normal checked-result contract.
    All outputs must be discarded after a checked failure.
    """
    error, result = checkify.checkify(
        lambda start, batched: _scan_likelihood(start, batched, return_trace)
    )(initial, inputs)
    checkify.check_error(error)
    return result
