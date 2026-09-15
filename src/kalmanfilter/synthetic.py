"""Explicit synthetic truth for the existing OIS/transition model (issue #9).

Initial, process, and observation draws are mutually independent synthetic
conventions; they do not resolve the PDF's joint-noise question. Numerical
checks follow the existing eager/checkify contract: for compiled execution use
``jax.jit(checkify.checkify(function))`` and discard all outputs on any error.
See docs/synthetic.md for sampling, named axes, and validation scenarios.
"""

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.typing import ArrayLike

from .ekf import EKFInputs, FilterState, initialize_filter
from .ois import OISInstrument, observation_quotes
from .transition import (
    DiagonalMatrix,
    StateCoordinates,
    StructuralStep,
    apply_map,
    split_state,
    transition_matrix,
    validate_step,
)

Covariance = DiagonalMatrix | Array


class SyntheticStepInputs(NamedTuple):
    """Mathematical inputs for one date, before generating its observations.

    theta_f, sigma_w, and sigma_v follow A.columns, D.columns, and G.columns.
    Instruments follow step.active_observations, with loading blocks in current
    PCA/step order. Supply immutable JAX arrays, as for EKFInputs. Each date is
    explicit; no global parameter extraction, tying, or lifecycle is inferred.
    """

    step: StructuralStep
    theta_f: Array
    sigma_w: Covariance
    theta_g: Array
    sigma_v: Covariance
    instruments: tuple[OISInstrument, ...]


class SyntheticDataset(NamedTuple):
    """Immutable truth and ordinary EKF inputs; tuples retain natural shapes.

    true_initial_state is sampled, while initial_filter holds its distribution.
    Entry t of every tuple describes inputs[t].step; that step identifies state,
    observation, and base-noise axes. process_noise / observation_noise are the
    mapped noises w / v; base_process_noise / base_observation_noise are eta_w /
    eta_v. The EKF inputs retain the supplied StructuralStep objects.
    """

    initial_filter: FilterState
    true_initial_state: Array
    true_states: tuple[Array, ...]
    base_process_noise: tuple[Array, ...]
    process_noise: tuple[Array, ...]
    noiseless_observations: tuple[Array, ...]
    base_observation_noise: tuple[Array, ...]
    observation_noise: tuple[Array, ...]
    observations: tuple[Array, ...]
    inputs: tuple[EKFInputs, ...]


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


def _noise_covariance(value: Covariance, size: int, name: str) -> Covariance:
    if isinstance(value, DiagonalMatrix):
        diagonal = _real64(value.diagonal, name, 1)
        if diagonal.shape != (size,):
            raise ValueError(f"{name} diagonal shape must be {(size,)}")
        checkify.check(jnp.all(diagonal >= 0), f"{name} diagonal must be nonnegative")
        return DiagonalMatrix(diagonal)
    covariance = _real64(value, name, 2)
    if covariance.shape != (size, size):
        raise ValueError(f"{name} shape must be {(size, size)}")
    checkify.check(jnp.all(covariance == covariance.T), f"{name} must be symmetric")
    return covariance


def _sample_gaussian(key: Array, covariance: Covariance, name: str) -> Array:
    """Sample validated base covariance; dense nonempty inputs must be SPD.

    Diagonal entries are variances and may be zero. No mapped covariance is
    factored. A singular dense covariance is unsupported: use DiagonalMatrix
    for zero base variances and explicit maps for shared sources/zero rows.
    """
    size = covariance.shape[0]
    if size == 0:
        return jnp.empty((0,), dtype=jnp.float64)
    normal = jax.random.normal(key, (size,), dtype=jnp.float64)
    if isinstance(covariance, DiagonalMatrix):
        draw = jnp.sqrt(covariance.diagonal) * normal
    else:
        factor = jnp.linalg.cholesky(covariance, symmetrize_input=False)
        checkify.check(jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.diag(factor) > 0),
                       f"{name} must admit a finite Cholesky factor with positive diagonal")
        draw = factor @ normal
    return _finite(draw, f"{name} draw")


def generate_synthetic_dataset(
    key: Array,
    initial_coordinates: StateCoordinates,
    a_x: ArrayLike,
    sigma_0: ArrayLike,
    steps: Sequence[SyntheticStepInputs],
) -> SyntheticDataset:
    """Sample x_0, then x_t = F_t x_prev + D_t eta_w and z_t = g + I_z x_u + G_t eta_v.

    The filter starts at (a_x, sigma_0), never at the sampled true x_0. sigma_0
    is dense SPD (or empty). Base sigma_w / sigma_v are nonnegative diagonal or
    dense SPD. Sampling introduces no jitter, floors, or covariance repair.

    Split key into (carry, initial_key), then split carry into (carry, w_key,
    v_key) once per date, including dates with empty observations. Identical
    keys and inputs reproduce bitwise in the locked environment. No keys are
    reused. An ordinary Python loop accommodates changing shapes without padding.
    """
    initial = initialize_filter(initial_coordinates, a_x, sigma_0)
    carry, initial_key = jax.random.split(key)
    true_initial = _finite(
        initial.state + _sample_gaussian(initial_key, initial.covariance, "initial covariance"),
        "true initial state",
    )
    previous_coordinates = initial.coordinates
    previous_state = true_initial
    states, base_w, noises_w, means, base_v, noises_v, observations, inputs = (
        [] for _ in range(8)
    )
    for item in steps:
        if not isinstance(item, SyntheticStepInputs):
            raise TypeError("steps must contain SyntheticStepInputs")
        step = item.step
        validate_step(step)
        if step.previous != previous_coordinates:
            raise ValueError("adjacent synthetic state coordinates must match exactly")
        if not isinstance(item.instruments, tuple):
            raise TypeError("instruments must be an ordered tuple")
        if len(item.instruments) != len(step.active_observations):
            raise ValueError("instruments must match active observation rows")
        theta_f = _real64(item.theta_f, "theta_f", 1)
        theta_g = _real64(item.theta_g, "theta_g", 1)
        sigma_w = _noise_covariance(item.sigma_w, len(step.process_noise_map.columns), "sigma_w")
        sigma_v = _noise_covariance(item.sigma_v, len(step.observation_noise_map.columns), "sigma_v")
        carry, w_key, v_key = jax.random.split(carry, 3)
        eta_w = _sample_gaussian(w_key, sigma_w, "sigma_w")
        w = apply_map(step.process_noise_map, eta_w)
        state = _finite(apply_map(transition_matrix(theta_f, step), previous_state) + w,
                        "true state")
        systematic, unsystematic = split_state(step.current, state)
        quotes = observation_quotes(theta_g, systematic, item.instruments)
        # Pricing validates ranks first. Also enforce named block sizes, even
        # when an incorrect PCA/step split has the right total systematic size.
        for instrument in item.instruments:
            if (jnp.shape(instrument.pca_loading_map)[1] != len(step.current.pca)
                    or jnp.shape(instrument.step_loading)[1] != len(step.current.steps)):
                raise ValueError("instrument PCA/step dimensions must match current coordinates")
        mean = _finite(quotes + apply_map(step.observation_selector, unsystematic),
                       "noiseless observation")
        eta_v = _sample_gaussian(v_key, sigma_v, "sigma_v")
        v = apply_map(step.observation_noise_map, eta_v)
        z = _finite(mean + v, "synthetic observation")
        states.append(state)
        base_w.append(eta_w)
        noises_w.append(w)
        means.append(mean)
        base_v.append(eta_v)
        noises_v.append(v)
        observations.append(z)
        inputs.append(EKFInputs(step, theta_f, sigma_w, theta_g, sigma_v, z, item.instruments))
        previous_coordinates, previous_state = step.current, state
    return SyntheticDataset(
        initial, true_initial, tuple(states), tuple(base_w), tuple(noises_w), tuple(means),
        tuple(base_v), tuple(noises_v), tuple(observations), tuple(inputs),
    )
