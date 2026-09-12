"""Innovation log-likelihood, PDF (57), reusing the EKF's Cholesky factors.

This evaluates the EKF Gaussian likelihood approximation, not an optimizer.
Compile fixed structures with ``jax.jit(checkify.checkify(function))`` and
inspect/throw the error outside JIT before using results. All numerical results
after any checkified failure are invalid and must be discarded. No repair or
penalty is substituted. See docs/likelihood.md for contracts and derivations.
"""

from collections.abc import Iterable, Iterator, Sequence
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from .ekf import EKFInputs, EKFStepResult, FilterState, MeasurementUpdate
from .ekf import ekf_step, initialize_filter


LOG_2PI = jnp.log(2 * jnp.asarray(jnp.pi, dtype=jnp.float64))


class InnovationLikelihood(NamedTuple):
    """One date's equation (57) terms; arrays are immutable JAX leaves.

    Innovation is (n_z,), L is (n_z,n_z) or None for n_z=0, and the three
    numerical terms are float64 scalars. The observation count is derived from
    the innovation shape, so it cannot disagree with that vector's dimension.
    """

    innovation: Array
    innovation_cholesky: Array | None
    log_determinant: Array
    quadratic_form: Array
    contribution: Array

    @property
    def n_observations(self) -> int:
        return self.innovation.shape[0]


class LikelihoodTrace(NamedTuple):
    """Matching per-date terms and existing EKF results, in sequence order.

    The terms reference each EKF update's innovation/factor; covariance arrays
    remain in the existing EKF result and are not copied into another record.
    """

    steps: tuple[InnovationLikelihood, ...]
    ekf_steps: tuple[EKFStepResult, ...]


class LikelihoodResult(NamedTuple):
    """Float64 total (), contributions (T,), and optional detailed trace."""

    total_log_likelihood: Array
    per_step_contributions: Array
    trace: LikelihoodTrace | None


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
    return _finite(array.astype(jnp.float64, copy=False), name)


def innovation_loglikelihood(
    innovation: ArrayLike, innovation_cholesky: ArrayLike | None,
) -> InnovationLikelihood:
    """One Gaussian innovation contribution from epsilon and its existing L.

    L must be finite lower triangular with positive diagonal, shape (n_z,n_z),
    and factor the covariance of the supplied innovation in the same order.
    No covariance is reconstructed or factored. The standalone boundary checks
    shapes, real values, finiteness, triangular structure, and positive diagonal;
    the production EKF already checks its own factor before reaching here.

    For n_z=0, require L=None and return three exact float64 zeros. This is an
    implementation convention for empty observations, not an explicit PDF rule.
    """
    epsilon = _real64(innovation, "likelihood innovation", 1)
    n_observations = epsilon.shape[0]
    if n_observations == 0:
        if innovation_cholesky is not None:
            raise ValueError("empty innovation requires innovation_cholesky=None")
        zero = jnp.zeros((), dtype=jnp.float64)
        return InnovationLikelihood(epsilon, None, zero, zero, zero)
    if innovation_cholesky is None:
        raise ValueError("nonempty innovation requires a Cholesky factor")
    factor = _real64(innovation_cholesky, "likelihood Cholesky factor", 2)
    if factor.shape != (n_observations, n_observations):
        raise ValueError("Cholesky factor shape must match the innovation dimension")
    checkify.check(jnp.all(factor == jnp.tril(factor)),
                   "likelihood Cholesky factor must be lower triangular")
    diagonal = jnp.diag(factor)
    checkify.check(jnp.all(diagonal > 0), "likelihood Cholesky diagonal must be positive")

    log_determinant = _finite(2 * jnp.sum(jnp.log(diagonal)), "log determinant")
    whitened = _finite(solve_triangular(factor, epsilon, lower=True), "whitened innovation")
    quadratic = _finite(jnp.dot(whitened, whitened), "innovation quadratic form")
    contribution = _finite(
        -0.5 * (n_observations * LOG_2PI + log_determinant + quadratic),
        "innovation log-likelihood contribution",
    )
    return InnovationLikelihood(epsilon, factor, log_determinant, quadratic, contribution)


def likelihood_contribution(update: MeasurementUpdate) -> InnovationLikelihood:
    """Evaluate (57) from a valid EKF update's existing innovation and factor.

    The caller must have checked any error returned with a precomputed update.
    This function cannot recover an earlier checkified error from numeric arrays.
    """
    return innovation_loglikelihood(update.innovation, update.innovation_cholesky)


def log_likelihood_from_trace(
    trace: Iterable[EKFStepResult], *, return_trace: bool = False,
) -> LikelihoodResult:
    """Sum contributions from valid EKF steps, without rerunning the filter.

    Accepts the trace from run_filter(..., return_trace=True), or an iterable
    of EKF step results. Sequence order is retained, including empty dates.
    An empty sequence has contributions shape (0,) and total zero. When trace
    retention is enabled, existing EKF results are referenced, never copied.
    """
    if trace is None:
        raise ValueError("an EKF trace is required; use run_filter(..., return_trace=True)")
    contributions = []
    terms = [] if return_trace else None
    ekf_steps = [] if return_trace else None
    for step in trace:
        likelihood = likelihood_contribution(step.update)
        contributions.append(likelihood.contribution)
        if terms is not None:
            terms.append(likelihood)
            ekf_steps.append(step)
    per_step = jnp.stack(contributions) if contributions else jnp.empty((0,), dtype=jnp.float64)
    total = _finite(jnp.sum(per_step), "total log-likelihood")
    full_trace = None if terms is None else LikelihoodTrace(tuple(terms), tuple(ekf_steps))
    return LikelihoodResult(total, per_step, full_trace)


def _forward_steps(initial: FilterState, inputs: Sequence[EKFInputs]) -> Iterator[EKFStepResult]:
    """Delegate initialization and every filtering equation to the existing EKF."""
    previous = initialize_filter(initial.coordinates, initial.state, initial.covariance)
    for item in inputs:
        step = ekf_step(previous, item)  # Also validates the exact coordinate chain.
        yield step
        previous = step.filtered


def run_likelihood(
    initial: FilterState, inputs: Sequence[EKFInputs], *, return_trace: bool = False,
) -> LikelihoodResult:
    """Run the existing EKF and accumulate (57) in an ordinary Python loop.

    Each date is filtered exactly once and its factor is reused immediately.
    Without trace retention, only scalar contributions and the current EKF step
    are retained by this driver; autodiff may retain intermediates as needed.
    Shapes/coordinates/instrument structures may change between dates. JIT is
    specialized to fixed input structures, with checkify applied first.
    """
    return log_likelihood_from_trace(_forward_steps(initial, inputs), return_trace=return_trace)
