# Innovation log-likelihood

This implements [issue #6](https://github.com/isakekbom/kalmanfilter/issues/6),
equation (57) on page 5 of [kalmanRante.pdf](../kalmanRante.pdf), in
[`likelihood.py`](../src/kalmanfilter/likelihood.py). It uses the existing
[EKF](ekf.md) from issue #5 without changing its recursion or computing another
Cholesky factorization. The mathematical specification is
[model_spec.md §7](model_spec.md#7-innovation-likelihood-equation-57).

This issue implements the **log-likelihood, not an optimizer**. Maximizing
`log_likelihood` is equivalent to minimizing `-log_likelihood`; no minimizer,
parameter packing/transforms, penalty policy, or optimization behavior is added.

## Equation (57) and its inputs

For the prediction innovation $\epsilon_t=z_t-\widehat z_t$ and its covariance
$S_t=H_tP_t^{t-1}H_t^T+R_t$, the PDF gives

$$
l=-\frac12\sum_{t=1}^T\left[
n_t^z\log(2\pi)+\log|S_t|+\epsilon_t^TS_t^{-1}\epsilon_t
\right]. \tag{57}
$$

The observation count $n_t^z$ is local to each date; it is not the state count.
The innovation is computed from the prediction, **before** the measurement
update. A residual recomputed at the filtered state would describe a different
quantity. Initialization affects all subsequent predictions, but (57) adds no
separate initial-state density or time-zero observation term.

Under a linear-Gaussian model and the required joint noise assumptions this is
the Gaussian innovation log-likelihood. For the nonlinear OIS model, it is the
**EKF Gaussian approximation** to the observation likelihood; this code does not
claim to evaluate the exact nonlinear marginal likelihood. The joint noise
questions in Q7 and estimation/initial-condition choices in Q9 remain open.

## Cholesky algebra and numerical evaluation

Issue #5 supplies `update.innovation`, `update.innovation_covariance`, and
`update.innovation_cholesky`, satisfying $S_t=L_tL_t^T$ for valid nonempty steps.
The likelihood consumes the existing innovation and lower factor directly.

The diagonal product identity for triangular matrices gives
$|L_tL_t^T|=\prod_i L_{t,i,i}^2$, hence

$$
\log|S_t|=2\sum_i\log L_{t,i,i}.
$$

This identity is also stated immediately after equation (57) in the PDF.
The implementation sums logarithms; it never computes the product or the
determinant. In particular, it does not call `det` or `slogdet`.

For the quadratic term, let $L_ty_t=\epsilon_t$. Then

$$
\epsilon_t^TS_t^{-1}\epsilon_t
=\epsilon_t^TL_t^{-T}L_t^{-1}\epsilon_t
=y_t^Ty_t=\|y_t\|^2.
$$

The code uses this derived algebra:

```text
log_determinant = 2 * sum(log(diag(L)))
y = solve_triangular(L, innovation, lower=True)
quadratic_form = dot(y, y)
contribution = -0.5 * (n_observations * LOG_2PI + log_determinant + quadratic_form)
```

`LOG_2PI` is computed with JAX in float64. No explicit inverse, pseudo-inverse,
solve using S, second factorization, or epsilon floor inside `log` is used.
The quadratic solve is separate from the EKF gain solve but reuses the same L.
Log densities may be positive; contributions are not capped at zero.

## API and shapes

| Function | Inputs | Result |
| --- | --- | --- |
| `innovation_loglikelihood(innovation, innovation_cholesky)` | Innovation `(n_z_t,)`, lower L `(n_z_t,n_z_t)` or None when empty | `InnovationLikelihood` with one date's terms |
| `likelihood_contribution(update)` | Existing valid `MeasurementUpdate` | Same one-date result, using its existing innovation and L |
| `log_likelihood_from_trace(trace, return_trace=False)` | Iterable of valid `EKFStepResult`, for example `run_filter(..., return_trace=True).trace` | `LikelihoodResult`; no filtering or factorization |
| `run_likelihood(initial, inputs, return_trace=False)` | Initial `FilterState`, sequence of `EKFInputs` | `LikelihoodResult` after one forward EKF pass |

All result containers are immutable named tuples and JAX pytrees. Numerical
arrays are immutable JAX arrays. The per-step `InnovationLikelihood` contains:

| Field / property | Shape | Meaning |
| --- | --- | --- |
| `n_observations` | Python integer from static shape | `innovation.shape[0]`, avoiding a separately supplied inconsistent count |
| `innovation` | `(n_z_t,)` | Existing EKF prediction innovation, in active observation order |
| `innovation_cholesky` | `(n_z_t,n_z_t)` or None | Existing lower factor, not a new copy of S |
| `log_determinant` | `()` | $2\sum_i\log L_{ii}$ |
| `quadratic_form` | `()` | Squared norm of the triangular-solve result |
| `contribution` | `()` | One summand of (57), including the minus one-half and dimension term |

The full `LikelihoodResult` contains `total_log_likelihood` (float64 scalar),
`per_step_contributions` (float64 `(T,)`), and `trace` (None by default).
The total is computed directly as `jnp.sum(per_step_contributions)`.

When `return_trace=True`, `trace` is a `LikelihoodTrace` with matching tuples:

- `trace.steps[t]` contains the one-date likelihood terms.
- `trace.ekf_steps[t]` references the corresponding existing `EKFStepResult`,
  including coordinate identities, active observations, S, and filtered values.

The likelihood terms reference the EKF's innovation/factor arrays. Large state
and covariance matrices stay in the existing EKF object; the likelihood does
not copy them into another record. With trace retention disabled, the driver
keeps scalar contributions and the current filter step rather than the full
EKF history. Autodiff may separately retain intermediates required for gradients.

## Empty dates, order, and changing dimensions

For `n_z_t == 0`, the existing EKF performs prediction only and supplies L=None.
The likelihood defines `log_determinant`, `quadratic_form`, and `contribution`
to be exact float64 scalar zero, and retains the empty innovation `(0,)`.
No solve or factorization is attempted. The standalone API requires L=None for
empty innovations, matching the EKF representation.

This is an **implementation convention**, consistent with a zero-dimensional
Gaussian contributing density one and log density zero. The PDF does not
explicitly discuss all-missing dates. Such a date still propagates the state
and covariance, and can therefore change later nonempty contributions. It is
not equivalent to skipping that date's prediction. An empty sequence yields
contributions `(0,)`, total zero, and empty trace tuples if requested; the
forward path still validates the supplied initial state.

The forward driver delegates initialization to `initialize_filter` and each
date to `ekf_step`. It only carries the returned filtered state and accumulates
the likelihood terms. It neither duplicates nor refactors EKF mathematics.
The existing step validates previous/current coordinate continuity, including
order, at each date. Rectangular transitions, changing observation counts, and
changing instrument/payment structures retain the issue #4/#5 contracts.

`step.active_observations` continues to define row order. The likelihood uses
the innovation/factor pair exactly in that order; it does not sort observations
or infer missingness from their values. An observed zero remains a measurement.

## Validation, failures, and JAX differentiation

Real numerical inputs are promoted to float64, matching the existing layers;
Boolean and complex inputs are rejected. Converting float32 cannot recover
precision lost before the call. The standalone innovation boundary checks
ranks, shape agreement, finite values, exact lower-triangular structure, and
strictly positive Cholesky diagonal entries. The production EKF already
validates S and its factor. No S reconstruction is done to verify a supplied
factor: its relationship to the covariance of the supplied innovation, in the
same order, is the standalone caller's precondition.

All numerical checks use `checkify`, including finite computed log determinant,
whitened innovation, quadratic form, contribution, and total. Invalid EKF
innovation covariance failures propagate through the full likelihood path.
Finite but poor likelihood values are returned as computed. Numerical failure
is not replaced with jitter, clipping, a finite penalty, or a silent `-inf`
fallback. Extreme arithmetic can still overflow float64 and is reported as a
checked failure. How a future optimizer reacts remains outside this issue/Q10.

Static shape/structure errors raise `ValueError` or `TypeError`. Numerical
checks raise `checkify.JaxRuntimeError` eagerly. For compiled execution apply
checkify before JIT and check the returned error outside the compiled function.
**All numerical outputs after any checkified failure are invalid and must be
discarded.** Precomputed EKF traces must already have passed their own error
check: a likelihood function cannot recover an earlier error object from arrays.

Production numerical code uses JAX, `jax.numpy`, and `jax.scipy` only. There are
no NumPy/SciPy host operations, Python float conversions, `.item()` calls,
callbacks, or stopped gradients. Differentiation flows through OIS pricing and
its state Jacobian, transition and covariance propagation, S, Cholesky, the
triangular solves, and the sum. To differentiate model parameters, run the
forward computation inside the differentiated function; a detached precomputed
trace cannot recreate dependence on parameters no longer in the computation.

The driver uses an ordinary Python loop through an iterator of existing EKF
steps. There is no `lax.scan`, padding, or fixed master state. A compiled kernel
is specialized to shapes and static metadata. A fixed finite sequence can be
traced/unrolled for a small gradient check, but changing its length, dimensions,
or instrument structure may require recompilation. This baseline does not
promise a single compilation or an efficient compiled graph for arbitrary
time-varying sequences.

## Example

This deterministic example uses caller-supplied algebraic inputs. It introduces
no market conventions or parameter transforms:

```python
import jax
import jax.numpy as jnp
from jax.experimental import checkify
from kalmanfilter.ekf import EKFInputs, initialize_filter
from kalmanfilter.likelihood import run_likelihood
from kalmanfilter.ois import OISInstrument
from kalmanfilter.transition import (
    StateCoordinates, StructuralStep, DiagonalMatrix, selection_map,
)

coordinates = StateCoordinates(("level",), (), ())
identity = selection_map(coordinates.all, coordinates.all, coordinates.all)
step = StructuralStep(
    coordinates, coordinates, identity, identity, identity,
    selection_map(("quote",), (), (None,)),
    selection_map(("quote",), ("noise",), ("noise",)),
)
instrument = OISInstrument(
    jnp.array([1.0]), jnp.array([[[0.0]], [[-1.0]]]), jnp.empty((2, 0)),
)
initial = initialize_filter(coordinates, [0.03], [[0.01]])
inputs = EKFInputs(
    step, jnp.array([0.8]), DiagonalMatrix(jnp.array([0.003])),
    jnp.array([0.7]), DiagonalMatrix(jnp.array([0.01])),
    jnp.array([0.025]), (instrument,),
)
result = run_likelihood(initial, (inputs,), return_trace=True)
assert result.per_step_contributions.shape == (1,)
assert result.trace.steps[0].n_observations == 1

def total(theta_g):
    return run_likelihood(
        initial, (inputs._replace(theta_g=theta_g),),
    ).total_log_likelihood

error, (value, gradient) = jax.jit(checkify.checkify(jax.value_and_grad(total)))(
    jnp.array([0.7]),
)
error.throw()  # Before consuming value or gradient.
assert value.dtype == gradient.dtype == jnp.float64
```

## Source status and scope

| Status | Content |
| --- | --- |
| Directly from the PDF | Equation (57), current observation dimension, Gaussian innovation terms, and the Cholesky log-determinant identity following (57). |
| Derived algebra | Quadratic form as the squared norm of a triangular solve; maximizing l equivalent to minimizing -l; exact Gaussian interpretation in the linear case and EKF approximation for nonlinear OIS. |
| Implementation convention | Reusing the existing factor and trace objects, immutable result layout, float64/checkify validation, explicit zero contribution for empty dates, failure without repair, and Python-loop accumulation. |
| Still unresolved | Joint noise assumptions (Q7), financial/coordinate construction and parameter-sharing questions, estimation/initial-condition choices (Q9), and broader admissibility/optimizer failure policy (Q10). |

Full-vector gradient validation is implemented separately in
[issue #8](gradient_validation.md); this issue includes only a small
single-parameter consistency test. Parameter transforms, optimizer
algorithms, Hessians, benchmarks, synthetic time-series generation, MATLAB
parity, and the noisy-optimization draft remain outside scope.

## Tests

[`tests/test_likelihood.py`](../tests/test_likelihood.py) covers hand-computable
scalar terms, independent multivariate NumPy `slogdet`/`solve` references,
extreme scales, exact EKF factor reuse, total/per-date agreement, empty dates,
changing state/observation dimensions, declared observation order including
zero, immutable traces without covariance copies, checked JIT, one-parameter
finite-difference agreement against an independent scalar nonlinear EKF,
invalid factors/innovation covariance, arithmetic overflow, and source guards
against refactorization, inverse/determinant calls, or host conversions.

Run the entire suite on this Windows checkout:

```text
uv run --locked --extra test pytest --basetemp .pytest_tmp -p no:cacheprovider
```
