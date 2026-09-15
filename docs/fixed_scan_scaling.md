# Fixed-shape EKF likelihood execution (#23)

The historical issue #10 benchmark measured **152.273925 seconds** for the first
checked JIT value/gradient call at only 24 dates, followed by millisecond-scale
optimization evaluations. That number is retained in
[the original results](../benchmarks/results/baseline_optimization.txt); it is
not a fresh measurement here. The ordinary Python time loop is unrolled during
JAX tracing. The explicit scan path stages one time-step body and iterates it for
the supplied segment length. JAX documents this lowering to a single loop in
[`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html).
Compilation still depends on shapes, checks and backend optimizations; no exact
constant-time compilation claim is made.

`run_filter` and `run_likelihood` remain the authoritative general drivers and
are unchanged. They support changing coordinate identities and dimensions,
active observation sets, and instrument payment structures. Call
`run_fixed_scan_likelihood` explicitly for a fixed structural segment. There is
no automatic routing between the two paths.

## Fixed segment contract

Within one segment, all dates must have identical:

- Previous and current `StateCoordinates`, including identities, order, PCA,
  central-bank-step and unsystematic block allocation. Previous must equal
  current, so a single date that changes coordinates is also unsupported.
- Active observation identities, count and order; the fixed instrument tuple
  follows that order.
- All five structural maps (`A`, `B`, `D`, `I^z`, `G`): map representation,
  row/column identities, source topology and **numerical weights/values**.
  Equal copies are accepted; changed weights fail even if shapes agree.
- Covariance representation (diagonal or dense), parameter/noise dimensions,
  instrument count, and every corresponding numerical leaf shape.
- Payment count `K_i` for each corresponding instrument. Different instruments
  may have different `K_i` in the same segment.

`stack_fixed_inputs` validates this contract and stacks a nonempty sequence once
during setup. It never silently ignores later structural steps. It rejects
changing observation counts, including mixtures of empty and nonempty dates.
An entirely empty observation segment is supported and performs prediction-only
steps with zero likelihood contributions.

There is no padding, NaN masking, segmentation, central-bank lifecycle inference
or global parameter-tying scheme. Numerical map weights cannot vary with time
in this first implementation. Numerical parameters, covariance entries,
observations, accrual factors and instrument loadings may vary explicitly in
batched arrays.

## Representation and execution

`FixedScanInputs` stores one unbatched `StructuralStep` and immutable JAX arrays:

| Quantity | Shape |
| --- | --- |
| `theta_f`, `theta_g` | `(T,n_f)`, `(T,n_g)` |
| `observations` | `(T,n_z)` |
| Diagonal `sigma_w.diagonal`, `sigma_v.diagonal` | `(T,n_w)`, `(T,n_v)` |
| Dense base covariances | `(T,n_w,n_w)`, `(T,n_v,n_v)` |
| Instrument `i` accrual factors | `(T,K_i)` |
| Instrument `i` PCA loading map | `(T,K_i+1,n_p,n_g)` |
| Instrument `i` step loading | `(T,K_i+1,n_c)` |

The covariance wrapper retains its ordinary `DiagonalMatrix` tree structure.
Its matrix `.shape` property describes an ordinary unbatched matrix and should
not be used on the batched wrapper; inspect `.diagonal.shape` instead. Scan
slices that leaf before passing it to the existing mathematical kernels.
Directly constructed `FixedScanInputs` must satisfy the same contract, with
leading time axes on all numerical leaves. There is only one structural step
to supply in that representation.

The scan carry contains filtered mean and covariance only. The body reconstructs
one numerical `EKFInputs`, calls existing `ekf_step`, and passes its update to
existing `likelihood_contribution`. OIS pricing, Jacobians, prediction, update,
and likelihood equations (38)–(57) have no second implementation here. Existing
innovation Cholesky factors are reused for the likelihood as before.

`FixedScanLikelihoodResult` contains `total_log_likelihood`,
`per_step_contributions`, `final` (`FilterState`), and optional `trace`.
`FixedScanTrace` holds predicted/filtered states and covariances, innovations,
innovation covariances, and likelihood contributions with leading time axes.
Static objects are not repeated along time. The default `return_trace=False`
emits only contributions alongside the final carry. Reverse-mode autodiff may
still retain intermediate residuals; disabling the diagnostic trace does not
imply constant-memory differentiation.

```python
import jax
from jax.experimental import checkify
from kalmanfilter.fixed_scan import stack_fixed_inputs, run_fixed_scan_likelihood

batch = stack_fixed_inputs(ekf_inputs)  # Once, outside JIT and differentiation.
checked = jax.jit(checkify.checkify(run_fixed_scan_likelihood))
error, result = checked(initial_filter, batch)
error.throw()  # Discard every output on failure.
log_likelihood = result.total_log_likelihood
```

For optimization, transform raw parameters once and broadcast explicitly shared
parameters into the batch with `jnp.broadcast_to`. See `make_scan_objective` in
[the benchmark](../benchmarks/fixed_scan_scaling.py). Its closure captures only
the batched representation; it never rebuilds a tuple of T per-date inputs.
The example fixes process variance and the initial distribution and estimates
persistence, two measurement variances and `theta_g`, exactly as in #10.
`unit_interval` persistence is only that synthetic convention. ParameterLayout
defaults and the unresolved global parameter question are unchanged.

## Checked numerical behavior

Array conversion preserves the existing real-valued float64 contract and checks
finite batched inputs. Existing kernel checks still reject invalid variances,
asymmetric supplied covariances, nonfinite pricing and invalid innovation
Cholesky factors. There are no repairs, jitter, clipping, penalties, zero-gradient
fallbacks or retries.

Scan stages its body even when called eagerly. The public function therefore
functionalizes its checks internally and reinjects the resulting error using
[`checkify.check_error`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.checkify.check_error.html).
Direct calls raise; an outer `jax.jit(checkify.checkify(...))` returns the usual
error/result pair, including when wrapped around `jax.value_and_grad`. Bare JIT
without outer checkification is not the checked API. The existing
`CompiledObjective` supplies the required outer checked value/gradient boundary.
All outputs remain invalid on any checked failure.

## Correctness validation

The new tests compare against the unchanged Python likelihood at T=1,5,24,100.
The main case has two PCA factors, one central-bank-step factor, one
unsystematic factor, two nonlinear quotes with different payment counts,
coupled transitions, and date-varying parameters, noise and instrument leaves.
Both diagonal noise and dense correlated noise are covered. Tests compare every
required trace array, every contribution and the final filtering distribution.
They also cover structural incompatibilities, setup order, float64, empty
observations and eager/compiled checked failures.

Trace and objective tolerances are `rtol=2e-12, atol=2e-13`. Raw gradient parity
uses `rtol=2e-11, atol=2e-12` on the #10 model at 12 dates, at the generating raw
vector and three perturbed vectors. These permit float64 rounding from staged
operation ordering and differentiation while remaining far below 1e-4. The
independently validated Python gradient from #8 is the reference; the full
finite-difference investigation is not repeated.

Two identical BFGS starts at T=12 use `maxiter=200`, `gtol=1e-6`. Both paths must
report success and gradient norm below the stopping tolerance. Final NLLs must
agree within absolute `2e-10`; fitted mathematical parameters must agree with
`rtol=2e-7, atol=2e-9`. Solver tolerances are distinct from pointwise derivative
tolerances because termination can occur at slightly different iterates.

The architecture test finds exactly one time scan, `unroll=1`, and equal outer
and body JAXPR equation counts at T=1 and T=100. It also checks the actual
benchmark objective for Python time loops/comprehensions and verifies it runs
without invoking the setup stacker. Existing ragged tests and source modules
remain unchanged. There are no wall-clock assertions in pytest.

Measured discrepancies and optimizer results are recorded below after the
benchmark completes.

## Reproduction and measured scaling

```text
uv run --locked --extra test pytest tests/test_fixed_scan.py -v --basetemp .pytest_tmp -p no:cacheprovider
uv run --locked --extra test pytest --basetemp .pytest_tmp -p no:cacheprovider
uv run --locked python benchmarks/fixed_scan_scaling.py
git diff --check
```

The benchmark uses seed 20261010 and generates 5000 genuine dates with the
existing synthetic generator once. Shorter runs use prefixes. Generation and
stacking are setup work and excluded from timings. The sole newly compiled
Python-loop value/gradient reference uses T=12; no long Python-loop gradient is
compiled. Plain scan is measured at T=12,24,100,500,1000,5000.

`CompiledObjective` times first checked `jax.value_and_grad` dispatch using
`perf_counter` and synchronizes with `block_until_ready`. The benchmark reports
the median and minimum of ten subsequent checked calls, each synchronized by
`evaluate` before stopping its timer. These include host conversion and error
handling. The first-call figure includes tracing, compilation and one
evaluation; it is not a compiler-only measurement. Each distinct T has its own
compiled specialization. Earlier trace parity checks may warm lower-level JAX
caches; numbers describe the reported process, not isolated fresh processes.

Exact stdout is saved in
[benchmarks/results/fixed_scan_scaling.txt](../benchmarks/results/fixed_scan_scaling.txt).
Timing and numerical result tables will be populated from that output.

## Limits

This establishes a fixed-shape primitive on modest state/observation dimensions,
not calibration on market data or large-matrix scalability. Reverse-mode memory
usage, backend and device behavior, and structural regime segmentation require
separate investigation. No checkpoint/rematerialization option is added unless
a concrete long-series failure makes it necessary. The general Python driver
remains the route for changing structure. #11 stays future research work; #12
remains blocked on MATLAB/reference material.
