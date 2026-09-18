# Baseline maximum-likelihood optimization

This document records the original #10 scope and measurements. The existing
methods remain available; [#25](curvature_optimization.md) adds HVP-based methods
and a separate conditioning study without changing these historical results.

[Issue #10](https://github.com/isakekbom/kalmanfilter/issues/10) adds baseline
optimization after the [full-gradient checks](gradient_validation.md) and
[synthetic validation](synthetic.md). The host orchestration lives in
[`optimization.py`](../src/kalmanfilter/optimization.py); it does not change
pricing, transitions, EKF, likelihood, parameter transforms, or generation.

The problem builder now lives in the shared configuration layer described in
[benchmark_config.md](benchmark_config.md); `make_problem(n_dates, seed)` is a
thin wrapper over the `reference` preset and reproduces the original builder
bitwise. The command-line flags documented there vary the series length, noise
levels, seed, starts and solver subset without changing the defaults below.

## Objective and methods

The estimation objective is

$$\min_r\operatorname{NLL}(r)=-\ell(\theta(r)).$$

The optimizer receives the **unconstrained raw vector** from `ParameterLayout`.
`raw_negative_log_likelihood` invokes the existing `unpack_parameters`, problem
builder, EKF, and innovation likelihood. Softplus coordinates represent
**variances**, not standard deviations. Reporting applies `unpack_parameters`
to the final raw vector; raw distances are secondary diagnostics.

Three methods are supported:

| Method | Explicit settings in the benchmark |
| --- | --- |
| BFGS | JAX gradient; `maxiter=200`, `gtol=1e-6`, `norm=2`, `xrtol=0`, `c1=1e-4`, `c2=0.9` |
| L-BFGS-B | JAX gradient; **no raw bounds**; `maxiter=200`, `gtol=1e-6`, `ftol=1e-12`, `maxcor=10`, `maxls=40`, `maxfun=20000` |
| GD | Plain `r_next = r - 1e-4 * gradient`; 200 iterations; gradient 2-norm tolerance `1e-6` |

BFGS tests the specified gradient norm; L-BFGS-B uses the infinity norm of
the projected gradient (the ordinary gradient without bounds), and can also
stop on relative objective reduction. See the official
[BFGS options](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html)
and [L-BFGS-B options](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
All reported gradient norms are 2-norms. Original SciPy success, numeric status,
and termination messages are retained; an objective-tolerance stop is not
relabeled as gradient convergence. GD has no line search, momentum, adaptation,
or repair. Its iteration limit is reported as non-success.

## Checked JAX / SciPy boundary

```python
def objective(raw):
    return raw_negative_log_likelihood(raw, layout, build_problem)

compiled = CompiledObjective(objective, starts[0])
results = run_multistart(compiled, starts, method="BFGS")
parameters = unpack_parameters(results.best.final_raw, layout)
```

The scalar callable closes over the static layout and problem builder.
`CompiledObjective` constructs JAX `value_and_grad`, adds finite-value/gradient
checks, functionalizes checks with `checkify`, and JIT-compiles the resulting
function. The objective must return a float64 scalar of shape `()`. Inputs are
validated real finite vectors, promoted to float64, and restricted to the
compiled shape. Mathematical model operations stay in JAX.

At the host boundary, each actual call synchronizes the checked result with
`jax.block_until_ready`, calls `error.throw()`, and only then converts the value
to a Python float and gradient to a NumPy float64 array. SciPy receives the
combined `(value, gradient)` callable with `jac=True`; it does not estimate
finite-difference gradients. This interface follows
[`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

Invalid objective proposals propagate as exceptions. There is no finite
penalty, zero fallback gradient, clipping, covariance jitter, or silent retry.
Multistart preserves ordinary solver non-success results but fails fast on a
checked numerical exception. Such a failed evaluation is never ranked as a
finite objective. Caller-selected interior starts make the controlled benchmark
valid; arbitrary objectives/starts are not guaranteed to finish.

SciPy **1.18.1** and NumPy **2.5.3** are now declared direct dependencies because
the optimizer imports them. Both were already resolved transitively in
`uv.lock`; its package versions and artifact hashes remain unchanged. JAX stays
at **0.11.1**. NumPy is permitted only in this host orchestration module and
test/example references, not in differentiable mathematical kernels.

## Timing, accounting, and immutable results

`CompiledObjective` times one valid first execution using `time.perf_counter()`.
Synchronization and checked validation complete before this timer stops.
`compilation_seconds` therefore includes tracing, compilation, dispatch, the
first execution, and boundary validation; it is **first-call/JIT time**, not
isolated compiler time. A run's `optimization_seconds` starts afterwards and
includes its initial evaluation, solver calls, host conversions, and logging.
Every JAX evaluation synchronizes before the timer can stop.

The same compiled object is reused for all methods and same-shape starts.
The compilation figure in each immutable `OptimizationResult` refers to that
same shared first call; these figures must not be summed across runs.

Each run keeps initial/final raw arrays, initial/final NLL, final gradient and
norm, iteration count, solver status/message, both evaluation counts, timings,
and a tuple of `IterationRecord` values. Records contain iteration number, raw
point, objective, gradient 2-norm, elapsed steady time, and raw step 2-norm.
Iteration 0 is the evaluated starting point. Its step norm is zero; its elapsed
time includes that initial evaluation. Raw arrays in results are immutable JAX
arrays. The compiled host handle has mutable counters and is for serial reuse.

A run-local cache stores the latest exact raw point, value, and gradient.
SciPy's initial request, accepted-iterate callbacks, and final reporting reuse
it when possible. A callback at a different point triggers a **counted** combined
evaluation. Returned gradients are copies so solver mutation cannot corrupt
the cache. `function_evaluations == gradient_evaluations` counts actual compiled
calls during that run, including its initial call and any logging cache misses.
SciPy's own `nfev/njev` are retained separately. GD has no solver-reported counts.

The first warmup call is excluded from run counts. This benchmark also evaluates
the generating truth once outside any run. Its overall count is **754**:
**752** run evaluations + **1** warmup + **1** truth diagnostic. In this run,
all SciPy callbacks reuse cached results, so wrapper and solver counts agree.
GD uses exactly 201 evaluations for its initial point and 200 steps.

`MultistartResult` retains every run independently. `best_index`/`best` identify
the lowest final NLL without discarding other runs or hiding their statuses.
Starts are supplied explicitly; the optimizer generates no random starts.

## Fixed synthetic estimation problem

The [benchmark](../benchmarks/baseline_optimization.py) generates **24 dates**
with seed **20261010** through `generate_synthetic_dataset`. There is one PCA
state `p`, no unsystematic/central-bank states, and two quotes per date. State
and observation dimensions remain fixed. The explicit maps give

$$x_t=\phi x_{t-1}+w_t,\qquad
z_t=(g_a(\theta_g,x_t),g_b(\theta_g,x_t))^T+v_t.$$

Each quote has one payment. For `qa`, accrual is 1 and PCA loading-map rows
are `(0,-1)`; for `qb`, accrual is 0.5 and rows are `(0.2,-1.5)`. Each row is
multiplied by the scalar `theta_g`, and central-bank loading blocks are empty.
Thus exact production pricing gives

$$g_a=\exp(\theta_g x)-1,\qquad
g_b=2[\exp(1.7\theta_g x)-1].$$

The generator computes these through `observation_quotes`; the equations here
explain the chosen inputs, not a second implementation. Observation noise has
independent named base coordinates and an identity selector into the two quotes.

| Mathematical quantity | Generating value | Estimated or fixed |
| --- | ---: | --- |
| Persistence `phi` | 0.9 | Estimated |
| Process variance | 0.0025 | **Fixed** |
| Observation variance `a` | 0.0004 | Estimated |
| Observation variance `b` | 0.0009 | Estimated |
| `theta_g` | 0.8 | Estimated |
| Initial mean | 0.35 | **Fixed distribution**, not sampled truth |
| Initial covariance | 0.0025 | **Fixed distribution** |

The layout is `ParameterLayout(n_f=1, n_w=0, n_v=2, n_x0=0, n_g=1,
theta_f_transform="unit_interval")`, with four raw coordinates. The builder
supplies the known initial `FilterState` and process variance; empty `n_w` and
`n_x0` blocks refer only to the estimated tuple, not zero process/state noise.
Fixing the process variance anchors the otherwise ambiguous scale between
latent fluctuations and pricing loadings. Fixing the initial distribution also
avoids estimating a distribution from one short trajectory. These are explicit
**synthetic benchmark conventions**, not assertions about known real-data inputs.

The optional unit-interval transition transform is likewise a **synthetic
optimization convention**. The generating persistence is strictly inside
`(0,1)`, and `pack_parameters` constructs the true raw vector. No hand-derived
logit, raw bounds, or package-wide transform change is used; the global default
remains identity. Fixed dimensions avoid introducing Q4 parameter tying.

The true raw vector is approximately `(2.19722458,-7.823846,-7.01266576,0.8)`.
The exact three starts are `pack_parameters(truth, layout)` plus these offsets,
in raw order `(phi, variance_a, variance_b, theta_g)`:

| Start | Raw offset | Initial NLL | Initial gradient 2-norm |
| --- | --- | ---: | ---: |
| 0, close | `(0.25,0.2,-0.2,0.1)` | -65.915597969 | 6.78672586 |
| 1, moderate | `(-0.8,0.8,-0.6,-0.2)` | -44.377498104 | 119.925039 |
| 2, larger | `(1,-1,1,0.3)` | -62.759064368 | 13.9406809 |

All methods receive exactly these starts and the same generated observations.

## Actual benchmark results

Measured on **2026-09-15**, Windows 11 CPU, CPython 3.12.14, JAX 0.11.1,
NumPy 2.5.3, and SciPy 1.18.1. The
[saved complete stdout](../benchmarks/results/baseline_optimization.txt) records
starts, timings, terminations, recovered parameters, and errors.

**Shared first-call/JIT time: 152.273925 seconds.** The unrolled 24-date checked
reverse-mode graph is expensive to compile. This cost is excluded from every
steady-state time below. No scan refactor or padding was introduced to alter
the existing model path. The smaller 12-date test case keeps CI compilation
cost lower.

Generating-truth NLL is **-65.952564789478**.

| Method | Start | Final NLL | Reduction | Gradient 2-norm | Iterations | Actual value / gradient calls | Steady seconds | Success/status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| BFGS | 0 | -66.733096254 | 0.817498285 | 4.90901917e-7 | 16 | 20 / 20 | 0.026932 | True / 0 |
| BFGS | 1 | -66.733096254 | 22.355598149 | 8.27453465e-7 | 24 | 31 / 31 | 0.042236 | True / 0 |
| BFGS | 2 | -66.733096254 | 3.974031886 | 4.38946353e-7 | 16 | 20 / 20 | 0.022388 | True / 0 |
| L-BFGS-B | 0 | -66.733096254 | 0.817498285 | 1.15435660e-5 | 21 | 25 / 25 | 0.044698 | True / 0 |
| L-BFGS-B | 1 | -66.733096254 | 22.355598149 | 6.85586818e-8 | 23 | 28 / 28 | 0.036241 | True / 0 |
| L-BFGS-B | 2 | -66.733096254 | 3.974031886 | 9.90008257e-7 | 21 | 25 / 25 | 0.035739 | True / 0 |
| GD | 0 | -66.292966491 | 0.377368522 | 2.61829700 | 200 | 201 / 201 | 0.238454 | False / 1 |
| GD | 1 | -61.323087969 | 16.945589865 | 9.46009397 | 200 | 201 / 201 | 0.219706 | False / 1 |
| GD | 2 | -65.233989521 | 2.474925154 | 7.19347354 | 200 | 201 / 201 | 0.202324 | False / 1 |

BFGS reports successful termination on its gradient criterion. L-BFGS-B start 0
reports `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`; its reported
2-norm is above `1e-6`, so this is explicitly **objective** convergence. Its
other starts report `CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`. All GD
runs report `Maximum iterations reached.` Solver counts for the six SciPy runs
equal the actual counts shown; GD solver counts are absent.

Recovered mathematical parameters, rounded for comparison:

| Method/start | Persistence | Observation variance a | Observation variance b | theta_g |
| --- | ---: | ---: | ---: | ---: |
| Generating truth | 0.90000000 | 0.00040000 | 0.00090000 | 0.80000000 |
| BFGS / 0 | 0.94351276 | 0.00039542 | 0.00029201 | 0.81475140 |
| BFGS / 1 | 0.94351276 | 0.00039542 | 0.00029201 | 0.81475140 |
| BFGS / 2 | 0.94351275 | 0.00039542 | 0.00029201 | 0.81475141 |
| L-BFGS-B / 0 | 0.94351280 | 0.00039542 | 0.00029201 | 0.81475155 |
| L-BFGS-B / 1 | 0.94351276 | 0.00039542 | 0.00029201 | 0.81475140 |
| L-BFGS-B / 2 | 0.94351276 | 0.00039542 | 0.00029201 | 0.81475141 |
| GD / 0 | 0.92160419 | 0.00046680 | 0.00073246 | 0.83458754 |
| GD / 1 | 0.83640567 | 0.00079064 | 0.00049148 | 0.91711789 |
| GD / 2 | 0.96030295 | 0.00015434 | 0.00251560 | 0.88862697 |

Process variance remains **0.0025** in every row. At the shared fitted solution,
absolute errors are approximately **0.04351276** in persistence,
**0.01475140** in `theta_g`, and **4.5794e-6 / 6.0799e-4** in observation
variances. The variance relative errors are about **1.145% / 67.554%**.

BFGS and L-BFGS-B reach the same reported minimum from all three starts, with
small gradients. GD improves the objective but remains slower to converge with
this fixed learning rate; its poor stopping diagnostics are retained. These
measurements describe this small controlled problem, not a universal speed ranking.

The fitted NLL is about **0.78053146 lower** than the generating-truth NLL.
Finite-sample MLE need not equal the generating parameter, so this is expected
and is not evidence of a model bug. Persistence, the first observation variance,
and the loading scale are reasonably recovered here. The second observation
variance is poorly recovered. Its strong movement when using only 12 dates
(below) is consistent with limited information about that component; this is
not a formal identifiability or uncertainty analysis. No tolerance forces that
estimate to equal truth. The nonlinear objective is still the EKF Gaussian
innovation approximation rather than an exact nonlinear marginal likelihood.

## Tests and reproduction

Run the benchmark and focused tests from the repository root:

```text
uv run --locked python benchmarks/baseline_optimization.py
uv run --locked --extra test pytest tests/test_optimization.py -v --basetemp .pytest_tmp -p no:cacheprovider
uv run --locked --extra test pytest --basetemp .pytest_tmp -p no:cacheprovider
```

Validation completed with **40 focused tests passed** and **436 full-suite
tests passed**, up from the verified merged-main baseline of 396.
An analytical positive-definite
quadratic is evaluated in centered form to avoid subtractive cancellation near
its optimum. BFGS and L-BFGS-B must recover the known solution with small
gradients. Separate tests cover caching and deliberately extra callback calls,
solver failure status, actual dispatch counts, one trace across starts/methods,
synchronization before timing stops, fixed-step GD, immutability, input precision,
checked invalid proposals, and absence of second-order/research functionality.

The end-to-end tests use the same construction/seed with **12 dates**, the first
two starts, and the same four estimated raw coordinates. Both methods reach
NLL **-32.1094101371**, versus **-29.7901255322** at generating truth and initial
NLLs **-30.3111181410 / -12.5817839801**. Final BFGS gradient norms are
**1.4072e-7 / 4.2475e-7**; L-BFGS-B gives **1.0450e-6 / 1.1593e-8**.
The fitted parameters are approximately `phi=0.98403341`, observation variances
`(0.00038391,0.00322639)`, and `theta_g=0.71793493`. From the moderate start,
persistence, `theta_g`, and the first observation variance move closer to their
generating values. The second variance is not constrained by a false recovery
assertion. Tests check objective reduction, a gradient reduction by at least
1000, finite/admissible mathematical parameters, and agreement of final NLLs
within `1e-7`; they make no wall-clock assertions.

No prerequisite mathematical bug was found. The existing source guard was
narrowed only to permit NumPy in the explicitly requested optimizer boundary;
inverse and callback bans still apply to all production modules.

## Limitations

This is synthetic calibration only: no real-data calibration, MATLAB/reference
parity, Hessian conditioning study, or PDF section 3 noisy optimization is
implemented. Issue #11 remains future work; #12 remains blocked by missing
reference material. Known initial/process distributions, loading construction,
unit-interval persistence, and independent noise are benchmark conventions.
The experiment establishes neither global convexity nor global optimality.
It gives no claim that compilation cost, runtime, conditioning, or recovery
scales unchanged to 20 years of data.
