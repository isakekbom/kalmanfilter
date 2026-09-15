# Curvature-aware optimization and local likelihood conditioning (#25)

This study extends the deterministic baseline from [#10](baseline_optimization.md)
using the fixed-shape [scan execution from #23](fixed_scan_scaling.md). It separates
solver behavior, local likelihood curvature, and derivative/optimization cost.
The objective remains the same unconstrained raw-coordinate EKF negative
innovation log-likelihood, `f(r) = -ell(theta(r))`. Existing parameter transforms,
EKF/OIS mathematics and ragged drivers are unchanged. Section 3 noisy
optimization (#11) is not implemented; MATLAB/reference parity (#12) remains
blocked on reference material.

## Algorithms and stopping rules

For a local quadratic with SPD Hessian H, a gradient-descent error component in
eigendirection i contracts by `1 - alpha * lambda_i`. Stability requires
`0 < alpha < 2/lambda_max`; a small `lambda_min` can then make progress along a
flatter direction slow. This explains a possible *local* mechanism for slow GD.
A nonlinear likelihood can move between regions with different curvature, and
an indefinite Hessian does not have this SPD contraction interpretation.

Newton's ideal step is `p_k = -H_k^{-1} g_k`. The implementation **never forms
this inverse**. Newton-CG approximately solves the Newton system using products
`H_k v` and a line search. Trust-krylov approximately solves a quadratic
trust-region subproblem in a Krylov subspace using the same products, including
when local curvature is indefinite. BFGS is quasi-Newton: it learns a dense
curvature approximation from successive steps and gradient changes, using
O(p^2) storage. L-BFGS-B keeps a limited history (here ten pairs), using O(mp)
storage. It is used without bounds in raw coordinates.

| Method | Configuration in this study | Native stopping behavior |
| --- | --- | --- |
| GD | Fixed step `1e-4 * 24/T`, 200 iterations | Gradient 2-norm <= 1e-6 or iteration limit |
| BFGS | Existing `norm=2`, `gtol=1e-6`, `xrtol=0`, `c1=1e-4`, `c2=0.9` | Gradient norm or native failure |
| L-BFGS-B | Existing `gtol=1e-6`, `ftol=1e-12`, `maxcor=10`, `maxls=40`, `maxfun=20000`; no bounds | Projected gradient infinity norm or relative objective reduction |
| Newton-CG | Exact JAX `hessp`, `xtol=1e-8`, `c1=1e-4`, `c2=0.9` | Native step tolerance, not a common gradient tolerance |
| trust-krylov | Exact JAX `hessp`, `gtol=1e-6`, `inexact=True`, initial/max radius 1/1000, acceptance eta=0.15 | Gradient 2-norm or native failure |

All use `maxiter=200`. `step_tolerance` is a separate host option for Newton-CG;
it is not substituted for `gradient_tolerance`. The GD rate equals #10's exact
rate at T=24 and is scaled by dates elsewhere to keep its per-date step scale
fixed. All methods still evaluate the *full sum* NLL. No objective rescaling,
parameter rescaling or preconditioner is hidden in the comparison.

Original SciPy success flags, numeric statuses and messages are preserved.
A solver reporting success can have a gradient 2-norm above 1e-6. The result
retains both diagnostics, rather than redefining success. Official references:
[Newton-CG](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html),
[trust-krylov](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustkrylov.html),
and the [SciPy optimization tutorial](https://docs.scipy.org/doc/scipy/tutorial/optimize.html).

## Checked HVP interface and accounting

`CompiledObjective` in [optimization.py](../src/kalmanfilter/optimization.py)
retains its checked float64 combined value/gradient interface. It adds

```python
compiled = CompiledObjective(objective, initial_raw)
value, gradient = compiled.evaluate(raw)
product = compiled.hessian_vector_product(raw, direction)
result = run_optimization(compiled, initial_raw, method="trust-krylov")
```

The product is exact AD:

```text
g(r) = grad(f)(r)
H(r) v = jax.jvp(g, (r,), (v,))[1]
```

The gradient closure also checks the primal scalar value and its gradient.
The HVP adds a finite-product check and is compiled under `checkify`. Both raw
point and direction must be finite real vectors of the compiled shape, promoted
to float64. The direction is used as supplied, including magnitude and zero;
it is never normalized or clipped by the production interface. Results are
synchronized, errors thrown, and only then copied to host NumPy float64 arrays.
There are no finite differences, dense Hessians, inverses, jitter, penalties,
replacement values or retries in this path. JAX describes the
[forward-over-reverse construction](https://docs.jax.dev/en/latest/jacobian-vector-products.html).

HVP compilation is lazy. BFGS, L-BFGS-B and GD incur no HVP compilation or
evaluation. The first HVP call has its own `hvp_compilation_seconds`; a curvature
run warms it with the initial point and an all-ones direction if needed, before
starting run timing/counts. An explicitly prewarmed HVP is reused. A zero-
iteration run needs no HVP.

- `total_evaluations` counts actual combined value/gradient dispatches.
- `total_hvp_evaluations` counts actual HVP dispatches, including numerical
  failures. A primal gradient internally computed by the HVP is counted as part
  of that HVP, not as another combined dispatch.
- `OptimizationResult.hvp_evaluations` excludes shared first-call warmup and
  external diagnostics, like the existing per-run value/gradient counts.
- `solver_hessian_evaluations` preserves SciPy's `nhev` separately. It need not
  equal actual HVP calls: the locked trust-krylov implementation reports an
  additional Hessian evaluation even with only `hessp` supplied.
- Value/gradient callbacks retain the existing last-point cache. HVPs are not
  cached or supplied as a dense matrix. Each requested product differentiates
  the original objective again; it does not reuse a stored linearization.

Iteration records remain immutable, including initial and accepted/reported
iterates, objective, gradient norm, elapsed time and step norm. Results add HVP
counts and the shared HVP first-call cost. `MultistartResult` exposes that shared
cost too. First-call figures referenced by multiple runs must not be summed.
Numerical exceptions propagate from the optimizer. The benchmark catches them
only at the boundary of an individual run, records `RUN_FAILURE`, and proceeds
to other independently specified runs without retrying the failed run.

## Independent derivative validation

An analytical three-dimensional SPD quadratic checks value, gradient, dense
Hessian, HVP and known optimum against NumPy algebra. Several points, directions
and finite-difference steps are used. Newton-CG and trust-krylov must recover
its known optimum. Separate tests check invalid values, gradients and second
derivatives, direction validation, float64, JIT reuse, lazy compilation,
synchronization, separate call counts, immutable results and native status
preservation. The first-order regression suite retains its original behavior.

The real 12-date raw EKF objective is also checked with a dense JAX Hessian and
directional central differences of its gradient:

```text
D_h g(r; v) = [g(r + h v) - g(r - h v)] / (2h)
h in {1e-3, 1e-4, 1e-5, 1e-6}
```

Benchmark directions are deterministic unit vectors: all ones, alternating
signed coordinate indices, and a normal vector from NumPy seed 25. Relative
component errors use `abs(a-b)/max(abs(a),abs(b))`, with zero only when both
components are zero. No denominator floor hides small components. Reported
relative maxima may concern a different component from absolute maxima.

Dense-HVP acceptance is `rtol=2e-11, atol=2e-11`. Directional-FD acceptance is
`rtol=2e-7, atol=2e-8` at h=1e-5; all four step sizes are reported. Truncation
and subtraction cancellation explain why the smallest step need not be best.
The pure quadratic additionally uses analytical tolerances 2e-15 and FD
tolerances 2e-9. Benchmark validation precedes curvature optimization for each
problem, including the larger families and long-series true points.

## Diagnostic Hessians

Dense `jax.hessian` appears only in the benchmark diagnostic helper and tests.
All dimensions studied here are small enough (p<=24) for this diagnostic. It is
never passed to Newton-CG/trust-krylov. The original maximum elementwise symmetry
error is reported, then `(H+H.T)/2` is used solely for spectral analysis; this
does not change any likelihood or covariance.

For eigenvalues lambda, define `tau = 1e-8 * max(1, max(abs(lambda)))`.
Eigenvalues greater than tau are materially positive, below -tau negative, and
the remainder near zero. An SPD condition number `lambda_max/lambda_min` is
reported only when **every** eigenvalue exceeds tau. Otherwise it is null;
the signed spectrum is retained. A separately named absolute spectral ratio
uses the largest/smallest absolute *material* eigenvalues, excluding near-zero
ones, and is not an SPD condition number.

Reports include the full spectrum, extrema, sign counts, threshold, raw Hessian
diagonal, extreme eigenvectors and the smallest-eigenvalue vector's squared
mass in each free parameter block. Generating parameters, each representative
start and a converged BFGS solution are inspected. Local flatness, scaling,
coupling and negative curvature can be described; one local Hessian establishes
neither global convexity/optimality nor formal non-identifiability.

## Benchmark design and reproduction

```text
uv run --locked python benchmarks/curvature_optimization.py
uv run --locked --extra test pytest tests/test_curvature_optimization.py tests/test_optimization.py -v --basetemp .pytest_tmp -p no:cacheprovider
uv run --locked --extra test pytest --basetemp .pytest_tmp -p no:cacheprovider
git diff --check
```

The benchmark emits tagged JSON lines retaining full precision, fitted raw and
mathematical parameters, derivative errors, timings, native statuses/messages
and failures. `--case reference`, `long100`, `long1000`, `long5000`, `larger3` or
`larger6` reproduces a subset. Full output is saved under
`benchmarks/results/curvature_optimization.txt`.

| Case | T | p | n_x | n_z | Starts per method |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact #10 reference | 24 | 4 | 1 | 2 | 3 original starts |
| Long reference family | 100 | 4 | 1 | 2 | 2 original starts |
| Long reference family | 1000 | 4 | 1 | 2 | 2 original starts |
| Long reference family | 5000 | 4 | 1 | 2 | 1 original start |
| Larger family | 100 | 12 | 3 | 6 | 2 explicit starts |
| Larger family | 100 | 24 | 6 | 12 | 2 explicit starts |

All five methods run for each listed start. The T=5000 subset limits total study
cost while retaining all methods; it provides less evidence about start
sensitivity. No large Python-unrolled objective is compiled. Reference data
are generated once with seed 20261010; shorter cases use exact prefixes. The
T=24 anchor retains #10's mathematical setup, starts and first-order options,
but uses the already validated scan execution. Comparisons to #10 use its saved
rounded NLL/parameter values with explicit tolerances, not its historical timing
as a current-machine comparator.

The larger family uses seed 202625, n independent PCA transitions, 2n nonlinear
quotes and 4n free raw parameters: n persistence values, 2n measurement
variances, n loading coefficients. A quote loads on its own factor and a cyclic
neighbor with weights 0.15 (quote a) or 0.3 (quote b). Loading maps are diagonal
in the n `theta_g` coefficients; start/payment multipliers and accruals follow
the #10 two-quote construction. These explicit loadings couple parameter
directions while retaining the existing pricing kernel. Free parameters are
broadcast across dates only within each synthetic benchmark.

Generating persistence spans 0.8–0.94, measurement variances 0.0004–0.001,
and loadings 0.65–0.9. Process variances 0.0015–0.003, initial means 0.2–0.35
and initial covariance 0.0025*I are fixed. The two raw starts use persistence
offsets +0.2/-0.3, measurement-variance offsets linearly spanning -0.2 to +0.2
or +0.3 to -0.3, and loading offsets +0.05/-0.08. Optional unit-interval
persistence and fixed scale anchors remain explicit synthetic conventions;
ParameterLayout and unresolved Q4/Q9 semantics are unchanged.

Timing uses `perf_counter` and checked host calls that synchronize JAX before
returning. First checked value+gradient and HVP calls include JIT plus first
execution. Warmed timings are median/minimum of five calls at the generating
point; full optimizer timing includes initial evaluation, solver, conversions
and iteration logging, and excludes shared compilations and separate dense/FD
diagnostics. Each T/dimension gets its own specialization. These are sequential
process measurements, not isolated compiler-only or peak-memory measurements.

## Measured results

Measurements are added from the reproducible benchmark output after validation.

## Limits

This is deterministic synthetic estimation of the EKF innovation approximation,
not exact nonlinear marginal likelihood or market calibration. The larger p=24
case is still modest: BFGS matrix storage is tiny at these sizes, so observations
here cannot establish a large-p storage crossover. Failure messages and common
gradient norms matter alongside speed. Numerical checks remain strict at every
solver proposal; an invalid proposal can abort a run. No error is replaced by a
finite objective. Regime segmentation, padding, lifecycle inference, global
parameter tying, new financial conventions and noisy optimization are outside
this study.
