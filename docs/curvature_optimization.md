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

The complete deterministic benchmark finished all requested cases and runs with
`noisy_optimization=false`. Results below are from the documented Windows/CPU,
float64 environment and should be interpreted as measurements of this benchmark,
not as market-calibration claims.

### Derivative correctness and #10 regression

The HVP implementation agreed with independently formed dense JAX
Hessian-vector products across all benchmark families. Maximum absolute
dense-HVP discrepancies were on the order of `1e-13` to `1.8e-12`; the
directional finite-difference check at `h=1e-5` also passed the documented
tolerances in every case. This supports using the HVP path as an exact-AD
second-order interface rather than as a finite-difference approximation.

The exact `T=24`, `p=4` reference problem preserved the historical #10
first-order result. Across the nine BFGS/L-BFGS-B/GD regression runs, the
maximum absolute NLL difference from the saved #10 output was
`4.78e-10`, and the maximum absolute fitted-parameter difference was
`4.73e-9`. The scan/HVP work therefore did not materially change the previous
baseline result.

### Derivative cost and time-series scaling

Warm checked value+gradient and HVP costs increased with the number of dates,
while first-call JIT costs remained separate from optimizer timing.

| Problem | T | p | Warm value+gradient | Warm HVP |
| --- | ---: | ---: | ---: | ---: |
| reference | 24 | 4 | 1.56 ms | 1.98 ms |
| reference | 100 | 4 | 2.80 ms | 6.02 ms |
| reference | 1000 | 4 | 24.3 ms | 54.5 ms |
| reference | 5000 | 4 | 122 ms | 266 ms |
| larger_n3 | 100 | 12 | 10.1 ms | 20.9 ms |
| larger_n6 | 100 | 24 | 15.2 ms | 36.5 ms |

The fixed-shape scan therefore keeps a 5000-date value+gradient evaluation
practical on CPU. An HVP costs roughly two to somewhat more than two
value+gradient evaluations in these cases, so curvature-aware methods must
recover that extra per-call cost through better optimization progress.

### Reference-family optimizer behavior

On the `T=24` anchor, BFGS, L-BFGS-B, Newton-CG and trust-krylov reached the
same local optimum to numerical precision from the documented starts. The
curvature methods often ended with smaller gradient norms, but required HVP
work. Plain fixed-step GD remained a diagnostic baseline rather than a
competitive solver.

The longest reference case (`T=5000`, `p=4`) makes the trade-off clearest:

| Method | Final NLL | Final gradient 2-norm | Iterations | Value+gradient calls | HVP calls | Warm optimization time | Native status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BFGS | -14572.801234 | 6.81e-8 | 12 | 18 | 0 | 2.15 s | success |
| L-BFGS-B | -14572.801234 | 1.54e-5 | 15 | 19 | 0 | 2.43 s | success by relative objective reduction |
| GD | -14544.566779 | 4.33e2 | 200 | 201 | 0 | 25.49 s | iteration limit |
| Newton-CG | -14572.801234 | 6.60e-10 | 11 | 13 | 29 | 9.15 s | success |
| trust-krylov | -14572.801234 | 2.97e-6 | 10 | 14 | 39 | 11.99 s | native status 2 |

BFGS, L-BFGS-B, Newton-CG and trust-krylov therefore agreed on the attained
objective, but their stopping diagnostics differed materially. In particular,
L-BFGS-B reported native success although its common gradient 2-norm remained
above `1e-6`, while trust-krylov reported a native failure message despite
reaching the same NLL with a small gradient. This is why native solver status,
objective value and a common gradient diagnostic are all retained.

GD was still far from the common solution after 200 iterations and took longer
than the other methods in this case. The result supports the original use of GD
as a conditioning diagnostic, not as the main estimator.

### Local curvature in the reference family

The local Hessian diagnostics show that poor or strongly varying curvature is a
plausible explanation for slow fixed-step GD. At `T=24`, the generating-point
SPD condition number was about `406`. Representative starts ranged from about
`165` to `9756`, and one start was locally indefinite with one materially
negative eigenvalue. At the converged BFGS solution the SPD condition number
was about `6005`; its smallest-curvature eigenvector was almost entirely in the
measurement-variance (`sigma_v`) block.

For the converged BFGS solutions at longer reference prefixes, the reported SPD
condition numbers were approximately `1250` at `T=100`, `109` at `T=1000`, and
`131` at `T=5000`. Thus increasing `T` did not make local conditioning
monotonically worse in this seeded synthetic family. These are local,
sample-specific Hessian measurements and should not be interpreted as a general
law that longer samples improve conditioning.

### Larger parameter problems

The `p=12` and `p=24` cases exposed behavior that is not visible in the
four-parameter reference problem.

For `larger_n3` (`T=100`, `p=12`), both BFGS runs reported success with final
gradient norms below `1e-6` and NLL near `-895.95307929`. L-BFGS-B reached
essentially the same objective and reported native success by relative objective
reduction, but its final gradient norms were about `1.5e-4` and `2.6e-4`.
GD remained far from stationarity after 200 iterations. Newton-CG and
trust-krylov reached essentially the same objective with gradient norms down to
approximately `1e-8`--`1e-7`, but required hundreds of HVPs and about
`5`--`6.5` seconds per run. The converged BFGS Hessian contained one near-zero
eigenvalue (`~9.8e-13`), with that direction entirely in the `sigma_v` block.
One fitted measurement variance was simultaneously driven extremely close to
zero.

For `larger_n6` (`T=100`, `p=24`), both BFGS runs again reported success,
reaching NLL `-1764.6018036` with gradient norms below about `5.1e-7` in
roughly `1.1`--`1.3` seconds. In contrast, both L-BFGS-B runs reached the
200-iteration limit. Their NLL values (`-1764.60017` and `-1764.59937`) were
close to the BFGS value, but their gradient norms (`0.072` and `0.213`) showed
that they had not reached the same stationary-point tolerance. GD also remained
far from stationarity.

Newton-CG and trust-krylov recovered essentially the same `p=24` objective as
BFGS from both starts. Newton-CG used `589`--`660` HVPs and took about
`22.9`--`23.5` seconds; trust-krylov used `490`--`496` HVPs and took about
`17.8`--`18.4` seconds. Their much larger wall times relative to BFGS reflect
the cost of repeatedly obtaining exact curvature products rather than a failure
of the HVP implementation.

At the converged `p=24` BFGS solution, two Hessian eigenvalues were near zero
(`5.75e-13` and `8.19e-13`) while the remaining 22 were materially positive.
The corresponding weakest directions were concentrated almost completely in
the measurement-variance block, and two fitted measurement variances were
driven close to the positive-transform boundary. This is evidence of local flat
curvature/boundary behavior in this synthetic realization. It is not, by
itself, evidence of formal non-identifiability or of global likelihood
geometry.

### Interpretation

The measurements do not support a single solver as uniformly preferable.
For the dimensions studied here (`p<=24`), dense BFGS storage is small and BFGS
provided a strong combination of runtime, final objective and common gradient
norm. L-BFGS-B retained its memory advantage but its native stopping rule could
declare success before the common gradient criterion was met, and at `p=24` it
hit the iteration limit from both starts. Newton-CG and trust-krylov demonstrate
that exact curvature information can be used without a dense Hessian or inverse
and can produce very small final gradients, but repeated HVPs made them more
expensive in these modest-dimensional tests. Fixed-step GD was consistently the
weakest optimization baseline.

The most important diagnostic finding is that the likelihood can contain
strongly different curvature scales, locally indefinite starts, and nearly flat
measurement-variance directions. These observations motivate retaining
curvature diagnostics and provide a characterized deterministic baseline for
#11. They do not yet establish how the methods behave on real market data,
very large parameter vectors, unresolved parameter-tying conventions, or the
PDF's noisy-optimization regime.

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
