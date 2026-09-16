# Kalman filter implementation roadmap

This repository aims to reproduce and then extend the term-structure model described in `kalmanRante.pdf`.

The core Python/JAX implementation through deterministic parameter estimation is now in place. The next phase is team-oriented: several independent workstreams can proceed in parallel, but research/real-data claims must respect the dependency order below.

We still do **not** have a trusted supervisor MATLAB implementation/reference dataset in the repository. Synthetic validation remains the current internal reference. Any unresolved market/model convention must be documented rather than guessed.

## Core development rules

For every issue:

1. Read `kalmanRante.pdf`, this roadmap, and prerequisite issue/docs.
2. State mathematical ambiguity explicitly instead of silently inventing a convention.
3. Prefer small, testable functions that map to mathematical objects.
4. Use JAX float64 for differentiable numerical code.
5. Avoid explicit matrix inverses; use solves/Cholesky factorizations.
6. Add tests in the same PR.
7. Preserve inspectable EKF intermediates for later parity work.
8. Keep experimental/noisy optimization separate from the production EKF model.
9. Preserve native numerical failures; do not hide them with clipping, jitter, finite penalties, or retries unless a separate issue explicitly justifies such behavior.
10. Follow the blocker sections in GitHub issues before starting dependent work.

---

# Completed foundation

## Phase 0 — Understand and scaffold

- **#1** Reconstruct/document the mathematical model — **completed**
- **#2** Reproducible Python/JAX project skeleton — **completed**

## Phase 1 — Mathematical model

- **#3** OIS pricing function and Jacobian — **completed**
- **#4** Time-varying structural matrices/maps — **completed**
- **#5** Extended Kalman Filter recursion — **completed**
- **#6** Stable innovation log-likelihood — **completed**

## Phase 2 — Parameterization and validation

- **#7** Constrained/raw parameterization — **completed**
- **#8** Full-likelihood JAX gradient validation — **completed**
- **#9** Synthetic generator/end-to-end validation — **completed**

## Phase 3 — Deterministic optimization baseline

- **#10** Baseline ML optimization (BFGS/L-BFGS-B/GD) — **completed**
- **#23** Fixed-shape `lax.scan` scaling to long time series — **completed**
- **#25** Curvature-aware optimization/conditioning (Newton-CG, trust-krylov, HVPs) — **completed**

These completed issues establish the deterministic baseline used by all research work below.

---

# Current parallel workstreams

After #25, the project deliberately branches into several parallel tracks.

## Track A — Numerical behavior and stronger deterministic diagnostics

### #33 Characterize the numerical precision floor
https://github.com/isakekbom/kalmanfilter/issues/33

Purpose: measure when finite-precision error becomes material in objective/gradient differences instead of assuming that numerical noise is already the dominant problem.

**Blocked by:** #23, #25 (completed).

**Blocks:** #37, #39.

### #40 Evaluate parameter scaling/preconditioning
https://github.com/isakekbom/kalmanfilter/issues/40

Purpose: separate poor coordinate scaling/conditioning from genuine finite-precision noise and establish the strongest fair deterministic baseline.

**Blocked by:** #25 (completed).

**Blocks:** #39.

## Track B — Section 3 mathematical specification and controlled noisy experiments

### #34 Formalize PDF Section 3 mathematics
https://github.com/isakekbom/kalmanfilter/issues/34

Purpose: convert equations (58)–(97) into an implementation-ready dimensioned specification and list unresolved choices explicitly.

**Blocked by:** #25 (completed).

**Blocks:** #35, #37.

### #36 Build a controlled noisy objective/gradient harness
https://github.com/isakekbom/kalmanfilter/issues/36

Purpose: create exact, seeded noise experiments where the true objective/gradient and optimum remain known.

**Blocked by:** #25 (completed).

**Blocks:** #37, #39.

### #35 Implement Section 3 algebra primitives
https://github.com/isakekbom/kalmanfilter/issues/35

Purpose: implement/test `vech`, local Taylor terms, `a/A/alpha/Sbar/y`, and stable solves independently of the outer algorithm.

**Blocked by:** #34.

**Blocks:** #37.

### #37 Implement the Section 3 local quadratic/noise model
https://github.com/isakekbom/kalmanfilter/issues/37

Purpose: assemble the local surrogate/noise model once its mathematics, numerical motivation, and controlled oracle are ready.

**Blocked by:** #33, #34, #35, #36.

**Blocks:** #38, #39.

### #38 Implement the Section 3 outer iteration/fixed-point solver
https://github.com/isakekbom/kalmanfilter/issues/38

Purpose: implement the iterative/full-system procedure around the local model with explicit initialization, stopping, history, and failure behavior.

**Blocked by:** #37.

**Blocks:** #39.

### #39 Benchmark noisy optimization against deterministic baselines
https://github.com/isakekbom/kalmanfilter/issues/39

Purpose: determine under which controlled or measured noise regimes the Section 3 method helps, does not help, or fails relative to the deterministic baseline.

**Blocked by:** #33, #36, #38, #40.

**Blocks:** completion of umbrella #11.

### #11 Section 3 noisy-optimization umbrella
https://github.com/isakekbom/kalmanfilter/issues/11

#11 is an umbrella, not a single implementation PR. It closes only after #33–#40 relevant to Section 3 are complete and #39 documents the final comparison.

## Track C — Visualization and communication

### #28 Visualize saved optimizer benchmark results
https://github.com/isakekbom/kalmanfilter/issues/28

Purpose: parse the saved baseline/curvature result files and produce reproducible plots/tables for NLL, gradients, runtime, evaluations, convergence, scaling, and conditioning.

**Ready now.** Does not block numerical implementation.

### #30 Visualize model-implied discount factors and rate curves
https://github.com/isakekbom/kalmanfilter/issues/30

Purpose: visualize discount factors, fitted OIS quotes, and rate curves where a valid maturity axis exists. Synthetic/node-index plots must not be mislabeled as market curves.

**Synthetic path ready now.** Real-market interpretation depends on #41/#42.

## Track D — External/reference model work

### #41 Resolve model and market-data conventions with the supervisor
https://github.com/isakekbom/kalmanfilter/issues/41

Purpose: resolve the remaining open questions from `docs/model_spec.md` (calendar/cash-flow conventions, loadings, state lifecycle, parameter tying, failure policy, Section 3 ambiguities, etc.) with source/supervisor evidence.

**Can run in parallel with all synthetic research.** Full completion depends on external information.

**Blocks:** #42 and any real-data/production claims.

### #12 MATLAB/reference-data parity
https://github.com/isakekbom/kalmanfilter/issues/12

Purpose: compare EKF intermediate quantities and likelihood against trusted supervisor/reference outputs.

**Externally blocked:** waiting for MATLAB/reference outputs/approved fixture.

**Blocks:** #42 and claims of reference parity.

### #42 Integrate trusted reference/market data into end-to-end calibration
https://github.com/isakekbom/kalmanfilter/issues/42

Purpose: build the final auditable data-adapter/calibration path from trusted inputs to fitted parameters, states, residuals, and curve outputs.

**Blocked by:** #41, #12, and availability/permission for a trusted dataset.

---

# Dependency graph / recommended order

```text
COMPLETED CORE
#1 + #2
   ↓
#3 + #4 + #7
   ↓
#5 → #6
├─────────────┐
↓             ↓
#8            #9
└──────┬──────┘
       ↓
      #10
       ↓
      #23
       ↓
      #25
       │
       ├──────────────┬──────────────┬──────────────┐
       ↓              ↓              ↓              ↓
      #33            #34            #36            #40
                       ↓
                      #35
       └───────────────┼──────────────┘
                       ↓
                      #37
                       ↓
                      #38
       #33 + #36 + #38 + #40
                       ↓
                      #39
                       ↓
                 close umbrella #11

PARALLEL VISUALIZATION
#10 + #25 → #28
#3 + #9   → #30 synthetic visualization

EXTERNAL / REAL-DATA TRACK
#41 ─────────────┐
                 ├→ #42 real/reference calibration
#12 [external] ──┘
#42 → real-data extension of #30
```

## What contributors can start immediately

Because #25 is completed, independent contributors can work in parallel on:

- #28 benchmark visualization;
- #30 synthetic curve/quote visualization;
- #33 numerical precision-floor measurements;
- #34 Section 3 mathematical specification;
- #36 controlled noisy-objective harness;
- #40 deterministic scaling/preconditioning diagnostics;
- #41 supervisor/model-convention resolution.

Do **not** start #35 before #34, #37 before #33/#34/#35/#36, #38 before #37, or #39 before #33/#36/#38/#40.

---

# Definition of deterministic baseline complete

The deterministic baseline is considered complete when:

- equations (8)–(57) have a documented implementation mapping;
- JAX runs in float64;
- stable solves/Cholesky are used instead of explicit inverses;
- synthetic end-to-end tests pass;
- full likelihood gradients pass independent checks;
- fixed-shape long-series execution is scalable;
- deterministic optimizers/curvature diagnostics are benchmarked;
- intermediate EKF quantities remain inspectable for later parity.

Issues #1–#10, #23, and #25 satisfy this baseline. Future research must preserve it as a regression reference.

# Known external limitations

Until #12/#41/#42 are resolved, the project must not claim:

- MATLAB/reference implementation parity;
- correctness of unresolved market calendar/cash-flow conventions;
- validated real-market parameter estimates;
- production-grade state lifecycle/parameter tying beyond documented conventions;
- real-data superiority of any optimizer.

Synthetic experiments remain valuable for numerical and algorithmic research, but they do not replace external/reference validation.
