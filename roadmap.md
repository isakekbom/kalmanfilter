# Kalman filter implementation roadmap

This repository aims to reproduce and then extend the term-structure model described in `kalmanRante.pdf`.

The current source of truth is **only the PDF**. We do not yet have the supervisor's MATLAB implementation or a reference dataset. Therefore the first objective is not to optimize the model; it is to reconstruct the mathematics carefully, implement a transparent baseline, and validate it on synthetic cases. MATLAB/reference parity comes later when those materials become available.

## Core principle for Codex

Work through the GitHub issues in dependency order. Do **not** jump directly to parameter optimization or the noisy-optimization section.

For every issue:

1. Read `kalmanRante.pdf`, `roadmap.md`, and the relevant prerequisite issues/docs.
2. State any mathematical ambiguity explicitly instead of silently inventing a convention.
3. Prefer small, testable functions whose names correspond to mathematical objects in the PDF.
4. Use JAX with 64-bit floating-point precision for differentiable numerical code.
5. Avoid explicit matrix inverses. Use linear solves / Cholesky factorizations where appropriate.
6. Add tests as part of the same change.
7. Preserve intermediate EKF quantities so later MATLAB parity debugging is possible.
8. Keep experimental optimization code separate from the core state-space model.

## Model summary

The latent state is

`x_t = (x_t^s, x_t^u)`

where `x_t^s` contains systematic term-structure factors and `x_t^u` instrument-specific deviations. The systematic state is further split into principal-component factors and central-bank-step factors.

The transition model is

`x_t = F_t(theta_F) x_{t-1} + w_t`

with

`F_t(theta_F) = A_t diag(theta_F) B_t`.

The observation model is nonlinear:

`z_t = g_t(theta_g, x_t^s) + I^z_t x_t^u + v_t`.

Therefore the implementation in equations (38)–(56) is an **Extended Kalman Filter (EKF)**: `g_t` is linearized around the predicted systematic state before the measurement update.

The Gaussian innovation log-likelihood in equation (57) is the baseline objective for parameter estimation. Section 3, equations (58)–(97), is a separate draft method for handling noisy objective/gradient evaluations and should only be attempted after the baseline likelihood and gradients have been validated.

---

# Phase 0 — Understand and scaffold

## #1 Reconstruct and document the mathematical model from `kalmanRante.pdf`
https://github.com/isakekbom/kalmanfilter/issues/1

**Priority: highest.**

Deliver `docs/model_spec.md` containing notation, dimensions, assumptions, open questions, and a mapping from equations to implementation objects.

Do not silently resolve ambiguous parts of the PDF.

## #2 Set up a reproducible Python/JAX project skeleton
https://github.com/isakekbom/kalmanfilter/issues/2

Can run in parallel with #1.

Target structure:

```text
src/kalmanfilter/
    __init__.py
    model.py
    ois.py
    transition.py
    ekf.py
    likelihood.py
    params.py
    synthetic.py
    optimization.py

tests/
docs/
```

Use JAX float64 and pytest from the start.

---

# Phase 1 — Implement the mathematical model

## #3 Implement OIS pricing function `g_t` and its Jacobian
https://github.com/isakekbom/kalmanfilter/issues/3

Depends on #1 and #2.

Implement equations (8)–(10), including discount factors and the nonlinear OIS quote function. Validate Jacobians against finite differences and, where practical, the analytical expression in the PDF.

## #4 Implement time-varying transition and observation-selection matrices
https://github.com/isakekbom/kalmanfilter/issues/4

Depends on #1 and #2.

Implement `A_t`, `B_t`, `D_t`, `I^z_t`, `G_t`, and `F_t(theta_F)`. The implementation must explicitly support changing state dimensions and missing observations.

## #5 Implement the Extended Kalman Filter recursion
https://github.com/isakekbom/kalmanfilter/issues/5

Depends on #3 and #4.

Implement equations (38)–(56). Return a detailed optional trace containing predicted/filtered states, covariances, innovations, `H_t`, `S_t`, and other intermediate quantities.

**Do not use explicit matrix inverses.**

## #6 Implement numerically stable innovation log-likelihood
https://github.com/isakekbom/kalmanfilter/issues/6

Depends on #5.

Implement equation (57). Use Cholesky factorization for the log determinant and solves. Keep per-time-step likelihood contributions.

---

# Phase 2 — Parameterization and validation

## #7 Design constrained parameterization
https://github.com/isakekbom/kalmanfilter/issues/7

Can begin after #1/#2 and should be complete before optimization.

Map an unconstrained optimizer vector to valid

`theta = (theta_F, Sigma_w, Sigma_v, a_x, Sigma_0, theta_g)`.

Ensure covariance parameters remain valid by construction.

## #9 Build synthetic data generator and end-to-end recovery tests
https://github.com/isakekbom/kalmanfilter/issues/9

Depends on #3–#6.

Synthetic data is essential while the MATLAB implementation is unavailable. Include a linear sanity case, nonlinear OIS case, and a time-varying/missing-data case.

## #8 Validate JAX autodiff gradients of the full likelihood
https://github.com/isakekbom/kalmanfilter/issues/8

Depends on #5, #6, #7.

Compare `jax.grad` / `jax.value_and_grad` against central finite differences and directional derivatives. Do not proceed to serious optimization until these checks pass.

---

# Phase 3 — Baseline parameter estimation

## #10 Add baseline maximum-likelihood optimization and benchmarks
https://github.com/isakekbom/kalmanfilter/issues/10

Depends on #8 and #9.

Benchmark at least L-BFGS/L-BFGS-B and BFGS with supplied gradients. Plain gradient descent may be retained as a diagnostic baseline because the supervisor has already observed that small gradient-descent steps can make convergence slow.

Measure:

- objective value,
- gradient norm,
- wall-clock time,
- function/gradient evaluations,
- termination reason,
- parameter recovery on synthetic data.

Report JIT compile cost separately from steady-state runtime.

## #23 Make fixed-dimension EKF likelihood scalable to long time series
https://github.com/isakekbom/kalmanfilter/issues/23

Added after #10 exposed a compilation scaling risk: its historical 24-date
Python-loop value/gradient first call took approximately 152 seconds. Preserve
that baseline and the general ragged drivers, and add an explicit `jax.lax.scan`
path for fixed structural segments. Batch numerical time data outside the
differentiated objective; reuse the existing EKF/likelihood kernels. Establish
state, covariance, innovation, raw-gradient and optimizer parity before measuring
first-call and warmed evaluation times through at least 1000 and preferably 5000
dates. See [fixed scan execution and measurements](docs/fixed_scan_scaling.md).

Automatic regime segmentation, padding, lifecycle inference and global parameter
tying remain outside this issue. Complete this scalability work before #11.

---

# Phase 4 — External parity when reference implementation arrives

## #12 Add MATLAB/reference-data parity tests
https://github.com/isakekbom/kalmanfilter/issues/12

**Currently blocked.**

Once supervisor MATLAB code or trusted outputs are available, compare intermediate quantities rather than only the final objective. This is the strongest test that the PDF has been interpreted correctly.

If MATLAB parity reveals a mismatch, fix the core model and add regression tests before continuing experimental optimization work.

---

# Phase 5 — Research / experimental optimization

## #11 Implement and evaluate the noisy-optimization method from section 3
https://github.com/isakekbom/kalmanfilter/issues/11

Depends on #10 and the subsequent #23 scalability work, and should ideally also wait for #12 if the reference implementation becomes available soon.

Implement equations (58)–(97) as an isolated research module. Compare it with standard optimizers using the same validated EKF objective and gradient.

Do not let this experimental layer change the baseline EKF mathematics.

---

# Recommended execution order

A practical order for Codex is:

```text
#1  model specification
#2  project skeleton
   ↓
#3  OIS observation function ─┐
#4  structural matrices       ├→ #5 EKF → #6 likelihood
                              │
#7  parameter transforms ─────┘

#6 + #7 → #8 gradient validation
#3–#6   → #9 synthetic validation

#8 + #9 → #10 baseline optimization

#5 + #6 → #12 MATLAB parity   [blocked until reference available]

#10 → #23 fixed-shape scan scalability → #11 noisy optimization experiment
```

# Definition of "baseline model complete"

The baseline implementation should not be considered complete until all of the following hold:

- equations (8)–(57) have a clear implementation mapping;
- matrix/vector dimensions are documented;
- JAX runs in float64;
- the filter uses stable solves rather than explicit inverses;
- covariance and innovation matrices remain numerically well behaved on tests;
- synthetic end-to-end tests pass;
- the full likelihood gradient passes independent numerical checks;
- intermediate values are inspectable for later MATLAB parity.

# Known limitations at repository start

At the time this roadmap was created, the repository contains only `kalmanRante.pdf`. In particular we do **not** yet have:

- the supervisor's MATLAB implementation;
- real market input data;
- trusted reference parameter values;
- trusted expected states or likelihood values;
- a complete operational specification for every time-varying matrix in the PDF.

Codex should treat these as genuine unknowns, not invitations to fabricate missing information. Any temporary convention required for synthetic tests must be documented and isolated so it can be replaced when the reference implementation/data becomes available.
