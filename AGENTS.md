# Agent Instructions

## Project context

This repository is a Python/JAX implementation of a term-structure model with
an OIS pricing layer, structural transitions, an extended Kalman filter,
likelihood evaluation, parameter transforms, synthetic validation, and
optimization utilities.

Read the relevant document in `docs/` before changing mathematical behavior.
The documentation describes the model contracts and equation-level conventions.

## Repository structure

```text
src/kalmanfilter/   Package implementation
tests/              Unit, numerical, and integration tests
docs/               Mathematical and implementation documentation
examples/           Runnable validation examples
benchmarks/         Benchmark scripts and recorded results
roadmap.md          Planned work and unresolved questions
pyproject.toml      Package metadata, dependencies, and pytest configuration
uv.lock             Locked dependency resolution for the uv workflow
```

### Main modules

- `model.py` defines dimensions and structural metadata.
- `ois.py` contains OIS pricing calculations and state Jacobians.
- `transition.py` contains structural maps, transitions, and noise covariance
  mappings.
- `ekf.py` implements the forward extended Kalman filter and Cholesky updates.
- `likelihood.py` computes innovation log-likelihoods from EKF results.
- `params.py` defines flat parameter layouts and parameter transformations.
- `gradient_validation.py` validates likelihood gradients against finite
  differences and directional derivatives.
- `synthetic.py` provides seeded synthetic states, noise, and observations.
- `optimization.py` contains the optimizer boundary and optimization traces.
- `fixed_scan.py` provides the fixed-shape `jax.lax.scan` likelihood path.

### Documentation map

Use the document closest to the behavior being changed:

- `docs/model_spec.md` for the mathematical model specification.
- `docs/ois.md` for pricing kernels and OIS derivatives.
- `docs/transition.md` for structural transitions and coordinate mappings.
- `docs/ekf.md` for the forward EKF equations and implementation details.
- `docs/likelihood.md` for innovation likelihood calculations.
- `docs/params.md` for parameter layout and transformations.
- `docs/gradient_validation.md` for derivative validation methodology.
- `docs/synthetic.md` for synthetic data and end-to-end validation.
- `docs/fixed_scan_scaling.md` for the fixed-shape scan implementation.
- `docs/baseline_optimization.md` for optimization workflows and benchmarks.

When a mathematical contract changes, update the relevant documentation and
focused tests in the same change. Keep `README.md` focused on onboarding and
`AGENTS.md` focused on repository and development guidance.

## Environment and validation

- The project requires Python 3.12 or newer.
- Run the test suite with `.venv\Scripts\python.exe -m pytest` on Windows.
- On Windows ARM64, use the x64 Python build because the pinned JAX version
  does not provide the required ARM64 wheels.
- Run a focused test first when changing a narrow module, then run the full
  suite when practical.
- Preserve the repository's JAX 64-bit configuration and existing dtype policy.

## Change guidelines

- Keep changes focused and fix the controlling behavior rather than adding
  surface-level workarounds.
- Preserve public APIs unless the task explicitly requires a contract change.
- Follow existing NumPy, SciPy, and JAX patterns in nearby code.
- Add or update focused tests for behavior changes.
- Do not modify generated benchmark results or unrelated files.
- Avoid broad formatting changes and unnecessary refactors.
- Do not add comments that merely narrate obvious code.

## Pull requests

- Work on a feature branch; do not commit directly to `main`.
- Keep commits focused and use a descriptive commit message.
- State the tests run and any limitations in the pull request description.
- Review `git diff` and `git status` before committing.
