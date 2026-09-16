# KalmanFilter: Code Overview

_A Python/JAX implementation of a term-structure model._

## Overview

This repository contains a Python/JAX implementation of the term-structure model in
[`kalmanRante.pdf`](kalmanRante.pdf). The current mathematical specification is
[`docs/model_spec.md`](docs/model_spec.md); the implementation sequence is in
[`roadmap.md`](roadmap.md).

The package includes the issue #2 foundation, the [OIS pricing kernel and state
Jacobians](docs/ois.md) from equations (8)–(10), and the [structural matrix
layer](docs/transition.md) from issue #4. The [forward Extended Kalman Filter](docs/ekf.md)
implements equations (38)–(56), with Cholesky gain solves, optional equation-level
traces, and support for changing state dimensions and empty observation sets.
Structural steps use explicit coordinate identities and active observations,
retaining compact selector/diagonal storage. The [innovation log-likelihood](docs/likelihood.md)
implements equation (57), reusing EKF Cholesky factors and retaining per-date
contributions with optional traces. The [parameter transforms](docs/params.md)
map a flat float64 vector to the six mathematical blocks, with compact positive
variances, a Cholesky initial covariance, and identity transition parameters by
default. [Full likelihood gradient validation](docs/gradient_validation.md)
compares the complete raw-vector derivative against independent central finite
differences and directional derivatives on controlled nonlinear EKF cases.
[Synthetic generation and end-to-end validation](docs/synthetic.md) (#9) cover
an exact linear Gaussian reference, nonlinear state tracking and likelihood
comparisons, and changing dimensions with missing observations.
[Baseline optimization](docs/baseline_optimization.md) (#10) supports BFGS,
L-BFGS-B, and diagnostic gradient descent with checked JAX gradients, explicit
multiple starts, and separate compilation/optimization timing. An explicit
[fixed-shape scan likelihood](docs/fixed_scan_scaling.md) (#23) batches a constant
structural segment for `jax.lax.scan`, while the general Python drivers retain
support for changing coordinates and ragged observations. Noisy optimization
(#11) is not implemented; MATLAB/reference parity (#12) remains blocked by
missing reference material. Financial conventions, factor construction, and
state lifecycle questions remain unresolved.

## Getting started

The usual first steps are:

1. Install the required Python and project dependencies using the setup below.
2. Run the test suite to verify the local environment.
3. Read the relevant document in `docs/` before changing model behavior.
4. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before making changes or opening
    a pull request.

### Run tests

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The full suite contains numerical and JAX compilation tests. The first run can
take several minutes; a successful run ends with a summary such as `489 passed`.
For a focused check, pass a test file or keyword, for example:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ois.py -q
.\.venv\Scripts\python.exe -m pytest -k likelihood -q
```

### Run an example

After installing the project, run the synthetic validation example with:

```powershell
.\.venv\Scripts\python.exe examples\synthetic_validation.py
```

For contributor workflow, coding conventions, and pull request requirements,
see [`CONTRIBUTING.md`](CONTRIBUTING.md). The repository structure and detailed
agent guidance are documented in [`AGENTS.md`](AGENTS.md).

## Reproducible setup

The package requires Python **3.12 or newer**. The reproducible development
environment uses **CPython 3.12.14**, **uv 0.12.13**, and the dependency versions
and artifact hashes in `uv.lock`. `.python-version` pins the development
interpreter; `pyproject.toml` declares the package requirements and build backend.
The default installation uses JAX CPU wheels and requires no accelerator setup.
See [JAX's supported platforms](https://docs.jax.dev/en/latest/installation.html)
for wheel availability and Windows runtime prerequisites.

Install the pinned uv release once if needed, using the
[official versioned installers](https://docs.astral.sh/uv/getting-started/installation/).
These do not require an existing Python installation.

Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/0.12.13/install.ps1 | iex"
```

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/0.12.13/install.sh | sh
```

Open a new terminal after installing uv. From the repository root, these commands
work on both platforms:

```text
uv --version
uv python install 3.12.14
uv sync --locked --extra test
uv run --locked --extra test pytest
```

`uv sync` creates `.venv` and installs this package in **editable mode** along with
the test extra. Source edits take effect without reinstalling. `--locked` rejects
an out-of-date lockfile rather than updating dependency resolution. No activation
script, `PYTHONPATH` adjustment, or local source-path injection is required.
For details, see [uv's sync behavior](https://docs.astral.sh/uv/concepts/projects/sync/).

To use pytest directly in that environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\pytest.exe
```

```bash
# macOS/Linux
.venv/bin/pytest
```

The equivalent module invocation is `uv run --locked --extra test python -m pytest`.
For a fresh environment without deleting an existing `.venv`, select another
environment directory before running the same sync and test commands:

### Windows ARM64

On Windows ARM64, use the x64 Python build for this project. The current JAX
dependencies do not provide the required ARM64 wheels, so Windows runs the x64
build through emulation. Install it with `winget` from PowerShell:

```powershell
winget install --id Python.Python.3.12 --exact --architecture x64 --scope user
```

Create the project environment with the installed x64 interpreter:

```powershell
$python = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
& $python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Run the tests directly from that environment:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

If Windows reports that a SciPy DLL was blocked, clear the downloaded-file
marker and run the test command again:

```powershell
Get-ChildItem .venv -Recurse -File | Unblock-File
.\.venv\Scripts\python.exe -m pytest
```

The first test run can take several minutes because JAX compiles numerical
functions. Source changes are available immediately because the package is
installed in editable mode.

```powershell
# Windows PowerShell; applies to this terminal session
$env:UV_PROJECT_ENVIRONMENT = '.venv-clean'
uv sync --locked --extra test
uv run --locked --extra test pytest
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

```bash
# macOS/Linux
UV_PROJECT_ENVIRONMENT=.venv-clean uv sync --locked --extra test
UV_PROJECT_ENVIRONMENT=.venv-clean uv run --locked --extra test pytest
```

### Standard editable installation with pip

With a Python 3.12+ interpreter already available, ordinary packaging tools also
work. This path pins direct requirements through `pyproject.toml`, but does not
read the transitive dependency lock; use the uv commands above to reproduce the
locked environment exactly.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv-pip
.\.venv-pip\Scripts\python.exe -m pip install -e ".[test]"
.\.venv-pip\Scripts\python.exe -m pytest
```

macOS/Linux:

```bash
python3.12 -m venv .venv-pip
.venv-pip/bin/python -m pip install -e ".[test]"
.venv-pip/bin/python -m pytest
```

## Precision policy and tests

Importing `kalmanfilter`, including any direct submodule import, centrally sets
`jax_enable_x64=True` in `src/kalmanfilter/__init__.py`. Import the package before
creating JAX arrays or compiling numerical functions. This is a process-wide
configuration change, as described in
[JAX's X64 documentation](https://docs.jax.dev/en/latest/default_dtypes.html).
It enables default floating arrays and explicit float64 arrays to use 64 bits;
it does not convert existing arrays or prevent explicitly requesting float32.

Verify the installed environment manually:

```text
uv run --locked --extra test python -c "import kalmanfilter; import jax; import jax.numpy as jnp; print(jax.config.x64_enabled, jnp.ones(1).dtype)"
```

Expected output: `True float64`.

The smoke tests start isolated Python processes outside the checkout, with X64
initially disabled, and check that root/submodule imports enable it. They also
import each skeleton module and execute a small float64 CPU JIT/autodiff
calculation. Dimension tests cover the documented state decomposition, changing
per-time counts, zero counts, and invalid metadata. These are infrastructure
checks. The OIS tests additionally compare the analytical gradient, JAX autodiff,
and central finite differences on deterministic algebraic examples with different
payment counts. Structural tests cover changing coordinate identities and shapes,
active observation ordering, covariance mappings, compact storage, and checked
JIT for individual steps. EKF tests cover independent linear/OIS references,
equation equivalence, changing dimensions, empty observations, covariance
diagnostics, checked Cholesky failure, and differentiation through a fixed step.
They do not select unresolved financial conventions.

## Package layout and scope

```text
src/kalmanfilter/
    __init__.py     # Central JAX X64 configuration and public metadata type
    model.py        # ModelDimensions: counts for one observation time
    ois.py          # Explicit OIS inputs, pricing, and state derivatives
    transition.py   # Named structural maps, transitions, and noise covariance maps
    ekf.py          # Forward EKF, Cholesky updates, and optional equation-level traces
    likelihood.py   # Innovation log-likelihood from EKF factors and per-date terms
    params.py       # Explicit flat parameter layout and JAX transforms/inverses
    gradient_validation.py  # Raw-vector NLL and independent derivative diagnostics
    synthetic.py    # Explicit seeded state/noise/observation truth and EKF inputs
    optimization.py # Host optimizer boundary, checked JAX gradients, and run traces
    fixed_scan.py   # Explicit batched likelihood for fixed structural segments
tests/
    test_smoke.py
    test_model.py
    test_ois.py
    test_transition.py
    test_ekf.py
    test_likelihood.py
    test_params.py
    test_gradient_validation.py
    test_synthetic.py
    test_optimization.py
    test_fixed_scan.py
docs/
    model_spec.md
    ois.md
    transition.md
    ekf.md
    likelihood.md
    params.md
    gradient_validation.md
    synthetic.md
    baseline_optimization.md
    fixed_scan_scaling.md
examples/
    synthetic_validation.py  # One-command generation, EKF, and likelihood
benchmarks/
    baseline_optimization.py # Fixed synthetic estimation and method comparison
    results/baseline_optimization.txt
    fixed_scan_scaling.py    # Checked JIT and warmed gradients through 5000 dates
    results/fixed_scan_scaling.txt
```

`ModelDimensions` takes required keyword arguments `n_p_t`, `n_c_t`, `n_u_t`, and
`n_z_t`, each a nonnegative Python integer. Its derived properties implement only
the specification's count identities: `n_s_t = n_p_t + n_c_t` and
`n_x_t = n_s_t + n_u_t`. Each instance describes one time; no fixed horizon,
cross-time mapping, parameter dimension, array padding, covariance structure, or
missing-observation behavior is imposed. Representing a zero count is metadata,
not a decision about how a future filter processes an empty observation set.

Mathematical model functions use JAX with explicit inputs and outputs. The
optimizer host boundary uses NumPy/SciPy. File reading and market-data parsing
belong outside the numerical model modules. The section 3 experimental
optimization framework remains unimplemented.
