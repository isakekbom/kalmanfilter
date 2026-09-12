# kalmanfilter

Python/JAX implementation of the term-structure model in
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
contributions with optional traces. Parameter transforms, synthetic time series,
and optimization are not implemented. Financial conventions, factor construction,
and state lifecycle questions remain unresolved.

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
    params.py       # Reserved for issue #7
    synthetic.py    # Reserved for issue #9
tests/
    test_smoke.py
    test_model.py
    test_ois.py
    test_transition.py
    test_ekf.py
    test_likelihood.py
docs/
    model_spec.md
    ois.md
    transition.md
    ekf.md
    likelihood.md
```

`ModelDimensions` takes required keyword arguments `n_p_t`, `n_c_t`, `n_u_t`, and
`n_z_t`, each a nonnegative Python integer. Its derived properties implement only
the specification's count identities: `n_s_t = n_p_t + n_c_t` and
`n_x_t = n_s_t + n_u_t`. Each instance describes one time; no fixed horizon,
cross-time mapping, parameter dimension, array padding, covariance structure, or
missing-observation behavior is imposed. Representing a zero count is metadata,
not a decision about how a future filter processes an empty observation set.

The reserved modules contain documentation only. Future differentiable numerical
functions belong in these modules and should use JAX with explicit inputs and
outputs. File reading, market-data parsing, and other I/O belong outside them.
No optimizer dependency or experimental optimization framework is included.
