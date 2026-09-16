# Contributing

This guide describes the expected
workflow for code, documentation, tests, and pull requests.

## Before you start

- Read the project overview and setup instructions in [`README.md`](README.md).
- Read [`AGENTS.md`](AGENTS.md) for repository structure and engineering
  guidance.
- Read the relevant document in `docs/` before changing mathematical behavior.
- Check `roadmap.md` and existing issues before starting larger work.

## Development setup

The project requires Python 3.12 or newer and uses JAX with 64-bit floating
point enabled. Follow the complete setup instructions in `README.md`.

On Windows ARM64, use the x64 Python build because the pinned JAX version does
not provide the required ARM64 wheels.

For the standard uv workflow:

```powershell
uv sync --locked --extra test
```

For the Windows ARM64 workflow, create the virtual environment with the x64
Python interpreter as described in `README.md`.

## Branches

Create a focused branch from the latest `main` branch:

```powershell
git switch main
git pull origin main
git switch -c feature/short-description
```

Use a descriptive branch name, such as:

- `feature/improve-likelihood-validation`
- `fix/cholesky-diagnostics`
- `docs/update-arm64-setup`
- `ci/run-tests`

Do not commit directly to `main`.

## Making changes

Keep changes focused on one problem or feature. Follow the engineering rules
and numerical conventions in [`AGENTS.md`](AGENTS.md), including the guidance
on public APIs, JAX precision, linear solves, and focused tests.

## Tests

Run a focused test while developing:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ois.py -q
```

Run the full test suite before opening a pull request:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The full suite includes JAX compilation and numerical tests. The first run can
take several minutes. Include the test command and result in the pull request
description.

## Documentation changes

Keep documentation close to the behavior it describes. Use `README.md` for
onboarding and common commands, `docs/` for mathematical specifications and
implementation details, and `roadmap.md` for project direction. Update focused
tests when a documented model contract changes.

## Commits

Keep commits small and descriptive. For example:

```powershell
git status
git diff
git add path/to/changed-file
git commit -m "Describe the change"
```

Do not include generated benchmark output or unrelated local files unless the
change specifically requires them.

## Pull requests

Before pushing, verify the branch and working tree:

```powershell
git status --short --branch
git diff --check
git log -1 --oneline
```

Push the branch and create a pull request against `main`:

```powershell
git push -u origin feature/short-description
```

A pull request should include:

- A short summary of the problem and solution.
- The files or behavior that changed.
- Tests that were run and their result.
- Known limitations or unresolved assumptions.
- Documentation updates when relevant.

Keep the pull request focused and respond to review feedback with follow-up
commits on the same branch.
