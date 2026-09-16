# Benchmark scenario configuration

[Issue #44](https://github.com/isakekbom/kalmanfilter/issues/44) adds a shared,
explicit configuration layer for the synthetic calibration benchmarks so that
problem size, noise levels, series length, seeds and starting points can be
varied from the command line or a configuration object without editing
benchmark source code. The implementation is
[`benchmarks/benchmark_config.py`](../benchmarks/benchmark_config.py); tests
are in [`tests/test_benchmark_config.py`](../tests/test_benchmark_config.py).

Everything in this layer is a **benchmark convention**. It reuses the
production OIS, transition, EKF, likelihood, parameter-transform and synthetic
generation code unchanged. No parameter tying, market-data convention or new
quote topology is introduced; the historical benchmark builders are reproduced
bitwise by the named presets.

## Dimensions and families

| Symbol | Meaning | Configuration |
| --- | --- | --- |
| `T` | number of dates | `n_dates` |
| `n_x` | latent (PCA) state dimension | `n_states` |
| `n_z` | observations (OIS quotes) per date | `n_states * quotes_per_state` |
| `p` | free raw optimizer parameters | derived: `n_x + n_z + n_x` |

The free raw vector is `(theta_f; sigma_v; theta_g)`: `n_x` persistence values
with the `unit_interval` transform, `n_z` observation variances with the
softplus transform, and `n_x` loading coefficients. The process variances and
the initial law `(a_x, Sigma_0)` are **fixed** known inputs, as in #10/#25.
`p` is derived from the family layout and reported explicitly; arbitrary `p`
values that do not correspond to a valid layout cannot be requested.

| Family | `n_states` | Quote topology | `n_z` | `p` | Historical builder |
| --- | --- | --- | ---: | ---: | --- |
| `reference` | exactly 1 | two hand-written nonlinear quotes on one factor | 2 | 4 | `baseline_optimization.make_problem` (#10) |
| `larger` | >= 2 | two quotes per state loading on the factor and its cyclic neighbour (weights 0.15 / 0.3) | `2 n_x` | `4 n_x` | `curvature_optimization.make_larger_problem` (#25) |

`quotes_per_state` is validated explicitly; only the value 2 is supported by
either family. Requesting another topology, `n_states != 1` for the reference
family or `n_states < 2` for the larger family fails at construction.

## Configuration fields

All fields are plain Python values (ints, floats, tuples of floats, strings),
so a configuration serializes to JSON and back exactly. `None` in a generating
field means "the family default"; `BenchmarkConfig.resolve()` makes every such
value explicit. Supported-by columns: B = `baseline_optimization.py`,
F = `fixed_scan_scaling.py`, C = `curvature_optimization.py`.

| Field | CLI flag | Unit / meaning | Default | Supported by |
| --- | --- | --- | --- | --- |
| `family` | `--family` | `reference` or `larger` | `reference` | B F C |
| `n_dates` | `--dates` | count, `T >= 1` (F: `>= 12`, the longest generated series) | 24 | B F C |
| `n_states` | `--states` | count, `n_x` (reference: 1; larger: >= 2) | 1 | B F C |
| `quotes_per_state` | `--quotes-per-state` | count; only 2 is supported | 2 | B F C |
| `seed` | `--seed` | nonnegative integer for `jax.random.key` | 20261010 | B F C |
| `persistence` | `--persistence` | generating `theta_f`, `n_x` values strictly inside (0, 1) | family default | B F C |
| `loading` | `--loading` | generating `theta_g`, `n_x` finite values | family default | B F C |
| `process_variances` | `--process-variances` | fixed `Sigma_w` diagonal, `n_x` **variances** > 0 | family default | B F C |
| `observation_variances` | `--observation-variances` | generating `Sigma_v` diagonal, `n_z` **variances** > 0 | family default | B F C |
| `process_variance_scale` | `--process-variance-scale` | multiplier applied to the process **variances** | 1.0 | B F C |
| `observation_variance_scale` | `--observation-variance-scale` | multiplier applied to the observation **variances** | 1.0 | B F C |
| `initial_mean` | `--initial-mean` | fixed `a_x`, `n_x` finite values | family default | B F C |
| `initial_covariance_scale` | `--initial-covariance-scale` | multiplier applied to `Sigma_0 = 0.0025 I` | 1.0 | B F C |
| `n_starts` | `--starts` | number of the family's preset raw starts to use | all (3 / 2) | B F C |
| `start_offsets` | `--start-offset ...` (repeatable) | explicit raw offsets from the generating raw vector, `p` values each | family default | B F C |
| `repeats` | `--repeats` | timing repetitions | script default (F: 10, C: 5) | F C |
| `methods` | `--methods` | solver subset from `BFGS`, `L-BFGS-B`, `GD`, `Newton-CG`, `trust-krylov` | script default (B: first three, C: all) | B C |

Noise conventions are never mixed silently: `process_variances` and
`observation_variances` are variances, the `*_variance_scale` fields multiply
variances, and therefore standard deviations scale by the square root. The
effective covariances are `diag(process_variances) * process_variance_scale`,
`diag(observation_variances) * observation_variance_scale` and
`0.0025 * initial_covariance_scale * I`. Because the generator draws
`sqrt(variance) * normal` with the same key sequence for every configuration
of the same `T` and `seed`, scaling a variance by 4 doubles the corresponding
base noise draws bitwise and leaves every other draw unchanged; this is tested.

`n_starts` and `start_offsets` describe the same thing from two directions:
`n_starts` selects a prefix of the family's preset offsets, `start_offsets`
supplies explicit offsets. When both are given they must agree in length.
`repeats` and `methods` are protocol fields: their defaults belong to the
benchmark script, `BenchmarkConfig.defaulted(...)` fills them, and a benchmark
that cannot honour a field refuses it with `reject_unsupported` instead of
ignoring it (`fixed_scan_scaling.py` refuses `methods`,
`baseline_optimization.py` refuses `repeats`).

### Family defaults

| Family | `persistence` | `loading` | `process_variances` | `observation_variances` | `initial_mean` | preset start offsets |
| --- | --- | --- | --- | --- | --- | --- |
| `reference` | (0.9) | (0.8) | (0.0025) | (0.0004, 0.0009) | (0.35) | three #10 offsets |
| `larger` | linspace(0.8, 0.94, n) | linspace(0.65, 0.9, n) | linspace(0.0015, 0.003, n) | linspace(0.0004, 0.001, 2n) | linspace(0.2, 0.35, n) | two #25 offsets |

The larger-family values use `jnp.linspace`, exactly as the historical builder
did, then store the resulting float64 values in the configuration.

## Presets

Presets are ordinary `BenchmarkConfig` values and resolve to the same
structure as custom runs. They reproduce the documented regression anchors.

| Preset | Family | `T` | `n_x` | `n_z` | `p` | Seed | Starts | Historical case |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `reference` | reference | 24 | 1 | 2 | 4 | 20261010 | 3 | #10 baseline; curvature `reference_T24` |
| `long100` | reference | 100 | 1 | 2 | 4 | 20261010 | 2 | curvature `reference_T100` |
| `long1000` | reference | 1000 | 1 | 2 | 4 | 20261010 | 2 | curvature `reference_T1000` |
| `long5000` | reference | 5000 | 1 | 2 | 4 | 20261010 | 1 | curvature `reference_T5000` |
| `larger3` | larger | 100 | 3 | 6 | 12 | 202625 | 2 | curvature `larger_n3` |
| `larger6` | larger | 100 | 6 | 12 | 24 | 202625 | 2 | curvature `larger_n6` |

`fixed_scan_scaling.py` defaults to the `reference` preset generated at
`T=5000` with all three starts and ten timing repeats, which is the #23
protocol. Generation splits one PRNG key per date, so a shorter `n_dates` with
the same seed is an exact prefix of a longer generation; `prefix_problem`
exploits this and is tested against direct generation.

## Python API

```python
from dataclasses import replace
from benchmark_config import BenchmarkConfig, PRESETS, build_problem, problem_record

config = BenchmarkConfig(family="larger", n_states=6, n_dates=1000,
                         observation_variance_scale=5.0, process_variance_scale=2.0, seed=123)
problem = build_problem(config)          # eager synthetic generation; nothing is compiled
problem.config                           # resolved configuration (family defaults explicit)
problem.layout, problem.true_raw, problem.starts, problem.dataset, problem.objective
record = problem_record(problem)         # T, n_x, n_z, p, seed, noise and generating values

custom = replace(PRESETS["long100"], observation_variance_scale=0.5)
text = custom.to_json()                  # exact round trip
assert BenchmarkConfig.from_json(text) == custom
```

`build_problem` returns the ragged Python-driver objective used by the
baseline benchmark; `make_scan_objective(initial, stack_fixed_inputs(inputs),
layout)` builds the fixed-shape scan objective used by the other two. The
`benchmarks/` directory is a plain script directory: run scripts from the
repository root, and in tests the module is importable because
`pyproject.toml` adds `benchmarks` to the pytest `pythonpath`.

## Command line

Every script accepts the flags in the table above. Precedence is
benchmark default < `--preset` (or `--case`) < `--config FILE` < explicit
flags. A configuration file is one JSON object of field names, optionally with
a `"preset"` key naming the base preset. Invalid values fail with a clear
`ValueError` before any data is generated or compiled.

```bash
# The documented reference cases (unchanged defaults)
uv run --locked python benchmarks/baseline_optimization.py
uv run --locked python benchmarks/fixed_scan_scaling.py
uv run --locked python benchmarks/curvature_optimization.py

# Non-default model size and noise levels
uv run --locked python benchmarks/curvature_optimization.py --case larger3 \
    --states 10 --dates 1000 --observation-variance-scale 5 --process-variance-scale 2 --seed 123
uv run --locked python benchmarks/baseline_optimization.py --dates 48 --observation-variance-scale 0.25 --methods BFGS L-BFGS-B
uv run --locked python benchmarks/fixed_scan_scaling.py --preset larger3 --dates 500 --repeats 3

# Explicit generating values and starts, or a configuration file
uv run --locked python benchmarks/baseline_optimization.py --persistence 0.95 --loading 0.7 \
    --start-offset 0.1 0.1 0.1 0.1 --start-offset -0.2 0.3 -0.3 0.0
uv run --locked python benchmarks/curvature_optimization.py --case reference --config experiment.json
```

`fixed_scan_scaling.py` always compiles the ragged Python-loop reference at
`T=12` for its parity check; that compile grows quickly with `p` (minutes for
the larger family), whereas the scan objective compiles in seconds.

`curvature_optimization.py` keeps `--case` (default `all`) for the documented
protocol. `--case all` runs every preset and accepts only the protocol flags
`--repeats` and `--methods`; generation flags and `--config` require a single
`--case`, which then serves as the base preset. A case whose generation equals
its preset keeps the preset name in the output; otherwise it is named
`custom_<family>_n<n_x>_T<T>`. The #10 regression anchor is evaluated only for
the unmodified `reference` preset when all three first-order methods run.

## Output records

Every run records its complete resolved configuration so raw result files are
self-describing:

- `baseline_optimization.py` and `fixed_scan_scaling.py` print a `CONFIG {json}`
  line before generation and a `PROBLEM {json}` line after it.
- `curvature_optimization.py` emits the resolved configuration of every case in
  its `ENV` record and the full problem record in each `PROBLEM` record; the
  `PROTOCOL` record lists the methods actually run.

The problem record contains `config`, `preset`, `T`, `n_x`, `n_z`, `p`, `seed`,
`generating_raw`, `generating_parameters` (`theta_f`, effective `sigma_v`,
`theta_g`), `fixed_process_variance` (effective), `fixed_initial_mean`,
`fixed_initial_covariance`, `starts`, `raw_order`, `theta_f_transform` and a
`noise_convention` sentence. Saved results under `benchmarks/results/` predate
this layer and keep their original format.

## Tests

`tests/test_benchmark_config.py` covers validation and serialization, bitwise
equivalence of the presets with the historical builders (kept verbatim in the
test), deterministic reproducibility, `n_dates` changing only the time
dimension, variance scales changing only the documented draws and covariances,
`n_states` changing `n_x`, `n_z` and `p` consistently, invalid dimensions,
scales and options failing before generation, command-line precedence and
configuration files, and two end-to-end script runs with a non-default state
dimension and non-default noise levels. No wall-clock thresholds are asserted.
