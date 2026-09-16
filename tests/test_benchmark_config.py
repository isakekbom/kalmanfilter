"""Shared benchmark configuration: validation, serialization, presets, generation and CLI."""

import argparse
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import runpy
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import benchmark_config as module
from benchmark_config import (
    PRESETS, BenchmarkConfig, BenchmarkProblem, add_config_arguments, argument_overrides,
    build_problem, config_from_arguments, prefix_problem, preset_name, problem_record,
    reject_unsupported, serialize,
)
from kalmanfilter.ois import OISInstrument
from kalmanfilter.params import ModelParameters, ParameterLayout, pack_parameters
from kalmanfilter.synthetic import SyntheticStepInputs, generate_synthetic_dataset
from kalmanfilter.transition import DiagonalMatrix, StateCoordinates, StructuralStep, selection_map


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = PRESETS["reference"]


def legacy_reference_problem(n_dates, seed=20261010):
    """The #10 builder exactly as it was before the shared configuration layer."""
    coordinates = StateCoordinates(("p",), (), ())
    step = StructuralStep(
        coordinates, coordinates,
        selection_map(("p",), ("phi",), ("phi",)),
        selection_map(("phi",), ("p",), ("p",)),
        selection_map(("p",), ("w",), ("w",)),
        selection_map(("qa", "qb"), (), (None, None)),
        selection_map(("qa", "qb"), ("va", "vb"), ("va", "vb")),
    )
    instruments = (
        OISInstrument(jnp.array([1.0]), jnp.array([[[0.0]], [[-1.0]]]), jnp.empty((2, 0))),
        OISInstrument(jnp.array([0.5]), jnp.array([[[0.2]], [[-1.5]]]), jnp.empty((2, 0))),
    )
    layout = ParameterLayout(n_f=1, n_w=0, n_v=2, n_x0=0, n_g=1, theta_f_transform="unit_interval")
    truth = ModelParameters(
        jnp.array([0.9]), DiagonalMatrix(jnp.empty((0,))),
        DiagonalMatrix(jnp.array([0.0004, 0.0009])), jnp.empty((0,)), jnp.empty((0, 0)),
        jnp.array([0.8]),
    )
    true_raw = pack_parameters(truth, layout)
    date = SyntheticStepInputs(step, truth.theta_f, DiagonalMatrix(jnp.array([0.0025])),
                               truth.theta_g, truth.sigma_v, instruments)
    dataset = generate_synthetic_dataset(jax.random.key(seed), coordinates,
                                         jnp.array([0.35]), jnp.array([[0.0025]]), (date,) * n_dates)
    offsets = jnp.array([[0.25, 0.2, -0.2, 0.1], [-0.8, 0.8, -0.6, -0.2], [1.0, -1.0, 1.0, 0.3]])
    return layout, truth, true_raw, dataset, tuple(true_raw + row for row in offsets)


def legacy_larger_problem(n_states, n_dates, seed=202625):
    """The #25 larger-family builder exactly as it was before the shared layer."""
    names = tuple(f"p{i}" for i in range(n_states))
    coordinates = StateCoordinates(names, (), ())
    quotes = tuple(f"q{i}{kind}" for i in range(n_states) for kind in ("a", "b"))
    identity = selection_map(names, names, names)
    step = StructuralStep(coordinates, coordinates, identity, identity, identity,
                          selection_map(quotes, (), (None,) * len(quotes)),
                          selection_map(quotes, quotes, quotes))
    instruments = []
    for i in range(n_states):
        for kind in range(2):
            weights = jnp.eye(n_states)[i] + (0.15 if kind == 0 else 0.3) * jnp.eye(n_states)[(i + 1) % n_states]
            loadings = jnp.stack((jnp.zeros((n_states, n_states)), -jnp.diag(weights)))
            if kind == 1:
                loadings = loadings.at[0].set(0.2 * jnp.diag(weights)).at[1].set(-1.5 * jnp.diag(weights))
            instruments.append(OISInstrument(jnp.array([1.0 if kind == 0 else 0.5]), loadings, jnp.empty((2, 0))))
    layout = ParameterLayout(n_f=n_states, n_w=0, n_v=2 * n_states, n_x0=0, n_g=n_states,
                             theta_f_transform="unit_interval")
    truth = ModelParameters(jnp.linspace(0.8, 0.94, n_states), DiagonalMatrix(jnp.empty(0)),
                            DiagonalMatrix(jnp.linspace(0.0004, 0.001, 2 * n_states)),
                            jnp.empty(0), jnp.empty((0, 0)), jnp.linspace(0.65, 0.9, n_states))
    raw = pack_parameters(truth, layout)
    date = SyntheticStepInputs(step, truth.theta_f, DiagonalMatrix(jnp.linspace(0.0015, 0.003, n_states)),
                               truth.theta_g, truth.sigma_v, tuple(instruments))
    data = generate_synthetic_dataset(jax.random.key(seed), coordinates,
                                      jnp.linspace(0.2, 0.35, n_states), jnp.eye(n_states) * 0.0025, (date,) * n_dates)
    first = jnp.concatenate((jnp.full(n_states, 0.2), jnp.linspace(-0.2, 0.2, 2 * n_states), jnp.full(n_states, 0.05)))
    second = jnp.concatenate((jnp.full(n_states, -0.3), jnp.linspace(0.3, -0.3, 2 * n_states), jnp.full(n_states, -0.08)))
    return layout, truth, raw, data, (raw + first, raw + second)


def assert_bitwise_equal(actual, expected):
    actual, expected = jax.tree.leaves(actual), jax.tree.leaves(expected)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        if isinstance(a, (jax.Array, np.ndarray)):
            assert a.dtype == b.dtype and a.shape == b.shape
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def assert_problem_matches_legacy(problem, legacy):
    layout, truth, true_raw, dataset, starts = legacy
    assert problem.layout == layout
    assert_bitwise_equal(problem.true_parameters, truth)
    assert_bitwise_equal(problem.true_raw, true_raw)
    assert_bitwise_equal(problem.dataset, dataset)
    assert_bitwise_equal(problem.starts, starts)


@pytest.fixture(scope="module")
def scripts():
    return {name: runpy.run_path(str(ROOT / "benchmarks" / f"{name}.py"))
            for name in ("baseline_optimization", "fixed_scan_scaling", "curvature_optimization")}


def test_default_configuration_is_the_reference_anchor_with_explicit_dimensions():
    config = BenchmarkConfig()
    assert config == REFERENCE and preset_name(config) == "reference"
    assert (config.n_dates, config.n_x, config.n_z, config.p, config.seed) == (24, 1, 2, 4, 20261010)
    assert config.layout == ParameterLayout(n_f=1, n_w=0, n_v=2, n_x0=0, n_g=1, theta_f_transform="unit_interval")
    resolved = config.resolve()
    assert resolved.persistence == (0.9,) and resolved.loading == (0.8,)
    assert resolved.process_variances == (0.0025,) and resolved.observation_variances == (0.0004, 0.0009)
    assert resolved.initial_mean == (0.35,) and resolved.n_starts == 3 and len(resolved.start_offsets) == 3
    assert resolved.repeats is None and resolved.methods is None
    assert resolved.resolve() == resolved and preset_name(resolved) == "reference"
    with pytest.raises(FrozenInstanceError):
        config.n_dates = 5


def test_presets_keep_the_documented_dimensions_seeds_and_starts():
    expected = {"reference": (24, 1, 2, 4, 20261010, 3), "long100": (100, 1, 2, 4, 20261010, 2),
                "long1000": (1000, 1, 2, 4, 20261010, 2), "long5000": (5000, 1, 2, 4, 20261010, 1),
                "larger3": (100, 3, 6, 12, 202625, 2), "larger6": (100, 6, 12, 24, 202625, 2)}
    assert set(PRESETS) == set(expected)
    for name, (t, n_x, n_z, p, seed, n_starts) in expected.items():
        config = PRESETS[name]
        assert (config.n_dates, config.n_x, config.n_z, config.p, config.seed) == (t, n_x, n_z, p, seed)
        assert config.resolve().n_starts == n_starts and preset_name(config) == name
        assert preset_name(replace(config, repeats=1, methods=("BFGS",))) == name
        assert preset_name(replace(config, seed=1)) is None
        assert BenchmarkConfig.from_json(config.resolve().to_json()) == config.resolve()


@pytest.mark.parametrize("n_states", [2, 3, 6])
def test_larger_family_derives_n_x_n_z_and_p_consistently(n_states):
    config = BenchmarkConfig(family="larger", n_states=n_states, n_dates=3, seed=202625)
    assert (config.n_x, config.n_z, config.p) == (n_states, 2 * n_states, 4 * n_states)
    assert config.layout.n_parameters == config.p
    resolved = config.resolve()
    assert len(resolved.persistence) == len(resolved.loading) == len(resolved.process_variances) == n_states
    assert len(resolved.observation_variances) == len(resolved.initial_mean) * 2 == 2 * n_states
    assert all(len(offset) == config.p for offset in resolved.start_offsets)
    problem = build_problem(config)
    assert problem.dataset.initial_filter.state.shape == (n_states,)
    assert problem.dataset.initial_filter.covariance.shape == (n_states, n_states)
    assert len(problem.dataset.inputs) == 3
    assert problem.dataset.observations[0].shape == (2 * n_states,)
    assert problem.true_raw.shape == (4 * n_states,)
    assert all(start.shape == (4 * n_states,) for start in problem.starts)
    record = problem_record(problem)
    assert (record["T"], record["n_x"], record["n_z"], record["p"]) == (3, n_states, 2 * n_states, 4 * n_states)


@pytest.mark.parametrize("values,error", [
    (dict(n_dates=0), ValueError), (dict(n_dates=2.0), TypeError), (dict(n_dates=True), TypeError),
    (dict(n_states=0), ValueError), (dict(n_states=2), ValueError), (dict(family="larger"), ValueError),
    (dict(family="larger", n_states=1), ValueError), (dict(family="other", n_states=2), ValueError),
    (dict(quotes_per_state=1), ValueError), (dict(quotes_per_state=3), ValueError),
    (dict(seed=-1), ValueError), (dict(seed=1.5), TypeError),
    (dict(observation_variance_scale=0.0), ValueError), (dict(observation_variance_scale=-2.0), ValueError),
    (dict(process_variance_scale=float("nan")), ValueError), (dict(process_variance_scale=float("inf")), ValueError),
    (dict(initial_covariance_scale=0), ValueError), (dict(initial_covariance_scale="1"), TypeError),
    (dict(persistence=(1.0,)), ValueError), (dict(persistence=(0.5, 0.5)), ValueError),
    (dict(persistence="0.9"), TypeError), (dict(loading=(float("nan"),)), ValueError),
    (dict(observation_variances=(0.1,)), ValueError), (dict(observation_variances=(0.1, 0.0)), ValueError),
    (dict(process_variances=(-0.1,)), ValueError), (dict(initial_mean=(1.0, 2.0)), ValueError),
    (dict(n_starts=0), ValueError), (dict(n_starts=4), ValueError),
    (dict(start_offsets=()), ValueError), (dict(start_offsets=((1.0, 2.0, 3.0),)), ValueError),
    (dict(start_offsets=((0.0, 0.0, 0.0, float("inf")),)), ValueError),
    (dict(n_starts=2, start_offsets=((0.0,) * 4,)), ValueError),
    (dict(repeats=0), ValueError), (dict(repeats=2.5), TypeError),
    (dict(methods=()), ValueError), (dict(methods=("BFGS", "BFGS")), ValueError),
    (dict(methods=("Adam",)), ValueError), (dict(methods="BFGS"), TypeError),
])
def test_invalid_configurations_fail_before_any_generation_or_compilation(monkeypatch, values, error):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid configuration reached data generation or compilation")
    monkeypatch.setattr(module, "generate_synthetic_dataset", forbidden)
    with pytest.raises(error):
        BenchmarkConfig(**values)
    with pytest.raises(error):
        replace(BenchmarkConfig(), **values)
    with pytest.raises(TypeError):
        build_problem(values)


def test_serialization_round_trips_and_rejects_unknown_fields(tmp_path):
    config = BenchmarkConfig(family="larger", n_states=2, n_dates=3, persistence=[0.5, 0.6],
                             observation_variances=np.array([1e-4, 2e-4, 3e-4, 4e-4]),
                             start_offsets=[[0.0] * 8, np.full(8, 0.1)], methods=["GD"], repeats=2)
    assert config.persistence == (0.5, 0.6) and config.start_offsets[1] == (0.1,) * 8
    assert config.methods == ("GD",) and isinstance(config.observation_variances, tuple)
    text = config.to_json()
    assert json.loads(text)["persistence"] == [0.5, 0.6] and json.loads(text)["methods"] == ["GD"]
    assert set(json.loads(text)) == set(config.to_dict())
    assert BenchmarkConfig.from_json(text) == config
    assert BenchmarkConfig.from_dict(json.loads(text)) == config
    assert BenchmarkConfig.from_dict({"n_dates": 9}, config) == replace(config, n_dates=9)
    assert config.resolve() == BenchmarkConfig.from_json(config.resolve().to_json())
    with pytest.raises(ValueError, match="unknown configuration fields"):
        BenchmarkConfig.from_dict({"n_dates": 9, "dates": 9})
    with pytest.raises(TypeError):
        BenchmarkConfig.from_dict([("n_dates", 9)])
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    assert BenchmarkConfig.from_json(path.read_text(encoding="utf-8")) == config


def test_reference_preset_reproduces_the_historical_builder_bitwise(scripts):
    problem = build_problem(REFERENCE)
    assert problem.config == REFERENCE.resolve()
    assert_problem_matches_legacy(problem, legacy_reference_problem(24))
    wrapper = scripts["baseline_optimization"]["make_problem"](12)
    assert isinstance(wrapper, BenchmarkProblem)
    assert wrapper.config == replace(REFERENCE, n_dates=12).resolve()
    assert_problem_matches_legacy(wrapper, legacy_reference_problem(12))
    assert_problem_matches_legacy(build_problem(replace(REFERENCE, n_dates=12, seed=7)), legacy_reference_problem(12, 7))


def test_larger_preset_reproduces_the_historical_builder_bitwise(scripts):
    # Generation is prefix-exact, so a shortened preset checks the same construction.
    config = replace(PRESETS["larger3"], n_dates=5)
    problem = build_problem(config)
    legacy = legacy_larger_problem(3, 5)
    assert_problem_matches_legacy(problem, legacy)
    curvature = scripts["curvature_optimization"]["make_larger_problem"](3, n_dates=5)
    assert curvature.config == config.resolve() and curvature.preset is None
    assert_bitwise_equal(curvature.batch.observations, jnp.stack(legacy[3].observations))
    assert_bitwise_equal(curvature.starts, legacy[4])
    assert_problem_matches_legacy(build_problem(replace(PRESETS["larger6"], n_dates=2)), legacy_larger_problem(6, 2))


def test_fixed_seed_and_configuration_reproduce_bitwise_and_seed_changes_data():
    config = BenchmarkConfig(family="larger", n_states=2, n_dates=4, seed=3)
    first, second = build_problem(config), build_problem(BenchmarkConfig.from_json(config.to_json()))
    assert first.config == second.config
    assert_bitwise_equal(first.dataset, second.dataset)
    assert_bitwise_equal(first.starts, second.starts)
    other = build_problem(replace(config, seed=4))
    assert_bitwise_equal(other.true_raw, first.true_raw)
    assert not np.array_equal(np.stack(other.dataset.observations), np.stack(first.dataset.observations))
    assert not np.array_equal(other.dataset.true_initial_state, first.dataset.true_initial_state)


def test_changing_n_dates_changes_only_the_time_dimension():
    short, long = build_problem(replace(REFERENCE, n_dates=6)), build_problem(replace(REFERENCE, n_dates=10))
    assert short.layout == long.layout and short.config == replace(long.config, n_dates=6)
    assert_bitwise_equal(short.true_raw, long.true_raw)
    assert_bitwise_equal(short.starts, long.starts)
    assert_bitwise_equal(short.dataset.initial_filter, long.dataset.initial_filter)
    assert_bitwise_equal(short.dataset.true_initial_state, long.dataset.true_initial_state)
    for name in ("true_states", "base_process_noise", "process_noise", "noiseless_observations",
                 "base_observation_noise", "observation_noise", "observations", "inputs"):
        assert len(getattr(short.dataset, name)) == 6 and len(getattr(long.dataset, name)) == 10
        assert_bitwise_equal(getattr(short.dataset, name), getattr(long.dataset, name)[:6])
    prefix = prefix_problem(long, replace(REFERENCE, n_dates=6, n_starts=2, repeats=3))
    assert prefix.config == replace(REFERENCE, n_dates=6, n_starts=2, repeats=3).resolve()
    assert_bitwise_equal(prefix.dataset, short.dataset)
    assert_bitwise_equal(prefix.starts, short.starts[:2])
    with pytest.raises(ValueError, match="cannot exceed"):
        prefix_problem(long, replace(REFERENCE, n_dates=11))
    with pytest.raises(ValueError, match="share every generating field"):
        prefix_problem(long, replace(REFERENCE, n_dates=6, seed=1))


def test_variance_scales_multiply_variances_so_noise_draws_scale_by_the_square_root():
    base = build_problem(replace(REFERENCE, n_dates=6))
    observation = build_problem(replace(REFERENCE, n_dates=6, observation_variance_scale=4.0))
    assert_bitwise_equal(observation.true_parameters.sigma_v.diagonal, 4.0 * base.true_parameters.sigma_v.diagonal)
    assert_bitwise_equal(observation.dataset.true_states, base.dataset.true_states)
    assert_bitwise_equal(observation.dataset.noiseless_observations, base.dataset.noiseless_observations)
    assert_bitwise_equal(observation.dataset.base_observation_noise,
                         tuple(2.0 * draw for draw in base.dataset.base_observation_noise))
    for item, reference in zip(observation.dataset.inputs, base.dataset.inputs, strict=True):
        assert_bitwise_equal(item.sigma_v.diagonal, 4.0 * reference.sigma_v.diagonal)
        assert_bitwise_equal(item.sigma_w.diagonal, reference.sigma_w.diagonal)
    assert not np.array_equal(observation.true_raw, base.true_raw)
    process = build_problem(replace(REFERENCE, n_dates=6, process_variance_scale=4.0))
    assert_bitwise_equal(process.fixed_process_variance, 4.0 * base.fixed_process_variance)
    assert_bitwise_equal(process.true_raw, base.true_raw)
    assert_bitwise_equal(process.dataset.base_process_noise, tuple(2.0 * draw for draw in base.dataset.base_process_noise))
    assert_bitwise_equal(process.dataset.base_observation_noise, base.dataset.base_observation_noise)
    assert not np.array_equal(np.stack(process.dataset.true_states), np.stack(base.dataset.true_states))
    for item in process.dataset.inputs:
        assert_bitwise_equal(item.sigma_w.diagonal, jnp.array([0.01]))
    initial = build_problem(replace(REFERENCE, n_dates=6, initial_covariance_scale=4.0))
    assert_bitwise_equal(initial.dataset.initial_filter.covariance, 4.0 * base.dataset.initial_filter.covariance)
    assert_bitwise_equal(initial.dataset.initial_filter.state, base.dataset.initial_filter.state)
    # The sampled x_0 adds the mean after scaling, so this comparison is not bitwise.
    np.testing.assert_allclose(initial.dataset.true_initial_state - 0.35,
                               2.0 * (base.dataset.true_initial_state - 0.35), rtol=1e-13)
    explicit = build_problem(replace(REFERENCE, n_dates=6, observation_variances=(0.001, 0.002),
                                     observation_variance_scale=3.0))
    assert_bitwise_equal(explicit.true_parameters.sigma_v.diagonal, jnp.array([0.001, 0.002]) * 3.0)


def test_explicit_generating_values_and_starts_are_used_verbatim():
    config = BenchmarkConfig(family="larger", n_states=2, n_dates=3, persistence=(0.5, 0.7), loading=(1.0, 1.5),
                             process_variances=(0.01, 0.02), initial_mean=(0.0, 1.0),
                             start_offsets=((0.0,) * 8, (0.5,) * 8))
    problem = build_problem(config)
    assert_bitwise_equal(problem.true_parameters.theta_f, jnp.array([0.5, 0.7]))
    assert_bitwise_equal(problem.true_parameters.theta_g, jnp.array([1.0, 1.5]))
    assert_bitwise_equal(problem.fixed_process_variance, jnp.array([0.01, 0.02]))
    assert_bitwise_equal(problem.dataset.initial_filter.state, jnp.array([0.0, 1.0]))
    assert_bitwise_equal(problem.starts, (problem.true_raw, problem.true_raw + 0.5))
    assert problem.config.n_starts == 2
    fewer = build_problem(replace(PRESETS["larger3"], n_dates=2, n_starts=1))
    assert len(fewer.starts) == 1
    assert_bitwise_equal(fewer.starts[0], build_problem(replace(PRESETS["larger3"], n_dates=2)).starts[0])


def test_defaulted_fills_only_protocol_fields_and_unsupported_fields_are_refused():
    config = BenchmarkConfig(repeats=2)
    filled = config.defaulted(repeats=5, methods=("BFGS",))
    assert filled.repeats == 2 and filled.methods == ("BFGS",)
    assert config.defaulted(repeats=5).methods is None
    with pytest.raises(ValueError, match="protocol fields"):
        config.defaulted(n_dates=5)
    reject_unsupported(config, "some_benchmark", "methods")
    with pytest.raises(ValueError, match="does not support the 'repeats'"):
        reject_unsupported(config, "some_benchmark", "repeats")
    with pytest.raises(ValueError, match="does not support the 'n_starts'"):
        reject_unsupported(BenchmarkConfig(n_starts=1), "some_benchmark", "methods", "n_starts")


def test_problem_record_is_self_describing_and_json_serializable():
    config = replace(PRESETS["larger3"], n_dates=2, observation_variance_scale=2.0, repeats=1, methods=("GD",))
    problem = build_problem(config)
    record = json.loads(serialize(problem_record(problem)))
    assert record["config"] == json.loads(config.resolve().to_json())
    assert (record["T"], record["n_x"], record["n_z"], record["p"], record["seed"]) == (2, 3, 6, 12, 202625)
    assert record["preset"] is None and record["raw_order"] == "theta_f; sigma_v; theta_g"
    assert record["generating_parameters"]["sigma_v"] == np.asarray(problem.true_parameters.sigma_v.diagonal).tolist()
    assert record["generating_parameters"]["theta_f"] == list(config.resolve().persistence)
    assert record["fixed_process_variance"] == list(config.resolve().process_variances)
    assert record["fixed_initial_mean"] == list(config.resolve().initial_mean)
    assert np.asarray(record["fixed_initial_covariance"]).shape == (3, 3)
    assert len(record["starts"]) == 2 and len(record["generating_raw"]) == 12
    assert "variances" in record["noise_convention"]
    assert json.loads(serialize(problem_record(build_problem(REFERENCE))))["preset"] == "reference"


def parse(argv, *, preset=True):
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, preset=preset)
    return parser.parse_args(argv)


def test_command_line_precedence_preset_then_file_then_flags(tmp_path):
    assert config_from_arguments(parse([])) == BenchmarkConfig()
    assert config_from_arguments(parse([]), PRESETS["larger3"]) == PRESETS["larger3"]
    assert argument_overrides(parse([])) == {}
    args = parse(["--preset", "larger3", "--dates", "7", "--observation-variance-scale", "5",
                  "--start-offset", *["0"] * 12, "--start-offset", *["0.1"] * 12, "--methods", "BFGS", "GD"])
    assert argument_overrides(args) == dict(n_dates=7, observation_variance_scale=5.0,
                                            start_offsets=[[0.0] * 12, [0.1] * 12], methods=["BFGS", "GD"])
    config = config_from_arguments(args)
    assert config == replace(PRESETS["larger3"], n_dates=7, observation_variance_scale=5.0,
                             start_offsets=((0.0,) * 12, (0.1,) * 12), methods=("BFGS", "GD"))
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"preset": "long100", "seed": 5, "repeats": 2}), encoding="utf-8")
    assert config_from_arguments(parse(["--config", str(path)])) == replace(PRESETS["long100"], seed=5, repeats=2)
    assert config_from_arguments(parse(["--config", str(path), "--seed", "6"])) == replace(PRESETS["long100"], seed=6, repeats=2)
    with pytest.raises(ValueError, match="preset either"):
        config_from_arguments(parse(["--config", str(path), "--preset", "reference"]))
    path.write_text(json.dumps({"preset": "nope"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown preset"):
        config_from_arguments(parse(["--config", str(path)]))
    path.write_text(json.dumps({"dates": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown configuration fields"):
        config_from_arguments(parse(["--config", str(path)]))
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON object"):
        config_from_arguments(parse(["--config", str(path)]))
    with pytest.raises(SystemExit):
        parse(["--preset", "reference"], preset=False)
    with pytest.raises(SystemExit):
        parse(["--methods", "Adam"])


@pytest.mark.parametrize("argv", [
    ["--states", "0"], ["--states", "2"], ["--family", "larger"], ["--dates", "0"],
    ["--observation-variance-scale", "0"], ["--process-variance-scale", "-1"], ["--quotes-per-state", "3"],
    ["--persistence", "1.5"], ["--starts", "9"], ["--start-offset", "1", "2"], ["--repeats", "0"],
])
def test_invalid_command_line_values_fail_before_generation(monkeypatch, argv):
    monkeypatch.setattr(module, "generate_synthetic_dataset",
                        lambda *args, **kwargs: pytest.fail("generation reached with invalid flags"))
    with pytest.raises(ValueError):
        config_from_arguments(parse(argv))


def test_fixed_scan_benchmark_refuses_unsupported_fields_before_generating(scripts, monkeypatch):
    script = scripts["fixed_scan_scaling"]
    assert script["scan_ladder"](5000) == (12, 24, 100, 500, 1000, 5000)
    assert script["scan_ladder"](300) == (12, 24, 100, 300)
    assert script["scan_ladder"](12) == (12,)
    monkeypatch.setattr(module, "generate_synthetic_dataset",
                        lambda *args, **kwargs: pytest.fail("generation must not start"))
    monkeypatch.setattr(sys, "argv", ["fixed_scan_scaling.py", "--methods", "BFGS"])
    with pytest.raises(ValueError, match="does not support the 'methods'"):
        script["main"]()
    monkeypatch.setattr(sys, "argv", ["fixed_scan_scaling.py", "--dates", "5"])
    with pytest.raises(SystemExit):
        script["main"]()
    assert script["DEFAULT"] == replace(REFERENCE, n_dates=5000)


def test_curvature_benchmark_runs_a_custom_larger_case_with_non_default_noise(scripts, monkeypatch, capsys):
    script = scripts["curvature_optimization"]
    monkeypatch.setattr(sys, "argv", ["curvature_optimization.py", "--case", "all", "--dates", "5"])
    with pytest.raises(SystemExit):
        script["main"]()
    monkeypatch.setattr(sys, "argv", ["curvature_optimization.py", "--case", "larger3", "--states", "2", "--dates", "6",
                                      "--observation-variance-scale", "3", "--repeats", "1", "--methods", "BFGS"])
    script["main"]()
    records = [line.split(" ", 1) for line in capsys.readouterr().out.splitlines() if " " in line]
    lines = {}
    for kind, payload in records:
        lines.setdefault(kind, []).append(json.loads(payload))
    expected = replace(PRESETS["larger3"], n_states=2, n_dates=6, observation_variance_scale=3.0,
                       repeats=1, methods=("BFGS",)).resolve()
    assert lines["ENV"][0]["configs"] == {"larger3": json.loads(expected.to_json())}
    (problem,) = lines["PROBLEM"]
    assert problem["name"] == "custom_larger_n2_T6" and problem["preset"] is None
    assert (problem["T"], problem["n_x"], problem["n_z"], problem["p"], problem["seed"]) == (6, 2, 4, 8, 202625)
    assert problem["config"] == json.loads(expected.to_json())
    assert problem["generating_parameters"]["sigma_v"] == pytest.approx([3 * v for v in expected.observation_variances])
    assert lines["PROTOCOL"][0]["methods"] == ["BFGS"]
    assert [(run["method"], run["start"]) for run in lines["RUN"]] == [("BFGS", 0), ("BFGS", 1)]
    assert lines["TIMING"][0]["warmed_hvp"]["repeats"] == 1
    assert lines["CASE_COMPLETE"][0]["expected_runs"] == 2 and "ISSUE10_REGRESSION" not in lines
    assert lines["COMPLETE"][0]["case"] == "larger3"


def test_baseline_benchmark_runs_the_reference_family_with_non_default_noise(scripts, monkeypatch, capsys):
    script = scripts["baseline_optimization"]
    monkeypatch.setattr(sys, "argv", ["baseline_optimization.py", "--dates", "3", "--process-variance-scale", "2",
                                      "--observation-variance-scale", "0.5", "--methods", "BFGS", "--starts", "1"])
    script["main"]()
    out = capsys.readouterr().out.splitlines()
    config = json.loads(next(line for line in out if line.startswith("CONFIG ")).split(" ", 1)[1])
    problem = json.loads(next(line for line in out if line.startswith("PROBLEM ")).split(" ", 1)[1])
    expected = replace(REFERENCE, n_dates=3, process_variance_scale=2.0, observation_variance_scale=0.5,
                       methods=("BFGS",), n_starts=1).resolve()
    assert config == json.loads(expected.to_json()) == problem["config"]
    assert (problem["T"], problem["n_x"], problem["n_z"], problem["p"], problem["seed"]) == (3, 1, 2, 4, 20261010)
    assert problem["fixed_process_variance"] == [0.005] and problem["generating_parameters"]["sigma_v"] == [0.0002, 0.00045]
    assert any(line.startswith("BFGS 0 ") for line in out) and not any(line.startswith("BFGS 1 ") for line in out)
    assert not any(line.startswith(("GD ", "L-BFGS-B ")) for line in out)
    monkeypatch.setattr(sys, "argv", ["baseline_optimization.py", "--repeats", "3"])
    with pytest.raises(ValueError, match="does not support the 'repeats'"):
        script["main"]()
