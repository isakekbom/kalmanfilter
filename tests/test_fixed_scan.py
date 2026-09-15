"""Fixed-shape execution parity against the authoritative ragged Python driver."""

import ast
from dataclasses import replace
import inspect
from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np
import pytest

import kalmanfilter.fixed_scan as fixed_scan
from kalmanfilter.ekf import EKFInputs, initialize_filter
from kalmanfilter.fixed_scan import FixedScanInputs, run_fixed_scan_likelihood, stack_fixed_inputs
from kalmanfilter.likelihood import run_likelihood
from kalmanfilter.ois import OISInstrument
from kalmanfilter.optimization import CompiledObjective, run_multistart
from kalmanfilter.params import unpack_parameters
from kalmanfilter.synthetic import SyntheticStepInputs, generate_synthetic_dataset
from kalmanfilter.transition import (
    DenseMap, DiagonalMatrix, StateCoordinates, StructuralStep, selection_map,
)


ROOT = Path(__file__).resolve().parents[1]
make_problem = runpy.run_path(str(ROOT / "benchmarks/baseline_optimization.py"))["make_problem"]
make_scan_objective = runpy.run_path(str(ROOT / "benchmarks/fixed_scan_scaling.py"))["make_scan_objective"]
COORDINATES = StateCoordinates(("p1", "p2"), ("c",), ("u",))
TRACE_RTOL, TRACE_ATOL = 2e-12, 2e-13
GRAD_RTOL, GRAD_ATOL = 2e-11, 2e-12


def structural_step(coordinates=COORDINATES, active=("qa", "qb")):
    axes = coordinates.all
    identity = selection_map(axes, axes, axes)
    b = jnp.eye(len(axes)).at[0, 1].set(0.03).at[1, 0].set(-0.02)
    return StructuralStep(
        coordinates, coordinates, identity, DenseMap(axes, axes, b), identity,
        selection_map(active, coordinates.unsystematic,
                      tuple(coordinates.unsystematic[0] for _ in active)),
        selection_map(active, ("va", "vb"), tuple({"qa": "va", "qb": "vb"}[a] for a in active)),
    )


@pytest.fixture(scope="module")
def dataset():
    step = structural_step()
    dates = []
    for t in range(100):
        instruments = tuple(OISInstrument(
            jnp.full((k,), 0.5 + t * 1e-4),
            -jnp.arange(k + 1, dtype=jnp.float64)[:, None, None]
            * jnp.array([[[1.0], [0.3 + t * 1e-4]]]),
            -0.2 * jnp.arange(k + 1, dtype=jnp.float64)[:, None],
        ) for k in (1, 2))
        dates.append(SyntheticStepInputs(
            step, jnp.array([0.9, 0.8, 0.95, 0.7]) + 1e-4 * np.sin(t),
            DiagonalMatrix(jnp.array([0.0001, 0.0002, 0.0001, 0.00001]) * (1 + t * 1e-3)),
            jnp.array([0.7 + t * 1e-4]),
            DiagonalMatrix(jnp.array([0.001, 0.002]) * (1 + t * 1e-3)), instruments,
        ))
    return generate_synthetic_dataset(jax.random.key(20260923), COORDINATES,
        jnp.array([0.03, -0.01, 0.005, 0.002]), jnp.eye(4) * 0.001, dates)


def reference_trace(reference):
    steps = reference.trace.ekf_steps
    return dict(
        predicted_states=jnp.stack([s.prediction.predicted_state for s in steps]),
        predicted_covariances=jnp.stack([s.prediction.predicted_covariance for s in steps]),
        filtered_states=jnp.stack([s.filtered.state for s in steps]),
        filtered_covariances=jnp.stack([s.filtered.covariance for s in steps]),
        innovations=jnp.stack([s.update.innovation for s in steps]),
        innovation_covariances=jnp.stack([s.update.innovation_covariance for s in steps]),
        per_step_contributions=reference.per_step_contributions,
    )


@pytest.mark.parametrize("n_dates", [1, 5, 24, 100])
@pytest.mark.parametrize("dense_noise", [False, True])
def test_full_trace_float64_and_final_parity(dataset, n_dates, dense_noise):
    inputs = dataset.inputs[:n_dates]
    if dense_noise:
        inputs = tuple(item._replace(
            sigma_w=jnp.diag(item.sigma_w.diagonal).at[0, 1].set(0.00001).at[1, 0].set(0.00001),
            sigma_v=jnp.diag(item.sigma_v.diagonal).at[0, 1].set(0.0002).at[1, 0].set(0.0002),
        ) for item in inputs)
    batched = stack_fixed_inputs(inputs)
    reference = run_likelihood(dataset.initial_filter, inputs, return_trace=True)
    checked = jax.jit(checkify.checkify(
        lambda start, batch: run_fixed_scan_likelihood(start, batch, return_trace=True)))
    error, actual = checked(dataset.initial_filter, batched)
    error.throw()
    np.testing.assert_allclose(actual.total_log_likelihood, reference.total_log_likelihood,
                               rtol=TRACE_RTOL, atol=TRACE_ATOL)
    for name, expected in reference_trace(reference).items():
        observed = getattr(actual.trace, name)
        assert observed.dtype == jnp.float64
        np.testing.assert_allclose(observed, expected, rtol=TRACE_RTOL, atol=TRACE_ATOL)
    np.testing.assert_array_equal(actual.per_step_contributions, actual.trace.per_step_contributions)
    np.testing.assert_array_equal(actual.final.state, actual.trace.filtered_states[-1])
    np.testing.assert_array_equal(actual.final.covariance, actual.trace.filtered_covariances[-1])
    assert actual.final.coordinates == COORDINATES


def test_stack_preserves_all_numerical_leaves_time_order_and_shared_step(dataset):
    inputs = dataset.inputs[:5]
    batch = stack_fixed_inputs(inputs)
    assert batch.step is inputs[0].step
    assert batch.n_steps == 5
    assert [i.accrual_factors.shape for i in batch.instruments] == [(5, 1), (5, 2)]
    for t, item in enumerate(inputs):
        sliced = jax.tree.map(lambda leaf: leaf[t], batch[1:])
        for actual, expected in zip(jax.tree.leaves(sliced), jax.tree.leaves(item[1:]), strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_distinct_but_identical_dense_and_weighted_maps_are_accepted(dataset):
    item = dataset.inputs[0]
    post = replace(item.step.transition_post_map, weights=jnp.ones(4))
    first = item._replace(step=replace(item.step, transition_post_map=post))
    second = first._replace(step=jax.tree.map(lambda a: a.copy(), first.step))
    assert second.step is not first.step
    assert stack_fixed_inputs((first, second)).n_steps == 2


@pytest.mark.parametrize("coordinates", [
    StateCoordinates(("p2", "p1"), ("c",), ("u",)),
    StateCoordinates(("p1", "other"), ("c",), ("u",)),
    StateCoordinates(("p1",), ("p2", "c"), ("u",)),
    StateCoordinates(("p1", "p2", "extra"), ("c",), ("u",)),
])
def test_coordinate_identity_order_blocks_and_dimension_changes_fail(dataset, coordinates):
    item = dataset.inputs[0]
    with pytest.raises(ValueError, match="coordinate identity/order"):
        stack_fixed_inputs((item, item._replace(step=structural_step(coordinates))))


@pytest.mark.parametrize("active", [("qb", "qa"), ("qa",), ()])
def test_active_identity_order_count_and_mixed_empty_fail(dataset, active):
    item = dataset.inputs[0]
    with pytest.raises(ValueError, match="active observation identity/order"):
        stack_fixed_inputs((item, item._replace(step=structural_step(active=active))))


@pytest.mark.parametrize("field", [
    "transition_post_map", "transition_pre_map", "process_noise_map", "observation_noise_map",
])
def test_changed_structural_map_weights_fail(dataset, field):
    item = dataset.inputs[0]
    mapping = getattr(item.step, field)
    if isinstance(mapping, DenseMap):
        changed = replace(mapping, values=mapping.values * 1.1)
    else:
        mapping = replace(mapping, weights=jnp.ones(len(mapping.rows)))
        item = item._replace(step=replace(item.step, **{field: mapping}))
        changed = replace(mapping, weights=mapping.weights * 1.1)
    other = item._replace(step=replace(item.step, **{field: changed}))
    with pytest.raises(ValueError, match="map weights"):
        stack_fixed_inputs((item, other))


@pytest.mark.parametrize("field", ["transition_post_map", "process_noise_map", "observation_selector", "observation_noise_map"])
def test_changed_structural_map_topology_fails(dataset, field):
    item = dataset.inputs[0]
    mapping = getattr(item.step, field)
    changed = replace(mapping, sources=(None,) + mapping.sources[1:])
    with pytest.raises(ValueError, match="map topology"):
        stack_fixed_inputs((item, item._replace(step=replace(item.step, **{field: changed}))))


@pytest.mark.parametrize("field", ["accrual_factors", "pca_loading_map", "step_loading"])
def test_corresponding_instrument_leaf_shapes_fail(dataset, field):
    item = dataset.inputs[0]
    instrument = item.instruments[0]
    changed = instrument._replace(**{field: getattr(instrument, field)[:-1]})
    with pytest.raises(ValueError, match="shapes must stay fixed"):
        stack_fixed_inputs((item, item._replace(instruments=(changed, item.instruments[1]))))


def test_container_structure_changes_fail(dataset):
    item = dataset.inputs[0]
    for other in (item._replace(instruments=item.instruments[:1]),
                  item._replace(sigma_v=jnp.diag(item.sigma_v.diagonal))):
        with pytest.raises(ValueError, match="container structure"):
            stack_fixed_inputs((item, other))


def test_stack_requires_nonempty_ekf_sequence():
    with pytest.raises(ValueError, match="nonempty"):
        stack_fixed_inputs(())
    with pytest.raises(TypeError, match="EKFInputs"):
        stack_fixed_inputs((1,))


@pytest.mark.parametrize("malformation,match", [
    (lambda b: b._replace(observations=b.observations[:, 0]), "rank two"),
    (lambda b: b._replace(theta_f=b.theta_f[:-1]), "theta_f shape"),
    (lambda b: b._replace(sigma_v=DiagonalMatrix(b.sigma_v.diagonal[:, :1])), "sigma_v diagonal shape"),
    (lambda b: b._replace(instruments=b.instruments[:1]), "instrument count"),
    (lambda b: b._replace(theta_g=b.theta_g[:, :0]), "pca_loading_map shape"),
])
def test_direct_batched_shapes_validated(dataset, malformation, match):
    batch = stack_fixed_inputs(dataset.inputs[:2])
    with pytest.raises(ValueError, match=match):
        run_fixed_scan_likelihood(dataset.initial_filter, malformation(batch))


def test_initial_coordinates_and_nonfixed_first_step_rejected(dataset):
    batch = stack_fixed_inputs(dataset.inputs[:2])
    different = StateCoordinates(("p2", "p1"), ("c",), ("u",))
    initial = initialize_filter(different, dataset.initial_filter.state, dataset.initial_filter.covariance)
    with pytest.raises(ValueError, match="initial filter coordinates"):
        run_fixed_scan_likelihood(initial, batch)
    step = batch.step
    changed = replace(step, previous=different,
                      transition_pre_map=replace(step.transition_pre_map, columns=different.all))
    with pytest.raises(ValueError, match="identical previous/current"):
        stack_fixed_inputs((dataset.inputs[0]._replace(step=changed),))


def test_all_empty_observations_are_prediction_only(dataset):
    inputs = tuple(item._replace(step=structural_step(active=()),
                                observations=jnp.empty(0), instruments=())
                   for item in dataset.inputs[:5])
    batch = stack_fixed_inputs(inputs)
    reference = run_likelihood(dataset.initial_filter, inputs, return_trace=True)
    actual = run_fixed_scan_likelihood(dataset.initial_filter, batch, return_trace=True)
    assert actual.total_log_likelihood == 0
    assert actual.trace.innovations.shape == (5, 0)
    assert actual.trace.innovation_covariances.shape == (5, 0, 0)
    for name, expected in reference_trace(reference).items():
        np.testing.assert_allclose(getattr(actual.trace, name), expected, rtol=TRACE_RTOL, atol=TRACE_ATOL)
    np.testing.assert_array_equal(actual.trace.filtered_states, actual.trace.predicted_states)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("failure", ["nonfinite", "negative_variance", "singular_innovation", "pricing_overflow"])
def test_numerical_failures_propagate_without_repairs(dataset, compiled, failure):
    batch = stack_fixed_inputs(dataset.inputs[:5])
    if failure == "nonfinite":
        batch = batch._replace(observations=batch.observations.at[3, 0].set(jnp.nan))
    elif failure == "negative_variance":
        batch = batch._replace(sigma_v=DiagonalMatrix(batch.sigma_v.diagonal.at[3, 0].set(-1)))
    elif failure == "singular_innovation":
        # Both rows become the same observation and R=0: S is singular.
        batch = batch._replace(instruments=(batch.instruments[0], batch.instruments[0]),
                               sigma_v=DiagonalMatrix(jnp.zeros_like(batch.sigma_v.diagonal)))
    else:
        batch = batch._replace(theta_g=jnp.full_like(batch.theta_g, 1e308))
    if compiled:
        function = jax.jit(checkify.checkify(run_fixed_scan_likelihood))
        error, _ = function(dataset.initial_filter, batch)
        with pytest.raises(checkify.JaxRuntimeError):
            error.throw()
    else:
        with pytest.raises(checkify.JaxRuntimeError):
            run_fixed_scan_likelihood(dataset.initial_filter, batch)


@pytest.fixture(scope="module")
def estimation_case():
    problem = make_problem(12)
    batch = stack_fixed_inputs(problem.dataset.inputs)
    scan = make_scan_objective(problem.dataset.initial_filter, batch, problem.layout)
    python_compiled = CompiledObjective(problem.objective, problem.true_raw)
    scan_compiled = CompiledObjective(scan, problem.true_raw)
    return problem, batch, scan, python_compiled, scan_compiled


@pytest.mark.parametrize("vector", [0, 1, 2, 3])
def test_raw_value_and_gradient_parity(estimation_case, vector):
    problem, _, _, python, scan = estimation_case
    raw = (problem.true_raw,) + problem.starts
    expected_value, expected_gradient = python.evaluate(raw[vector])
    actual_value, actual_gradient = scan.evaluate(raw[vector])
    np.testing.assert_allclose(actual_value, expected_value, rtol=TRACE_RTOL, atol=TRACE_ATOL)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=GRAD_RTOL, atol=GRAD_ATOL)


def test_invalid_raw_objective_does_not_become_finite_penalty_or_zero_gradient(estimation_case):
    problem, _, _, _, scan = estimation_case
    invalid = problem.true_raw.at[1].set(-1000.0)  # softplus underflow
    with pytest.raises(checkify.JaxRuntimeError):
        scan.evaluate(invalid)


def test_bfgs_two_start_mathematical_parameter_parity(estimation_case):
    problem, _, _, python, scan = estimation_case
    old = run_multistart(python, problem.starts[:2], method="BFGS")
    new = run_multistart(scan, problem.starts[:2], method="BFGS")
    for expected, actual in zip(old.runs, new.runs, strict=True):
        assert expected.success, expected.message
        assert actual.success, actual.message
        assert expected.gradient_norm < 1e-6 and actual.gradient_norm < 1e-6
        np.testing.assert_allclose(actual.final_objective, expected.final_objective, rtol=0, atol=2e-10)
        expected_params = unpack_parameters(expected.final_raw, problem.layout)
        actual_params = unpack_parameters(actual.final_raw, problem.layout)
        for a, b in zip(jax.tree.leaves(actual_params), jax.tree.leaves(expected_params), strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-7, atol=2e-9)


def test_scan_objective_reuses_prepared_arrays_and_has_no_python_time_loop(estimation_case, monkeypatch):
    problem, _, objective, _, _ = estimation_case
    def forbidden(*args, **kwargs):
        pytest.fail("stacking must remain outside the differentiated objective")
    monkeypatch.setattr(fixed_scan, "stack_fixed_inputs", forbidden)
    # The actual benchmark closure's body must only transform/broadcast batched
    # inputs, without a for/while/comprehension rebuilding per-date objects.
    source = ast.parse(inspect.getsource(make_scan_objective))
    objective_ast = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "objective")
    assert not any(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ast.walk(objective_ast))
    error, result = checkify.checkify(jax.value_and_grad(objective))(problem.true_raw)
    error.throw()
    assert np.isfinite(result[1]).all()
    assert "jax.lax.scan(" in inspect.getsource(fixed_scan._scan_likelihood)


def test_traced_graph_contains_one_time_scan_and_does_not_expand_with_t(estimation_case):
    problem, batch, _, _, _ = estimation_case
    graphs = []
    for t in (1, 100):
        # Setup-only repeat gives two lengths of identical numerical structure.
        resized = FixedScanInputs(batch.step, *jax.tree.map(
            lambda a: jnp.broadcast_to(a[0], (t,) + a.shape[1:]), batch[1:]))
        objective = make_scan_objective(problem.dataset.initial_filter, resized, problem.layout)
        graph = jax.make_jaxpr(checkify.checkify(objective))(problem.true_raw).jaxpr
        scans = [e for e in graph.eqns if e.primitive.name == "scan"]
        assert len(scans) == 1
        assert scans[0].params["length"] == t
        assert scans[0].params["unroll"] == 1
        graphs.append((len(graph.eqns), len(scans[0].params["jaxpr"].jaxpr.eqns)))
    assert graphs[0] == graphs[1]
