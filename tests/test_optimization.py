"""Optimizer plumbing and a generated nonlinear raw-parameter likelihood."""

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import runpy
from types import SimpleNamespace

import kalmanfilter.optimization as optimization
import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np
import pytest

from kalmanfilter.optimization import CompiledObjective, run_multistart, run_optimization
from kalmanfilter.params import unpack_parameters


MATRIX = jnp.array([[3.0, 0.4], [0.4, 1.0]])
LINEAR = jnp.array([-1.0, 2.0])
OPTIMUM = -np.linalg.solve(np.asarray(MATRIX), np.asarray(LINEAR))


def quadratic(raw):
    # Algebraically 0.5*x.T*A*x+b.T*x minus its minimum. Center the expression
    # to avoid cancellation of nonzero objective terms near the known optimum.
    displacement = raw - OPTIMUM
    return 0.5 * displacement @ MATRIX @ displacement


@pytest.fixture(scope="module")
def quadratic_runs():
    starts = (jnp.array([4.0, -3.0]), jnp.array([-2.0, 5.0]))
    objective = CompiledObjective(quadratic, starts[0])
    groups = {method: run_multistart(objective, starts, method=method, gradient_tolerance=1e-9,
                                     function_tolerance=1e-15)
              for method in ("BFGS", "L-BFGS-B")}
    return objective, starts, groups


@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B"])
def test_supplied_gradient_solvers_recover_analytical_quadratic_optimum(quadratic_runs, method):
    _, starts, groups = quadratic_runs
    result = groups[method]
    assert len(result.runs) == len(starts)
    for start, run in zip(starts, result.runs, strict=True):
        assert run.success, run.message
        assert run.status == 0
        assert run.message
        assert run.gradient_norm < 1e-7
        np.testing.assert_allclose(run.final_raw, OPTIMUM, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(run.final_gradient, MATRIX @ run.final_raw + LINEAR, atol=1e-14)
        np.testing.assert_allclose(run.final_objective, quadratic(OPTIMUM), atol=1e-13)
        np.testing.assert_array_equal(run.initial_raw, start)
        assert run.final_objective < run.initial_objective


def test_iteration_zero_accepted_history_and_separate_timings(quadratic_runs):
    objective, _, groups = quadratic_runs
    for group in groups.values():
        assert group.compilation_seconds == objective.compilation_seconds >= 0
        for run in group.runs:
            assert run.compilation_seconds == group.compilation_seconds
            assert run.optimization_seconds >= 0
            assert len(run.history) == run.iterations + 1
            np.testing.assert_array_equal(run.history[0].raw, run.initial_raw)
            np.testing.assert_array_equal(run.history[-1].raw, run.final_raw)
            assert run.history[0].step_norm == 0
            assert run.history[0].objective == run.initial_objective
            assert run.history[-1].objective == run.final_objective
            previous_time = 0
            for i, record in enumerate(run.history):
                assert record.iteration == i
                assert previous_time <= record.elapsed_seconds <= run.optimization_seconds
                previous_time = record.elapsed_seconds
                assert np.isfinite([record.objective, record.gradient_norm, record.step_norm]).all()
                np.testing.assert_allclose(record.objective, quadratic(record.raw), atol=1e-13)
                np.testing.assert_allclose(record.gradient_norm, np.linalg.norm(MATRIX @ record.raw + LINEAR), atol=1e-13)


def test_actual_evaluations_include_initial_points_and_exclude_shared_warmup(quadratic_runs):
    objective, _, groups = quadratic_runs
    runs = tuple(run for group in groups.values() for run in group.runs)
    assert objective.total_evaluations == 1 + sum(run.function_evaluations for run in runs)
    for run in runs:
        assert run.function_evaluations == run.gradient_evaluations > 0
        assert run.solver_function_evaluations == run.solver_gradient_evaluations > 0
        # These ordinary accepted-iterate callbacks reuse the latest evaluation.
        assert run.function_evaluations == run.solver_function_evaluations


def test_multistart_preserves_all_runs_and_selects_lowest_objective(quadratic_runs):
    _, _, groups = quadratic_runs
    for group in groups.values():
        assert len(group.runs) == 2
        assert group.best is group.runs[group.best_index]
        assert group.best.final_objective == min(run.final_objective for run in group.runs)
        with pytest.raises(FrozenInstanceError):
            group.runs = ()


def test_same_shape_starts_and_methods_trace_once_and_count_real_dispatches(monkeypatch):
    traces = []
    def objective(raw):
        traces.append(raw.shape)
        return quadratic(raw)
    compiled = CompiledObjective(objective, [4, -3])
    assert traces == [(2,)]
    dispatched = []
    actual = compiled._compiled
    def counted(raw):
        dispatched.append(raw.dtype)
        return actual(raw)
    monkeypatch.setattr(compiled, "_compiled", counted)
    groups = [run_multistart(compiled, ([4, -3], [-2, 5]), method=method)
              for method in ("BFGS", "L-BFGS-B")]
    assert traces == [(2,)]
    assert len(dispatched) == sum(run.function_evaluations for group in groups for run in group.runs)
    assert compiled.total_evaluations == 1 + len(dispatched)
    assert all(dtype == jnp.float64 for dtype in dispatched)


def test_bridge_passes_combined_jax_gradient_and_no_bounds(monkeypatch):
    scipy_minimize = optimization.minimize
    calls = []
    def inspected(fun, initial, **kwargs):
        calls.append(kwargs)
        assert kwargs["jac"] is True
        assert kwargs["bounds"] is None
        assert "hess" not in kwargs and "hessp" not in kwargs
        value, gradient = fun(initial)
        assert isinstance(value, float)
        assert gradient.dtype == np.float64
        np.testing.assert_allclose(gradient, MATRIX @ initial + LINEAR, atol=1e-14)
        return scipy_minimize(fun, initial, **kwargs)
    monkeypatch.setattr(optimization, "minimize", inspected)
    compiled = CompiledObjective(quadratic, [4, -3])
    for method in ("BFGS", "L-BFGS-B"):
        assert run_optimization(compiled, [4, -3], method=method).success
    assert len(calls) == 2


def test_callback_cache_miss_is_counted_and_solver_failure_is_preserved(monkeypatch):
    def solver(fun, initial, callback, **kwargs):
        fun(initial)  # Cache hit after iteration zero.
        fun(initial + 1)  # Different trial point.
        callback(initial)  # Accepted point needs one explicitly counted call.
        return SimpleNamespace(x=initial, nit=1, success=False, status=2,
                               message="Deliberate line-search failure for accounting test", nfev=2, njev=2)
    monkeypatch.setattr(optimization, "minimize", solver)
    compiled = CompiledObjective(quadratic, [4, -3])
    run = run_optimization(compiled, [4, -3])
    assert run.function_evaluations == run.gradient_evaluations == 3
    assert run.solver_function_evaluations == run.solver_gradient_evaluations == 2
    assert compiled.total_evaluations == 4
    assert not run.success and run.status == 2
    assert run.message == "Deliberate line-search failure for accounting test"
    assert run.final_objective == run.initial_objective


def test_compilation_timing_synchronizes_before_stopping_clock(monkeypatch):
    events = []
    actual_sync = jax.block_until_ready
    def clock():
        events.append("clock")
        return float(len(events))
    def synchronize(result):
        events.append("synchronize")
        return actual_sync(result)
    monkeypatch.setattr(optimization, "perf_counter", clock)
    monkeypatch.setattr(jax, "block_until_ready", synchronize)
    compiled = CompiledObjective(quadratic, [4, -3])
    assert events == ["clock", "synchronize", "clock"]
    assert compiled.compilation_seconds == 2
    events.clear()
    result = run_optimization(compiled, [4, -3], max_iterations=0)
    assert events[-1] == "clock"
    assert events.index("synchronize") < len(events) - 1
    assert result.function_evaluations == 1
    assert result.compilation_seconds == 2


def test_fixed_step_gradient_descent_is_a_plain_diagnostic():
    start = np.array([4.0, -3.0])
    compiled = CompiledObjective(quadratic, start)
    result = run_optimization(compiled, start, method="GD", learning_rate=0.1,
                              max_iterations=300, gradient_tolerance=1e-8)
    assert result.success
    assert result.gradient_norm < 1e-8
    assert result.function_evaluations == result.iterations + 1
    assert result.solver_function_evaluations is result.solver_gradient_evaluations is None
    np.testing.assert_allclose(result.final_raw, OPTIMUM, atol=2e-8)
    for previous, current in zip(result.history, result.history[1:]):
        expected = previous.raw - 0.1 * (MATRIX @ previous.raw + LINEAR)
        np.testing.assert_allclose(current.raw, expected, atol=2e-15)
    np.testing.assert_array_equal(start, [4, -3])


@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B", "GD"])
def test_iteration_limit_is_not_rewritten_as_success(method):
    compiled = CompiledObjective(quadratic, [4, -3])
    result = run_optimization(compiled, [4, -3], method=method, max_iterations=1, gradient_tolerance=1e-12)
    assert result.iterations == 1
    assert not result.success
    assert result.status != 0
    assert result.message


def test_results_and_raw_arrays_are_immutable(quadratic_runs):
    result = quadratic_runs[2]["BFGS"].runs[0]
    with pytest.raises(FrozenInstanceError):
        result.success = False
    with pytest.raises(FrozenInstanceError):
        result.history[0].objective = 0
    with pytest.raises(TypeError):
        result.final_raw[0] = 0
    assert result.final_raw.dtype == result.final_gradient.dtype == jnp.float64


@pytest.mark.parametrize("point", [[], [[1, 2]], [True, False], [1j, 2], [np.nan, 1], [np.inf, 1]])
def test_invalid_raw_vectors_are_rejected(point):
    with pytest.raises((ValueError, TypeError)):
        CompiledObjective(quadratic, point)


def test_reuse_rejects_different_shape_without_recompilation():
    compiled = CompiledObjective(quadratic, [4, -3])
    with pytest.raises(ValueError, match="shape"):
        run_optimization(compiled, [1])
    assert compiled.total_evaluations == 1


@pytest.mark.parametrize("objective,error,match", [
    (lambda x: x, ValueError, "scalar"),
    (lambda x: jnp.sum(x).astype(jnp.float32), TypeError, "float64"),
    (lambda x: jnp.sum(x) * jnp.inf, checkify.JaxRuntimeError, "finite"),
    (lambda x: jnp.sqrt(x[0]), checkify.JaxRuntimeError, "gradient must be finite"),
])
def test_invalid_objective_value_or_gradient_fails(objective, error, match):
    with pytest.raises(error, match=match):
        CompiledObjective(objective, [0.0])


@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B", "GD"])
def test_checked_invalid_optimizer_proposal_is_never_a_finite_penalty(method):
    def positive_domain(raw):
        checkify.check(jnp.all(raw > 0), "test objective requires positive input")
        return jnp.log(raw[0])
    compiled = CompiledObjective(positive_domain, [1.0])
    with pytest.raises(checkify.JaxRuntimeError, match="positive input"):
        run_optimization(compiled, [1.0], method=method, learning_rate=2)


@pytest.mark.parametrize("options", [
    {"method": "Newton"}, {"max_iterations": -1}, {"max_iterations": True},
    {"gradient_tolerance": 0}, {"function_tolerance": np.inf}, {"learning_rate": -1},
])
def test_invalid_options_fail_before_run_evaluation(options):
    compiled = CompiledObjective(quadratic, [4, -3])
    with pytest.raises(ValueError):
        run_optimization(compiled, [4, -3], **options)
    assert compiled.total_evaluations == 1


def test_empty_multistart_fails():
    compiled = CompiledObjective(quadratic, [4, -3])
    with pytest.raises(ValueError, match="at least one"):
        run_multistart(compiled, ())


@pytest.fixture(scope="module")
def synthetic_runs():
    # Load the standalone benchmark explicitly; no production code imports tests.
    make_problem = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks" / "baseline_optimization.py"))["make_problem"]
    problem = make_problem(n_dates=12)
    compiled = CompiledObjective(problem.objective, problem.starts[0])
    truth_value, _ = compiled.evaluate(problem.true_raw)
    groups = {method: run_multistart(compiled, problem.starts[:2], method=method)
              for method in ("BFGS", "L-BFGS-B")}
    return problem, truth_value, groups


@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B"])
def test_generated_nonlinear_raw_likelihood_optimization(synthetic_runs, method):
    problem, truth_value, groups = synthetic_runs
    assert problem.layout.n_x0 == problem.layout.n_w == 0
    assert problem.layout.theta_f_transform == "unit_interval"
    assert len(problem.dataset.inputs) == 12
    for result in groups[method].runs:
        print(f"{method}: initial={result.initial_objective:.12g} final={result.final_objective:.12g} "
              f"truth={truth_value:.12g} grad={result.gradient_norm:.9g} status={result.status}: {result.message}")
        assert result.final_objective < result.initial_objective
        assert result.gradient_norm < 1e-3 * result.history[0].gradient_norm
        assert result.gradient_norm < 1e-4
        assert result.success, result.message
        assert result.final_objective < truth_value
        assert np.isfinite(result.final_raw).all()
        fitted = unpack_parameters(result.final_raw, problem.layout)
        for leaf in jax.tree_util.tree_leaves(fitted):
            assert leaf.dtype == jnp.float64
            assert np.isfinite(leaf).all()
        assert np.all((fitted.theta_f > 0) & (fitted.theta_f < 1))
        assert np.all(fitted.sigma_v.diagonal > 0)
        print("  fitted phi, observation variances, theta_g:", fitted.theta_f, fitted.sigma_v.diagonal, fitted.theta_g)


def test_reasonable_starts_and_methods_reach_similar_synthetic_objectives(synthetic_runs):
    _, _, groups = synthetic_runs
    objectives = [run.final_objective for group in groups.values() for run in group.runs]
    assert max(objectives) - min(objectives) < 1e-7


def test_selected_mathematical_parameters_recover_from_moderate_start(synthetic_runs):
    problem, _, groups = synthetic_runs
    initial = unpack_parameters(problem.starts[1], problem.layout)
    truth = problem.true_parameters
    for method in ("BFGS", "L-BFGS-B"):
        fitted = unpack_parameters(groups[method].runs[1].final_raw, problem.layout)
        assert abs(fitted.theta_f[0] - truth.theta_f[0]) < abs(initial.theta_f[0] - truth.theta_f[0])
        assert abs(fitted.theta_g[0] - truth.theta_g[0]) < abs(initial.theta_g[0] - truth.theta_g[0])
        assert abs(fitted.sigma_v.diagonal[0] - truth.sigma_v.diagonal[0]) < abs(initial.sigma_v.diagonal[0] - truth.sigma_v.diagonal[0])
        # The second variance is poorly recovered on 12 dates; do not force it
        # toward truth when the measured finite-sample optimum lies elsewhere.


def test_optimizer_source_excludes_dense_hessians_research_and_hidden_starts():
    tree = ast.parse(Path(optimization.__file__).read_text(encoding="utf-8"))
    forbidden = {"hessian", "jacfwd", "jacrev", "hess", "eigh", "eigvalsh", "inv", "pinv",
                 "random", "clip", "nan_to_num", "central_difference_gradient", "stop_gradient"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            assert name not in forbidden
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("jaxopt", "optax", "random"))
