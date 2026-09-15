"""Exact HVP validation and curvature solvers, independent of timing thresholds."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import runpy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np
import pytest

import kalmanfilter.optimization as optimization
from kalmanfilter.fixed_scan import stack_fixed_inputs
from kalmanfilter.optimization import CompiledObjective, run_multistart, run_optimization
from kalmanfilter.params import unpack_parameters


A = jnp.array([[4.0, 0.6, -0.2], [0.6, 2.0, 0.3], [-0.2, 0.3, 1.2]])
B = jnp.array([1.0, -2.0, 0.5])
OPTIMUM = -np.linalg.solve(np.asarray(A), np.asarray(B))
POINTS = (jnp.array([0.2, -0.4, 0.7]), jnp.array([-0.8, 1.1, -0.5]))
DIRECTIONS = (jnp.array([1.0, 2.0, -1.0]), jnp.array([-0.5, 0.25, 1.0]), jnp.ones(3))
METHODS = ("Newton-CG", "trust-krylov")


def quadratic(raw):
    return 0.5 * raw @ A @ raw + B @ raw + 2.0


def test_spd_quadratic_value_gradient_dense_hessian_and_optimum():
    assert np.linalg.eigvalsh(np.asarray(A)).min() > 0
    compiled = CompiledObjective(quadratic, POINTS[0])
    for point in POINTS + (jnp.asarray(OPTIMUM),):
        value, gradient = compiled.evaluate(point)
        np.testing.assert_allclose(value, 0.5 * np.asarray(point) @ np.asarray(A) @ point + B @ point + 2,
                                   rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(gradient, A @ point + B, rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(jax.hessian(quadratic)(point), A, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(compiled.evaluate(OPTIMUM)[1], 0, atol=1e-15)


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_quadratic_hvp_analytical_dense_and_gradient_finite_differences(point, direction):
    compiled = CompiledObjective(quadratic, point)
    actual = compiled.hessian_vector_product(point, direction)
    np.testing.assert_allclose(actual, A @ direction, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(actual, jax.hessian(quadratic)(point) @ direction, rtol=2e-15, atol=2e-15)
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        numerical = (compiled.evaluate(point + h * direction)[1]
                     - compiled.evaluate(point - h * direction)[1]) / (2 * h)
        np.testing.assert_allclose(numerical, actual, rtol=2e-9, atol=2e-9)
    assert actual.dtype == np.float64


def test_hvp_is_linear_in_supplied_direction_and_result_is_a_copy():
    compiled = CompiledObjective(quadratic, POINTS[0])
    v, w = DIRECTIONS[:2]
    first = compiled.hessian_vector_product(POINTS[0], v)
    np.testing.assert_allclose(compiled.hessian_vector_product(POINTS[0], 2 * v - w),
                               2 * first - A @ w, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(compiled.hessian_vector_product(POINTS[0], jnp.zeros(3)), 0)
    first[:] = 999
    np.testing.assert_allclose(compiled.hessian_vector_product(POINTS[0], v), A @ v)


@pytest.mark.parametrize("method", ["GD", "BFGS", "L-BFGS-B"])
def test_first_order_methods_never_compile_or_dispatch_hvp(method):
    compiled = CompiledObjective(quadratic, POINTS[0])
    def forbidden(*args):
        pytest.fail("first-order method requested HVP")
    compiled._hvp_function = forbidden
    run = run_optimization(compiled, POINTS[0], method=method, max_iterations=3)
    assert compiled._compiled_hvp is None
    assert compiled.total_hvp_evaluations == run.hvp_evaluations == 0
    assert compiled.hvp_compilation_seconds == run.hvp_compilation_seconds == 0
    assert run.solver_hessian_evaluations is None


def test_lazy_checked_hvp_jit_traces_once_and_counts_failures(monkeypatch):
    compiled = CompiledObjective(lambda x: jnp.sum(x ** 1.5), [1.0])
    traces = []
    actual = compiled._hvp_function
    def counted(*args):
        traces.append(1)
        return actual(*args)
    compiled._hvp_function = counted
    for point in ([1.0], [2.0]):
        compiled.hessian_vector_product(point, [1])
    # f and gradient are finite at zero; its second derivative is not.
    assert compiled.evaluate([0.0]) == (0.0, np.array([0.0]))
    with pytest.raises(checkify.JaxRuntimeError, match="HVP must be finite"):
        compiled.hessian_vector_product([0.0], [1.0])
    assert traces == [1]
    assert compiled.total_hvp_evaluations == 3
    assert compiled.total_evaluations == 2
    error, _ = compiled._compiled_hvp(jnp.array([0.0]), jnp.array([1.0]))
    with pytest.raises(checkify.JaxRuntimeError, match="HVP must be finite"):
        error.throw()


@pytest.mark.parametrize("function,bad_point,match", [
    (lambda x: jnp.sqrt(x[0]), [-1.0], "objective must be finite"),
    (lambda x: jnp.sqrt(x[0]), [0.0], "gradient must be finite"),
    (lambda x: jnp.exp(x[0]), [1000.0], "objective must be finite"),
])
def test_hvp_checks_primal_objective_and_gradient(function, bad_point, match):
    compiled = CompiledObjective(function, [1.0])
    with pytest.raises(checkify.JaxRuntimeError, match=match):
        compiled.hessian_vector_product(bad_point, [1.0])
    assert compiled.total_hvp_evaluations == 1


@pytest.mark.parametrize("vector", [[], [[1, 2, 3]], [1, 2], [True, False, True],
                                     [1j, 2, 3], [np.nan, 2, 3], [np.inf, 2, 3]])
def test_invalid_directions_fail_before_dispatch(vector):
    compiled = CompiledObjective(quadratic, POINTS[0])
    with pytest.raises((ValueError, TypeError)):
        compiled.hessian_vector_product(POINTS[0], vector)
    assert compiled.total_hvp_evaluations == 0 and compiled._compiled_hvp is None


def test_hvp_first_call_timer_synchronizes_and_is_not_recharged(monkeypatch):
    compiled = CompiledObjective(quadratic, POINTS[0])
    events = []
    sync = jax.block_until_ready
    def clock():
        events.append("clock")
        return float(len(events))
    def synchronize(value):
        events.append("sync")
        return sync(value)
    monkeypatch.setattr(optimization, "perf_counter", clock)
    monkeypatch.setattr(jax, "block_until_ready", synchronize)
    compiled.hessian_vector_product(POINTS[0], DIRECTIONS[0])
    assert events == ["clock", "sync", "clock"]
    assert compiled.hvp_compilation_seconds == 2
    events.clear()
    compiled.hessian_vector_product(POINTS[1], DIRECTIONS[1])
    assert events == ["sync"] and compiled.hvp_compilation_seconds == 2


@pytest.mark.parametrize("method", METHODS)
def test_curvature_solvers_recover_spd_quadratic_optimum_and_count_dispatches(method):
    compiled = CompiledObjective(quadratic, POINTS[0])
    group = run_multistart(compiled, POINTS, method=method, gradient_tolerance=1e-8, step_tolerance=1e-10)
    assert compiled.total_hvp_evaluations == 1 + sum(r.hvp_evaluations for r in group.runs)
    assert compiled.total_evaluations == 1 + sum(r.function_evaluations for r in group.runs)
    for result in group.runs:
        assert result.success, result.message
        np.testing.assert_allclose(result.final_raw, OPTIMUM, rtol=0, atol=2e-7)
        np.testing.assert_allclose(result.final_objective, quadratic(OPTIMUM), atol=2e-13)
        assert result.gradient_norm < 1e-6
        assert result.hvp_evaluations > 0
        assert result.solver_hessian_evaluations > 0
        assert len(result.history) == result.iterations + 1
        assert result.hvp_compilation_seconds == group.hvp_compilation_seconds
        assert result.final_raw.dtype == result.final_gradient.dtype == jnp.float64
        with pytest.raises(FrozenInstanceError):
            result.hvp_evaluations = 0


@pytest.mark.parametrize("method", METHODS)
def test_solver_gets_hessp_without_dense_hessian_and_native_status_is_preserved(monkeypatch, method):
    def solver(fun, initial, **kwargs):
        assert kwargs["jac"] is True and kwargs["bounds"] is None
        assert "hess" not in kwargs
        np.testing.assert_allclose(kwargs["hessp"](initial, DIRECTIONS[0]), A @ DIRECTIONS[0])
        kwargs["hessp"](initial, DIRECTIONS[1])
        assert ("gtol" in kwargs["options"]) == (method == "trust-krylov")
        assert ("xtol" in kwargs["options"]) == (method == "Newton-CG")
        value, _ = fun(initial)
        # Native success is not relabelled even when the common gradient is large.
        return SimpleNamespace(x=initial, fun=value, nit=0, success=True, status=0,
                               message="Native success, deliberately large gradient.", nfev=1, njev=1, nhev=7)
    monkeypatch.setattr(optimization, "minimize", solver)
    compiled = CompiledObjective(quadratic, POINTS[0])
    result = run_optimization(compiled, POINTS[0], method=method)
    assert result.success and result.gradient_norm > 1e-6
    assert result.message == "Native success, deliberately large gradient."
    assert result.hvp_evaluations == 2 and result.solver_hessian_evaluations == 7
    assert result.function_evaluations == result.gradient_evaluations == 1


@pytest.mark.parametrize("method", METHODS)
def test_zero_iteration_run_does_not_request_hvp(method):
    compiled = CompiledObjective(quadratic, POINTS[0])
    result = run_optimization(compiled, POINTS[0], method=method, max_iterations=0)
    assert result.iterations == 0 and not result.success
    assert result.hvp_evaluations == compiled.total_hvp_evaluations == 0
    assert compiled._compiled_hvp is None


@pytest.fixture(scope="module")
def ekf_case():
    root = Path(__file__).resolve().parents[1]
    make = runpy.run_path(str(root / "benchmarks/baseline_optimization.py"))["make_problem"]
    make_scan = runpy.run_path(str(root / "benchmarks/fixed_scan_scaling.py"))["make_scan_objective"]
    problem = make(12)
    objective = make_scan(problem.dataset.initial_filter, stack_fixed_inputs(problem.dataset.inputs), problem.layout)
    return problem, objective, CompiledObjective(objective, problem.true_raw)


def test_real_raw_ekf_hvp_matches_dense_hessian_and_directional_gradient_differences(ekf_case):
    problem, objective, compiled = ekf_case
    dense = jax.jit(checkify.checkify(jax.hessian(objective)))
    directions = (jnp.array([1.0, -2.0, 3.0, -4.0]), jnp.array([-2.0, 0.5, 1.0, 3.0]))
    for point in (problem.true_raw, *problem.starts[:2]):
        error, hessian = dense(point)
        error.throw()
        for direction in directions:
            direction = direction / jnp.linalg.norm(direction)
            actual = compiled.hessian_vector_product(point, direction)
            np.testing.assert_allclose(actual, hessian @ direction, rtol=2e-11, atol=2e-11)
            for h in (1e-3, 1e-4, 1e-5, 1e-6):
                numerical = (compiled.evaluate(point + h * direction)[1]
                             - compiled.evaluate(point - h * direction)[1]) / (2 * h)
                assert np.isfinite(numerical).all()
                if h == 1e-5:
                    np.testing.assert_allclose(numerical, actual, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("method", METHODS)
def test_curvature_optimization_of_real_ekf_raw_objective(ekf_case, method):
    problem, _, compiled = ekf_case
    result = run_optimization(compiled, problem.starts[0], method=method)
    assert result.success, result.message
    assert result.final_objective < result.initial_objective
    np.testing.assert_allclose(result.final_objective, -32.109410137131, rtol=0, atol=2e-8)
    assert result.gradient_norm < 1e-4
    assert result.hvp_evaluations > 0
    fitted = unpack_parameters(result.final_raw, problem.layout)
    assert np.all(fitted.sigma_v.diagonal > 0)
    assert np.all((fitted.theta_f > 0) & (fitted.theta_f < 1))
    np.testing.assert_allclose(fitted.theta_g, [0.71793493], atol=2e-6)


def test_invalid_raw_transform_still_raises_through_hvp(ekf_case):
    problem, _, compiled = ekf_case
    with pytest.raises(checkify.JaxRuntimeError):
        compiled.hessian_vector_product(problem.true_raw.at[1].set(-1000), jnp.ones(4))


@pytest.fixture(scope="module")
def benchmark_module():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/curvature_optimization.py"))


@pytest.mark.parametrize("spectrum,counts,condition", [
    ([1, 4, 10], (3, 0, 0), 10),
    ([-2, 4, 10], (2, 1, 0), None),
    ([1e-12, 4, 10], (2, 0, 1), None),
    ([0, 0, 0], (0, 0, 3), None),
])
def test_diagnostics_do_not_assign_spd_condition_number_to_indefinite_or_flat_hessian(
    benchmark_module, spectrum, counts, condition,
):
    report = benchmark_module["hessian_diagnostics"](np.diag(spectrum))
    assert (report["positive"], report["negative"], report["near_zero"]) == counts
    assert report["spd_condition_number"] == condition
    np.testing.assert_array_equal(report["eigenvalues"], spectrum)
    assert report["threshold"] == 1e-8 * max(1, max(abs(x) for x in spectrum))


def test_diagnostic_symmetry_error_is_reported_without_mutating_input(benchmark_module):
    hessian = np.array([[2.0, 0.2], [0.2 + 1e-12, 1.0]])
    before = hessian.copy()
    report = benchmark_module["hessian_diagnostics"](hessian)
    assert report["symmetry_error"] == abs(hessian[0, 1] - hessian[1, 0])
    np.testing.assert_array_equal(hessian, before)
    for invalid in (np.zeros((0, 0)), np.ones((2, 3)), np.array([[np.nan]])):
        with pytest.raises(ValueError):
            benchmark_module["hessian_diagnostics"](invalid)


def test_larger_family_has_explicit_free_blocks_and_all_parameters_enter_likelihood(benchmark_module):
    problem = benchmark_module["make_larger_problem"](3, n_dates=5)
    assert problem.layout.n_parameters == 12
    assert problem.initial.state.shape == (3,)
    assert problem.batch.observations.shape == (5, 6)
    assert problem.layout.n_x0 == problem.layout.n_w == 0
    assert len(problem.starts) == 2
    compiled = CompiledObjective(problem.objective, problem.true_raw)
    _, gradient = compiled.evaluate(problem.true_raw)
    assert np.all(np.abs(gradient) > 1e-6)
    hvp = compiled.hessian_vector_product(problem.true_raw, np.ones(12))
    dense = benchmark_module["dense_hessian_function"](problem.objective)(problem.true_raw)
    np.testing.assert_allclose(hvp, dense @ np.ones(12), rtol=2e-11, atol=2e-11)
