"""Equation (57) references use scalar/NumPy algebra, never production likelihood."""

import ast
import math
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

import kalmanfilter.ekf as ekf
import kalmanfilter.likelihood as likelihood
from kalmanfilter.ekf import EKFInputs, initialize_filter, run_filter
from kalmanfilter.likelihood import (
    innovation_loglikelihood, likelihood_contribution, log_likelihood_from_trace, run_likelihood,
)
from kalmanfilter.ois import OISInstrument
from kalmanfilter.transition import (
    DenseMap, DiagonalMatrix, StateCoordinates, StructuralStep, select_observations, selection_map,
)


def array(value):
    return jnp.asarray(value, dtype=jnp.float64)


def gaussian_reference(innovation, covariance):
    """Independent oracle via S, with no Cholesky-based likelihood computation."""
    epsilon, s = np.asarray(innovation), np.asarray(covariance)
    if epsilon.size == 0:
        return 0.0, 0.0, 0.0
    sign, logdet = np.linalg.slogdet(s)
    assert sign == 1
    quadratic = epsilon @ np.linalg.solve(s, epsilon)
    contribution = -0.5 * (epsilon.size * np.log(2 * np.pi) + logdet + quadratic)
    return logdet, quadratic, contribution


def scalar_sequence():
    """A one-state nonlinear example: g(x)=exp(theta_g*x)-1, J=theta_g*exp(theta_g*x)."""
    coordinates = StateCoordinates(("level",), (), ())
    identity = selection_map(coordinates.all, coordinates.all, coordinates.all)
    step = StructuralStep(
        coordinates, coordinates, identity, identity, identity,
        selection_map(("quote",), (), (None,)),
        selection_map(("quote",), ("v",), ("v",)),
    )
    instrument = OISInstrument(array([1]), array([[[0]], [[-1]]]), jnp.empty((2, 0)))
    inputs = EKFInputs(step, array([0.8]), DiagonalMatrix(array([0.003])), array([0.7]),
                       DiagonalMatrix(array([0.01])), array([0.025]), (instrument,))
    initial = initialize_filter(coordinates, [0.03], [[0.01]])
    return initial, (inputs, inputs._replace(observations=array([0])))


def scalar_forward_reference(theta_g):
    """Complete two-date scalar EKF and likelihood, independent of production code."""
    x, p = 0.03, 0.01
    contributions = []
    for z in (0.025, 0.0):
        x_pred = 0.8 * x
        p_pred = 0.8**2 * p + 0.003
        exponential = np.exp(theta_g * x_pred)
        g = exponential - 1
        jacobian = theta_g * exponential
        epsilon = z - g
        s = jacobian**2 * p_pred + 0.01
        contributions.append(-0.5 * (np.log(2 * np.pi) + np.log(s) + epsilon**2 / s))
        gain = p_pred * jacobian / s
        x = x_pred + gain * epsilon
        p = p_pred - gain * jacobian * p_pred
    return np.array(contributions)


def changing_sequence():
    """Issue #5's supplied removal/introduction/reordering algebra, with active-only rows."""
    c0 = StateCoordinates(("p",), ("old",), ("ua",))
    c1 = StateCoordinates(("p",), (), ("ua",))
    c2 = StateCoordinates(("p",), ("new",), ("ua", "ub"))
    c3 = StateCoordinates(("p",), (), ("ua", "ub"))
    c4 = StateCoordinates(("p",), (), ("ub", "ua"))
    cases = (
        (c0, c1, np.eye(2), [[1, 0.2, 0], [0, 0, 1]], ("qa",), ("ua",)),
        (c1, c2, [[1, 0], [0.25, 0.5], [0, 1], [0, 0]], np.eye(2), ("qb", "qa"), ("ub", "ua")),
        (c2, c3, np.eye(3), [[1, 0.1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], (), ()),
        (c3, c4, np.eye(3), [[1, 0, 0], [0, 0, 1], [0, 1, 0]], ("qa",), ("ua",)),
    )
    inputs = []
    for previous, current, a, b, active, deviations in cases:
        parameters = tuple(f"f{i}" for i in range(np.shape(a)[1]))
        step = StructuralStep(
            previous, current, DenseMap(current.all, parameters, array(a)),
            DenseMap(parameters, previous.all, array(b)),
            selection_map(current.all, current.all, current.all),
            selection_map(active, current.unsystematic, deviations),
            selection_map(active, ("va", "vb"), tuple({"qa": "va", "qb": "vb"}[i] for i in active)),
        )
        instruments = []
        for i in range(len(active)):
            dates = np.arange(i + 2, dtype=float)
            instruments.append(OISInstrument(
                array(np.full(i + 1, 0.5)), array(-(dates[:, None, None] + 0.1)),
                array(-0.2 * dates[:, None] * np.ones((1, len(current.steps)))),
            ))
        inputs.append(EKFInputs(
            step, jnp.full(len(parameters), 0.8), DiagonalMatrix(jnp.full(len(current.all), 0.001)),
            array([0.7]), DiagonalMatrix(array([0.002, 0.005])),
            select_observations(step, ("qa", "qb"), array([0, 0.04])), tuple(instruments),
        ))
    initial = initialize_filter(c0, [0.03, -0.01, 0.002], np.eye(3) * 0.01)
    return initial, tuple(inputs)


def empty_observations(inputs):
    step = inputs.step
    return inputs._replace(
        step=replace(step, observation_selector=selection_map((), step.current.unsystematic, ()),
                     observation_noise_map=selection_map((), step.observation_noise_map.columns, ())),
        observations=array([]), instruments=(),
    )


def test_scalar_gaussian_hand_computable_terms_and_sign():
    # e=3, S=4, L=2: quadratic=9/4, logdet=log(4).
    result = innovation_loglikelihood([3], [[2]])
    expected = -0.5 * (math.log(2 * math.pi) + math.log(4) + 9 / 4)
    assert result.n_observations == 1
    np.testing.assert_allclose(result.log_determinant, math.log(4), rtol=1e-15)
    np.testing.assert_allclose(result.quadratic_form, 9 / 4, rtol=1e-15)
    np.testing.assert_allclose(result.contribution, expected, rtol=1e-15)
    assert result.contribution < 0
    assert likelihood.LOG_2PI.dtype == jnp.float64
    # A density can exceed one, so valid log-likelihoods are not capped at zero.
    assert innovation_loglikelihood([0], [[0.01]]).contribution > 0


@pytest.mark.parametrize("epsilon,s", [
    ([0.3, -0.7], [[2, 0.6], [0.6, 1]]),
    ([1.2, -0.8, 0.2], [[3, 0.4, -0.2], [0.4, 2, 0.5], [-0.2, 0.5, 1]]),
])
def test_multivariate_terms_match_independent_covariance_reference(epsilon, s):
    factor = np.linalg.cholesky(s)
    result = innovation_loglikelihood(epsilon, factor)
    logdet, quadratic, contribution = gaussian_reference(epsilon, s)
    assert result.n_observations == len(epsilon)
    for actual, expected in ((result.log_determinant, logdet), (result.quadratic_form, quadratic),
                             (result.contribution, contribution)):
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_extreme_scales_use_log_diagonal_and_solve_without_floors(scale):
    # A direct 3D determinant under/overflows at these scales; logdet stays finite.
    diagonal = np.array([scale, 2 * scale, 3 * scale])
    whitened = np.array([0.5, -1, 2])
    result = innovation_loglikelihood(diagonal * whitened, np.diag(diagonal))
    logdet = 2 * sum(math.log(d) for d in diagonal)
    quadratic = whitened @ whitened
    np.testing.assert_allclose(result.log_determinant, logdet, rtol=2e-15)
    np.testing.assert_allclose(result.quadratic_form, quadratic, rtol=2e-15)
    np.testing.assert_allclose(result.contribution,
                               -0.5 * (3 * math.log(2 * math.pi) + logdet + quadratic), rtol=2e-15)
    assert np.isfinite(result.contribution)


def test_forward_path_uses_exactly_one_ekf_cholesky_per_nonempty_date(monkeypatch):
    initial, inputs = changing_sequence()
    original = ekf.jnp.linalg.cholesky
    factors = []
    expected_count = sum(len(item.step.active_observations) > 0 for item in inputs)

    def counted(s, **kwargs):
        assert s.shape[0] > 0, "empty dates must not attempt Cholesky"
        assert len(factors) < expected_count, "unexpected second factorization for likelihood"
        factor = original(s, **kwargs)
        factors.append(factor)
        return factor

    monkeypatch.setattr(ekf.jnp.linalg, "cholesky", counted)
    result = run_likelihood(initial, inputs, return_trace=True)
    assert len(factors) == expected_count == 3
    factor_index = 0
    for terms, step in zip(result.trace.steps, result.trace.ekf_steps, strict=True):
        assert terms.innovation is step.update.innovation
        assert terms.innovation_cholesky is step.update.innovation_cholesky
        if terms.n_observations:
            assert terms.innovation_cholesky is factors[factor_index]
            factor_index += 1


def test_existing_trace_is_reused_without_filtering_or_factorization(monkeypatch):
    initial, inputs = changing_sequence()
    filtered = run_filter(initial, inputs, return_trace=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("likelihood of an existing trace must reuse the EKF results")

    monkeypatch.setattr(likelihood, "ekf_step", forbidden)
    monkeypatch.setattr(ekf.jnp.linalg, "cholesky", forbidden)
    result = log_likelihood_from_trace(filtered.trace, return_trace=True)
    for original, retained, terms in zip(filtered.trace, result.trace.ekf_steps, result.trace.steps, strict=True):
        assert retained is original
        assert terms.innovation is original.update.innovation
        assert terms.innovation_cholesky is original.update.innovation_cholesky
    direct = likelihood_contribution(filtered.trace[0].update)
    np.testing.assert_array_equal(direct.contribution, result.per_step_contributions[0])


def test_changing_dimensions_order_zero_observations_and_independent_total():
    initial, inputs = changing_sequence()
    result = run_likelihood(initial, inputs, return_trace=True)
    compact = run_likelihood(initial, inputs)
    reference_filter = run_filter(initial, inputs, return_trace=True)
    reference = []
    for index, (terms, step, oracle_step, n_x, n_z) in enumerate(zip(
        result.trace.steps, result.trace.ekf_steps, reference_filter.trace, (2, 4, 3, 3), (1, 2, 0, 1), strict=True,
    )):
        expected = gaussian_reference(oracle_step.update.innovation, oracle_step.update.innovation_covariance)
        reference.append(expected[2])
        assert terms.n_observations == n_z
        assert terms.innovation.shape == (n_z,)
        assert step.filtered.coordinates == inputs[index].step.current
        assert step.prediction.transition.shape == (n_x, len(inputs[index].step.previous.all))
        for actual, value in zip((terms.log_determinant, terms.quadratic_form, terms.contribution), expected, strict=True):
            np.testing.assert_allclose(actual, value, rtol=3e-14, atol=3e-15)
        np.testing.assert_allclose(-2 * terms.contribution - terms.log_determinant - terms.quadratic_form,
                                   n_z * math.log(2 * math.pi), rtol=3e-14, atol=3e-15)
    assert result.trace.ekf_steps[1].structural_step.active_observations == ("qb", "qa")
    np.testing.assert_array_equal(result.trace.ekf_steps[1].observations, [0.04, 0])
    ordered = result.trace.steps[1]
    wrong_order = gaussian_reference(ordered.innovation[::-1], result.trace.ekf_steps[1].update.innovation_covariance)[2]
    assert abs(wrong_order - float(ordered.contribution)) > 1e-5
    np.testing.assert_allclose(result.per_step_contributions, reference, rtol=3e-14, atol=3e-15)
    np.testing.assert_allclose(result.total_log_likelihood, sum(reference), rtol=3e-14)
    np.testing.assert_array_equal(result.total_log_likelihood, jnp.sum(result.per_step_contributions))
    np.testing.assert_array_equal(compact.per_step_contributions, result.per_step_contributions)
    np.testing.assert_array_equal(compact.total_log_likelihood, result.total_log_likelihood)
    assert compact.trace is None
    assert result.total_log_likelihood.dtype == result.per_step_contributions.dtype == jnp.float64


def test_two_date_scalar_total_matches_fully_independent_nonlinear_filter():
    initial, inputs = scalar_sequence()
    result = run_likelihood(initial, inputs)
    expected = scalar_forward_reference(0.7)
    np.testing.assert_allclose(result.per_step_contributions, expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(result.total_log_likelihood, np.sum(expected), rtol=2e-14)


def test_empty_date_adds_zero_but_propagates_state_to_later_observation():
    initial, inputs = scalar_sequence()
    missing = empty_observations(inputs[0])
    result = run_likelihood(initial, (inputs[0], missing, inputs[1]), return_trace=True)
    prefix = run_likelihood(initial, inputs[:1])
    prefix_with_missing = run_likelihood(initial, (inputs[0], missing))
    skipped_date = run_likelihood(initial, inputs, return_trace=True)
    empty = result.trace.steps[1]
    assert empty.n_observations == 0
    assert empty.innovation.shape == (0,)
    assert empty.innovation_cholesky is None
    for value in (empty.log_determinant, empty.quadratic_form, empty.contribution):
        assert value.shape == () and value.dtype == jnp.float64
        assert value == 0
    np.testing.assert_array_equal(prefix.total_log_likelihood, prefix_with_missing.total_log_likelihood)
    assert abs(float(result.per_step_contributions[2] - skipped_date.per_step_contributions[1])) > 1e-5
    assert not np.array_equal(result.trace.ekf_steps[2].prediction.predicted_state,
                              skipped_date.trace.ekf_steps[1].prediction.predicted_state)
    empty_only = run_likelihood(initial, (missing, missing), return_trace=True)
    np.testing.assert_array_equal(empty_only.per_step_contributions, [0, 0])
    assert empty_only.total_log_likelihood == 0


def test_empty_sequence_has_zero_total_and_optional_empty_trace():
    initial, _ = scalar_sequence()
    for result in (run_likelihood(initial, ()), log_likelihood_from_trace(())):
        assert result.total_log_likelihood == 0
        assert result.total_log_likelihood.dtype == result.per_step_contributions.dtype == jnp.float64
        assert result.per_step_contributions.shape == (0,)
        assert result.trace is None
    result = run_likelihood(initial, (), return_trace=True)
    assert result.trace.steps == result.trace.ekf_steps == ()
    with pytest.raises(checkify.JaxRuntimeError, match="initial covariance must be finite"):
        run_likelihood(replace(initial, covariance=array([[np.nan]])), ())
    with pytest.raises(ValueError, match="an EKF trace is required"):
        log_likelihood_from_trace(None)


def test_forward_path_preserves_ekf_coordinate_validation():
    initial, inputs = changing_sequence()
    with pytest.raises(ValueError, match="previous filtered coordinates"):
        run_likelihood(initial, (inputs[0], inputs[2]))


@pytest.mark.parametrize("empty", [False, True])
def test_innovation_kernel_checked_jit_and_float64_promotion(empty):
    epsilon = jnp.empty((0,), dtype=jnp.float32) if empty else jnp.array([3], dtype=jnp.float32)
    factor = None if empty else jnp.array([[2]], dtype=jnp.float32)
    error, result = jax.jit(checkify.checkify(innovation_loglikelihood))(epsilon, factor)
    error.throw()
    assert result.n_observations == (0 if empty else 1)
    for leaf in jax.tree.leaves(result):
        assert leaf.dtype == jnp.float64
    expected = 0 if empty else -0.5 * (math.log(2 * math.pi) + math.log(4) + 9 / 4)
    np.testing.assert_allclose(result.contribution, expected, rtol=1e-15)


def test_fixed_structure_forward_jit_and_single_parameter_gradient_smoke():
    initial, inputs = scalar_sequence()

    def total(theta):
        items = tuple(item._replace(theta_g=jnp.reshape(theta, (1,))) for item in inputs)
        return run_likelihood(initial, items).total_log_likelihood

    error, (value, gradient) = jax.jit(checkify.checkify(jax.value_and_grad(total)))(array(0.7))
    error.throw()
    delta = 1e-5
    expected_gradient = (scalar_forward_reference(0.7 + delta).sum()
                         - scalar_forward_reference(0.7 - delta).sum()) / (2 * delta)
    assert value.dtype == gradient.dtype == jnp.float64
    assert np.isfinite(gradient)
    np.testing.assert_allclose(value, scalar_forward_reference(0.7).sum(), rtol=2e-14)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-8, atol=2e-10)
    error, traced = jax.jit(checkify.checkify(lambda state, items: run_likelihood(
        state, items, return_trace=True)))(initial, inputs)
    error.throw()
    assert len(traced.trace.steps) == len(traced.trace.ekf_steps) == 2
    np.testing.assert_allclose(traced.total_log_likelihood, value, rtol=2e-14)


def test_ekf_cholesky_failure_propagates_without_penalty_or_repair():
    initial, inputs = scalar_sequence()
    initial = replace(initial, covariance=array([[0]]))
    invalid_input = inputs[0]._replace(sigma_w=DiagonalMatrix(array([0])), sigma_v=DiagonalMatrix(array([0])))
    with pytest.raises(checkify.JaxRuntimeError, match="innovation Cholesky failed"):
        run_likelihood(initial, (invalid_input,))
    error, invalid = jax.jit(checkify.checkify(run_likelihood))(initial, (invalid_input,))
    with pytest.raises(checkify.JaxRuntimeError, match="innovation Cholesky failed"):
        error.throw()
    # Inspect only to verify the failure wasn't substituted with a penalty or -inf.
    assert np.isnan(invalid.total_log_likelihood)
    assert np.isnan(invalid.per_step_contributions[0])


@pytest.mark.parametrize("epsilon,factor,kind,message", [
    ([], np.empty((0, 0)), ValueError, "empty innovation requires"),
    ([1], None, ValueError, "nonempty innovation requires"),
    ([[1]], [[1]], ValueError, "rank 1"),
    ([1], [1], ValueError, "rank 2"),
    ([1, 2], [[1]], ValueError, "shape must match"),
    ([True], [[1]], TypeError, "real numbers"),
    ([1j], [[1]], TypeError, "real numbers"),
    ([1], [[True]], TypeError, "real numbers"),
    ([1], [[1j]], TypeError, "real numbers"),
    ([np.inf], [[1]], checkify.JaxRuntimeError, "likelihood innovation must be finite"),
    ([1], [[np.nan]], checkify.JaxRuntimeError, "likelihood Cholesky factor must be finite"),
    ([1], [[np.inf]], checkify.JaxRuntimeError, "likelihood Cholesky factor must be finite"),
    ([1], [[0]], checkify.JaxRuntimeError, "diagonal must be positive"),
    ([1], [[-1]], checkify.JaxRuntimeError, "diagonal must be positive"),
    ([1, 2], [[1, 0.1], [0, 1]], checkify.JaxRuntimeError, "must be lower triangular"),
])
def test_standalone_input_validation(epsilon, factor, kind, message):
    with pytest.raises(kind, match=message):
        innovation_loglikelihood(epsilon, factor)


@pytest.mark.parametrize("epsilon,factor,message", [
    ([1], [[0]], "diagonal must be positive"),
    ([1], [[np.nan]], "likelihood Cholesky factor must be finite"),
    ([1e200], [[1]], "innovation quadratic form must be finite"),
    ([1e200], [[1e-200]], "whitened innovation must be finite"),
])
def test_checked_standalone_failures_are_reported(epsilon, factor, message):
    error, _ = jax.jit(checkify.checkify(innovation_loglikelihood))(array(epsilon), array(factor))
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        error.throw()


def test_accumulation_overflow_is_reported_without_finite_penalty():
    initial, inputs = scalar_sequence()
    step = run_filter(initial, inputs[:1], return_trace=True).trace[0]
    # Valid factor and finite single contribution; their sum exceeds float64.
    large = step._replace(update=step.update._replace(innovation=array([1.2e154]), innovation_cholesky=array([[1]])))
    assert np.isfinite(likelihood_contribution(large.update).contribution)
    error, invalid = checkify.checkify(log_likelihood_from_trace)((large,) * 3)
    with pytest.raises(checkify.JaxRuntimeError, match="total log-likelihood must be finite"):
        error.throw()
    assert not np.isfinite(invalid.total_log_likelihood)


def test_results_are_immutable_and_inputs_not_mutated():
    initial, inputs = scalar_sequence()
    before = [np.array(leaf) for leaf in jax.tree.leaves((initial, inputs))]
    result = run_likelihood(initial, inputs, return_trace=True)
    with pytest.raises(AttributeError):
        result.total_log_likelihood = array(0)
    with pytest.raises(AttributeError):
        result.trace.steps[0].contribution = array(0)
    for actual, expected in zip(jax.tree.leaves((initial, inputs)), before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_production_likelihood_has_no_factorization_inverse_determinant_or_host_conversion():
    tree = ast.parse(Path(likelihood.__file__).read_text(encoding="utf-8"))
    forbidden = {"cholesky", "inv", "pinv", "inverse", "det", "slogdet", "matrix_power", "prod",
                 "float", "item", "pure_callback", "io_callback", "stop_gradient", "scan"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            assert name not in forbidden
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert all(name.split(".")[0] not in {"numpy", "scipy", "math"} for name in names)
