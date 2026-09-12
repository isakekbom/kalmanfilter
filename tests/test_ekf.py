"""Independent equation references for issue #5; no generated time series."""

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

import kalmanfilter.ekf as ekf
from kalmanfilter.ekf import (
    EKFInputs, FilterState, build_observation_linearization, ekf_step,
    initialize_filter, measurement_update, predict, run_filter,
)
from kalmanfilter.ois import OISInstrument, observation_quotes, quote_state_jacobian
from kalmanfilter.transition import (
    CoordinateMap, DenseMap, DiagonalMatrix, StateCoordinates, StructuralStep,
    select_observations, selection_map,
)


def array(value):
    return jnp.asarray(value, dtype=jnp.float64)


def numpy_map(mapping):
    """Independent dense oracle; no production map application/materialization."""
    if isinstance(mapping, DenseMap):
        return np.asarray(mapping.values)
    result = np.zeros(mapping.shape)
    weights = np.ones(len(mapping.rows)) if mapping.weights is None else np.asarray(mapping.weights)
    for row, source in enumerate(mapping.sources):
        if source is not None:
            result[row, mapping.columns.index(source)] = weights[row]
    return result


def numpy_covariance(covariance):
    return (np.diag(np.asarray(covariance.diagonal)) if isinstance(covariance, DiagonalMatrix)
            else np.asarray(covariance))


def step_with_maps(previous, current, a, b, d, active, deviations, g=None):
    """Assemble explicitly supplied algebra, with local named parameter axes."""
    parameters = tuple(f"f{i}" for i in range(np.shape(a)[1]))
    noises = tuple(f"w{i}" for i in range(np.shape(d)[1]))
    selector = selection_map(active, current.unsystematic, deviations)
    return StructuralStep(
        previous, current, DenseMap(current.all, parameters, array(a)),
        DenseMap(parameters, previous.all, array(b)),
        DenseMap(current.all, noises, array(d)), selector,
        selection_map(active, active, active) if g is None else g,
    )


def removal_step():
    # Same named removal case as issue #4: keep non-prefix coordinates.
    previous = StateCoordinates(("p",), ("old", "keep"), ("ua", "ub"))
    current = StateCoordinates(("p",), ("keep",), ("ub",))
    params = ("f_ub", "f_keep", "f_p")
    return StructuralStep(
        previous, current,
        selection_map(current.all, params, ("f_p", "f_keep", "f_ub")),
        selection_map(params, previous.all, ("ub", "keep", "p")),
        selection_map(current.all, current.all, current.all),
        selection_map(("qb",), ("ub",), ("ub",)),
        selection_map(("qb",), ("vb", "va"), ("vb",)),
    )


def introduction_step():
    # Issue #4's explicit new-row mixing; no automatic initialization policy.
    previous = StateCoordinates(("p",), (), ("u",))
    current = StateCoordinates(("p",), ("new",), ("u",))
    return step_with_maps(previous, current, [[1, 0], [0.25, 0.5], [0, 1]],
                          np.eye(2), [[1, 0], [0.5, 2], [0, 1]], ("q",), ("u",))


def instrument_for(coordinates, scale=1.0, payments=1):
    # Deterministic algebraic inputs, not financial conventions or a generator.
    dates = np.arange(payments + 1, dtype=float)
    pca = -scale * (dates[:, None, None] + 0.1) * np.ones((1, len(coordinates.pca), 1))
    steps = -0.2 * scale * dates[:, None] * np.ones((1, len(coordinates.steps)))
    return OISInstrument(array(np.full(payments, 0.5)), array(pca), array(steps))


def inputs_for(step):
    return EKFInputs(
        step, jnp.full(step.transition_pre_map.shape[0], 0.8),
        DiagonalMatrix(jnp.full(step.process_noise_map.shape[1], 0.001)),
        array([0.7]), DiagonalMatrix(jnp.full(step.observation_noise_map.shape[1], 0.002)),
        jnp.linspace(0.01, 0.03, len(step.active_observations)),
        tuple(instrument_for(step.current, i + 1, i + 1)
              for i in range(len(step.active_observations))),
    )


def nonlinear_case(dense_noise=False):
    coordinates = StateCoordinates(("p",), ("c",), ("ua", "ub", "uc"))
    active = ("qc", "qa")
    ids = coordinates.all
    identity = selection_map(ids, ids, ids)
    noise = (DenseMap(active, ("vb", "va", "vc"), array([[0.3, 0.1, 1.1], [0.2, 0.9, -0.1]]))
             if dense_noise else selection_map(active, ("vb", "va", "vc"), ("vc", "va")))
    step = StructuralStep(coordinates, coordinates, identity, identity, identity,
                          selection_map(active, coordinates.unsystematic, ("uc", "ua")), noise)
    inputs = inputs_for(step)._replace(
        theta_f=array([0.9, 0.8, 0.7, 0.6, 0.5]),
        observations=select_observations(step, ("qa", "qb", "qc"), array([0, 999, 0.03])),
    )
    if dense_noise:
        inputs = inputs._replace(
            sigma_w=array([[0.002, 0.0003, 0, 0, 0], [0.0003, 0.001, 0, 0, 0],
                           [0, 0, 0.003, 0, 0], [0, 0, 0, 0.004, 0], [0, 0, 0, 0, 0.005]]),
            sigma_v=array([[0.002, 0.0001, 0.0003], [0.0001, 0.003, 0.0002],
                           [0.0003, 0.0002, 0.004]]),
        )
    initial = initialize_filter(coordinates, array([0.03, -0.01, 0.002, 0.004, -0.003]),
                                array(np.eye(5) * 0.01 + np.ones((5, 5)) * 0.001))
    return initial, inputs


def numpy_quote_and_gradient(theta_g, state, instrument):
    loadings = np.concatenate((np.asarray(instrument.pca_loading_map) @ np.asarray(theta_g),
                               np.asarray(instrument.step_loading)), axis=1)
    discounts = np.exp(loadings @ state)
    accruals = np.asarray(instrument.accrual_factors)
    annuity = accruals @ discounts[1:]
    numerator = discounts[0] - discounts[-1]
    gradient = ((discounts[0] * loadings[0] - discounts[-1] * loadings[-1]) * annuity
                - numerator * ((accruals * discounts[1:]) @ loadings[1:])) / annuity**2
    return numerator / annuity, gradient


def numpy_reference(previous_state, previous_covariance, inputs):
    step = inputs.step
    a, b, d, selector, g = map(numpy_map, (
        step.transition_post_map, step.transition_pre_map, step.process_noise_map,
        step.observation_selector, step.observation_noise_map,
    ))
    f = a @ np.diag(np.asarray(inputs.theta_f)) @ b
    q = d @ numpy_covariance(inputs.sigma_w) @ d.T
    r = g @ numpy_covariance(inputs.sigma_v) @ g.T
    x = f @ previous_state
    p = f @ previous_covariance @ f.T + q
    boundary = len(step.current.systematic)
    pairs = [numpy_quote_and_gradient(inputs.theta_g, x[:boundary], instrument)
             for instrument in inputs.instruments]
    quotes = np.array([pair[0] for pair in pairs])
    jacobian = np.array([pair[1] for pair in pairs]).reshape(len(pairs), boundary)
    h = np.concatenate((jacobian, selector), axis=1)
    z_hat = quotes + selector @ x[boundary:]
    innovation = np.asarray(inputs.observations) - z_hat
    s = h @ p @ h.T + r
    gain = np.linalg.solve(s, h @ p).T if len(pairs) else np.empty((len(x), 0))
    return dict(f=f, q=q, r=r, x=x, p=p, quotes=quotes, jacobian=jacobian, h=h,
                z_hat=z_hat, innovation=innovation, s=s, gain=gain,
                filtered_state=x + gain @ innovation, filtered_covariance=p - gain @ h @ p)


def assert_reference(result, reference):
    p, lin, update = result.prediction, result.linearization, result.update
    pairs = (
        (numpy_map(p.transition), reference["f"]),
        (numpy_covariance(p.process_covariance), reference["q"]),
        (numpy_covariance(update.observation_covariance), reference["r"]),
        (p.predicted_state, reference["x"]), (p.predicted_covariance, reference["p"]),
        (lin.modeled_quotes, reference["quotes"]), (lin.quote_jacobian, reference["jacobian"]),
        (lin.observation_jacobian, reference["h"]), (lin.predicted_observation, reference["z_hat"]),
        (update.innovation, reference["innovation"]), (update.innovation_covariance, reference["s"]),
        (update.kalman_gain, reference["gain"]), (update.filtered_state, reference["filtered_state"]),
        (update.filtered_covariance, reference["filtered_covariance"]),
    )
    for actual, expected in pairs:
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-15)


def test_hand_computable_linear_gaussian_prediction_and_update():
    coords = StateCoordinates(("x",), (), ())
    step = step_with_maps(coords, coords, [[1]], [[1]], [[1]], (), ())
    initial = initialize_filter(coords, [2], [[3]])
    prediction = predict(initial, step, array([0.5]), DiagonalMatrix(array([0.25])))
    # x-=1, P-=1; H=2 and affine offset=.25 => z_hat=2.25.
    # z=3.25 => eps=1, S=5, K=2/5, x+=7/5, P+=1/5.
    update = measurement_update(prediction.predicted_state, prediction.predicted_covariance,
                                [3.25], [2.25], [[2]], DiagonalMatrix(array([1])))
    for actual, expected in (
        (prediction.predicted_state, [1]), (prediction.predicted_covariance, [[1]]),
        (update.innovation, [1]), (update.innovation_covariance, [[5]]),
        (update.kalman_gain, [[2 / 5]]), (update.filtered_state, [7 / 5]),
        (update.filtered_covariance, [[1 / 5]]), (update.innovation_cholesky, [[np.sqrt(5)]]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)


def test_prediction_equations_42_and_43_for_rectangular_mixing():
    step = introduction_step()
    initial = initialize_filter(step.previous, [0.2, -0.1], [[0.4, 0.07], [0.07, 0.3]])
    theta = array([0.8, 0.6])
    sigma_w = array([[0.03, 0.004], [0.004, 0.02]])
    prediction = predict(initial, step, theta, sigma_w)
    a, b, d = map(numpy_map, (step.transition_post_map, step.transition_pre_map, step.process_noise_map))
    diagonal = np.diag(theta)
    p = np.asarray(initial.covariance)
    f = a @ diagonal @ b
    q = d @ np.asarray(sigma_w) @ d.T
    eq42 = f @ p @ f.T + q
    eq43 = a @ diagonal @ b @ p @ b.T @ diagonal @ a.T + q
    np.testing.assert_allclose(eq42, eq43, rtol=2e-15, atol=1e-16)
    np.testing.assert_allclose(prediction.predicted_covariance, eq43, rtol=2e-15, atol=1e-16)
    assert prediction.transition.shape == (3, 2)


def test_two_dimensional_linear_update_and_pdf_covariance_equivalence():
    x = np.array([0.2, -0.3])
    p = np.array([[2, 0.4], [0.4, 1]])
    h = np.array([[1, 2], [-0.5, 1]])
    r = np.array([[0.5, 0.1], [0.1, 0.3]])
    z = np.array([0.7, -0.2])
    z_hat = h @ x + [0.1, -0.05]
    s = h @ p @ h.T + r
    gain = np.linalg.solve(s, h @ p).T
    result = measurement_update(x, p, z, z_hat, h, r)
    eq54 = (np.eye(2) - gain @ h) @ p
    eq55 = p - gain @ (h @ p)
    eq56 = p - p @ h.T @ np.linalg.solve(s, h @ p)
    for expected in (eq54, eq55, eq56):
        np.testing.assert_allclose(result.filtered_covariance, expected, rtol=2e-14, atol=3e-16)
    for actual, expected in ((result.innovation, z - z_hat), (result.innovation_covariance, s),
                             (result.kalman_gain, gain), (result.filtered_state, x + gain @ (z - z_hat))):
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=3e-16)
    np.testing.assert_allclose(result.innovation_cholesky @ result.innovation_cholesky.T, s, rtol=1e-15)


@pytest.mark.parametrize("dense_noise", [False, True])
def test_nonlinear_ois_step_reference_and_observation_order(dense_noise):
    initial, inputs = nonlinear_case(dense_noise)
    result = ekf_step(initial, inputs)
    reference = numpy_reference(np.asarray(initial.state), np.asarray(initial.covariance), inputs)
    assert_reference(result, reference)
    p, lin, update = result.prediction, result.linearization, result.update
    assert result.structural_step.active_observations == ("qc", "qa")
    np.testing.assert_array_equal(result.observations, [0.03, 0])
    assert lin.quote_jacobian.shape == (2, 2)
    assert lin.observation_jacobian.shape == (2, 5)
    assert lin.systematic_state.shape == (2,)
    assert lin.unsystematic_state.shape == (3,)
    assert update.kalman_gain.shape == (5, 2)
    np.testing.assert_array_equal(lin.observation_jacobian[:, 2:], [[0, 0, 1], [1, 0, 0]])
    np.testing.assert_allclose(lin.modeled_quotes,
                               observation_quotes(inputs.theta_g, lin.systematic_state, inputs.instruments))
    np.testing.assert_allclose(lin.quote_jacobian,
                               quote_state_jacobian(inputs.theta_g, lin.systematic_state, inputs.instruments))
    np.testing.assert_allclose(lin.linearization_offset,
                               reference["quotes"] - reference["jacobian"] @ reference["x"][:2])
    # Equations (46)-(49): retain the affine term; it is nonzero in this case.
    assert np.max(np.abs(lin.linearization_offset)) > 1e-6
    affine_innovation = result.observations - lin.observation_jacobian @ p.predicted_state - lin.linearization_offset
    np.testing.assert_allclose(update.innovation, affine_innovation, rtol=2e-14, atol=2e-16)
    assert np.all(np.isfinite(update.filtered_state))
    assert np.all(np.isfinite(update.filtered_covariance))


@pytest.mark.parametrize("compiled", [False, True])
def test_empty_observations_prediction_only_without_cholesky(monkeypatch, compiled):
    initial, inputs = nonlinear_case()
    step = replace(inputs.step,
                   observation_selector=selection_map((), inputs.step.current.unsystematic, ()),
                   observation_noise_map=selection_map((), ("vb", "va", "vc"), ()))
    inputs = inputs._replace(step=step, observations=array([]), instruments=())

    def forbidden(*args, **kwargs):
        raise AssertionError("Cholesky must not be called for empty observations")

    monkeypatch.setattr(ekf.jnp.linalg, "cholesky", forbidden)
    if compiled:
        error, result = jax.jit(checkify.checkify(ekf_step))(initial, inputs)
        error.throw()
    else:
        result = ekf_step(initial, inputs)
    p, lin, u = result.prediction, result.linearization, result.update
    for actual, shape in ((lin.modeled_quotes, (0,)), (lin.quote_jacobian, (0, 2)),
                           (lin.observation_jacobian, (0, 5)), (lin.linearization_offset, (0,)),
                           (lin.predicted_observation, (0,)), (u.innovation, (0,)),
                           (u.innovation_covariance, (0, 0)), (u.kalman_gain, (5, 0))):
        assert actual.shape == shape
        assert actual.dtype == jnp.float64
    assert u.observation_covariance.shape == (0, 0)
    assert u.innovation_cholesky is None
    np.testing.assert_array_equal(u.filtered_state, p.predicted_state)
    np.testing.assert_array_equal(u.filtered_covariance, p.predicted_covariance)


@pytest.mark.parametrize("step_factory", [removal_step, introduction_step])
def test_changing_dimensions_preserves_named_coordinates(step_factory):
    step = step_factory()
    n_previous, n_current = len(step.previous.all), len(step.current.all)
    initial = initialize_filter(step.previous, jnp.arange(n_previous) * 0.01,
                                jnp.eye(n_previous) * 0.02)
    inputs = inputs_for(step)
    result = ekf_step(initial, inputs)
    assert_reference(result, numpy_reference(np.asarray(initial.state), np.asarray(initial.covariance), inputs))
    assert result.prediction.transition.shape == (n_current, n_previous)
    assert result.prediction.predicted_state.shape == result.filtered.state.shape == (n_current,)
    assert result.prediction.predicted_covariance.shape == result.filtered.covariance.shape == (n_current, n_current)
    assert result.filtered.coordinates == step.current
    assert result.structural_step.previous == step.previous


def changing_sequence():
    # Removal, introduction, removal on an all-missing date, then reordered IDs.
    c0 = StateCoordinates(("p",), ("old",), ("ua",))
    c1 = StateCoordinates(("p",), (), ("ua",))
    c2 = StateCoordinates(("p",), ("new",), ("ua", "ub"))
    c3 = StateCoordinates(("p",), (), ("ua", "ub"))
    c4 = StateCoordinates(("p",), (), ("ub", "ua"))
    steps = (
        step_with_maps(c0, c1, np.eye(2), [[1, 0.2, 0], [0, 0, 1]], np.eye(2), ("qa",), ("ua",)),
        step_with_maps(c1, c2, [[1, 0], [0.25, 0.5], [0, 1], [0, 0]], np.eye(2),
                       np.eye(4), ("qb", "qa"), ("ub", "ua")),
        step_with_maps(c2, c3, np.eye(3), [[1, 0.1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                       np.eye(3), (), ()),
        step_with_maps(c3, c4, np.eye(3), [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                       np.eye(3), ("qa",), ("ua",)),
    )
    return tuple(inputs_for(step) for step in steps)


def test_forward_sequence_ragged_outputs_and_optional_trace():
    inputs = changing_sequence()
    initial = initialize_filter(inputs[0].step.previous, [0.03, -0.01, 0.002], np.eye(3) * 0.01)
    result = run_filter(initial, inputs, return_trace=True)
    compact = run_filter(initial, inputs)
    assert compact.trace is None
    assert len(result.trace) == len(result.filtered) == 4
    x, p = np.asarray(initial.state), np.asarray(initial.covariance)
    for item, trace, filtered, without_trace, size in zip(inputs, result.trace, result.filtered,
                                                         compact.filtered, (2, 4, 3, 3), strict=True):
        reference = numpy_reference(x, p, item)
        assert_reference(trace, reference)
        x, p = reference["filtered_state"], reference["filtered_covariance"]
        assert filtered.coordinates == item.step.current
        assert filtered.state.shape == (size,)
        assert filtered.covariance.shape == (size, size)
        np.testing.assert_array_equal(filtered.state, without_trace.state)
        np.testing.assert_array_equal(filtered.covariance, without_trace.covariance)
    assert result.final is result.filtered[-1]
    assert result.final.coordinates.unsystematic == ("ub", "ua")


def test_forward_driver_rejects_same_size_coordinate_order_mismatch_before_numerics(monkeypatch):
    inputs = changing_sequence()
    initial = initialize_filter(inputs[0].step.previous, [0, 0, 0], np.eye(3))
    wrong = inputs[-1]._replace(step=replace(
        inputs[-1].step, previous=inputs[-1].step.current,
        transition_pre_map=replace(inputs[-1].step.transition_pre_map,
                                   columns=inputs[-1].step.current.all),
    ))

    def forbidden(*args, **kwargs):
        raise AssertionError("driver should validate the whole chain before filtering")

    monkeypatch.setattr(ekf, "ekf_step", forbidden)
    with pytest.raises(ValueError, match="sequence coordinate mismatch"):
        run_filter(initial, inputs[:-1] + (wrong,))


def test_empty_sequence_and_empty_state_spaces():
    coordinates = StateCoordinates((), (), ())
    initial = initialize_filter(coordinates, [], jnp.empty((0, 0)))
    result = run_filter(initial, (), return_trace=True)
    assert result.filtered == result.trace == ()
    assert result.final is result.initial
    empty_map = selection_map((), (), ())
    step = StructuralStep(coordinates, coordinates, empty_map, empty_map, empty_map, empty_map, empty_map)
    result = ekf_step(initial, inputs_for(step))
    assert result.filtered.state.shape == (0,)
    assert result.filtered.covariance.shape == (0, 0)
    assert float(result.prediction.covariance_asymmetry.relative) == 0
    assert result.update.innovation_cholesky is None


def test_covariance_symmetry_and_measured_roundoff():
    initial, inputs = nonlinear_case(dense_noise=True)
    result = ekf_step(initial, inputs)
    p, lin, u = result.prediction, result.linearization, result.update
    hp = lin.observation_jacobian @ p.predicted_covariance
    raw_s = hp @ lin.observation_jacobian.T + numpy_covariance(u.observation_covariance)
    raw_filtered = p.predicted_covariance - u.kalman_gain @ hp
    f = numpy_map(p.transition)
    raw_prediction = f @ np.asarray(initial.covariance) @ f.T + numpy_covariance(p.process_covariance)
    for covariance, raw, diagnostic in (
        (p.predicted_covariance, raw_prediction, p.covariance_asymmetry),
        (u.innovation_covariance, raw_s, u.innovation_covariance_asymmetry),
        (u.filtered_covariance, raw_filtered, u.filtered_covariance_asymmetry),
    ):
        np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-16)
        # Valid, moderately scaled examples: this is a test bound, not a kernel threshold.
        assert 0 <= float(diagnostic.max_absolute) < 1e-15
        assert 0 <= float(diagnostic.relative) < 1e-13
        np.testing.assert_allclose(diagnostic.max_absolute,
                                   np.max(np.abs(raw - raw.T)), rtol=0, atol=1e-17)
    assert 0 <= float(result.observation_noise_asymmetry.relative) < 1e-13


@pytest.mark.parametrize("p,r", [([[0.0]], [[0.0]]), ([[-2.0]], [[1.0]])])
def test_non_positive_definite_innovation_covariance_is_checkified_failure(p, r):
    args = (array([0]), array(p), array([1]), array([0]), array([[1]]), array(r))
    with pytest.raises(checkify.JaxRuntimeError, match="innovation Cholesky failed"):
        measurement_update(*args)
    error, invalid = jax.jit(checkify.checkify(measurement_update))(*args)
    with pytest.raises(checkify.JaxRuntimeError, match="innovation Cholesky failed"):
        error.throw()
    # Inspect only to prove there was no repair; never consume failed outputs as estimates.
    np.testing.assert_array_equal(invalid.innovation_covariance, np.asarray(p) + np.asarray(r))
    assert not np.all(np.isfinite(invalid.innovation_cholesky))


def test_nonfinite_computed_innovation_covariance_reports_quantity():
    args = (array([0]), array([[1e308]]), array([0]), array([0]), array([[2]]), array([[0]]))
    error, _ = jax.jit(checkify.checkify(measurement_update))(*args)
    with pytest.raises(checkify.JaxRuntimeError, match="innovation covariance must be finite"):
        error.throw()


@pytest.mark.parametrize("bad_factor", [0.0, -1.0])
def test_invalid_finite_cholesky_diagonal_has_explicit_check(monkeypatch, bad_factor):
    # Fault injection isolates the diagonal check from the preceding finite-factor check.
    monkeypatch.setattr(ekf.jnp.linalg, "cholesky", lambda s, **kwargs: jnp.full_like(s, bad_factor))
    error, _ = checkify.checkify(measurement_update)([0], [[1]], [1], [0], [[1]], [[1]])
    with pytest.raises(checkify.JaxRuntimeError, match="Cholesky diagonal must be finite and positive"):
        error.throw()


def test_tiny_positive_innovation_variance_is_used_without_floor():
    result = measurement_update([0], [[1e-24]], [1e-12], [0], [[1]], [[1e-24]])
    np.testing.assert_allclose(result.innovation_covariance, [[2e-24]], rtol=1e-15, atol=0)
    np.testing.assert_allclose(result.kalman_gain, [[0.5]], rtol=1e-15)


def test_fixed_shape_jit_checkify_matches_eager_with_float64_dynamic_inputs():
    initial, inputs = nonlinear_case()
    compiled = jax.jit(checkify.checkify(ekf_step))
    for theta_f in (inputs.theta_f, inputs.theta_f * 0.95):
        item = inputs._replace(theta_f=theta_f)
        error, result = compiled(initial, item)
        error.throw()
        assert result.filtered.coordinates == inputs.step.current
        eager = ekf_step(initial, item)
        for actual, expected in zip(jax.tree.leaves(result), jax.tree.leaves(eager), strict=True):
            assert actual.dtype == expected.dtype == jnp.float64
            np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-15)


def test_differentiation_through_complete_step_matches_independent_finite_difference():
    initial, inputs = nonlinear_case()
    # Differentiate a scalar diagnostic of state AND covariance through prediction,
    # the OIS Jacobian, S, both solves, and the covariance update; no likelihood.
    def diagnostic(theta_f, theta_g, variances, state):
        item = inputs._replace(theta_f=theta_f, theta_g=theta_g, sigma_v=DiagonalMatrix(variances))
        result = ekf_step(replace(initial, state=state), item).update
        return jnp.sum(result.filtered_state**2) + jnp.trace(result.filtered_covariance)

    args = (inputs.theta_f, inputs.theta_g, inputs.sigma_v.diagonal, initial.state)
    error, gradients = jax.jit(checkify.checkify(jax.grad(diagnostic, argnums=(0, 1, 2, 3))))(*args)
    error.throw()

    def reference(values):
        theta_f, theta_g, variances, state = values
        item = inputs._replace(theta_f=theta_f, theta_g=theta_g, sigma_v=DiagonalMatrix(variances))
        result = numpy_reference(state, np.asarray(initial.covariance), item)
        return np.sum(result["filtered_state"]**2) + np.trace(result["filtered_covariance"])

    for arg_index, gradient in enumerate(gradients):
        expected = np.empty(gradient.shape)
        for index in range(len(gradient)):
            plus, minus = [np.array(arg) for arg in args], [np.array(arg) for arg in args]
            delta = 1e-7
            plus[arg_index][index] += delta
            minus[arg_index][index] -= delta
            expected[index] = (reference(plus) - reference(minus)) / (2 * delta)
        assert gradient.dtype == jnp.float64
        assert np.all(np.isfinite(gradient))
        np.testing.assert_allclose(gradient, expected, rtol=2e-6, atol=2e-9)


def test_compact_path_materializes_only_observation_selector(monkeypatch):
    initial, inputs = nonlinear_case()
    original = ekf.materialize_map
    calls = []

    def limited(mapping):
        assert mapping is inputs.step.observation_selector
        calls.append(mapping)
        return original(mapping)

    def forbidden_diagonal(*args, **kwargs):
        raise AssertionError("Do not construct a dense diagonal matrix")

    # Cholesky diagonal extraction remains allowed; diagonal matrix creation does not.
    original_diag = jnp.diag

    def only_extract(value, *args, **kwargs):
        if jnp.ndim(value) == 1:
            forbidden_diagonal()
        return original_diag(value, *args, **kwargs)

    monkeypatch.setattr(ekf, "materialize_map", limited)
    monkeypatch.setattr(ekf.jnp, "diag", only_extract)
    result = ekf_step(initial, inputs)
    assert len(calls) == 1
    assert isinstance(result.prediction.transition, CoordinateMap)
    assert isinstance(result.prediction.process_covariance, DiagonalMatrix)
    assert isinstance(result.update.observation_covariance, DiagonalMatrix)


def test_production_has_no_inverse_numpy_or_callback_calls():
    for path in Path(ekf.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                assert name not in {"inv", "pinv", "inverse", "matrix_power", "pure_callback", "io_callback"}, path
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert all(name != "numpy" and not name.startswith("numpy.") for name in names), path


def test_initialization_promotes_float64_and_preserves_supplied_values():
    coords = StateCoordinates(("a", "b"), (), ())
    initial = initialize_filter(coords, jnp.array([1, 2], dtype=jnp.float32), [[1, 0], [0, 0]])
    assert initial.state.dtype == initial.covariance.dtype == jnp.float64
    np.testing.assert_array_equal(initial.state, [1, 2])
    np.testing.assert_array_equal(initial.covariance, [[1, 0], [0, 0]])
    error, compiled = jax.jit(checkify.checkify(lambda x, p: initialize_filter(coords, x, p)))(
        initial.state, initial.covariance)
    error.throw()
    np.testing.assert_array_equal(compiled.covariance, initial.covariance)


@pytest.mark.parametrize("state,covariance,error,match", [
    ([1], [[1]], ValueError, "state shape"),
    ([1, 2], [[1]], ValueError, "covariance shape"),
    ([[1, 2]], np.eye(2), ValueError, "rank 1"),
    ([1, 2], [1, 2], ValueError, "rank 2"),
    ([True, False], np.eye(2), TypeError, "real numbers"),
    ([1j, 2], np.eye(2), TypeError, "real numbers"),
    ([1, 2], np.eye(2, dtype=complex), TypeError, "real numbers"),
    ([np.nan, 2], np.eye(2), checkify.JaxRuntimeError, "initial state must be finite"),
    ([1, 2], [[1, 0], [0, np.inf]], checkify.JaxRuntimeError, "initial covariance must be finite"),
    ([1, 2], [[1, 0.1], [0, 1]], checkify.JaxRuntimeError, "initial covariance must be symmetric"),
])
def test_initialization_rejects_invalid_supplied_values(state, covariance, error, match):
    with pytest.raises(error, match=match):
        initialize_filter(StateCoordinates(("a", "b"), (), ()), state, covariance)


def test_measurement_update_shape_and_covariance_validation():
    args = ([0, 0], np.eye(2), [1], [0], [[1, 2]], DiagonalMatrix(array([1])))
    for index, value in ((3, [0, 0]), (4, [[1]]), (5, DiagonalMatrix(array([1, 2])))):
        changed = list(args)
        changed[index] = value
        with pytest.raises(ValueError, match="shape"):
            measurement_update(*changed)
    with pytest.raises(checkify.JaxRuntimeError, match="nonnegative"):
        measurement_update(*args[:-1], DiagonalMatrix(array([-1])))
    with pytest.raises(checkify.JaxRuntimeError, match="symmetric"):
        measurement_update([0, 0], [[1, 0.1], [0, 1]], *args[2:])


def test_ois_block_sizes_and_instrument_count_checked_against_coordinates():
    initial, inputs = nonlinear_case()
    wrong = OISInstrument(array([0.5]), jnp.ones((2, 2, 1)), jnp.empty((2, 0)))
    with pytest.raises(ValueError, match="PCA/step block dimensions"):
        build_observation_linearization(inputs.step, inputs.theta_g, initial.state, (wrong, wrong))
    with pytest.raises(ValueError, match="one entry per active"):
        build_observation_linearization(inputs.step, inputs.theta_g, initial.state, inputs.instruments[:1])
    with pytest.raises(TypeError, match="ordered tuple"):
        build_observation_linearization(inputs.step, inputs.theta_g, initial.state, list(inputs.instruments))


def test_one_step_rejects_wrong_previous_identity_and_immutable_results():
    initial, inputs = nonlinear_case()
    wrong = replace(initial, coordinates=StateCoordinates(("other",), ("c",), ("ua", "ub", "uc")))
    with pytest.raises(ValueError, match="previous filtered coordinates"):
        ekf_step(wrong, inputs)
    before = [np.array(leaf) for leaf in jax.tree.leaves((initial, inputs))]
    result = ekf_step(initial, inputs)
    with pytest.raises(FrozenInstanceError):
        result.filtered.state = array([1])
    with pytest.raises(AttributeError):
        result.prediction = None
    for actual, expected in zip(jax.tree.leaves((initial, inputs)), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
