"""Seeded truth, independent algebra, and complete production EKF validation."""

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import kalmanfilter  # Enable float64 before constructing any JAX arrays.
import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np
import pytest

from kalmanfilter.ekf import EKFInputs, run_filter
from kalmanfilter.likelihood import run_likelihood
from kalmanfilter.ois import OISInstrument, observation_quotes
from kalmanfilter.synthetic import SyntheticStepInputs, generate_synthetic_dataset
from kalmanfilter.transition import (
    CoordinateMap, DenseMap, DiagonalMatrix, StateCoordinates, StructuralStep,
    selection_map,
)


def array(value):
    return jnp.asarray(value, dtype=jnp.float64)


def numpy_map(mapping):
    """Independent expansion of named map entries; no production materializer."""
    if isinstance(mapping, DenseMap):
        return np.asarray(mapping.values)
    result = np.zeros(mapping.shape)
    for row, source in enumerate(mapping.sources):
        if source is not None:
            result[row, mapping.columns.index(source)] = (
                1 if mapping.weights is None else np.asarray(mapping.weights)[row]
            )
    return result


def zero_quote():
    # One valid payment, all discounts exactly one, hence g=0 and J has 0 columns.
    return OISInstrument(array([1]), jnp.empty((2, 0, 0)), jnp.empty((2, 0)))


def linear_spec():
    coordinates = StateCoordinates((), (), ("u",))
    step = StructuralStep(
        coordinates, coordinates,
        selection_map(("u",), ("phi",), ("phi",)),
        selection_map(("phi",), ("u",), ("u",)),
        selection_map(("u",), ("w",), ("w",)),
        selection_map(("quote",), ("u",), ("u",)),
        selection_map(("quote",), ("v",), ("v",)),
    )
    item = SyntheticStepInputs(step, array([0.8]), DiagonalMatrix(array([0.04])),
                               array([]), DiagonalMatrix(array([0.09])), (zero_quote(),))
    return coordinates, array([0.3]), array([[0.25]]), (item,) * 12


def nonlinear_instruments(with_policy=False):
    step_a = array([[0], [-0.2], [-0.4]]) if with_policy else jnp.empty((3, 0))
    step_b = array([[0.1], [-0.4], [-0.9]]) if with_policy else jnp.empty((3, 0))
    return (
        OISInstrument(array([0.5, 0.5]), array([[[0.1]], [[-0.5]], [[-1.1]]]), step_a),
        OISInstrument(array([0.5, 1.0]), array([[[0.15]], [[-1.0]], [[-2.0]]]), step_b),
    )


def nonlinear_spec():
    coordinates = StateCoordinates(("p",), (), ("ua", "ub"))
    f_axes, w_axes, quotes, v_axes = ("fp", "fa", "fb"), ("wp", "wa", "wb"), ("qa", "qb"), ("va", "vb")
    step = StructuralStep(
        coordinates, coordinates,
        selection_map(coordinates.all, f_axes, f_axes),
        selection_map(f_axes, coordinates.all, coordinates.all),
        selection_map(coordinates.all, w_axes, w_axes),
        selection_map(quotes, coordinates.unsystematic, ("ua", "ub")),
        selection_map(quotes, v_axes, v_axes),
    )
    item = SyntheticStepInputs(
        step, array([0.95, 0.65, 0.55]), DiagonalMatrix(array([2e-4, 1e-5, 1e-5])),
        array([0.8]), DiagonalMatrix(array([1e-5, 2e-5])), nonlinear_instruments(),
    )
    factor = array([[0.02, 0, 0], [0.003, 0.005, 0], [-0.002, 0.001, 0.006]])
    return coordinates, array([0.03, 0, 0]), factor @ factor.T, (item,) * 24


def changing_spec():
    base = StateCoordinates(("p",), (), ("ua", "ub"))
    policy = StateCoordinates(("p",), ("policy",), ("ua", "ub"))
    f3, f4 = ("fp", "fa", "fb"), ("fp", "fc", "fa", "fb")
    w3, w4 = ("wp", "wa", "wb"), ("wp", "wc", "wa", "wb")
    v_axes = ("va", "vb")
    plain, augmented = nonlinear_instruments(), nonlinear_instruments(True)
    # Every row, local parameter axis, and instrument order is supplied here.
    dates = (
        (base, base, f3, f3, w3, ("qa", "qb"), ("ua", "ub"), v_axes, plain, [0.95, 0.65, 0.55]),
        (base, policy, f3, ("fp", None, "fa", "fb"), w4,
         ("qa", "qb"), ("ua", "ub"), v_axes, augmented, [0.95, 0.65, 0.55]),
        (policy, policy, f4, f4, w4, ("qb",), ("ub",), ("vb",),
         (augmented[1],), [0.95, 0.8, 0.65, 0.55]),
        (policy, base, f4, ("fp", "fa", "fb"), w3,
         ("qa", "qb"), ("ua", "ub"), v_axes, plain, [0.95, 0.8, 0.65, 0.55]),
        (base, base, f3, f3, w3, (), (), (), (), [0.95, 0.65, 0.55]),
        (base, base, f3, f3, w3, ("qb", "qa"), ("ub", "ua"), ("vb", "va"),
         (plain[1], plain[0]), [0.95, 0.65, 0.55]),
    )
    items = []
    for previous, current, f_axes, a_sources, w_axes, quotes, u_sources, v_sources, instruments, theta in dates:
        step = StructuralStep(
            previous, current,
            selection_map(current.all, f_axes, a_sources),
            selection_map(f_axes, previous.all, previous.all),
            selection_map(current.all, w_axes, w_axes),
            selection_map(quotes, current.unsystematic, u_sources),
            selection_map(quotes, v_axes, v_sources),
        )
        w = [2e-4, 5e-5, 1e-5, 1e-5] if current == policy else [2e-4, 1e-5, 1e-5]
        items.append(SyntheticStepInputs(step, array(theta), DiagonalMatrix(array(w)), array([0.8]),
                                         DiagonalMatrix(array([1e-5, 2e-5])), instruments))
    _, mean, covariance, _ = nonlinear_spec()
    return base, mean, covariance, tuple(items)


@pytest.fixture(scope="module")
def linear():
    dataset = generate_synthetic_dataset(jax.random.key(42), *linear_spec())
    return dataset, run_likelihood(dataset.initial_filter, dataset.inputs, return_trace=True)


@pytest.fixture(scope="module")
def nonlinear():
    dataset = generate_synthetic_dataset(jax.random.key(20260909), *nonlinear_spec())
    return dataset, run_likelihood(dataset.initial_filter, dataset.inputs, return_trace=True)


@pytest.fixture(scope="module")
def changing():
    dataset = generate_synthetic_dataset(jax.random.key(314), *changing_spec())
    return dataset, run_likelihood(dataset.initial_filter, dataset.inputs, return_trace=True)


def assert_truth_equal(first, second):
    np.testing.assert_array_equal(first.true_initial_state, second.true_initial_state)
    for name in ("true_states", "base_process_noise", "process_noise", "noiseless_observations",
                 "base_observation_noise", "observation_noise", "observations"):
        assert len(getattr(first, name)) == len(getattr(second, name))
        for a, b in zip(getattr(first, name), getattr(second, name), strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("key_factory", [jax.random.key, jax.random.PRNGKey])
def test_fixed_key_reproduces_all_truth_bitwise(key_factory):
    spec = changing_spec()
    first = generate_synthetic_dataset(key_factory(314), *spec)
    second = generate_synthetic_dataset(key_factory(314), *spec)
    assert_truth_equal(first, second)
    for supplied, generated in zip(spec[3], first.inputs, strict=True):
        assert generated.step is supplied.step
        assert generated.instruments is supplied.instruments


def test_different_key_changes_stochastic_realizations(linear):
    first, _ = linear
    second = generate_synthetic_dataset(jax.random.key(43), *linear_spec())
    for name in ("true_initial_state", "true_states", "process_noise", "observation_noise", "observations"):
        assert not np.array_equal(getattr(first, name), getattr(second, name))


def test_initial_truth_is_sampled_but_filter_receives_distribution(nonlinear):
    dataset, _ = nonlinear
    coordinates, mean, covariance, _ = nonlinear_spec()
    _, initial_key = jax.random.split(jax.random.key(20260909))
    normal = np.asarray(jax.random.normal(initial_key, (3,), dtype=jnp.float64))
    expected = np.asarray(mean) + np.linalg.cholesky(np.asarray(covariance)) @ normal
    np.testing.assert_allclose(dataset.true_initial_state, expected, rtol=2e-15, atol=2e-17)
    np.testing.assert_array_equal(dataset.initial_filter.state, mean)
    np.testing.assert_array_equal(dataset.initial_filter.covariance, covariance)
    assert dataset.initial_filter.coordinates == coordinates
    assert not np.array_equal(dataset.true_initial_state, mean)


def assert_generation_algebra(dataset):
    previous = np.asarray(dataset.true_initial_state)
    for t, item in enumerate(dataset.inputs):
        step = item.step
        a, b, d, selector, g = map(numpy_map, (
            step.transition_post_map, step.transition_pre_map, step.process_noise_map,
            step.observation_selector, step.observation_noise_map,
        ))
        state = np.asarray(dataset.true_states[t])
        expected = a @ np.diag(np.asarray(item.theta_f)) @ b @ previous + np.asarray(dataset.process_noise[t])
        np.testing.assert_allclose(state, expected, rtol=2e-15, atol=2e-17)
        np.testing.assert_allclose(dataset.process_noise[t], d @ np.asarray(dataset.base_process_noise[t]),
                                   rtol=2e-15, atol=2e-17)
        np.testing.assert_allclose(dataset.observation_noise[t], g @ np.asarray(dataset.base_observation_noise[t]),
                                   rtol=2e-15, atol=2e-17)
        n_s = len(step.current.systematic)
        # Direct NumPy OIS equation, independent of production pricing/splitting.
        quotes = []
        for instrument in item.instruments:
            loadings = np.concatenate((
                np.asarray(instrument.pca_loading_map) @ np.asarray(item.theta_g),
                np.asarray(instrument.step_loading),
            ), axis=1)
            discounts = np.exp(loadings @ state[:n_s])
            quotes.append((discounts[0] - discounts[-1]) / (np.asarray(instrument.accrual_factors) @ discounts[1:]))
        expected_mean = np.asarray(quotes) + selector @ state[n_s:]
        np.testing.assert_allclose(dataset.noiseless_observations[t], expected_mean, rtol=2e-13, atol=3e-16)
        np.testing.assert_array_equal(dataset.observations[t], dataset.noiseless_observations[t] + dataset.observation_noise[t])
        assert item.observations is dataset.observations[t]
        previous = state


@pytest.mark.parametrize("case", ["linear", "nonlinear", "changing"])
def test_generator_equations_against_independent_dense_algebra(case, request):
    dataset, _ = request.getfixturevalue(case)
    assert_generation_algebra(dataset)


@pytest.mark.parametrize("case", ["linear", "nonlinear", "changing"])
def test_complete_sequences_are_finite_float64_with_exact_coordinates(case, request):
    dataset, result = request.getfixturevalue(case)
    for leaf in jax.tree_util.tree_leaves((dataset, result)):
        assert np.isfinite(leaf).all()
        assert leaf.dtype == jnp.float64
    previous = dataset.initial_filter.coordinates
    for t, (item, trace) in enumerate(zip(dataset.inputs, result.trace.ekf_steps, strict=True)):
        assert isinstance(item, EKFInputs)
        assert item.step.previous == previous
        assert trace.structural_step is item.step
        assert trace.filtered.coordinates == item.step.current
        nx, nz = len(item.step.current.all), len(item.step.active_observations)
        assert dataset.true_states[t].shape == dataset.process_noise[t].shape == trace.filtered.state.shape == (nx,)
        assert trace.filtered.covariance.shape == (nx, nx)
        assert dataset.observations[t].shape == dataset.noiseless_observations[t].shape == dataset.observation_noise[t].shape == (nz,)
        assert trace.update.innovation.shape == (nz,)
        assert trace.update.innovation_covariance.shape == (nz, nz)
        assert dataset.base_process_noise[t].shape == (len(item.step.process_noise_map.columns),)
        assert dataset.base_observation_noise[t].shape == (len(item.step.observation_noise_map.columns),)
        previous = item.step.current


def test_exact_linear_gaussian_production_path_against_scalar_reference(linear):
    dataset, result = linear
    mean, variance, total = 0.3, 0.25, 0.0
    errors = dict.fromkeys(("predicted_mean", "filtered_mean", "predicted_variance",
                           "filtered_variance", "innovation", "innovation_variance"), 0.0)
    for z, trace in zip(dataset.observations, result.trace.ekf_steps, strict=True):
        predicted_mean = 0.8 * mean
        predicted_variance = 0.8**2 * variance + 0.04
        innovation = np.asarray(z)[0] - predicted_mean
        innovation_variance = predicted_variance + 0.09
        gain = predicted_variance / innovation_variance
        mean = predicted_mean + gain * innovation
        variance = (1 - gain) * predicted_variance
        total += -0.5 * (np.log(2 * np.pi) + np.log(innovation_variance) + innovation**2 / innovation_variance)
        expected = [predicted_mean, mean, predicted_variance, variance, innovation, innovation_variance]
        actual = [trace.prediction.predicted_state[0], trace.update.filtered_state[0],
                  trace.prediction.predicted_covariance[0, 0], trace.update.filtered_covariance[0, 0],
                  trace.update.innovation[0], trace.update.innovation_covariance[0, 0]]
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)
        for name, a, e in zip(errors, actual, expected, strict=True):
            errors[name] = max(errors[name], abs(float(a) - e))
        np.testing.assert_array_equal(trace.linearization.modeled_quotes, [0])
        np.testing.assert_array_equal(trace.linearization.observation_jacobian, [[1]])
        assert trace.linearization.quote_jacobian.shape == (1, 0)
    np.testing.assert_allclose(result.total_log_likelihood, total, rtol=2e-14, atol=2e-14)
    errors["total_log_likelihood"] = abs(float(result.total_log_likelihood) - total)
    print("linear max absolute errors:", errors, "log-likelihood:", float(result.total_log_likelihood))


def tracking_rmse(dataset, result):
    # Concatenate per-coordinate errors, not padded states; equal weight per entry.
    filtered = np.concatenate([np.asarray(trace.filtered.state - truth)
                               for truth, trace in zip(dataset.true_states, result.trace.ekf_steps, strict=True)])
    predicted = np.concatenate([np.asarray(trace.prediction.predicted_state - truth)
                                for truth, trace in zip(dataset.true_states, result.trace.ekf_steps, strict=True)])
    return np.sqrt(np.mean(filtered**2)), np.sqrt(np.mean(predicted**2))


def test_nonlinear_observations_improve_aggregate_state_tracking(nonlinear):
    dataset, result = nonlinear
    filtered, predicted = tracking_rmse(dataset, result)
    print(f"nonlinear RMSE: filtered={filtered:.12g}, predicted={predicted:.12g}")
    assert filtered < predicted
    # The exact quote has curvature; this scenario does not reduce to linear g.
    item = dataset.inputs[0]
    g = lambda x: observation_quotes(item.theta_g, array([x]), item.instruments)
    assert np.max(np.abs(g(0.1) + g(-0.1) - 2 * g(0))) > 1e-3


@pytest.mark.parametrize("alternative", ["low_persistence", "inflated_noise"])
def test_true_likelihood_exceeds_prespecified_poor_alternatives(nonlinear, alternative):
    dataset, result = nonlinear
    if alternative == "low_persistence":
        wrong = tuple(item._replace(theta_f=array([0.2, 0.1, 0.1])) for item in dataset.inputs)
    else:
        wrong = tuple(item._replace(
            sigma_w=DiagonalMatrix(25 * item.sigma_w.diagonal),
            sigma_v=DiagonalMatrix(25 * item.sigma_v.diagonal),
        ) for item in dataset.inputs)
    assert all(a.observations is b.observations for a, b in zip(dataset.inputs, wrong, strict=True))
    other = run_likelihood(dataset.initial_filter, wrong).total_log_likelihood
    assert np.isfinite(other)
    print(f"nonlinear likelihood: true={float(result.total_log_likelihood):.12g}, {alternative}={float(other):.12g}")
    # Controlled seeded comparison, not a claim that finite-sample MLE equals truth.
    assert result.total_log_likelihood > other


def test_changing_dimensions_introduce_and_remove_policy_state(changing):
    dataset, result = changing
    assert [len(item.step.current.all) for item in dataset.inputs] == [3, 4, 4, 3, 3, 3]
    assert dataset.inputs[1].step.state_change.introduced == ("policy",)
    assert dataset.inputs[3].step.state_change.removed == ("policy",)
    # The explicitly zero transition row means the new state is exactly its noise.
    np.testing.assert_array_equal(dataset.true_states[1][1], dataset.process_noise[1][1])
    filtered, predicted = tracking_rmse(dataset, result)
    print(f"changing RMSE: filtered={filtered:.12g}, predicted={predicted:.12g}; LL={float(result.total_log_likelihood):.12g}")
    assert filtered < predicted


def test_partial_empty_and_reordered_observation_dates(changing):
    dataset, result = changing
    assert [item.step.active_observations for item in dataset.inputs] == [
        ("qa", "qb"), ("qa", "qb"), ("qb",), ("qa", "qb"), (), ("qb", "qa"),
    ]
    assert [z.shape for z in dataset.observations] == [(2,), (2,), (1,), (2,), (0,), (2,)]
    assert dataset.inputs[2].step.observation_selector.sources == ("ub",)
    assert dataset.inputs[2].step.observation_noise_map.sources == ("vb",)
    assert dataset.inputs[4].instruments == ()
    assert dataset.base_observation_noise[4].shape == (2,)
    assert dataset.observation_noise[4].shape == (0,)
    assert result.per_step_contributions[4] == 0
    empty = result.trace.ekf_steps[4]
    assert empty.update.innovation_cholesky is None
    np.testing.assert_array_equal(empty.filtered.state, empty.prediction.predicted_state)
    np.testing.assert_array_equal(empty.filtered.covariance, empty.prediction.predicted_covariance)
    filtered = run_filter(dataset.initial_filter, dataset.inputs)
    for state, likelihood_step in zip(filtered.filtered, result.trace.ekf_steps, strict=True):
        np.testing.assert_array_equal(state.state, likelihood_step.filtered.state)


def test_empty_observation_date_does_not_shift_future_process_randomness():
    coordinates, mean, covariance, steps = linear_spec()
    item = steps[0]
    empty_step = replace(item.step,
        observation_selector=selection_map((), ("u",), ()),
        observation_noise_map=selection_map((), ("v",), ()),
    )
    masked = item._replace(step=empty_step, instruments=())
    full = generate_synthetic_dataset(jax.random.key(42), coordinates, mean, covariance, steps[:3])
    partial = generate_synthetic_dataset(jax.random.key(42), coordinates, mean, covariance, (item, masked, item))
    for name in ("true_states", "base_process_noise", "process_noise", "base_observation_noise"):
        np.testing.assert_array_equal(getattr(full, name), getattr(partial, name))
    np.testing.assert_array_equal(full.observations[2], partial.observations[2])


def test_diagonal_sampling_uses_sqrt_variance_and_distinct_date_keys(linear):
    dataset, _ = linear
    carry, _ = jax.random.split(jax.random.key(42))
    for w, v in zip(dataset.base_process_noise, dataset.base_observation_noise, strict=True):
        carry, wk, vk = jax.random.split(carry, 3)
        np.testing.assert_array_equal(w, 0.2 * jax.random.normal(wk, (1,), dtype=jnp.float64))
        np.testing.assert_array_equal(v, 0.3 * jax.random.normal(vk, (1,), dtype=jnp.float64))


def test_repeated_noise_sources_weights_and_zero_rows_never_factor_mapped_covariance(monkeypatch):
    coordinates = StateCoordinates((), (), ("a", "b", "c"))
    step = StructuralStep(
        coordinates, coordinates,
        selection_map(coordinates.all, coordinates.all, coordinates.all),
        selection_map(coordinates.all, coordinates.all, coordinates.all),
        CoordinateMap(coordinates.all, ("w",), ("w", "w", None), array([2, -1, 1])),
        selection_map(("qa", "qb", "qc"), coordinates.all, coordinates.all),
        selection_map(("qa", "qb", "qc"), ("v",), ("v", "v", None)),
    )
    item = SyntheticStepInputs(step, array([0.5] * 3), DiagonalMatrix(array([0.25])),
                               array([]), DiagonalMatrix(array([0.09])), (zero_quote(),) * 3)
    original, calls = jnp.linalg.cholesky, []
    def counted(matrix, **kwargs):
        calls.append(matrix.shape)
        return original(matrix, **kwargs)
    monkeypatch.setattr(jnp.linalg, "cholesky", counted)
    dataset = generate_synthetic_dataset(jax.random.key(7), coordinates, array([0] * 3), jnp.eye(3), (item,) * 2)
    assert calls == [(3, 3)]  # Only the supplied initial covariance.
    for w, v in zip(dataset.process_noise, dataset.observation_noise, strict=True):
        assert w[0] == -2 * w[1]
        assert v[0] == v[1]
        assert w[2] == v[2] == 0
    assert_generation_algebra(dataset)


def test_dense_base_covariances_and_dense_structural_maps():
    coordinates, mean, covariance, steps = nonlinear_spec()
    item = steps[0]
    step = replace(item.step,
        transition_post_map=DenseMap(coordinates.all, ("fp", "fa", "fb"), array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])),
        transition_pre_map=DenseMap(("fp", "fa", "fb"), coordinates.all, array([[1, 0, 0], [0.2, 1, 0], [0, 0, 1]])),
        process_noise_map=DenseMap(coordinates.all, ("wp", "wa", "wb"), array([[1, 0, 0], [0.1, 1, 0], [0, 0.2, 1]])),
        observation_noise_map=DenseMap(("qa", "qb"), ("va", "vb"), array([[1, 0.3], [0.2, 1]])),
    )
    sw = array([[2e-4, 1e-5, 0], [1e-5, 1e-5, 2e-6], [0, 2e-6, 1e-5]])
    sv = array([[1e-5, 3e-6], [3e-6, 2e-5]])
    dataset = generate_synthetic_dataset(jax.random.key(11), coordinates, mean, covariance,
                                         (item._replace(step=step, sigma_w=sw, sigma_v=sv),) * 2)
    carry, _ = jax.random.split(jax.random.key(11))
    for t in range(2):
        carry, wk, vk = jax.random.split(carry, 3)
        for actual, cov, key in ((dataset.base_process_noise[t], sw, wk), (dataset.base_observation_noise[t], sv, vk)):
            expected = np.linalg.cholesky(np.asarray(cov)) @ np.asarray(jax.random.normal(key, actual.shape, dtype=jnp.float64))
            np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-17)
    assert_generation_algebra(dataset)
    assert np.isfinite(run_likelihood(dataset.initial_filter, dataset.inputs).total_log_likelihood)


def test_zero_diagonal_variances_are_exact_and_valid():
    coordinates, mean, covariance, steps = linear_spec()
    item = steps[0]._replace(sigma_w=DiagonalMatrix(array([0])), sigma_v=DiagonalMatrix(array([0])))
    dataset = generate_synthetic_dataset(jax.random.key(8), coordinates, mean, covariance, (item,) * 3)
    for name in ("base_process_noise", "process_noise", "base_observation_noise", "observation_noise"):
        np.testing.assert_array_equal(getattr(dataset, name), np.zeros((3, 1)))
    assert_generation_algebra(dataset)


@pytest.mark.parametrize("covariance,match", [
    (array([[0]]), "Cholesky"), (array([[-1]]), "Cholesky"),
    (array([[np.nan]]), "finite"), (array([[np.inf]]), "finite"),
    (array([1]), "rank"), (jnp.eye(2), "shape"), (jnp.array([[True]]), "real"),
])
def test_invalid_initial_covariance_fails(covariance, match):
    coordinates, mean, _, steps = linear_spec()
    with pytest.raises((ValueError, TypeError, checkify.JaxRuntimeError), match=match):
        generate_synthetic_dataset(jax.random.key(1), coordinates, mean, covariance, steps[:1])


@pytest.mark.parametrize("field", ["sigma_w", "sigma_v"])
@pytest.mark.parametrize("value,match", [
    (DiagonalMatrix(array([-1])), "nonnegative"),
    (DiagonalMatrix(array([np.nan])), "finite"),
    (DiagonalMatrix(array([np.inf])), "finite"),
    (DiagonalMatrix(array([1, 2])), "shape"),
    (DiagonalMatrix(jnp.array([True])), "real"),
    (array([[0]]), "Cholesky"), (array([[-1]]), "Cholesky"),
    (array([[np.inf]]), "finite"), (array([1]), "rank"), (jnp.eye(2), "shape"),
])
def test_invalid_base_noise_fails_without_repair(field, value, match):
    coordinates, mean, covariance, steps = linear_spec()
    with pytest.raises((ValueError, TypeError, checkify.JaxRuntimeError), match=match):
        generate_synthetic_dataset(jax.random.key(1), coordinates, mean, covariance,
                                   (steps[0]._replace(**{field: value}),))


@pytest.mark.parametrize("target", ["initial", "sigma_w", "sigma_v"])
@pytest.mark.parametrize("kind", ["asymmetric", "indefinite"])
def test_dense_covariance_symmetry_and_positive_definiteness(target, kind):
    coordinates, mean, covariance, steps = nonlinear_spec()
    size = 2 if target == "sigma_v" else 3
    bad = jnp.eye(size).at[0, 1].set(2)
    if kind == "indefinite":
        bad = bad.at[1, 0].set(2)
    if target == "initial":
        covariance = bad
    else:
        steps = (steps[0]._replace(**{target: bad}),)
    with pytest.raises(checkify.JaxRuntimeError, match="symmetric" if kind == "asymmetric" else "Cholesky"):
        generate_synthetic_dataset(jax.random.key(1), coordinates, mean, covariance, steps[:1])


def test_empty_sequence_and_fully_empty_spaces(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("empty covariance must not be factorized")
    monkeypatch.setattr(jnp.linalg, "cholesky", forbidden)
    coordinates = StateCoordinates((), (), ())
    empty_map = selection_map((), (), ())
    step = StructuralStep(coordinates, coordinates, empty_map, empty_map, empty_map, empty_map, empty_map)
    item = SyntheticStepInputs(step, array([]), DiagonalMatrix(array([])), array([]), array([]).reshape(0, 0), ())
    for steps in ((), (item,) * 2):
        dataset = generate_synthetic_dataset(jax.random.key(1), coordinates, array([]), jnp.empty((0, 0)), steps)
        assert dataset.true_initial_state.shape == (0,)
        assert len(dataset.inputs) == len(steps)
        assert all(x.shape == (0,) for x in dataset.true_states + dataset.observations)
        result = run_likelihood(dataset.initial_filter, dataset.inputs, return_trace=True)
        assert result.total_log_likelihood == 0
        np.testing.assert_array_equal(result.per_step_contributions, np.zeros(len(steps)))


def test_inputs_results_and_arrays_are_immutable(linear):
    dataset, _ = linear
    spec = linear_spec()[3][0]
    with pytest.raises(AttributeError):
        spec.theta_f = array([0])
    with pytest.raises(AttributeError):
        dataset.true_states = ()
    with pytest.raises(FrozenInstanceError):
        dataset.initial_filter.state = array([0])
    with pytest.raises(TypeError):
        dataset.true_states[0][0] = 0
    assert isinstance(dataset.inputs, tuple)
    assert isinstance(dataset.inputs[0].instruments, tuple)


def test_empty_sequence_still_samples_the_nonempty_initial_distribution():
    coordinates, mean, covariance, _ = linear_spec()
    dataset = generate_synthetic_dataset(jax.random.key(42), coordinates, mean, covariance, ())
    assert dataset.true_initial_state.shape == (1,)
    assert not np.array_equal(dataset.true_initial_state, mean)
    for field in dataset[2:]:
        assert field == ()
    assert run_likelihood(dataset.initial_filter, dataset.inputs).total_log_likelihood == 0


@pytest.mark.parametrize("map_field", ["process_noise_map", "observation_noise_map"])
def test_overflow_in_mapped_noise_fails_explicitly(map_field):
    coordinates, mean, covariance, steps = linear_spec()
    mapping = getattr(steps[0].step, map_field)
    huge = CoordinateMap(mapping.rows, mapping.columns, mapping.sources, array([1e308]))
    item = steps[0]._replace(
        step=replace(steps[0].step, **{map_field: huge}),
        sigma_w=DiagonalMatrix(array([1e308])), sigma_v=DiagonalMatrix(array([1e308])),
    )
    with pytest.raises(checkify.JaxRuntimeError, match="finite"):
        generate_synthetic_dataset(jax.random.key(1), coordinates, mean, covariance, (item,))


def test_invalid_adjacent_coordinate_identity_fails():
    coordinates, mean, covariance, steps = linear_spec()
    wrong = StateCoordinates((), (), ("other",))
    with pytest.raises(ValueError, match="coordinates must match"):
        generate_synthetic_dataset(jax.random.key(1), wrong, mean, covariance, steps[:1])


@pytest.mark.parametrize("instruments,match", [([], "tuple"), ((), "active observation"), ((object(),), "OISInstrument")])
def test_invalid_instrument_structure_fails(instruments, match):
    coordinates, mean, covariance, steps = linear_spec()
    with pytest.raises((ValueError, TypeError), match=match):
        generate_synthetic_dataset(jax.random.key(1), coordinates, mean, covariance,
                                   (steps[0]._replace(instruments=instruments),))


def test_incorrect_pca_step_split_with_same_total_size_fails():
    coordinates, mean, covariance, steps = nonlinear_spec()
    wrong = OISInstrument(array([1]), jnp.empty((2, 0, 1)), array([[0], [-1]]))
    with pytest.raises(ValueError, match="PCA/step dimensions"):
        generate_synthetic_dataset(jax.random.key(1), coordinates, mean, covariance,
                                   (steps[0]._replace(instruments=(wrong, wrong)),))


def test_checked_jit_validates_dynamic_noise_and_promotes_float64():
    coordinates, mean, covariance, steps = linear_spec()
    def generate(key, variance):
        item = steps[0]._replace(sigma_w=DiagonalMatrix(variance))
        return generate_synthetic_dataset(key, coordinates, mean, covariance, (item,))
    compiled = jax.jit(checkify.checkify(generate))
    error, dataset = compiled(jax.random.key(1), jnp.array([0.04], dtype=jnp.float32))
    error.throw()
    assert dataset.inputs[0].sigma_w.diagonal.dtype == jnp.float64
    assert dataset.true_states[0].dtype == jnp.float64
    second_error, second = compiled(jax.random.key(1), jnp.array([0.04], dtype=jnp.float32))
    second_error.throw()
    assert_truth_equal(dataset, second)
    error, _ = compiled(jax.random.key(1), jnp.array([-1], dtype=jnp.float32))
    assert "nonnegative" in error.get()


def test_synthetic_source_stays_at_mathematical_model_layer():
    source = Path(kalmanfilter.__file__).with_name("synthetic.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"minimize", "optimize", "hessian", "hess", "inv", "pinv", "inverse", "clip", "scan", "nan_to_num",
                 "pure_callback", "io_callback", "ParameterLayout", "pack_parameters", "unpack_parameters"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            assert (node.id if isinstance(node, ast.Name) else node.attr) not in forbidden
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith(("numpy", "scipy", "random")) for name in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("numpy", "scipy", "random", "params", "optimization"))
