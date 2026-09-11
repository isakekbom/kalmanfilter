"""Deterministic algebraic examples, not a synthetic financial time series."""

from math import exp

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from kalmanfilter.ois import (
    OISInstrument,
    discount_factors,
    discount_loadings,
    observation_quotes,
    ois_quote,
    quote_state_gradient,
    quote_state_jacobian,
)


def array(value):
    return jnp.asarray(value, dtype=jnp.float64)


def case(n_p, n_c, n_g, counts):
    """Arbitrary dimensionless inputs; no calendars, PCA, or step construction."""
    theta = array(np.linspace(-0.4, 0.6, n_g))
    state = array(np.linspace(-0.15, 0.23, n_p + n_c))
    instruments = []
    for i, count in enumerate(counts):
        pca_shape = (count + 1, n_p, n_g)
        step_shape = (count + 1, n_c)
        pca = np.sin(np.arange(np.prod(pca_shape)).reshape(pca_shape) + 0.3 + i) / 3
        steps = np.cos(np.arange(np.prod(step_shape)).reshape(step_shape) + 0.7 + i) / 4
        instruments.append(
            OISInstrument(
                accrual_factors=array(np.linspace(0.17, 0.53, count)),
                pca_loading_map=array(pca),
                step_loading=array(steps),
            )
        )
    return theta, state, tuple(instruments)


def numpy_quotes(theta, state, instruments):
    """Independent scalar valuation path for finite differences and quote checks."""
    result = []
    for instrument in instruments:
        maps = np.asarray(instrument.pca_loading_map)
        fixed = np.asarray(instrument.step_loading)
        discounts = []
        for k in range(len(instrument.accrual_factors) + 1):
            loading = np.concatenate((maps[k].dot(theta), fixed[k]))
            discounts.append(exp(float(loading.dot(state))))
        annuity = sum(
            float(accrual) * discount
            for accrual, discount in zip(instrument.accrual_factors, discounts[1:])
        )
        result.append((discounts[0] - discounts[-1]) / annuity)
    return np.array(result, dtype=np.float64)


def central_difference(function, point, step=1e-5):
    point = np.asarray(point, dtype=np.float64)
    columns = []
    for j in range(len(point)):
        delta = np.zeros_like(point)
        delta[j] = step * max(1.0, abs(point[j]))
        columns.append((function(point + delta) - function(point - delta)) / (2 * delta[j]))
    return np.stack(columns, axis=1)


def test_loading_blocks_discount_sign_and_nonunit_start_discount():
    theta = array([2.0, -1.0])
    maps = array([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 1.0], [0.5, 2.0]]])
    fixed = array([[0.5], [-0.25]])
    loadings = discount_loadings(theta, maps, fixed)
    expected_loadings = np.array([[0.0, 2.0, 0.5], [-3.0, -1.0, -0.25]])
    np.testing.assert_array_equal(loadings, expected_loadings)
    state = array([0.1, -0.2, 0.3])
    discounts = discount_factors(loadings, state)
    np.testing.assert_allclose(discounts, [exp(-0.25), exp(-0.175)], rtol=2e-15)
    assert discounts[0] != 1.0
    assert loadings.dtype == discounts.dtype == jnp.float64
    instrument = OISInstrument(array([0.4]), maps, fixed)
    expected_quote = (exp(-0.25) - exp(-0.175)) / (0.4 * exp(-0.175))
    assert ois_quote(theta, state, instrument).shape == ()
    np.testing.assert_allclose(ois_quote(theta, state, instrument), expected_quote, rtol=2e-14)


def test_zero_state_has_unit_discounts_zero_quote_and_known_nonzero_gradient():
    instrument = OISInstrument(array([0.25]), array([[[1.0]], [[3.0]]]), array([[0.5], [-1.0]]))
    theta, state = array([1.0]), array([0.0, 0.0])
    loadings = discount_loadings(theta, instrument.pca_loading_map, instrument.step_loading)
    np.testing.assert_array_equal(discount_factors(loadings, state), [1.0, 1.0])
    assert ois_quote(theta, state, instrument) == 0
    np.testing.assert_array_equal(quote_state_gradient(theta, state, instrument), [-8.0, 6.0])
    np.testing.assert_array_equal(quote_state_jacobian(theta, state, (instrument,)), [[-8.0, 6.0]])


@pytest.mark.parametrize(
    "dimensions,counts",
    [((2, 1, 2), (1, 3, 2)), ((1, 2, 3), (2, 4)), ((2, 0, 2), (3, 1)), ((0, 2, 0), (1, 2, 4))],
    ids=["mixed-two-pca", "mixed-two-steps", "pca-only", "steps-only"],
)
@pytest.mark.parametrize("state_scale", [0.0, 1.0, -2.0])
def test_ragged_quotes_and_three_independent_state_derivatives(dimensions, counts, state_scale):
    theta, state, instruments = case(*dimensions, counts)
    state = state * state_scale
    quotes = observation_quotes(theta, state, instruments)
    autodiff = quote_state_jacobian(theta, state, instruments)
    analytical = jnp.stack([quote_state_gradient(theta, state, item) for item in instruments])
    finite_difference = central_difference(
        lambda x: numpy_quotes(np.asarray(theta), x, instruments), state
    )

    assert quotes.shape == (len(counts),)
    assert autodiff.shape == analytical.shape == (len(counts), dimensions[0] + dimensions[1])
    assert quotes.dtype == autodiff.dtype == analytical.dtype == jnp.float64
    np.testing.assert_allclose(
        quotes, numpy_quotes(np.asarray(theta), np.asarray(state), instruments),
        rtol=2e-13, atol=2e-15,
    )
    np.testing.assert_allclose(autodiff, analytical, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(autodiff, finite_difference, rtol=2e-8, atol=2e-10)


def test_parameter_derivatives_propagate_through_pca_maps():
    theta, state, instruments = case(2, 1, 3, (1, 2, 4))
    actual = jax.jacfwd(observation_quotes, argnums=0)(theta, state, instruments)
    expected = central_difference(
        lambda parameters: numpy_quotes(parameters, np.asarray(state), instruments), theta
    )
    assert actual.shape == (3, 3)
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-10)
    reverse = jax.grad(
        lambda parameters: jnp.sum(observation_quotes(parameters, state, instruments))
    )(theta)
    np.testing.assert_allclose(reverse, expected.sum(axis=0), rtol=2e-8, atol=2e-10)


def test_instrument_arrays_remain_differentiable_pytree_inputs():
    theta, state, (item,) = case(2, 1, 2, (3,))
    gradient = jax.grad(lambda data: ois_quote(theta, state, data))(item)
    for derivative, value in zip(gradient, item):
        assert derivative.shape == value.shape
        assert derivative.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(derivative))
    # Scaling all accrual factors scales the quote inversely (equation 9).
    np.testing.assert_allclose(
        jnp.dot(gradient.accrual_factors, item.accrual_factors),
        -ois_quote(theta, state, item), rtol=2e-13,
    )


def test_active_instrument_order_and_counts_can_change_between_calls():
    theta, state, instruments = case(2, 1, 2, (1, 3, 2))
    original = observation_quotes(theta, state, instruments)
    selected = (instruments[2], instruments[0])
    selection = jnp.array([2, 0])
    np.testing.assert_array_equal(
        observation_quotes(theta, state, selected), original[selection]
    )
    np.testing.assert_array_equal(
        quote_state_jacobian(theta, state, selected),
        quote_state_jacobian(theta, state, instruments)[selection],
    )
    _, _, next_instruments = case(2, 1, 2, (4,))
    assert observation_quotes(theta, state, next_instruments).shape == (1,)


def test_empty_active_tuple_and_empty_state_blocks_have_mathematical_shapes():
    theta, state, _ = case(2, 1, 2, (1,))
    assert observation_quotes(theta, state, ()).shape == (0,)
    assert quote_state_jacobian(theta, state, ()).shape == (0, 3)
    empty_theta, empty_state, instruments = case(0, 0, 0, (1, 2))
    np.testing.assert_array_equal(
        observation_quotes(empty_theta, empty_state, instruments), [0.0, 0.0]
    )
    assert quote_state_jacobian(empty_theta, empty_state, instruments).shape == (2, 0)


def test_checkified_compilation_preserves_values_jacobians_and_dynamic_inputs():
    theta, state, instruments = case(2, 1, 2, (1, 3, 2))
    compiled_quotes = jax.jit(checkify.checkify(observation_quotes))
    compiled_jacobian = jax.jit(checkify.checkify(quote_state_jacobian))
    for multiplier in (1.0, 1.7):
        changed = tuple(
            item._replace(accrual_factors=item.accrual_factors * multiplier)
            for item in instruments
        )
        error, quotes = compiled_quotes(theta * multiplier, state, changed)
        error.throw()
        error, jacobian = compiled_jacobian(theta * multiplier, state, changed)
        error.throw()
        np.testing.assert_allclose(
            quotes, numpy_quotes(np.asarray(theta * multiplier), np.asarray(state), changed),
            rtol=2e-13, atol=2e-15,
        )
        np.testing.assert_allclose(
            jacobian,
            jnp.stack([quote_state_gradient(theta * multiplier, state, item) for item in changed]),
            rtol=2e-12, atol=2e-13,
        )


def test_small_nonzero_and_negative_annuities_are_not_modified():
    theta, state, instruments = case(1, 1, 1, (1,))
    item = instruments[0]
    normal = ois_quote(theta, state, item)
    normal_gradient = quote_state_gradient(theta, state, item)
    for scale in (1e-14, -1.0):
        scaled = item._replace(accrual_factors=item.accrual_factors * scale)
        np.testing.assert_allclose(ois_quote(theta, state, scaled) * scale, normal, rtol=2e-14)
        np.testing.assert_allclose(
            quote_state_gradient(theta, state, scaled) * scale, normal_gradient, rtol=2e-13
        )
        np.testing.assert_allclose(
            quote_state_jacobian(theta, state, (scaled,))[0] * scale,
            normal_gradient, rtol=2e-13,
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("accrual_factors", np.ones((2, 1)), "accrual_factors must have rank 1"),
        ("accrual_factors", np.empty(0), "at least one payment"),
        ("accrual_factors", np.ones(3), "K\\+1 rows"),
        ("pca_loading_map", np.ones((3, 2)), "pca_loading_map must have rank 3"),
        ("pca_loading_map", np.ones((3, 2, 3)), "last dimension"),
        ("step_loading", np.ones(3), "step_loading must have rank 2"),
        ("step_loading", np.ones((2, 1)), "equal row counts"),
        ("step_loading", np.ones((3, 2)), "columns must equal len"),
    ],
)
def test_instrument_shape_errors_fail_clearly(field, value, message):
    theta, state, (item,) = case(2, 1, 2, (2,))
    invalid = item._replace(**{field: array(value)})
    with pytest.raises(ValueError, match=message):
        ois_quote(theta, state, invalid)


@pytest.mark.parametrize(
    "argument,value,message",
    [
        ("theta", [[1.0, 2.0]], "theta_g must have rank 1"),
        ("state", [[1.0, 2.0, 3.0]], "x_s must have rank 1"),
        ("state", [1.0, 2.0], "columns must equal len"),
    ],
)
def test_parameter_and_state_shape_errors(argument, value, message):
    theta, state, instruments = case(2, 1, 2, (2,))
    if argument == "theta":
        theta = array(value)
    else:
        state = array(value)
    with pytest.raises(ValueError, match=message):
        observation_quotes(theta, state, instruments)


def test_mismatched_pca_step_partition_is_rejected_even_with_equal_total_size():
    theta, state, first = case(2, 1, 2, (1,))
    _, _, second = case(1, 2, 2, (2,))
    with pytest.raises(ValueError, match="share PCA and step block dimensions"):
        observation_quotes(theta, state, first + second)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "target", ["theta_g", "x_s", "accrual_factors", "pca_loading_map", "step_loading"]
)
def test_nonfinite_inputs_are_rejected(target, bad):
    theta, state, (item,) = case(2, 1, 2, (2,))
    if target == "theta_g":
        theta = theta.at[0].set(bad)
    elif target == "x_s":
        state = state.at[0].set(bad)
    else:
        value = getattr(item, target)
        value = value.at[(0,) * value.ndim].set(bad)
        item = item._replace(**{target: value})
    with pytest.raises(checkify.JaxRuntimeError, match=f"{target} must be finite"):
        ois_quote(theta, state, item)


@pytest.mark.parametrize("invalid", [jnp.array([1.0 + 1j]), jnp.array([True])])
def test_nonreal_or_boolean_inputs_are_not_silently_cast(invalid):
    _, state, (item,) = case(1, 1, 1, (1,))
    with pytest.raises(TypeError, match="theta_g must contain real numbers"):
        ois_quote(invalid, state, item)


@pytest.mark.parametrize("function", [observation_quotes, quote_state_jacobian])
@pytest.mark.parametrize(
    "accrual,message",
    [
        ([0.0, 0.0], "annuity must be nonzero"),
        ([1.0, -1.0], "annuity must be nonzero"),
        ([1e308, 1e308], "annuity must be finite"),
    ],
)
def test_invalid_annuity_fails_eager_and_in_compiled_error_result(function, accrual, message):
    theta, state, (item,) = case(1, 1, 1, (2,))
    state = jnp.zeros_like(state)
    item = item._replace(accrual_factors=array(accrual))
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        function(theta, state, (item,))
    compiled = jax.jit(checkify.checkify(function))
    error, _ = compiled(theta, state, (item,))
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        error.throw()


@pytest.mark.parametrize("state", [[1000.0], [-1000.0]])
def test_exponential_overflow_or_underflow_fails_without_repair(state):
    with pytest.raises(checkify.JaxRuntimeError, match="exponential overflow/underflow"):
        discount_factors(array([[1.0], [2.0]]), array(state))


def test_quote_overflow_fails_without_clamping():
    item = OISInstrument(array([1e-200]), array([[[500.0]], [[0.0]]]), jnp.empty((2, 0)))
    with pytest.raises(checkify.JaxRuntimeError, match="OIS quote must be finite"):
        ois_quote(array([1.0]), array([1.0]), item)


def test_instrument_container_is_immutable_and_kernels_do_not_mutate_inputs():
    theta, state, instruments = case(2, 1, 2, (1, 2))
    before = [np.array(value) for value in jax.tree.leaves(instruments)]
    with pytest.raises(AttributeError):
        instruments[0].accrual_factors = array([1.0])
    observation_quotes(theta, state, instruments)
    for expected, actual in zip(before, jax.tree.leaves(instruments)):
        np.testing.assert_array_equal(actual, expected)


def test_real_inputs_are_promoted_to_float64():
    item = OISInstrument(
        jnp.array([1], dtype=jnp.int32),
        jnp.ones((2, 1, 1), dtype=jnp.float32),
        jnp.zeros((2, 1), dtype=jnp.float32),
    )
    theta, state = jnp.ones(1, dtype=jnp.float32), jnp.ones(2, dtype=jnp.float32)
    assert ois_quote(theta, state, item).dtype == jnp.float64
    assert quote_state_jacobian(theta, state, (item,)).dtype == jnp.float64


def test_invalid_instrument_containers_fail_clearly():
    theta, state, instruments = case(1, 1, 1, (1,))
    with pytest.raises(TypeError, match="ordered tuple"):
        observation_quotes(theta, state, list(instruments))
    with pytest.raises(TypeError, match="must be an OISInstrument"):
        ois_quote(theta, state, None)
