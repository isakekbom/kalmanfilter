"""Parameter conventions, independent covariance references, and small AD smokes."""

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

import kalmanfilter.params as params_module
from kalmanfilter.ekf import EKFInputs, initialize_filter
from kalmanfilter.likelihood import run_likelihood
from kalmanfilter.ois import OISInstrument
from kalmanfilter.params import (
    ModelParameters, ParameterLayout, inverse_logit, inverse_softplus,
    pack_parameters, unpack_parameters,
)
from kalmanfilter.transition import DenseMap, DiagonalMatrix, StateCoordinates, StructuralStep, selection_map


def array(value):
    return jnp.asarray(value, dtype=jnp.float64)


def layout(n_f=2, n_w=2, n_v=1, n_x0=2, n_g=2, mode="identity"):
    return ParameterLayout(n_f=n_f, n_w=n_w, n_v=n_v, n_x0=n_x0, n_g=n_g, theta_f_transform=mode)


def assert_parameters_close(actual, expected, **kwargs):
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert left.dtype == jnp.float64
        np.testing.assert_allclose(left, right, **kwargs)


@pytest.mark.parametrize("counts,expected", [
    ((0, 0, 0, 0, 0), 0), ((1, 1, 1, 1, 1), 6),
    ((2, 2, 1, 2, 2), 12), ((3, 2, 4, 3, 5), 23),
    ((0, 7, 2, 4, 0), 23), ((5, 0, 0, 0, 2), 7),
])
def test_parameter_count_and_slices(counts, expected):
    configuration = layout(*counts)
    assert configuration.n_parameters == expected
    assert configuration.n_cholesky == counts[3] * (counts[3] + 1) // 2
    sizes = (counts[0], counts[1], counts[2], counts[3], counts[3] * (counts[3] + 1) // 2, counts[4])
    start = 0
    for block, size in zip(configuration.slices, sizes, strict=True):
        assert block == slice(start, start + size)
        start += size
    assert start == expected


def test_hand_labelled_pdf_block_order_and_variance_not_standard_deviation():
    configuration = layout()
    raw = array([-2.5, 1.5, -1, 2, -3, 0.25, -0.75, 0.1, -0.2, 0.3, 4, -5])
    p = unpack_parameters(raw, configuration)
    assert configuration.slices.theta_f == slice(0, 2)
    assert configuration.slices.sigma_w == slice(2, 4)
    assert configuration.slices.sigma_v == slice(4, 5)
    assert configuration.slices.a_x == slice(5, 7)
    assert configuration.slices.sigma_0 == slice(7, 10)
    assert configuration.slices.theta_g == slice(10, 12)
    np.testing.assert_array_equal(p.theta_f, [-2.5, 1.5])
    np.testing.assert_allclose(p.sigma_w.diagonal, np.logaddexp(0, [-1, 2]), rtol=1e-15)
    np.testing.assert_allclose(p.sigma_v.diagonal, np.logaddexp(0, [-3]), rtol=1e-15)
    np.testing.assert_array_equal(p.a_x, [0.25, -0.75])
    np.testing.assert_array_equal(p.theta_g, [4, -5])
    expected_l = np.array([[np.logaddexp(0, 0.1), 0], [-0.2, np.logaddexp(0, 0.3)]])
    np.testing.assert_allclose(p.sigma_0, expected_l @ expected_l.T, rtol=1e-15)
    assert not np.allclose(p.sigma_w.diagonal, np.logaddexp(0, [-1, 2])**2)


def test_explicit_three_by_three_row_major_lower_triangle():
    configuration = layout(0, 0, 0, 3, 0)
    assert configuration.lower_triangle_order == ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
    raw = array([0.1, 0.2, 0.3, -0.3, 0.25, 0.7, -0.1, 0.4, -0.2])
    p = unpack_parameters(raw, configuration)
    expected_l = np.array([[np.logaddexp(0, -0.3), 0, 0],
                           [0.25, np.logaddexp(0, 0.7), 0],
                           [-0.1, 0.4, np.logaddexp(0, -0.2)]])
    np.testing.assert_allclose(p.sigma_0, expected_l @ expected_l.T, rtol=2e-15, atol=1e-16)
    np.testing.assert_allclose(pack_parameters(p, configuration), raw, rtol=2e-14, atol=2e-15)


@pytest.mark.parametrize("mode", ["identity", "unit_interval"])
@pytest.mark.parametrize("n", [0, 1, 3])
@pytest.mark.parametrize("offset", [-0.5, 0.4])
def test_raw_round_trip_for_representative_layouts(mode, n, offset):
    configuration = layout(3, 2, 2, n, 2, mode)
    raw = jnp.linspace(-0.6, 0.7, configuration.n_parameters) + offset
    recovered = pack_parameters(unpack_parameters(raw, configuration), configuration)
    np.testing.assert_allclose(recovered, raw, rtol=5e-14, atol=5e-15)


@pytest.mark.parametrize("mode", ["identity", "unit_interval"])
def test_interior_mathematical_parameter_round_trip(mode):
    configuration = layout(3, 2, 2, 3, 2, mode)
    factor = np.array([[0.4, 0, 0], [-0.1, 0.5, 0], [0.2, -0.15, 0.7]])
    p = ModelParameters(
        array([-2.2, 1, 2.3] if mode == "identity" else [0.2, 0.5, 0.8]),
        DiagonalMatrix(array([0.01, 0.4])), DiagonalMatrix(array([2, 0.02])),
        array([0.02, -0.01, 0.03]), array(factor @ factor.T), array([-0.3, 0.8]),
    )
    raw = pack_parameters(p, configuration)
    recovered = unpack_parameters(raw, configuration)
    assert_parameters_close(recovered, p, rtol=2e-14, atol=2e-16)


@pytest.mark.parametrize("offset", [-1.5, 0.0, 2.0])
def test_compact_positive_variances_and_dense_spd_covariance(offset):
    configuration = layout(2, 5, 4, 3, 2)
    raw = jnp.linspace(-0.4, 0.4, configuration.n_parameters) + offset
    p = unpack_parameters(raw, configuration)
    for covariance, size in ((p.sigma_w, 5), (p.sigma_v, 4)):
        assert isinstance(covariance, DiagonalMatrix)
        assert covariance.diagonal.shape == (size,)
        assert covariance.diagonal.dtype == jnp.float64
        assert np.all(np.isfinite(covariance.diagonal))
        assert np.all(covariance.diagonal > 0)
    assert p.sigma_0.shape == (3, 3)
    assert p.sigma_0.dtype == jnp.float64
    np.testing.assert_array_equal(p.sigma_0, p.sigma_0.T)
    assert np.all(np.linalg.eigvalsh(np.asarray(p.sigma_0)) > 0)


def test_stable_softplus_and_inverse_across_representable_extremes():
    raw_values = array([-700, -100, -40, 0, 40, 100, 700])
    configuration = layout(0, 7, 7, 0, 0)
    p = unpack_parameters(jnp.concatenate((raw_values, raw_values)), configuration)
    expected = np.logaddexp(0, np.asarray(raw_values))
    for covariance in (p.sigma_w, p.sigma_v):
        assert np.all(covariance.diagonal > 0)
        assert np.all(np.isfinite(covariance.diagonal))
        np.testing.assert_allclose(covariance.diagonal, expected, rtol=2e-14, atol=0)
        np.testing.assert_allclose(inverse_softplus(covariance.diagonal), raw_values, rtol=2e-14, atol=2e-15)
    interior = array([1e-250, 1e-12, 0.5, 50, 1e250])
    recovered = jax.nn.softplus(inverse_softplus(interior))
    np.testing.assert_allclose(recovered, interior, rtol=5e-14, atol=0)


@pytest.mark.parametrize("diagonal_raw", [-100, 100])
def test_extreme_initial_factor_diagonals_without_floor(diagonal_raw):
    configuration = layout(0, 0, 0, 2, 0)
    raw = array([0, 0, diagonal_raw, 0, diagonal_raw])
    p = unpack_parameters(raw, configuration)
    expected = np.eye(2) * np.logaddexp(0, diagonal_raw)**2
    np.testing.assert_allclose(p.sigma_0, expected, rtol=2e-14, atol=0)
    np.testing.assert_allclose(pack_parameters(p, configuration), raw, rtol=2e-14, atol=1e-14)


def test_theta_f_default_and_identity_blocks_preserve_unbounded_values():
    configuration = layout(4, 0, 0, 1, 3)
    assert configuration.theta_f_transform == "identity"
    raw = array([-100, -2, 1, 100, -25, 0.1, -500, 0, 500])
    p = unpack_parameters(raw, configuration)
    np.testing.assert_array_equal(p.theta_f, [-100, -2, 1, 100])
    np.testing.assert_array_equal(p.a_x, [-25])
    np.testing.assert_array_equal(p.theta_g, [-500, 0, 500])
    np.testing.assert_array_equal(pack_parameters(p, configuration)[configuration.slices.theta_f], p.theta_f)


def test_explicit_unit_interval_and_stable_inverse_logit():
    configuration = layout(5, 0, 0, 0, 0, "unit_interval")
    raw = array([-5, -1, 0, 1, 5])
    p = unpack_parameters(raw, configuration)
    assert np.all((p.theta_f > 0) & (p.theta_f < 1))
    np.testing.assert_allclose(p.theta_f, 1 / (1 + np.exp(-np.asarray(raw))), rtol=2e-15)
    np.testing.assert_allclose(pack_parameters(p, configuration), raw, rtol=5e-14, atol=5e-15)
    probabilities = array([1e-12, 0.1, 0.5, 0.9, 1 - 1e-12])
    expected = np.log(probabilities) - np.log1p(-np.asarray(probabilities))
    np.testing.assert_allclose(inverse_logit(probabilities), expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("mode", ["identity", "unit_interval"])
def test_fixed_layout_checked_jit_pack_unpack_and_gradient_smoke(mode):
    configuration = layout(mode=mode)
    raw = jnp.linspace(0.1, 0.8, configuration.n_parameters)
    error, p = jax.jit(checkify.checkify(unpack_parameters))(raw, configuration)
    error.throw()
    error, recovered = jax.jit(checkify.checkify(pack_parameters))(p, configuration)
    error.throw()
    np.testing.assert_allclose(recovered, raw, rtol=5e-14, atol=5e-15)

    def diagnostic(vector):
        p = unpack_parameters(vector, configuration)
        return (jnp.sum(p.theta_f**2) + jnp.sum(p.sigma_w.diagonal**2)
                + jnp.sum(p.sigma_v.diagonal) + jnp.sum(p.a_x**2)
                + jnp.sum(p.sigma_0**2) + jnp.sum(p.theta_g**2))

    gradient = jax.grad(diagnostic)(raw)
    error, (value, compiled_gradient) = jax.jit(checkify.checkify(jax.value_and_grad(diagnostic)))(raw)
    error.throw()
    assert gradient.shape == raw.shape
    assert value.dtype == gradient.dtype == compiled_gradient.dtype == jnp.float64
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient != 0)
    np.testing.assert_allclose(compiled_gradient, gradient, rtol=2e-14, atol=2e-14)


def test_forward_unpack_and_gradient_avoid_cholesky(monkeypatch):
    configuration = layout(2, 2, 1, 3, 2)
    raw = jnp.linspace(-0.4, 0.4, configuration.n_parameters)

    def forbidden(*args, **kwargs):
        raise AssertionError("forward parameters must not refactor the Gram product")

    monkeypatch.setattr(params_module.jnp.linalg, "cholesky", forbidden)
    eager = unpack_parameters(raw, configuration)
    # Fresh wrappers force tracing while the factorization guard is active.
    error, compiled = jax.jit(checkify.checkify(
        lambda vector: unpack_parameters(vector, configuration)
    ))(raw)
    error.throw()
    assert_parameters_close(compiled, eager, rtol=2e-14, atol=2e-15)
    diagnostic = lambda vector: jnp.sum(unpack_parameters(vector, configuration).sigma_0)
    error, (value, gradient) = jax.jit(checkify.checkify(jax.value_and_grad(diagnostic)))(raw)
    error.throw()
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert gradient.shape == raw.shape and gradient.dtype == jnp.float64


def test_empty_layout_avoids_cholesky_and_supports_checked_jit(monkeypatch):
    configuration = layout(0, 0, 0, 0, 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("no factorization for an empty initial space")

    monkeypatch.setattr(params_module.jnp.linalg, "cholesky", forbidden)
    error, p = jax.jit(checkify.checkify(unpack_parameters))(array([]), configuration)
    error.throw()
    assert p.sigma_0.shape == (0, 0)
    error, raw = jax.jit(checkify.checkify(pack_parameters))(p, configuration)
    error.throw()
    assert raw.shape == (0,) and raw.dtype == jnp.float64


def test_layout_is_static_and_containers_are_immutable_pytrees():
    configuration = layout()
    assert jax.tree.leaves(configuration) == []
    assert configuration == replace(configuration)
    assert hash(configuration) == hash(replace(configuration))
    with pytest.raises(FrozenInstanceError):
        configuration.n_x0 = 99
    p = unpack_parameters(jnp.zeros(configuration.n_parameters), configuration)
    assert len(jax.tree.leaves(p)) == 6
    assert all(leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(p))
    with pytest.raises(AttributeError):
        p.theta_f = array([1])


def test_no_dense_process_or_observation_diagonal_matrices(monkeypatch):
    configuration = layout(2, 64, 48, 3, 1)
    raw = jnp.zeros(configuration.n_parameters)
    original_diag = jnp.diag

    def extract_only(value, *args, **kwargs):
        assert jnp.ndim(value) == 2, "do not construct dense diagonal variances"
        return original_diag(value, *args, **kwargs)

    monkeypatch.setattr(params_module.jnp, "diag", extract_only)
    p = unpack_parameters(raw, configuration)
    assert isinstance(p.sigma_w, DiagonalMatrix) and isinstance(p.sigma_v, DiagonalMatrix)
    traced = jax.make_jaxpr(checkify.checkify(unpack_parameters))(raw, configuration)
    for equation in traced.jaxpr.eqns:
        for variable in equation.outvars:
            assert getattr(variable.aval, "shape", None) not in {(64, 64), (48, 48)}


@pytest.mark.parametrize("name", ["n_f", "n_w", "n_v", "n_x0", "n_g"])
@pytest.mark.parametrize("value,error", [(-1, ValueError), (True, TypeError), (1.5, TypeError)])
def test_invalid_static_dimensions(name, value, error):
    with pytest.raises(error, match=name):
        replace(layout(), **{name: value})


@pytest.mark.parametrize("mode,error", [("stationary", ValueError), (None, TypeError)])
def test_invalid_transform_mode(mode, error):
    with pytest.raises(error, match="theta_f_transform"):
        layout(mode=mode)


@pytest.mark.parametrize("raw,error,message", [
    (0.0, ValueError, "rank"), ([[0.0]], ValueError, "rank"),
    ([0.0], ValueError, "shape"), ([True] * 12, TypeError, "real numbers"),
    ([1j] * 12, TypeError, "real numbers"),
])
def test_invalid_raw_structure(raw, error, message):
    with pytest.raises(error, match=message):
        unpack_parameters(raw, layout())


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32])
def test_real_input_promotion_to_float64(dtype):
    configuration = layout()
    p = unpack_parameters(jnp.zeros(configuration.n_parameters, dtype=dtype), configuration)
    assert all(leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(p))
    assert pack_parameters(p, configuration).dtype == jnp.float64


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf, -jnp.inf])
def test_checked_jit_rejects_nonfinite_raw_in_each_block(value):
    configuration = layout()
    checked = jax.jit(checkify.checkify(unpack_parameters))
    for block in configuration.slices:
        raw = jnp.zeros(configuration.n_parameters).at[block.start].set(value)
        error, _ = checked(raw, configuration)
        with pytest.raises(checkify.JaxRuntimeError, match="raw parameter vector must be finite"):
            error.throw()


@pytest.mark.parametrize("function,value,message", [
    (inverse_softplus, 0, "positive"), (inverse_softplus, -1, "positive"),
    (inverse_softplus, jnp.inf, "finite"), (inverse_softplus, jnp.nan, "finite"),
    (inverse_logit, 0, "strictly inside"), (inverse_logit, 1, "strictly inside"),
    (inverse_logit, -0.1, "strictly inside"), (inverse_logit, 1.1, "strictly inside"),
    (inverse_logit, jnp.inf, "finite"), (inverse_logit, jnp.nan, "finite"),
])
def test_checked_inverse_domains(function, value, message):
    error, _ = jax.jit(checkify.checkify(function))(array(value))
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        error.throw()


@pytest.mark.parametrize("function", [inverse_softplus, inverse_logit])
@pytest.mark.parametrize("value", [True, 0.5 + 0j])
def test_inverse_rejects_boolean_and_complex(function, value):
    with pytest.raises(TypeError, match="real numbers"):
        function(value)


@pytest.mark.parametrize("field,value,error,message", [
    ("theta_f", array([[1, 2]]), ValueError, "rank"),
    ("theta_f", array([1]), ValueError, "shape"),
    ("a_x", array([1]), ValueError, "shape"),
    ("theta_g", jnp.array([True, False]), TypeError, "real numbers"),
    ("sigma_w", array([1, 1]), TypeError, "DiagonalMatrix"),
    ("sigma_v", array([[1]]), TypeError, "DiagonalMatrix"),
    ("sigma_w", DiagonalMatrix(array([1])), ValueError, "shape"),
    ("sigma_v", DiagonalMatrix(array([[1]])), ValueError, "rank"),
    ("sigma_0", array([1, 1]), ValueError, "rank"),
    ("sigma_0", array([[1]]), ValueError, "shape"),
    ("sigma_0", jnp.eye(2, dtype=jnp.complex128), TypeError, "real numbers"),
])
def test_inverse_rejects_invalid_model_structure(field, value, error, message):
    configuration = layout()
    p = unpack_parameters(jnp.zeros(configuration.n_parameters), configuration)
    with pytest.raises(error, match=message):
        pack_parameters(p._replace(**{field: value}), configuration)


@pytest.mark.parametrize("field,value,message", [
    ("sigma_w", DiagonalMatrix(array([0, 1])), "positive"),
    ("sigma_w", DiagonalMatrix(array([-1, 1])), "positive"),
    ("sigma_v", DiagonalMatrix(array([0])), "positive"),
    ("sigma_v", DiagonalMatrix(array([-1])), "positive"),
    ("sigma_0", array([[1, 0.1], [0, 1]]), "symmetric"),
    ("sigma_0", array([[1, 0], [0, 0]]), "Cholesky failed"),
    ("sigma_0", array([[1, 2], [2, 1]]), "Cholesky failed"),
    ("sigma_0", array([[-1, 0], [0, -1]]), "Cholesky failed"),
])
def test_checked_inverse_rejects_invalid_covariances_without_repair(field, value, message):
    configuration = layout()
    p = unpack_parameters(jnp.zeros(configuration.n_parameters), configuration)
    error, _ = jax.jit(checkify.checkify(pack_parameters))(p._replace(**{field: value}), configuration)
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        error.throw()


@pytest.mark.parametrize("field", ModelParameters._fields)
def test_checked_inverse_rejects_nonfinite_in_every_model_block(field):
    configuration = layout()
    p = unpack_parameters(jnp.zeros(configuration.n_parameters), configuration)
    original = getattr(p, field)
    value = (DiagonalMatrix(original.diagonal.at[0].set(jnp.nan))
             if isinstance(original, DiagonalMatrix) else original.at[0].set(jnp.nan))
    error, _ = jax.jit(checkify.checkify(pack_parameters))(p._replace(**{field: value}), configuration)
    with pytest.raises(checkify.JaxRuntimeError, match="finite"):
        error.throw()


@pytest.mark.parametrize("theta_f", [-0.1, 0, 1, 1.1])
def test_unit_interval_pack_rejects_endpoints_and_outside(theta_f):
    configuration = layout(mode="unit_interval")
    p = unpack_parameters(jnp.zeros(configuration.n_parameters), configuration)
    error, _ = jax.jit(checkify.checkify(pack_parameters))(
        p._replace(theta_f=array([theta_f, 0.5])), configuration,
    )
    with pytest.raises(checkify.JaxRuntimeError, match="strictly inside"):
        error.throw()


@pytest.mark.parametrize("field,value,message", [
    ("sigma_w", -1000, "softplus underflow"),
    ("sigma_v", -1000, "softplus underflow"),
    ("sigma_0", -1000, "softplus underflow"),
    ("sigma_0", 1e200, "Sigma_0 must be finite"),
    ("theta_f", -1000, "floating-point saturation"),
    ("theta_f", 100, "floating-point saturation"),
])
def test_unrepresentable_extremes_fail_instead_of_clipping(field, value, message):
    configuration = layout(1, 1, 1, 1, 1, "unit_interval")
    raw = jnp.zeros(configuration.n_parameters).at[getattr(configuration.slices, field).start].set(value)
    error, _ = jax.jit(checkify.checkify(unpack_parameters))(raw, configuration)
    with pytest.raises(checkify.JaxRuntimeError, match=message):
        error.throw()


def test_finite_gram_underflow_is_not_repaired_or_refactorized_forward():
    configuration = layout(0, 0, 0, 1, 0)
    raw = array([0, -500])
    assert jax.nn.softplus(raw[1]) > 0
    error, p = jax.jit(checkify.checkify(unpack_parameters))(raw, configuration)
    error.throw()
    assert np.all(np.isfinite(p.sigma_0))
    np.testing.assert_array_equal(p.sigma_0, [[0]])
    # The externally supplied covariance still must pass the inverse SPD check.
    error, _ = jax.jit(checkify.checkify(pack_parameters))(p, configuration)
    with pytest.raises(checkify.JaxRuntimeError, match="Cholesky failed"):
        error.throw()


def test_parameter_boundary_requires_explicit_container_types():
    configuration = layout()
    raw = jnp.zeros(configuration.n_parameters)
    p = unpack_parameters(raw, configuration)
    with pytest.raises(TypeError, match="layout must be ParameterLayout"):
        unpack_parameters(raw, configuration.block_sizes)
    with pytest.raises(TypeError, match="layout must be ParameterLayout"):
        pack_parameters(p, configuration.block_sizes)
    with pytest.raises(TypeError, match="parameters must be ModelParameters"):
        pack_parameters(tuple(p), configuration)


def small_model(raw, *, drop_unsystematic=False):
    """Explicit local maps: layout supplies no coordinate identities or tying."""
    p = unpack_parameters(raw, layout(2, 2, 1, 2, 1))
    initial_coordinates = StateCoordinates(("level",), (), ("u",))
    current = (StateCoordinates(("level",), (), ()) if drop_unsystematic else initial_coordinates)
    f_axes, w_axes = ("f-level", "f-u"), ("w-level", "w-u")
    a = (DenseMap(current.all, f_axes, array([[1, 0]])) if drop_unsystematic
         else selection_map(current.all, f_axes, f_axes))
    step = StructuralStep(
        initial_coordinates, current, a,
        selection_map(f_axes, initial_coordinates.all, initial_coordinates.all),
        selection_map(current.all, w_axes, ("w-level",) if drop_unsystematic else w_axes),
        selection_map(("quote",), current.unsystematic, (None,) if drop_unsystematic else ("u",)),
        selection_map(("quote",), ("v",), ("v",)),
    )
    instrument = OISInstrument(array([0.5]), array([[[0.1]], [[-0.5]]]), jnp.empty((2, 0)))
    initial = initialize_filter(initial_coordinates, p.a_x, p.sigma_0)
    inputs = EKFInputs(step, p.theta_f, p.sigma_w, p.theta_g, p.sigma_v, array([0.025]), (instrument,))
    return initial, inputs


def test_raw_vector_to_ekf_likelihood_gradient_smoke():
    # Smoke only: issue #8 owns comprehensive full-likelihood gradient validation.
    raw = array([0.9, 0.8, -6, -8, -7, 0.03, 0.002, -2.5, 0.01, -3, 0.7])

    def objective(vector):
        initial, inputs = small_model(vector)
        return run_likelihood(initial, (inputs,)).total_log_likelihood

    error, (value, gradient) = jax.jit(checkify.checkify(jax.value_and_grad(objective)))(raw)
    error.throw()
    assert value.shape == () and gradient.shape == raw.shape
    assert value.dtype == gradient.dtype == jnp.float64
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert np.all(gradient != 0)


def test_initial_layout_does_not_force_later_state_dimensions():
    raw = array([0.9, 0.8, -6, -8, -7, 0.03, 0.002, -2.5, 0.01, -3, 0.7])
    initial, inputs = small_model(raw, drop_unsystematic=True)
    result = run_likelihood(initial, (inputs,), return_trace=True)
    assert initial.state.shape == (2,) and initial.covariance.shape == (2, 2)
    assert result.trace.ekf_steps[0].filtered.state.shape == (1,)
    assert result.trace.ekf_steps[0].filtered.covariance.shape == (1, 1)
    assert np.isfinite(result.total_log_likelihood)
    assert inputs.theta_f.shape == (2,) and inputs.sigma_w.diagonal.shape == (2,)


def test_production_parameter_layer_stays_jax_only_and_in_scope():
    tree = ast.parse(Path(params_module.__file__).read_text(encoding="utf-8"))
    banned_calls = {"float", "item", "tolist", "pure_callback", "io_callback", "clip", "abs",
                    "eig", "eigh", "eigvals", "eigvalsh", "inv", "pinv", "minimize"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in banned_calls
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(m.split(".")[0] in {"numpy", "scipy", "optax", "jaxopt", "likelihood", "ekf"}
                           for m in modules)
