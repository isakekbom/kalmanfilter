"""Explicit algebraic structural examples; no calendar or filtering simulation."""

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from kalmanfilter.ois import OISInstrument, observation_quotes, quote_state_jacobian
from kalmanfilter.transition import (
    CoordinateMap,
    DenseMap,
    DiagonalMatrix,
    StateChange,
    StateCoordinates,
    StructuralStep,
    apply_map,
    covariance_dense,
    mapped_covariance,
    materialize_map,
    observation_covariance,
    process_covariance,
    select_observations,
    selection_map,
    split_state,
    transition_matrix,
    validate_step,
)


def array(value):
    return jnp.asarray(value, dtype=jnp.float64)


def constant_step():
    coordinates = StateCoordinates(("p",), ("c",), ("u",))
    parameters = ("f_p", "f_c", "f_u")
    return StructuralStep(
        previous=coordinates,
        current=coordinates,
        transition_post_map=selection_map(coordinates.all, parameters, parameters),
        transition_pre_map=selection_map(parameters, coordinates.all, coordinates.all),
        process_noise_map=selection_map(coordinates.all, ("wp", "wc", "wu"), ("wp", "wc", "wu")),
        observation_selector=selection_map(("quote",), ("u",), ("u",)),
        observation_noise_map=selection_map(("quote",), ("noise",), ("noise",)),
    )


def numpy_map(mapping):
    """Independent small dense oracle, used only in tests."""
    if isinstance(mapping, DenseMap):
        return np.asarray(mapping.values)
    result = np.zeros(mapping.shape, dtype=np.float64)
    weights = np.ones(len(mapping.rows)) if mapping.weights is None else np.asarray(mapping.weights)
    for row, source in enumerate(mapping.sources):
        if source is not None:
            result[row, mapping.columns.index(source)] = weights[row]
    return result


def test_constant_dimension_diagonal_transition_exactly():
    step = constant_step()
    theta = array([0.9, 1.0, 0.6])
    transition = transition_matrix(theta, step)
    assert isinstance(transition, CoordinateMap)
    assert transition.shape == (3, 3)
    assert transition.rows == transition.columns == ("p", "c", "u")
    np.testing.assert_array_equal(materialize_map(transition), np.diag([0.9, 1.0, 0.6]))
    np.testing.assert_array_equal(apply_map(transition, array([2, 3, 5])), [1.8, 3.0, 3.0])
    assert step.state_change.surviving == (("p", 0, 0), ("c", 1, 1), ("u", 2, 2))
    assert step.state_change.introduced == step.state_change.removed == ()


def removal_step():
    previous = StateCoordinates(("p",), ("old", "keep"), ("ua", "ub"))
    current = StateCoordinates(("p",), ("keep",), ("ub",))
    # Parameter ordering is deliberately different from current state ordering.
    parameters = ("f_ub", "f_keep", "f_p")
    return StructuralStep(
        previous, current,
        selection_map(current.all, parameters, ("f_p", "f_keep", "f_ub")),
        selection_map(parameters, previous.all, ("ub", "keep", "p")),
        selection_map(current.all, ("wp", "wk", "wu"), ("wp", "wk", "wu")),
        selection_map(("qb",), ("ub",), ("ub",)),
        selection_map(("qb",), ("vb", "va"), ("vb",)),
    )


def test_removal_preserves_named_survivors_not_array_prefixes():
    step = removal_step()
    assert step.state_change.surviving == (("p", 0, 0), ("keep", 2, 1), ("ub", 4, 2))
    assert step.state_change.removed == ("old", "ua")
    assert step.state_change.introduced == ()
    assert step.transition_post_map.shape == (3, 3)
    assert step.transition_pre_map.shape == (3, 5)
    expected = [[0.9, 0, 0, 0, 0], [0, 0, 0.8, 0, 0], [0, 0, 0, 0, 0.7]]
    transition = transition_matrix(array([0.7, 0.8, 0.9]), step)
    np.testing.assert_array_equal(materialize_map(transition), expected)
    np.testing.assert_allclose(
        apply_map(transition, array([1, 100, 2, 200, 3])), [0.9, 1.6, 2.1], rtol=1e-15
    )


def introduction_step():
    previous = StateCoordinates(("p",), (), ("u",))
    current = StateCoordinates(("p",), ("new",), ("u",))
    parameters = ("fp", "fu")
    # Arbitrary supplied algebra: new depends on two previous coordinates.
    # This is not a proposed financial initialization rule.
    return StructuralStep(
        previous, current,
        DenseMap(current.all, parameters, array([[1, 0], [0.25, 0.5], [0, 1]])),
        selection_map(parameters, previous.all, ("p", "u")),
        DenseMap(current.all, ("w1", "w2"), array([[1, 0], [0.5, 2], [0, 1]])),
        selection_map(("q",), ("u",), ("u",)),
        selection_map(("q",), ("v",), ("v",)),
    )


def test_introduction_requires_supplied_linear_map_without_initialization():
    step = introduction_step()
    assert step.state_change.introduced == ("new",)
    assert step.state_change.surviving == (("p", 0, 0), ("u", 1, 2))
    assert step.state_change.removed == ()
    assert step.transition_post_map.shape == (3, 2)
    assert step.transition_pre_map.shape == (2, 2)
    result = transition_matrix(array([0.8, 0.6]), step)
    np.testing.assert_array_equal(materialize_map(result), [[0.8, 0], [0.2, 0.3], [0, 0.6]])
    assert result.shape == (3, 2)
    # Changing the caller's new-row coefficients changes F; nothing overrides it.
    changed = replace(step, transition_post_map=replace(
        step.transition_post_map, values=array([[1, 0], [-2, 3], [0, 1]])
    ))
    np.testing.assert_allclose(materialize_map(transition_matrix(array([0.8, 0.6]), changed))[1], [-1.6, 1.8])


def test_simultaneous_removal_and_introduction_with_equal_counts():
    previous = StateCoordinates(("p",), ("old",), ("ua",))
    current = StateCoordinates(("p",), ("new",), ("ub",))
    parameters = ("fp", "fc", "fu")
    step = StructuralStep(
        previous, current,
        selection_map(current.all, parameters, parameters),
        # None explicitly specifies a zero deterministic row, not a distribution.
        selection_map(parameters, previous.all, ("p", "old", None)),
        selection_map(current.all, ("wp", "wc", "wu"), ("wp", "wc", "wu")),
        selection_map(("qb",), ("ub",), ("ub",)),
        selection_map(("qb",), ("vb",), ("vb",)),
    )
    assert step.state_change.surviving == (("p", 0, 0),)
    assert step.state_change.removed == ("old", "ua")
    assert step.state_change.introduced == ("new", "ub")
    result = transition_matrix(array([0.9, 0.5, 0.8]), step)
    assert result.sources == ("p", "old", None)
    np.testing.assert_array_equal(materialize_map(result), np.diag([0.9, 0.5, 0]))


@pytest.mark.parametrize("a_dense,b_dense", [(False, False), (False, True), (True, False), (True, True)])
def test_all_map_combinations_and_repeated_sources_match_independent_product(a_dense, b_dense):
    step = constant_step()
    parameters = ("r0", "r1", "r2", "r3")
    a = CoordinateMap(step.current.all, parameters, ("r3", "r0", None), array([2, -1, 7]))
    b = CoordinateMap(parameters, step.previous.all, ("u", "p", None, "u"), array([3, -2, 8, 4]))
    if a_dense:
        a = DenseMap(a.rows, a.columns, array([[1, 2, -1, 3], [0.5, -1, 2, 4], [1, 0, 0, -2]]))
    if b_dense:
        b = DenseMap(b.rows, b.columns, array([[1, 0, 2], [-1, 3, 0], [0, 1, 1], [2, 0, -3]]))
    step = replace(step, transition_post_map=a, transition_pre_map=b)
    theta = array([0.5, -0.25, 2, 0.75])
    expected = numpy_map(a) @ np.diag(np.asarray(theta)) @ numpy_map(b)
    result = transition_matrix(theta, step)
    np.testing.assert_allclose(materialize_map(result), expected, rtol=1e-14, atol=1e-14)
    operand = array([[1, 2], [3, -1], [0.5, 4]])
    np.testing.assert_allclose(apply_map(result, operand), expected @ np.asarray(operand), atol=1e-14)


def missing_step():
    coordinates = StateCoordinates(("p",), (), ("ua", "ub", "uc"))
    parameters = ("fp", "fa", "fb", "fc")
    return StructuralStep(
        coordinates, coordinates,
        selection_map(coordinates.all, parameters, parameters),
        selection_map(parameters, coordinates.all, coordinates.all),
        selection_map(coordinates.all, coordinates.all, coordinates.all),
        selection_map(("qc", "qa"), coordinates.unsystematic, ("uc", "ua")),
        selection_map(("qc", "qa"), ("noise-b", "noise-a", "noise-c"), ("noise-c", "noise-a")),
    )


def test_only_explicit_active_observations_are_selected_in_declared_order():
    step = missing_step()
    assert step.active_observations == ("qc", "qa")
    assert step.observation_selector.shape == step.observation_noise_map.shape == (2, 3)
    np.testing.assert_array_equal(materialize_map(step.observation_selector), [[0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(materialize_map(step.observation_noise_map), [[0, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(
        select_observations(step, ("qa", "qb", "qc"), array([0, 999, 0.03])), [0.03, 0]
    )
    np.testing.assert_array_equal(apply_map(step.observation_selector, array([10, 20, 30])), [30, 10])
    with pytest.raises(ValueError, match="unknown source identity 'qc'"):
        select_observations(step, ("qa", "qb"), array([0, 1]))


def test_ois_interface_uses_same_active_and_systematic_order_without_filtering():
    step = missing_step()
    systematic, unsystematic = split_state(step.current, array([0.1, 2, 3, 4]))
    data_by_id = {
        "qa": OISInstrument(array([0.25]), array([[[0]], [[-1]]]), jnp.empty((2, 0))),
        "qc": OISInstrument(array([0.2, 0.3]), array([[[0.1]], [[-0.5]], [[-2]]]), jnp.empty((3, 0))),
    }
    instruments = tuple(data_by_id[identity] for identity in step.active_observations)
    quotes = observation_quotes(array([0.7]), systematic, instruments)
    jacobian = quote_state_jacobian(array([0.7]), systematic, instruments)
    assert quotes.shape == (step.dimensions.n_z_t,)
    assert jacobian.shape == (2, 1)
    assert unsystematic.shape == (3,)
    assert apply_map(step.observation_selector, unsystematic).shape == quotes.shape


@pytest.mark.parametrize("dense_base", [False, True])
@pytest.mark.parametrize("kind", ["unique-selector", "repeated-selector", "dense"])
def test_covariance_mapping_matches_independent_reference(kind, dense_base):
    rows, columns = ("a", "b", "c"), ("w0", "w1")
    if kind == "dense":
        mapping = DenseMap(rows, columns, array([[1, 2], [-1, 0.5], [0.25, -2]]))
    else:
        sources = ("w1", "w0", None) if kind == "unique-selector" else ("w1", "w1", None)
        mapping = CoordinateMap(rows, columns, sources, array([2, -3, 7]))
    base = array([[4, 1], [1, 9]]) if dense_base else DiagonalMatrix(array([4, 9]))
    reference_base = np.asarray(base) if dense_base else np.diag([4, 9])
    expected = numpy_map(mapping) @ reference_base @ numpy_map(mapping).T
    result = mapped_covariance(mapping, base)
    assert result.shape == (3, 3)
    np.testing.assert_allclose(covariance_dense(result), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(covariance_dense(result), covariance_dense(result).T, atol=1e-14)
    assert isinstance(result, DiagonalMatrix) == (kind == "unique-selector" and not dense_base)
    if kind == "repeated-selector":
        assert covariance_dense(result)[0, 1] == -54  # shared noise is correlated


def test_process_and_observation_covariances_have_correct_different_axes():
    step = introduction_step()
    q = process_covariance(step, DiagonalMatrix(array([4, 9])))
    np.testing.assert_array_equal(q, [[4, 2, 0], [2, 37, 18], [0, 18, 9]])
    r = observation_covariance(missing_step(), DiagonalMatrix(array([2, 3, 5])))
    assert isinstance(r, DiagonalMatrix)
    assert r.shape == (2, 2)
    np.testing.assert_array_equal(r.diagonal, [5, 3])


def test_g_equals_selector_only_when_explicitly_supplied():
    step = missing_step()
    specialized = replace(step, observation_noise_map=step.observation_selector)
    # Variances here follow (ua,ub,uc), the explicitly chosen G column order.
    result = observation_covariance(specialized, DiagonalMatrix(array([2, 3, 5])))
    np.testing.assert_array_equal(result.diagonal, [5, 2])
    general = replace(step, observation_noise_map=DenseMap(
        step.active_observations, ("shared",), array([[2], [-1]])
    ))
    np.testing.assert_array_equal(
        observation_covariance(general, DiagonalMatrix(array([3]))), [[12, -6], [-6, 3]]
    )


def timeline():
    states = (
        StateCoordinates(("p",), ("c0",), ("u0",)),
        StateCoordinates(("p",), (), ("u0",)),
        StateCoordinates(("p",), ("c1",), ("u0", "u1")),
        StateCoordinates(("p",), ("c2",), ("u1",)),
        StateCoordinates(("p",), ("c2",), ("u1",)),
    )
    # A common parameter axis for THIS explicit example, independent of state
    # storage. Connections, including introduced rows, are specified individually.
    parameters = ("fp", "fc", "fu0", "fu1")
    connections = (
        (("fp", "fu0"), ("p", None, "u0", None)),
        (("fp", "fc", "fu0", "fu1"), ("p", "p", "u0", None)),
        (("fp", "fc", "fu1"), ("p", "c1", None, "u1")),
        (("fp", "fc", "fu1"), ("p", "c2", None, "u1")),
    )
    active = (("q0",), ("q1", "q0"), (), ("q1",))
    deviations = (("u0",), ("u1", "u0"), (), ("u1",))
    noise = (("v0",), ("v1", "v0"), (), ("v1",))
    return tuple(
        StructuralStep(
            previous, current,
            selection_map(current.all, parameters, a_sources),
            selection_map(parameters, previous.all, b_sources),
            selection_map(current.all, parameters, a_sources),
            selection_map(obs, current.unsystematic, dev),
            selection_map(obs, ("v0", "v1"), obs_noise),
        )
        for previous, current, (a_sources, b_sources), obs, dev, obs_noise in zip(
            states[:-1], states[1:], connections, active, deviations, noise
        )
    )


def product_shape(*shapes):
    """Dimension audit only; evaluates no EKF state/covariance recursion."""
    for left, right in zip(shapes, shapes[1:]):
        assert left[1] == right[0]
    return shapes[0][0], shapes[-1][1]


def test_time_varying_sequence_and_all_future_ekf_product_shapes():
    steps = timeline()
    assert [(len(s.previous.all), s.dimensions.n_x_t, s.dimensions.n_z_t) for s in steps] == [
        (3, 2, 1), (2, 4, 2), (4, 3, 0), (3, 3, 1)
    ]
    for previous, current in zip(steps, steps[1:]):
        assert previous.current == current.previous  # identities, not just counts
    theta = array([0.9, 0.8, 0.7, 0.6])
    for step in steps:
        a, b = step.transition_post_map, step.transition_pre_map
        f = transition_matrix(theta, step)
        q = process_covariance(step, DiagonalMatrix(array([1, 2, 3, 4])))
        r = observation_covariance(step, DiagonalMatrix(array([5, 6])))
        n_prev, n = len(step.previous.all), step.dimensions.n_x_t
        s, u, m = step.dimensions.n_s_t, step.dimensions.n_u_t, step.dimensions.n_z_t
        assert f.shape == (n, n_prev) == product_shape(a.shape, (4, 4), b.shape)  # (30)
        assert product_shape(f.shape, (n_prev, 1)) == (n, 1)  # (41)
        assert product_shape(f.shape, (n_prev, n_prev), f.shape[::-1]) == q.shape == (n, n)  # (42)
        assert product_shape(a.shape, (4, 4), b.shape, (n_prev, n_prev),
                             b.shape[::-1], (4, 4), a.shape[::-1]) == q.shape  # (43)
        assert step.observation_selector.shape == (m, u)
        h = (m, s + u)  # (44), concatenate J=(m,s) and I^z=(m,u)
        assert h == (m, n)
        assert product_shape((m, s), (s, 1)) == (m, 1)  # (45), (47)-(48)
        assert product_shape(h, (n, 1)) == (m, 1)  # (46)
        assert product_shape((m, u), (u, 1)) == (m, 1)  # (49)
        assert product_shape(h, (n, n), h[::-1]) == r.shape == (m, m)  # (50)
        gain = product_shape((n, n), h[::-1], (m, m))
        assert gain == (n, m)  # (51)
        assert product_shape(h, (n, n)) == (m, n)  # (52) solve RHS
        assert product_shape(gain, (m, 1)) == (n, 1)  # (53)
        assert product_shape(gain, h, (n, n)) == (n, n)  # (54)-(55)
        assert product_shape((n, n), h[::-1], (m, m), h, (n, n)) == (n, n)  # (56)


@pytest.mark.parametrize("step_factory", [constant_step, introduction_step, missing_step])
def test_fixed_step_jit_and_float64_dynamic_parameters_and_covariances(step_factory):
    step = step_factory()
    theta = jnp.ones(step.transition_post_map.shape[1], dtype=jnp.float32)
    state = jnp.arange(len(step.previous.all), dtype=jnp.int32)
    w = DiagonalMatrix(jnp.ones(step.process_noise_map.shape[1], dtype=jnp.float32))
    v = DiagonalMatrix(jnp.ones(step.observation_noise_map.shape[1], dtype=jnp.float32))

    def kernel(parameters, structure, operand, sigma_w, sigma_v):
        f = transition_matrix(parameters, structure)
        return (
            f, apply_map(f, operand),
            process_covariance(structure, sigma_w),
            observation_covariance(structure, sigma_v),
        )

    compiled = jax.jit(checkify.checkify(kernel))
    for scale in (1.0, 1.5):
        changed = step
        if isinstance(step.transition_post_map, DenseMap):
            changed = replace(step, transition_post_map=replace(
                step.transition_post_map, values=step.transition_post_map.values * scale
            ))
        error, result = compiled(theta * scale, changed, state, w, v)
        error.throw()
        f, applied, q, r = result
        reference = (
            numpy_map(changed.transition_post_map)
            @ np.diag(np.asarray(theta * scale))
            @ numpy_map(changed.transition_pre_map)
        )
        np.testing.assert_allclose(materialize_map(f), reference, rtol=1e-14)
        np.testing.assert_allclose(applied, reference @ np.asarray(state), rtol=1e-14)
        for leaf in jax.tree.leaves(result):
            assert leaf.dtype == jnp.float64
        assert q.shape == (step.dimensions.n_x_t,) * 2
        assert r.shape == (step.dimensions.n_z_t,) * 2


def test_jit_accepts_changing_step_shapes_with_recompilation():
    compiled = jax.jit(checkify.checkify(transition_matrix))
    for step in timeline():
        error, f = compiled(array([0.9, 0.8, 0.7, 0.6]), step)
        error.throw()
        assert f.shape == (len(step.current.all), len(step.previous.all))


def test_structural_parameter_and_covariance_derivatives():
    step = introduction_step()
    theta = array([0.8, 0.6])
    x = array([2, 3])
    derivative = jax.jacfwd(lambda t: apply_map(transition_matrix(t, step), x))(theta)
    np.testing.assert_array_equal(derivative, [[2, 0], [0.5, 1.5], [0, 3]])
    gradient = jax.grad(
        lambda variances: jnp.sum(process_covariance(step, DiagonalMatrix(variances)))
    )(array([4, 9]))
    np.testing.assert_array_equal(gradient, [2.25, 9])


def test_selectors_and_diagonal_covariances_never_allocate_quadratic_arrays():
    n = 64
    ids = tuple(f"p{i}" for i in range(n))
    state = StateCoordinates(ids, (), ())
    identity = selection_map(ids, ids, ids)
    step = StructuralStep(state, state, identity, identity, identity,
                          selection_map((), (), ()), selection_map((), (), ()))

    def kernel(theta, variances, operand):
        f = transition_matrix(theta, step)
        return f, process_covariance(step, DiagonalMatrix(variances)), apply_map(f, operand)

    args = (jnp.ones(n), jnp.ones(n), jnp.ones(n))
    result = kernel(*args)
    assert isinstance(result[0], CoordinateMap)
    assert isinstance(result[1], DiagonalMatrix)
    assert all(leaf.shape == (n,) for leaf in jax.tree.leaves(result))
    traced = jax.make_jaxpr(checkify.checkify(kernel))(*args)
    assert all(getattr(var.aval, "shape", None) != (n, n)
               for equation in traced.jaxpr.eqns for var in equation.outvars)


def test_empty_spaces_and_explicit_zero_rows():
    empty = StateCoordinates((), (), ())
    current = StateCoordinates(("new",), (), ())
    zero = selection_map(current.all, (), (None,))
    step = StructuralStep(empty, current, zero, selection_map((), (), ()), zero,
                          selection_map((), (), ()), selection_map((), (), ()))
    f = transition_matrix(array([]), step)
    assert f.shape == (1, 0)
    np.testing.assert_array_equal(apply_map(f, array([])), [0])
    np.testing.assert_array_equal(process_covariance(step, DiagonalMatrix(array([]))).diagonal, [0])
    assert observation_covariance(step, DiagonalMatrix(array([]))).shape == (0, 0)
    assert select_observations(step, (), array([])).shape == (0,)
    empty_map = selection_map((), (), ())
    assert materialize_map(empty_map).shape == (0, 0)
    assert mapped_covariance(empty_map, jnp.empty((0, 0))).shape == (0, 0)


@pytest.mark.parametrize(
    "ids,error",
    [(["p"], TypeError), (("p", "p"), ValueError), (("",), TypeError), ((1,), TypeError)],
)
def test_invalid_coordinate_identities_fail(ids, error):
    with pytest.raises(error):
        StateCoordinates(ids, (), ())


def test_duplicate_cross_block_and_changed_block_identities_fail():
    with pytest.raises(ValueError, match="unique"):
        StateCoordinates(("same",), ("same",), ())
    with pytest.raises(ValueError, match="changed factor block"):
        StateChange(StateCoordinates(("same",), (), ()), StateCoordinates((), ("same",), ()))


@pytest.mark.parametrize("sources", [("unknown",), (), ["p"], (True,)])
def test_invalid_selector_sources_fail(sources):
    with pytest.raises(ValueError):
        selection_map(("row",), ("p",), sources)


@pytest.mark.parametrize("field,axis", [
    ("transition_post_map", "rows"), ("transition_pre_map", "columns"),
    ("transition_post_map", "columns"), ("process_noise_map", "rows"),
    ("observation_selector", "columns"), ("observation_noise_map", "rows"),
])
def test_same_size_wrong_named_axes_fail(field, axis):
    step = constant_step()
    mapping = getattr(step, field)
    ids = getattr(mapping, axis)
    changed_ids = tuple(reversed(ids)) if len(ids) > 1 else ("different",)
    kwargs = {axis: changed_ids}
    if axis == "columns":
        rename = dict(zip(ids, changed_ids))
        kwargs["sources"] = tuple(rename.get(s, s) for s in mapping.sources)
    with pytest.raises(ValueError, match="coordinate order mismatch"):
        replace(step, **{field: replace(mapping, **kwargs)})


@pytest.mark.parametrize("mapping", [
    DenseMap(("a",), ("b",), array([[1, 2]])),
    CoordinateMap(("a",), ("b",), ("b",), array([1, 2])),
])
def test_array_shapes_are_validated_at_numerical_boundary(mapping):
    with pytest.raises(ValueError, match="map data shape"):
        apply_map(mapping, array([1]))


def test_whole_step_validation_includes_noise_maps_before_use():
    step = introduction_step()
    assert validate_step(step) is step
    malformed = replace(step, process_noise_map=replace(
        step.process_noise_map, values=jnp.ones((2, 3))
    ))
    with pytest.raises(ValueError, match="map data shape"):
        validate_step(malformed)
    nonfinite = replace(step, process_noise_map=replace(
        step.process_noise_map, values=step.process_noise_map.values.at[0, 0].set(jnp.inf)
    ))
    error, _ = jax.jit(checkify.checkify(validate_step))(nonfinite)
    with pytest.raises(checkify.JaxRuntimeError, match="map values must be finite"):
        error.throw()


def test_wrong_operand_parameter_covariance_and_state_shapes_fail():
    step = constant_step()
    with pytest.raises(ValueError, match="operand leading dimension"):
        apply_map(step.transition_pre_map, array([1, 2]))
    with pytest.raises(ValueError, match="theta_f shape"):
        transition_matrix(array([1, 2]), step)
    with pytest.raises(ValueError, match="base variances shape"):
        process_covariance(step, DiagonalMatrix(array([1, 2])))
    with pytest.raises(ValueError, match="base covariance shape"):
        process_covariance(step, jnp.eye(2))
    with pytest.raises(ValueError, match="state shape"):
        split_state(step.current, array([1, 2]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_parameters_fail_eager_and_compiled(bad):
    step = constant_step()
    theta = array([0.9, bad, 0.6])
    with pytest.raises(checkify.JaxRuntimeError, match="theta_f must be finite"):
        transition_matrix(theta, step)
    error, _ = jax.jit(checkify.checkify(transition_matrix))(theta, step)
    with pytest.raises(checkify.JaxRuntimeError, match="theta_f must be finite"):
        error.throw()


def test_bad_covariance_data_fails_without_repair():
    step = constant_step()
    for diagonal in ([1, -1, 2], [1, np.nan, 2]):
        with pytest.raises(checkify.JaxRuntimeError):
            process_covariance(step, DiagonalMatrix(array(diagonal)))
    with pytest.raises(checkify.JaxRuntimeError, match="must be symmetric"):
        process_covariance(step, array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]))
    with pytest.raises(checkify.JaxRuntimeError, match="mapped variances must be finite"):
        mapped_covariance(CoordinateMap(("a",), ("b",), ("b",), array([1e200])), DiagonalMatrix(array([1])))
    # Singular PSD noise is permitted; positive floors would be an extra policy.
    np.testing.assert_array_equal(
        process_covariance(step, DiagonalMatrix(array([0, 0, 0]))).diagonal, [0, 0, 0]
    )


@pytest.mark.parametrize("bad", [array([1]).astype(jnp.complex128), jnp.array([True])])
def test_nonreal_and_boolean_values_are_rejected(bad):
    with pytest.raises(TypeError, match="real numbers"):
        apply_map(selection_map(("a",), ("b",), ("b",)), bad)


def test_metadata_is_immutable_and_arrays_are_not_mutated():
    step = introduction_step()
    before = [np.array(leaf) for leaf in jax.tree.leaves(step)]
    with pytest.raises(FrozenInstanceError):
        step.current = step.previous
    with pytest.raises(FrozenInstanceError):
        step.current.steps = ("replacement",)
    transition_matrix(array([0.8, 0.6]), step)
    process_covariance(step, DiagonalMatrix(array([1, 2])))
    for expected, actual in zip(before, jax.tree.leaves(step)):
        np.testing.assert_array_equal(actual, expected)
