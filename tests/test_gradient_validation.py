"""Independent function-value derivatives of the complete raw-vector NLL."""

import ast
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

import kalmanfilter.gradient_validation as validation
from kalmanfilter.ekf import EKFInputs, initialize_filter
from kalmanfilter.gradient_validation import (
    central_difference_gradient, directional_central_difference, gradient_errors,
    raw_negative_log_likelihood,
)
from kalmanfilter.likelihood import run_likelihood
from kalmanfilter.ois import OISInstrument
from kalmanfilter.params import ParameterLayout, unpack_parameters
from kalmanfilter.transition import StateCoordinates, StructuralStep, selection_map


STEPS = (1e-4, 1e-5, 1e-6, 1e-7)


def array(value):
    return jnp.asarray(value, dtype=jnp.float64)


def make_layout(mode="identity"):
    return ParameterLayout(n_f=2, n_w=2, n_v=1, n_x0=2, n_g=1, theta_f_transform=mode)


def raw_cases(mode="identity"):
    center = np.array([0.85, 0.6, -5, -6, -5.5, 0.03, -0.01, -2.3, 0.025, -2.8, 0.75])
    scales = np.array([0.03, 0.03, 0.15, 0.15, 0.15, 0.005, 0.004, 0.1, 0.005, 0.1, 0.03])
    perturbed = center + np.random.default_rng(20260915).normal(size=(3, center.size)) * scales
    if mode == "unit_interval":
        perturbed = perturbed[:1].copy()
        perturbed[0, :2] = [1.2, 0.4]  # Explicit option, safely inside sigmoid's useful range.
    return tuple(array(row) for row in perturbed)


def build_problem(p):
    """Three explicitly supplied dates, two states, one nonlinear quote per date."""
    coordinates = StateCoordinates(("level",), (), ("u",))
    f_axes, w_axes = ("f-level", "f-u"), ("w-level", "w-u")
    step = StructuralStep(
        coordinates, coordinates,
        selection_map(coordinates.all, f_axes, f_axes),
        selection_map(f_axes, coordinates.all, coordinates.all),
        selection_map(coordinates.all, w_axes, w_axes),
        selection_map(("quote",), coordinates.unsystematic, ("u",)),
        selection_map(("quote",), ("v",), ("v",)),
    )
    inputs = tuple(
        EKFInputs(step, p.theta_f, p.sigma_w, p.theta_g, p.sigma_v, array([quote]),
                  (OISInstrument(array([0.5, 0.5]), array(loadings)[:, None, None],
                                 jnp.empty((3, 0), dtype=jnp.float64)),))
        for quote, loadings in (
            (0.04, (0.1, -0.5, -1.1)),
            (0.025, (0.08, -0.45, -1.05)),
            (0.055, (0.12, -0.55, -1.2)),
        )
    )
    return initialize_filter(coordinates, p.a_x, p.sigma_0), inputs


def checked_call(function):
    compiled = jax.jit(checkify.checkify(function))

    def call(*args):
        error, result = compiled(*args)
        error.throw()
        return result

    return call


def block_name(configuration, index):
    return next(name for name in configuration.slices._fields
                if getattr(configuration.slices, name).start <= index < getattr(configuration.slices, name).stop)


@pytest.fixture(scope="module", params=["identity", "unit_interval"])
def measurements(request):
    mode = request.param
    configuration = make_layout(mode)

    def objective(raw):
        return raw_negative_log_likelihood(raw, configuration, build_problem)

    value_only = checked_call(objective)
    value_and_gradient = checked_call(jax.value_and_grad(objective))
    gradient_only = checked_call(jax.grad(objective))
    rng = np.random.default_rng(17)
    directions = (np.arange(1, 12) * np.where(np.arange(11) % 2, -1, 1),
                  rng.normal(size=11), rng.normal(size=11))
    cases = []
    for raw in raw_cases(mode):
        value, automatic = value_and_gradient(raw)
        numerical = tuple(central_difference_gradient(value_only, raw, h) for h in STEPS)
        directional = tuple(tuple(directional_central_difference(value_only, raw, d, h)
                                  for h in STEPS) for d in directions)
        cases.append(dict(raw=raw, value=value, automatic=automatic, grad_only=gradient_only(raw),
                          numerical=numerical, directional=directional))
    return mode, configuration, objective, cases


def test_full_raw_vector_gradients_and_component_reports(measurements):
    mode, configuration, _, cases = measurements
    for case_index, case in enumerate(cases):
        raw, value, automatic = case["raw"], case["value"], case["automatic"]
        assert value.shape == () and automatic.shape == raw.shape == (11,)
        assert value.dtype == automatic.dtype == jnp.float64
        assert np.isfinite(value) and np.all(np.isfinite(automatic))
        np.testing.assert_allclose(case["grad_only"], automatic, rtol=2e-13, atol=2e-13)
        for h, numerical in zip(STEPS, case["numerical"], strict=True):
            errors = gradient_errors(automatic, numerical.gradient)
            worst = int(errors.worst_absolute_index)
            worst_relative = int(errors.worst_relative_index)
            print(f"\n{mode} case={case_index} h={h:.0e}: abs={errors.max_absolute:.9e} "
                  f"at {worst}/{block_name(configuration, worst)} "
                  f"AD={automatic[worst]:.12e} FD={numerical.gradient[worst]:.12e}; "
                  f"rel={errors.max_relative:.9e} at {worst_relative}/{block_name(configuration, worst_relative)}")
            np.testing.assert_allclose(numerical.steps, h * np.maximum(1, np.abs(raw)), rtol=1e-15)
            for name, block in zip(configuration.slices._fields, configuration.slices, strict=True):
                block_errors = gradient_errors(automatic[block], numerical.gradient[block])
                assert np.all(np.abs(automatic[block]) > 1e-6), f"inactive block: {name}"
                print(f"  {name}: abs={block_errors.max_absolute:.9e} rel={block_errors.max_relative:.9e}")


def test_full_gradient_directional_reports(measurements):
    mode, _, _, cases = measurements
    for case_index, case in enumerate(cases):
        for direction_index, estimates in enumerate(case["directional"]):
            for h, numerical in zip(STEPS, estimates, strict=True):
                ad_directional = jnp.dot(case["automatic"], numerical.direction)
                errors = gradient_errors(jnp.atleast_1d(ad_directional), jnp.atleast_1d(numerical.derivative))
                np.testing.assert_allclose(np.linalg.norm(numerical.direction), 1, rtol=2e-15)
                assert np.all(numerical.direction != 0)
                print(f"\n{mode} case={case_index} direction={direction_index} h={h:.0e}: "
                      f"AD={ad_directional:.12e} FD={numerical.derivative:.12e} "
                      f"abs={errors.max_absolute:.9e} rel={errors.max_relative:.9e}")


def test_central_difference_matches_analytical_smooth_function_without_ad(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("finite differences must use objective values only")

    for name in ("grad", "value_and_grad", "jacfwd", "jacrev", "jvp", "vjp"):
        monkeypatch.setattr(jax, name, forbidden)
    point = array([-1.3, 0.7, 2.1])

    def function(x):
        return jnp.sin(x[0]) + x[0] * x[1] + jnp.exp(0.2 * x[2]) + x[1]**3

    analytical = np.array([np.cos(point[0]) + point[1], point[0] + 3 * point[1]**2,
                           0.2 * np.exp(0.2 * point[2])])
    errors = []
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-10):
        numerical = central_difference_gradient(function, point, h)
        errors.append(np.max(np.abs(numerical.gradient - analytical)))
        if h in (1e-4, 1e-5, 1e-6):
            np.testing.assert_allclose(numerical.gradient, analytical, rtol=1e-7, atol=2e-10)
        directional = directional_central_difference(function, point, array([2, -1, 3]), h)
        if h in (1e-4, 1e-5, 1e-6):
            np.testing.assert_allclose(directional.derivative, analytical @ directional.direction,
                                       rtol=1e-7, atol=2e-10)
    assert errors[2] < errors[0]  # Truncation decreases into a useful region.
    assert errors[4] > errors[2]  # Very small steps suffer cancellation.


def test_gradient_errors_have_true_relative_errors_and_both_zero_convention():
    result = gradient_errors([0, 1e-12, 2, -4], [0, 2e-12, 2.25, -3])
    np.testing.assert_array_equal(result.absolute, [0, 1e-12, 0.25, 1])
    np.testing.assert_allclose(result.relative, [0, 0.5, 1 / 9, 0.25], rtol=1e-15)
    assert result.max_absolute == 1 and result.worst_absolute_index == 3
    assert result.max_relative == 0.5 and result.worst_relative_index == 1
    # Near-zero derivatives can pass an absolute criterion despite a large relative error.
    assert result.absolute[1] <= 2e-12 + 1e-8 * 2e-12
    empty = gradient_errors([], [])
    assert empty.max_absolute == empty.max_relative == 0
    assert empty.worst_absolute_index is empty.worst_relative_index is None
