"""Verify the installed package and its precision policy in fresh processes."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("entry_module", ["kalmanfilter", "kalmanfilter.model"])
def test_installed_import_enables_float64(entry_module, tmp_path):
    # A fresh, isolated interpreter avoids a previous test masking bad setup.
    # Running outside the checkout also exercises the editable installation.
    script = textwrap.dedent(
        f"""
        import importlib
        from importlib.metadata import version

        import jax
        assert not jax.config.x64_enabled
        importlib.import_module({entry_module!r})
        import jax.numpy as jnp

        assert version("kalmanfilter")
        assert jax.config.x64_enabled
        values = jnp.array([1.0, 2.0])
        assert values.dtype == jnp.float64
        assert jnp.ones(2, dtype=jnp.float64).dtype == jnp.float64

        # Exercise the CPU runtime, JIT, and autodiff without model logic.
        gradient = jax.jit(jax.grad(lambda x: jnp.sum(x * x)))(values)
        assert gradient.dtype == jnp.float64
        assert gradient.tolist() == [2.0, 4.0]

        for module in (
            "model", "ois", "transition", "ekf", "likelihood", "params", "synthetic"
        ):
            importlib.import_module("kalmanfilter." + module)
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "JAX_ENABLE_X64": "false", "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
