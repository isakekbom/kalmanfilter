"""Package foundation with process-wide JAX float64 enabled on import.

Import this package before creating arrays or compiling numerical functions.
The setting also applies when importing a kalmanfilter submodule directly.
"""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .model import ModelDimensions

__all__ = ["ModelDimensions"]
