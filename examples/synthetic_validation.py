"""Run the exact scalar synthetic case through the production EKF/likelihood.

From the repository root: uv run --locked python examples/synthetic_validation.py
"""

import kalmanfilter  # Enable float64 before creating JAX arrays.
import jax
import jax.numpy as jnp

from kalmanfilter.likelihood import run_likelihood
from kalmanfilter.ois import OISInstrument
from kalmanfilter.synthetic import SyntheticStepInputs, generate_synthetic_dataset
from kalmanfilter.transition import DiagonalMatrix, StateCoordinates, StructuralStep, selection_map


def main():
    coordinates = StateCoordinates((), (), ("u",))
    step = StructuralStep(
        coordinates, coordinates,
        selection_map(("u",), ("phi",), ("phi",)),
        selection_map(("phi",), ("u",), ("u",)),
        selection_map(("u",), ("w",), ("w",)),
        selection_map(("quote",), ("u",), ("u",)),
        selection_map(("quote",), ("v",), ("v",)),
    )
    instrument = OISInstrument(jnp.array([1.0]), jnp.empty((2, 0, 0)), jnp.empty((2, 0)))
    date = SyntheticStepInputs(
        step, jnp.array([0.8]), DiagonalMatrix(jnp.array([0.04])),
        jnp.empty((0,)), DiagonalMatrix(jnp.array([0.09])), (instrument,),
    )
    dataset = generate_synthetic_dataset(
        jax.random.key(42), coordinates, jnp.array([0.3]), jnp.array([[0.25]]), (date,) * 12,
    )
    result = run_likelihood(dataset.initial_filter, dataset.inputs, return_trace=True)
    print(f"Dates: {len(dataset.inputs)}")
    print(f"Log-likelihood: {float(result.total_log_likelihood):.12f}")
    print("Final true state:", dataset.true_states[-1])
    print("Final filtered state:", result.trace.ekf_steps[-1].filtered.state)


if __name__ == "__main__":
    main()
