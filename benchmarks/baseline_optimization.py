"""Controlled fixed-dimension baseline; no real-data calibration or global claim.

Run from the repository root:
    uv run --locked python benchmarks/baseline_optimization.py
"""

import argparse
from collections.abc import Callable
import platform
from typing import NamedTuple

import kalmanfilter  # Enable float64 before creating JAX arrays.
import jax
import jax.numpy as jnp
import numpy as np
import scipy

from kalmanfilter.gradient_validation import raw_negative_log_likelihood
from kalmanfilter.ois import OISInstrument
from kalmanfilter.optimization import CompiledObjective, run_multistart
from kalmanfilter.params import ModelParameters, ParameterLayout, pack_parameters, unpack_parameters
from kalmanfilter.synthetic import SyntheticDataset, SyntheticStepInputs, generate_synthetic_dataset
from kalmanfilter.transition import DiagonalMatrix, StateCoordinates, StructuralStep, selection_map


class BenchmarkProblem(NamedTuple):
    layout: ParameterLayout
    true_parameters: ModelParameters
    true_raw: jax.Array
    dataset: SyntheticDataset
    objective: Callable[[jax.Array], jax.Array]
    starts: tuple[jax.Array, ...]
    fixed_process_variance: jax.Array


def make_problem(n_dates=24, seed=20261010):
    """Estimate phi, two measurement variances, and theta_g; fix x_0 law and Q.

    Fixing the process variance anchors the latent/loading scale. The empty
    n_w/n_x0 blocks describe the estimated parameter tuple only; the builder
    explicitly supplies the known process variance and initial distribution.
    This is a synthetic estimation convention, not a general masking system.
    """
    if isinstance(n_dates, bool) or not isinstance(n_dates, int) or n_dates < 1:
        raise ValueError("n_dates must be a positive integer")
    coordinates = StateCoordinates(("p",), (), ())
    step = StructuralStep(
        coordinates, coordinates,
        selection_map(("p",), ("phi",), ("phi",)),
        selection_map(("phi",), ("p",), ("p",)),
        selection_map(("p",), ("w",), ("w",)),
        selection_map(("qa", "qb"), (), (None, None)),
        selection_map(("qa", "qb"), ("va", "vb"), ("va", "vb")),
    )
    instruments = (
        OISInstrument(jnp.array([1.0]), jnp.array([[[0.0]], [[-1.0]]]), jnp.empty((2, 0))),
        OISInstrument(jnp.array([0.5]), jnp.array([[[0.2]], [[-1.5]]]), jnp.empty((2, 0))),
    )
    layout = ParameterLayout(n_f=1, n_w=0, n_v=2, n_x0=0, n_g=1,
                             theta_f_transform="unit_interval")
    truth = ModelParameters(
        jnp.array([0.9]), DiagonalMatrix(jnp.empty((0,))),
        DiagonalMatrix(jnp.array([0.0004, 0.0009])), jnp.empty((0,)), jnp.empty((0, 0)),
        jnp.array([0.8]),
    )
    true_raw = pack_parameters(truth, layout)
    fixed_process_variance = jnp.array([0.0025])
    date = SyntheticStepInputs(step, truth.theta_f, DiagonalMatrix(fixed_process_variance),
                               truth.theta_g, truth.sigma_v, instruments)
    dataset = generate_synthetic_dataset(jax.random.key(seed), coordinates,
                                         jnp.array([0.35]), jnp.array([[0.0025]]), (date,) * n_dates)

    def build_problem(parameters):
        inputs = tuple(item._replace(theta_f=parameters.theta_f, sigma_v=parameters.sigma_v,
                                     theta_g=parameters.theta_g) for item in dataset.inputs)
        return dataset.initial_filter, inputs

    def objective(raw):
        return raw_negative_log_likelihood(raw, layout, build_problem)

    offsets = jnp.array([[0.25, 0.2, -0.2, 0.1], [-0.8, 0.8, -0.6, -0.2], [1.0, -1.0, 1.0, 0.3]])
    starts = tuple(true_raw + row for row in offsets)
    return BenchmarkProblem(layout, truth, true_raw, dataset, objective, starts, fixed_process_variance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    print(f"Platform: {platform.platform()}; Python {platform.python_version()}; "
          f"JAX {jax.__version__}; NumPy {np.__version__}; SciPy {scipy.__version__}", flush=True)
    problem = make_problem(args.dates, args.seed)
    print(f"Seed={args.seed}; dates={args.dates}; raw order=(phi, variance_a, variance_b, theta_g)")
    print("Generating raw:", np.asarray(problem.true_raw))
    print("Fixed initial mean=[0.35], covariance=[[0.0025]], process variance=[0.0025]")
    print("Generating estimated parameters: phi=[0.9], observation variances=[0.0004,0.0009], theta_g=[0.8]")
    for index, start in enumerate(problem.starts):
        print(f"Start {index}:", np.asarray(start))
    print("Compiling one checked value-and-gradient function...", flush=True)
    compiled = CompiledObjective(problem.objective, problem.starts[0])
    true_value, _ = compiled.evaluate(problem.true_raw)
    print(f"Shared first-call/JIT seconds: {compiled.compilation_seconds:.6f}; generating-truth NLL: {true_value:.12f}", flush=True)
    print("All methods: maxiter=200, gtol=1e-6; L-BFGS-B ftol=1e-12; GD fixed learning_rate=1e-4")
    print("method start initial_NLL final_NLL final_grad2 iterations actual_f/g solver_f/g steady_seconds success status", flush=True)
    for method in ("BFGS", "L-BFGS-B", "GD"):
        group = run_multistart(compiled, problem.starts, method=method, max_iterations=200,
                               gradient_tolerance=1e-6, function_tolerance=1e-12, learning_rate=1e-4)
        for index, result in enumerate(group.runs):
            print(f"{method} {index} {result.initial_objective:.9f} {result.final_objective:.9f} "
                  f"{result.gradient_norm:.9g} {result.iterations} "
                  f"{result.function_evaluations}/{result.gradient_evaluations} "
                  f"{result.solver_function_evaluations}/{result.solver_gradient_evaluations} "
                  f"{result.optimization_seconds:.6f} {result.success} {result.status}", flush=True)
            print("  reason:", result.message)
            parameters = unpack_parameters(result.final_raw, problem.layout)
            print("  phi:", np.asarray(parameters.theta_f), "observation variances:", np.asarray(parameters.sigma_v.diagonal),
                  "theta_g:", np.asarray(parameters.theta_g), "process variance (fixed):", np.asarray(problem.fixed_process_variance))
            print("  absolute errors: phi=", np.asarray(jnp.abs(parameters.theta_f - problem.true_parameters.theta_f)),
                  "observation variances=", np.asarray(jnp.abs(parameters.sigma_v.diagonal - problem.true_parameters.sigma_v.diagonal)),
                  "theta_g=", np.asarray(jnp.abs(parameters.theta_g - problem.true_parameters.theta_g)))
            print("  relative observation-variance errors:", np.asarray(
                jnp.abs(parameters.sigma_v.diagonal / problem.true_parameters.sigma_v.diagonal - 1)))
            print(f"  initial grad2={result.history[0].gradient_norm:.9g}; "
                  f"NLL reduction={result.initial_objective-result.final_objective:.9f}; "
                  f"raw distance to generating vector={float(jnp.linalg.norm(result.final_raw-problem.true_raw)):.9g}", flush=True)
        print(f"{method} best start: {group.best_index}", flush=True)
    print("Total actual compiled evaluations (includes one warmup and one truth diagnostic):", compiled.total_evaluations)


if __name__ == "__main__":
    main()
