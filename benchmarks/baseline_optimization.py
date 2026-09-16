"""Controlled fixed-dimension baseline; no real-data calibration or global claim.

Run from the repository root:
    uv run --locked python benchmarks/baseline_optimization.py
The default arguments reproduce the #10 reference (preset ``reference``).
Configuration flags are documented in docs/benchmark_config.md.
"""

import argparse
from dataclasses import replace
import platform

import kalmanfilter  # Enable float64 before creating JAX arrays.
import jax
import jax.numpy as jnp
import numpy as np
import scipy

from benchmark_config import (
    PRESETS, BenchmarkProblem, add_config_arguments, build_problem, config_from_arguments,
    problem_record, reject_unsupported, serialize,
)
from kalmanfilter.optimization import CompiledObjective, run_multistart
from kalmanfilter.params import unpack_parameters


METHODS = ("BFGS", "L-BFGS-B", "GD")


def make_problem(n_dates=24, seed=20261010) -> BenchmarkProblem:
    """The #10 reference family: phi, two measurement variances and theta_g are free.

    Fixing the process variance anchors the latent/loading scale. This is the
    shared reference builder with only its historical arguments exposed; use
    BenchmarkConfig directly for other sizes, noise levels or starts.
    """
    if isinstance(n_dates, bool) or not isinstance(n_dates, int) or n_dates < 1:
        raise ValueError("n_dates must be a positive integer")
    return build_problem(replace(PRESETS["reference"], n_dates=n_dates, seed=seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    args = parser.parse_args()
    config = config_from_arguments(args, PRESETS["reference"]).defaulted(methods=METHODS)
    reject_unsupported(config, "baseline_optimization", "repeats")
    print(f"Platform: {platform.platform()}; Python {platform.python_version()}; "
          f"JAX {jax.__version__}; NumPy {np.__version__}; SciPy {scipy.__version__}", flush=True)
    print("CONFIG", serialize(config.resolve().to_dict()), flush=True)
    problem = build_problem(config)
    config = problem.config
    print("PROBLEM", serialize(problem_record(problem)), flush=True)
    print(f"Seed={config.seed}; dates={config.n_dates}; n_x={config.n_x}; n_z={config.n_z}; p={config.p}; "
          "raw order=(theta_f, observation variances, theta_g)")
    print("Generating raw:", np.asarray(problem.true_raw))
    initial = problem.dataset.initial_filter
    print(f"Fixed initial mean={np.asarray(initial.state)}, covariance={np.asarray(initial.covariance).tolist()}, "
          f"process variance={np.asarray(problem.fixed_process_variance)}")
    truth = problem.true_parameters
    print(f"Generating estimated parameters: phi={np.asarray(truth.theta_f)}, "
          f"observation variances={np.asarray(truth.sigma_v.diagonal)}, theta_g={np.asarray(truth.theta_g)}")
    for index, start in enumerate(problem.starts):
        print(f"Start {index}:", np.asarray(start))
    print("Compiling one checked value-and-gradient function...", flush=True)
    compiled = CompiledObjective(problem.objective, problem.starts[0])
    true_value, _ = compiled.evaluate(problem.true_raw)
    print(f"Shared first-call/JIT seconds: {compiled.compilation_seconds:.6f}; generating-truth NLL: {true_value:.12f}", flush=True)
    print("All methods: maxiter=200, gtol=1e-6; L-BFGS-B ftol=1e-12; GD fixed learning_rate=1e-4")
    print("method start initial_NLL final_NLL final_grad2 iterations actual_f/g solver_f/g steady_seconds success status", flush=True)
    for method in config.methods:
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
            print("  absolute errors: phi=", np.asarray(jnp.abs(parameters.theta_f - truth.theta_f)),
                  "observation variances=", np.asarray(jnp.abs(parameters.sigma_v.diagonal - truth.sigma_v.diagonal)),
                  "theta_g=", np.asarray(jnp.abs(parameters.theta_g - truth.theta_g)))
            print("  relative observation-variance errors:", np.asarray(
                jnp.abs(parameters.sigma_v.diagonal / truth.sigma_v.diagonal - 1)))
            print(f"  initial grad2={result.history[0].gradient_norm:.9g}; "
                  f"NLL reduction={result.initial_objective-result.final_objective:.9f}; "
                  f"raw distance to generating vector={float(jnp.linalg.norm(result.final_raw-problem.true_raw)):.9g}", flush=True)
        print(f"{method} best start: {group.best_index}", flush=True)
    print("Total actual compiled evaluations (includes one warmup and one truth diagnostic):", compiled.total_evaluations)


if __name__ == "__main__":
    main()
