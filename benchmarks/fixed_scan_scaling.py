"""Checked fixed-shape value/gradient scaling; run from the repository root.

    uv run --locked python benchmarks/fixed_scan_scaling.py

Input generation/stacking is setup work, outside every differentiated objective.
"""

import argparse
import platform
from time import perf_counter

import kalmanfilter  # Enable float64 before constructing arrays.
import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jax.experimental import checkify

from kalmanfilter.fixed_scan import FixedScanInputs, run_fixed_scan_likelihood, stack_fixed_inputs
from kalmanfilter.likelihood import run_likelihood
from kalmanfilter.optimization import CompiledObjective, run_multistart
from kalmanfilter.params import unpack_parameters
from kalmanfilter.transition import DiagonalMatrix


def make_scan_objective(initial, batched, layout):
    """Issue #10's explicit shared-parameter convention, with no time expansion.

    Estimate phi, measurement variances and theta_g; the supplied batch retains
    fixed process variance and the supplied initial distribution. This helper
    neither infers parameter ties nor changes ParameterLayout defaults.
    """
    def objective(raw):
        parameters = unpack_parameters(raw, layout)
        inputs = batched._replace(
            theta_f=jnp.broadcast_to(parameters.theta_f, batched.theta_f.shape),
            sigma_v=DiagonalMatrix(jnp.broadcast_to(
                parameters.sigma_v.diagonal, batched.sigma_v.diagonal.shape)),
            theta_g=jnp.broadcast_to(parameters.theta_g, batched.theta_g.shape),
        )
        return -run_fixed_scan_likelihood(initial, inputs).total_log_likelihood

    return objective


def measure(label, n_dates, objective, raw, repeats):
    # CompiledObjective times its first checked value_and_grad dispatch with
    # perf_counter and block_until_ready. evaluate() also synchronizes and checks
    # every result before returning host scalars/arrays; host transfer is included.
    compiled = CompiledObjective(objective, raw)
    seconds = []
    for _ in range(repeats):
        started = perf_counter()
        value, gradient = compiled.evaluate(raw)
        seconds.append(perf_counter() - started)
    print(f"{label} {n_dates} {compiled.compilation_seconds:.6f} "
          f"{np.median(seconds):.6f} {min(seconds):.6f} "
          f"{value:.12f} {np.linalg.norm(gradient):.9g}", flush=True)
    return compiled


def prefix_inputs(batched, n_dates):
    return FixedScanInputs(batched.step, *jax.tree.map(lambda leaf: leaf[:n_dates], batched[1:]))


def report_trace_parity(problem, batched):
    """Report actual maxima on the benchmark model, independently of CI fixtures."""
    maxima = dict.fromkeys(("predicted_state", "predicted_covariance", "filtered_state",
                          "filtered_covariance", "innovation", "innovation_covariance",
                          "per_step_LL", "total_LL"), 0.0)
    checked = jax.jit(checkify.checkify(
        lambda initial, batch: run_fixed_scan_likelihood(initial, batch, return_trace=True)))
    for t in (1, 5, 24, 100):
        reference = run_likelihood(problem.dataset.initial_filter, problem.dataset.inputs[:t], return_trace=True)
        error, actual = checked(problem.dataset.initial_filter, prefix_inputs(batched, t))
        error.throw()
        steps = reference.trace.ekf_steps
        pairs = (
            (actual.trace.predicted_states, jnp.stack([s.prediction.predicted_state for s in steps])),
            (actual.trace.predicted_covariances, jnp.stack([s.prediction.predicted_covariance for s in steps])),
            (actual.trace.filtered_states, jnp.stack([s.filtered.state for s in steps])),
            (actual.trace.filtered_covariances, jnp.stack([s.filtered.covariance for s in steps])),
            (actual.trace.innovations, jnp.stack([s.update.innovation for s in steps])),
            (actual.trace.innovation_covariances, jnp.stack([s.update.innovation_covariance for s in steps])),
            (actual.per_step_contributions, reference.per_step_contributions),
            (actual.total_log_likelihood, reference.total_log_likelihood),
        )
        for name, (a, b) in zip(maxima, pairs, strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-13)
            maxima[name] = max(maxima[name], float(jnp.max(jnp.abs(a - b))))
    print("Trace parity T=1,5,24,100; rtol=2e-12, atol=2e-13; maximum absolute differences:", flush=True)
    for name, maximum in maxima.items():
        print(f"  {name}: {maximum:.12g}", flush=True)


def report_estimation_parity(problem, python, scan):
    value_error = gradient_error = 0.0
    for raw in (problem.true_raw,) + problem.starts:
        old_value, old_gradient = python.evaluate(raw)
        new_value, new_gradient = scan.evaluate(raw)
        np.testing.assert_allclose(new_value, old_value, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(new_gradient, old_gradient, rtol=2e-11, atol=2e-12)
        value_error = max(value_error, abs(new_value - old_value))
        gradient_error = max(gradient_error, float(np.max(np.abs(new_gradient - old_gradient))))
    print(f"Raw parity T=12, four vectors: max_NLL_abs={value_error:.12g}; "
          f"max_gradient_abs={gradient_error:.12g}; gradient rtol=2e-11, atol=2e-12", flush=True)
    old = run_multistart(python, problem.starts[:2], method="BFGS")
    new = run_multistart(scan, problem.starts[:2], method="BFGS")
    print("BFGS parity T=12, maxiter=200, gtol=1e-6; NLL atol=2e-10; parameter rtol=2e-7, atol=2e-9:", flush=True)
    for i, (a, b) in enumerate(zip(old.runs, new.runs, strict=True)):
        assert a.success, a.message
        assert b.success, b.message
        np.testing.assert_allclose(b.final_objective, a.final_objective, rtol=0, atol=2e-10)
        old_p = unpack_parameters(a.final_raw, problem.layout)
        new_p = unpack_parameters(b.final_raw, problem.layout)
        differences = []
        for x, y in zip(jax.tree.leaves(old_p), jax.tree.leaves(new_p), strict=True):
            np.testing.assert_allclose(y, x, rtol=2e-7, atol=2e-9)
            if x.size:
                differences.append(float(jnp.max(jnp.abs(x - y))))
        print(f"  start={i} python_NLL={a.final_objective:.12f} scan_NLL={b.final_objective:.12f} "
              f"NLL_abs={abs(a.final_objective-b.final_objective):.12g} "
              f"parameter_max_abs={max(differences):.12g} "
              f"python_grad2={a.gradient_norm:.9g} scan_grad2={b.gradient_norm:.9g}", flush=True)
        print(f"    scan fitted phi={np.asarray(new_p.theta_f)}, variances={np.asarray(new_p.sigma_v.diagonal)}, "
              f"theta_g={np.asarray(new_p.theta_g)}", flush=True)


def main():
    from baseline_optimization import make_problem

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(f"Platform: {platform.platform()}; Python {platform.python_version()}; "
          f"JAX {jax.__version__}; NumPy {np.__version__}; SciPy {scipy.__version__}", flush=True)
    print(f"Device: {jax.devices()[0]}; float64={jax.config.x64_enabled}; seed={args.seed}", flush=True)
    print("Model: one PCA state, two nonlinear OIS quotes; issue #10 convention.", flush=True)
    print("Fixed initial mean=[0.35], covariance=[[0.0025]], process variance=[0.0025].", flush=True)
    print("Evaluate generating raw: phi=0.9, observation variances=[0.0004,0.0009], theta_g=0.8.", flush=True)
    print("Generating 5000 genuine synthetic dates once; all shorter runs use prefixes.", flush=True)
    problem = make_problem(5000, args.seed)
    batched = stack_fixed_inputs(problem.dataset.inputs)
    print("Setup complete; generation/stacking excluded from timings. Plain scan; no remat.", flush=True)
    report_trace_parity(problem, batched)
    print(f"First call includes tracing, compilation and synchronized evaluation; warmed median/min of {args.repeats} calls.", flush=True)
    print("path T first_checked_JIT_seconds warm_median_seconds warm_min_seconds NLL gradient_norm2", flush=True)
    # The sole freshly compiled Python reference uses 12 dates. Never compile
    # the old raw objective at long T, even though make_problem exposes one.
    small = make_problem(12, args.seed)
    python = measure("python", 12, small.objective, small.true_raw, args.repeats)
    for n_dates in (12, 24, 100, 500, 1000, 5000):
        prefix = prefix_inputs(batched, n_dates)
        objective = make_scan_objective(problem.dataset.initial_filter, prefix, problem.layout)
        compiled = measure("scan", n_dates, objective, problem.true_raw, args.repeats)
        if n_dates == 12:
            report_estimation_parity(small, python, compiled)
    print("T=5000 checked value-and-gradient succeeded. No checkpoint/remat was needed.", flush=True)


if __name__ == "__main__":
    main()
