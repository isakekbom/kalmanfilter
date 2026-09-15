"""Deterministic optimizer/conditioning study; no market or identification claims.

Run: uv run --locked python benchmarks/curvature_optimization.py
JSON records on stdout retain failures, native solver messages and full precision.
Use --case to reproduce a subset; default all runs the documented protocol.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import NamedTuple

import kalmanfilter
import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np
import scipy

from kalmanfilter.fixed_scan import FixedScanInputs, stack_fixed_inputs, run_fixed_scan_likelihood
from kalmanfilter.ois import OISInstrument
from kalmanfilter.optimization import CompiledObjective, run_optimization
from kalmanfilter.params import ModelParameters, ParameterLayout, pack_parameters, unpack_parameters
from kalmanfilter.synthetic import SyntheticStepInputs, generate_synthetic_dataset
from kalmanfilter.transition import DiagonalMatrix, StateCoordinates, StructuralStep, selection_map


METHODS = ("BFGS", "L-BFGS-B", "GD", "Newton-CG", "trust-krylov")
H_STEPS = (1e-3, 1e-4, 1e-5, 1e-6)


class Problem(NamedTuple):
    name: str
    layout: ParameterLayout
    true_parameters: ModelParameters
    true_raw: jax.Array
    initial: object
    batch: FixedScanInputs
    objective: object
    starts: tuple[jax.Array, ...]


def emit(kind, **values):
    def convert(value):
        if isinstance(value, (jax.Array, np.ndarray, np.generic)):
            return np.asarray(value).tolist()
        raise TypeError(type(value).__name__)
    print(kind + " " + json.dumps(values, default=convert, allow_nan=False, sort_keys=True), flush=True)


def make_objective(initial, batch, layout):
    """Explicit benchmark-only time sharing: phi, measurement variance, loading.

    Fixed process covariance and initial distribution anchor the latent scale.
    All raw blocks use existing transforms; no global extraction rule is added.
    """
    def objective(raw):
        p = unpack_parameters(raw, layout)
        numerical = batch._replace(
            theta_f=jnp.broadcast_to(p.theta_f, batch.theta_f.shape),
            sigma_v=DiagonalMatrix(jnp.broadcast_to(p.sigma_v.diagonal, batch.sigma_v.diagonal.shape)),
            theta_g=jnp.broadcast_to(p.theta_g, batch.theta_g.shape),
        )
        return -run_fixed_scan_likelihood(initial, numerical).total_log_likelihood
    return objective


def reference_case(reference, batch, n_dates):
    prefix = FixedScanInputs(batch.step, *jax.tree.map(lambda a: a[:n_dates], batch[1:]))
    initial, layout = reference.dataset.initial_filter, reference.layout
    # T=24 retains all three original starts; long cases use the first two,
    # except T=5000 where every method receives just the identical first start.
    starts = reference.starts if n_dates == 24 else reference.starts[:1 if n_dates == 5000 else 2]
    return Problem(f"reference_T{n_dates}", layout, reference.true_parameters, reference.true_raw,
                   initial, prefix, make_objective(initial, prefix, layout), starts)


def make_larger_problem(n_states, n_dates=100, seed=202625):
    """Coupled nonlinear quotes, p=4*n_states, n_z=2*n_states; fixed Q and x0 law.

    Two quotes per state load on that factor and its cyclic neighbor with
    different explicit weights. Each loading column shares one theta_g across
    quotes/dates. These are synthetic inputs, not financial/model conventions.
    """
    if n_states < 2 or n_dates < 1:
        raise ValueError("larger family requires at least two states and one date")
    names = tuple(f"p{i}" for i in range(n_states))
    coordinates = StateCoordinates(names, (), ())
    quotes = tuple(f"q{i}{kind}" for i in range(n_states) for kind in ("a", "b"))
    identity = selection_map(names, names, names)
    step = StructuralStep(coordinates, coordinates, identity, identity, identity,
                          selection_map(quotes, (), (None,) * len(quotes)),
                          selection_map(quotes, quotes, quotes))
    instruments = []
    for i in range(n_states):
        for kind in range(2):
            weights = jnp.eye(n_states)[i] + (0.15 if kind == 0 else 0.3) * jnp.eye(n_states)[(i + 1) % n_states]
            loadings = jnp.stack((jnp.zeros((n_states, n_states)), -jnp.diag(weights)))
            if kind == 1:
                loadings = loadings.at[0].set(0.2 * jnp.diag(weights)).at[1].set(-1.5 * jnp.diag(weights))
            instruments.append(OISInstrument(jnp.array([1.0 if kind == 0 else 0.5]),
                                             loadings, jnp.empty((2, 0))))
    layout = ParameterLayout(n_f=n_states, n_w=0, n_v=2 * n_states, n_x0=0, n_g=n_states,
                             theta_f_transform="unit_interval")
    truth = ModelParameters(jnp.linspace(0.8, 0.94, n_states), DiagonalMatrix(jnp.empty(0)),
                           DiagonalMatrix(jnp.linspace(0.0004, 0.001, 2 * n_states)),
                           jnp.empty(0), jnp.empty((0, 0)), jnp.linspace(0.65, 0.9, n_states))
    raw = pack_parameters(truth, layout)
    date = SyntheticStepInputs(step, truth.theta_f, DiagonalMatrix(jnp.linspace(0.0015, 0.003, n_states)),
                               truth.theta_g, truth.sigma_v, tuple(instruments))
    data = generate_synthetic_dataset(jax.random.key(seed), coordinates,
        jnp.linspace(0.2, 0.35, n_states), jnp.eye(n_states) * 0.0025, (date,) * n_dates)
    batch = stack_fixed_inputs(data.inputs)
    first = jnp.concatenate((jnp.full(n_states, 0.2), jnp.linspace(-0.2, 0.2, 2 * n_states), jnp.full(n_states, 0.05)))
    second = jnp.concatenate((jnp.full(n_states, -0.3), jnp.linspace(0.3, -0.3, 2 * n_states), jnp.full(n_states, -0.08)))
    return Problem(f"larger_n{n_states}", layout, truth, raw, data.initial_filter, batch,
                   make_objective(data.initial_filter, batch, layout), (raw + first, raw + second))


def dense_hessian_function(objective):
    """Diagnostic only; the production solver interface never calls this."""
    def dense(point):
        hessian = jax.hessian(objective)(point)
        checkify.check(jnp.all(jnp.isfinite(hessian)), "diagnostic Hessian must be finite")
        return hessian
    checked = jax.jit(checkify.checkify(dense))
    def evaluate(point):
        error, hessian = checked(point)
        jax.block_until_ready((error, hessian))
        error.throw()
        return np.asarray(hessian)
    return evaluate


def error_metrics(actual, reference):
    delta = np.abs(np.asarray(actual) - np.asarray(reference))
    denominator = np.maximum(np.abs(actual), np.abs(reference))
    relative = np.divide(delta, denominator, out=np.zeros_like(delta), where=denominator != 0)
    return float(np.max(delta)), float(np.max(relative))


def validate_hvp(name, objective, compiled, points):
    """Dense-AD and directional-FD references, before second-order optimization."""
    dense = dense_hessian_function(objective)
    p = len(points[0])
    directions = np.stack((np.ones(p), np.arange(1, p + 1) * (-1.0) ** np.arange(p),
                           np.random.default_rng(25).normal(size=p)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    maxima = {"dense": [0.0, 0.0], **{str(h): [0.0, 0.0] for h in H_STEPS}}
    for point in points:
        hessian = dense(point)
        for direction in directions:
            product = compiled.hessian_vector_product(point, direction)
            reference = hessian @ direction
            np.testing.assert_allclose(product, reference, rtol=2e-11, atol=2e-11)
            maxima["dense"] = np.maximum(maxima["dense"], error_metrics(product, reference)).tolist()
            for h in H_STEPS:
                numerical = (compiled.evaluate(point + h * direction)[1]
                             - compiled.evaluate(point - h * direction)[1]) / (2 * h)
                maxima[str(h)] = np.maximum(maxima[str(h)], error_metrics(numerical, product)).tolist()
                if h == 1e-5:
                    np.testing.assert_allclose(numerical, product, rtol=2e-7, atol=2e-8)
    emit("HVP_VALIDATION", problem=name, points=len(points), directions=len(directions),
         errors_abs_rel=maxima, accepted_h=1e-5, fd_rtol=2e-7, fd_atol=2e-8,
         dense_rtol=2e-11, dense_atol=2e-11)
    return dense


def hessian_diagnostics(hessian):
    """Classify local curvature with tau=1e-8*max(1,max(abs(eigenvalues)))."""
    hessian = np.asarray(hessian, dtype=np.float64)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or hessian.size == 0:
        raise ValueError("diagnostic Hessian must be a nonempty square matrix")
    if not np.isfinite(hessian).all():
        raise ValueError("diagnostic Hessian must be finite")
    # Symmetrization is solely for diagnostic eigendecomposition; report the
    # original error first. This never changes the objective or any covariance.
    symmetry = float(np.max(np.abs(hessian - hessian.T)))
    eigenvalues, vectors = np.linalg.eigh(0.5 * hessian + 0.5 * hessian.T)
    threshold = 1e-8 * max(1.0, float(np.max(np.abs(eigenvalues))))
    positive = int(np.sum(eigenvalues > threshold))
    negative = int(np.sum(eigenvalues < -threshold))
    near_zero = len(eigenvalues) - positive - negative
    material = np.abs(eigenvalues[np.abs(eigenvalues) > threshold])
    return dict(symmetry_error=symmetry, eigenvalues=eigenvalues,
                lambda_min=eigenvalues[0], lambda_max=eigenvalues[-1], threshold=threshold,
                positive=positive, negative=negative, near_zero=near_zero,
                spd_condition_number=float(eigenvalues[-1] / eigenvalues[0]) if positive == len(eigenvalues) else None,
                material_absolute_spectral_ratio=float(material.max() / material.min()) if material.size else None,
                diagonal=np.diag(hessian), minimum_eigenvector=vectors[:, 0], maximum_eigenvector=vectors[:, -1])


def report_hessian(problem, dense, label, point):
    diagnostics = hessian_diagnostics(dense(point))
    blocks = {name: [part.start, part.stop] for name, part in problem.layout.slices.items() if part.stop > part.start}
    masses = {name: float(np.sum(diagnostics["minimum_eigenvector"][start:stop] ** 2))
              for name, (start, stop) in blocks.items()}
    emit("HESSIAN", problem=problem.name, point=label, raw=point, blocks=blocks,
         minimum_eigenvector_block_mass=masses, **diagnostics)


def time_calls(call, repeats):
    times = []
    for _ in range(repeats):
        started = perf_counter()
        call()  # CompiledObjective synchronizes, checks and copies before return.
        times.append(perf_counter() - started)
    return dict(median_seconds=float(np.median(times)), minimum_seconds=min(times), repeats=repeats)


def parameters_record(parameters):
    return dict(theta_f=parameters.theta_f, sigma_w=parameters.sigma_w.diagonal,
                sigma_v=parameters.sigma_v.diagonal, a_x=parameters.a_x,
                sigma_0=parameters.sigma_0, theta_g=parameters.theta_g)


def regression_anchor(runs):
    """Compare against saved #10 results, including their printed precision."""
    lines = (Path(__file__).parent / "results/baseline_optimization.txt").read_text(encoding="utf-8").splitlines()
    expected = {}
    for i, line in enumerate(lines):
        fields = line.split()
        if len(fields) == 11 and fields[0] in METHODS[:3] and fields[1].isdigit():
            values = lines[i + 2]
            # Parameter values are printed to ~8 digits in the historical file.
            import re
            numbers = [float(x) for x in re.findall(r"\[([^\]]+)\]", values)[0].split()]
            groups = re.findall(r"\[([^\]]+)\]", values)
            parameters = np.array(numbers + [float(x) for x in groups[1].split()] + [float(groups[2])])
            expected[fields[0], int(fields[1])] = (float(fields[3]), parameters)
    objective_error = parameter_error = 0.0
    for (method, index), run in runs.items():
        if method not in METHODS[:3]:
            continue
        nll, parameters = expected[method, index]
        fitted = run["fitted"]
        actual = np.concatenate((fitted["theta_f"], fitted["sigma_v"], fitted["theta_g"]))
        np.testing.assert_allclose(run["final_objective"], nll, rtol=0, atol=2e-8)
        np.testing.assert_allclose(actual, parameters, rtol=2e-6, atol=2e-7)
        objective_error = max(objective_error, abs(run["final_objective"] - nll))
        parameter_error = max(parameter_error, float(np.max(np.abs(actual - parameters))))
    assert len([key for key in runs if key[0] in METHODS[:3]]) == 9
    emit("ISSUE10_REGRESSION", runs=9, max_NLL_abs=objective_error, max_parameter_abs=parameter_error,
         NLL_atol=2e-8, parameter_rtol=2e-6, parameter_atol=2e-7,
         source="historical rounded #10 stdout; same problem/starts, current scan execution")


def benchmark(problem, repeats):
    p, t = problem.layout.n_parameters, problem.batch.n_steps
    emit("PROBLEM", name=problem.name, T=t, p=p, n_x=len(problem.initial.state),
         n_z=problem.batch.observations.shape[1], starts=problem.starts, generating_raw=problem.true_raw,
         generating_parameters=parameters_record(problem.true_parameters),
         fixed_initial_mean=problem.initial.state, fixed_initial_covariance=problem.initial.covariance,
         fixed_process_variance=problem.batch.sigma_w.diagonal[0],
         raw_order="theta_f; sigma_v; theta_g", theta_f_transform=problem.layout.theta_f_transform)
    compiled = CompiledObjective(problem.objective, problem.true_raw)
    direction = jnp.ones(p) / jnp.sqrt(float(p))
    compiled.hessian_vector_product(problem.true_raw, direction)
    emit("TIMING", problem=problem.name, first_value_gradient_seconds=compiled.compilation_seconds,
         first_hvp_seconds=compiled.hvp_compilation_seconds,
         warmed_value_gradient=time_calls(lambda: compiled.evaluate(problem.true_raw), repeats),
         warmed_hvp=time_calls(lambda: compiled.hessian_vector_product(problem.true_raw, direction), repeats))
    # Each family is validated before its curvature optimizations. For long p=4
    # cases, use the true point; reference and larger families also use starts.
    points = (problem.true_raw, *problem.starts) if t <= 100 else (problem.true_raw,)
    dense = validate_hvp(problem.name, problem.objective, compiled, points)
    for label, point in (("generating", problem.true_raw), *[(f"start_{i}", x) for i, x in enumerate(problem.starts)]):
        report_hessian(problem, dense, label, point)
    # Keep #10's exact fixed rate at T=24. For other T use its average-per-date
    # scale, declared for GD only; every method still evaluates the full NLL.
    rate = 1e-4 * 24 / t
    emit("PROTOCOL", problem=problem.name, methods=METHODS, max_iterations=200,
         gradient_tolerance=1e-6, function_tolerance=1e-12, step_tolerance=1e-8,
         GD_learning_rate=rate, starts=len(problem.starts), retries=0)
    runs = {}
    for method in METHODS:
        for index, start in enumerate(problem.starts):
            evaluations, hvps = compiled.total_evaluations, compiled.total_hvp_evaluations
            begun = perf_counter()
            try:
                result = run_optimization(compiled, start, method=method, learning_rate=rate)
            except (checkify.JaxRuntimeError, ValueError, FloatingPointError) as error:
                emit("RUN_FAILURE", problem=problem.name, method=method, start=index,
                     exception_type=type(error).__name__, message=str(error), initial_raw=start,
                     actual_value_gradient_calls=compiled.total_evaluations - evaluations,
                     actual_hvp_calls=compiled.total_hvp_evaluations - hvps,
                     warmed_seconds=perf_counter() - begun)
                continue  # Independent subsequent runs; never retry or repair this one.
            record = asdict(result)
            record.pop("history")
            record["NLL_reduction"] = result.initial_objective - result.final_objective
            record["fitted"] = parameters_record(unpack_parameters(result.final_raw, problem.layout))
            emit("RUN", problem=problem.name, start=index, **record)
            runs[method, index] = record
    bfgs = [r for (method, _), r in runs.items() if method == "BFGS" and r["success"]]
    if bfgs:
        best = min(bfgs, key=lambda r: r["final_objective"])
        report_hessian(problem, dense, "converged_BFGS", best["final_raw"])
    else:
        emit("DIAGNOSTIC_UNAVAILABLE", problem=problem.name, point="converged_BFGS",
             reason="No BFGS run reported success; no converged point is fabricated.")
    if problem.name == "reference_T24":
        regression_anchor(runs)
    emit("CASE_COMPLETE", problem=problem.name, completed_runs=len(runs), expected_runs=len(METHODS) * len(problem.starts),
         total_value_gradient_dispatches=compiled.total_evaluations,
         total_hvp_dispatches=compiled.total_hvp_evaluations)


def main():
    from baseline_optimization import make_problem

    parser = argparse.ArgumentParser(description=__doc__)
    choices = ("all", "reference", "long100", "long1000", "long5000", "larger3", "larger6")
    parser.add_argument("--case", choices=choices, default="all")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    emit("ENV", platform=platform.platform(), python=platform.python_version(), jax=jax.__version__,
         numpy=np.__version__, scipy=scipy.__version__, device=str(jax.devices()[0]), float64=jax.config.x64_enabled,
         reference_seed=20261010, larger_seed=202625, case=args.case,
         timing="perf_counter; synchronized checked calls; setup and dense diagnostics excluded from optimizer timings")
    a = jnp.array([[4.0, 0.6, -0.2], [0.6, 2.0, 0.3], [-0.2, 0.3, 1.2]])
    b = jnp.array([1.0, -2.0, 0.5])
    quadratic = lambda x: 0.5 * x @ a @ x + b @ x + 2.0
    points = (jnp.array([0.2, -0.4, 0.7]), jnp.array([-0.8, 1.1, -0.5]))
    validate_hvp("analytical_SPD_quadratic", quadratic, CompiledObjective(quadratic, points[0]), points)
    lengths = {"reference": 24, "long100": 100, "long1000": 1000, "long5000": 5000}
    selected = tuple(lengths) if args.case == "all" else ((args.case,) if args.case in lengths else ())
    if selected:
        emit("SETUP", message="Generate genuine dates once; shorter reference cases use prefixes.")
        reference = make_problem(max(lengths[key] for key in selected))
        batch = stack_fixed_inputs(reference.dataset.inputs)
        for key in selected:
            benchmark(reference_case(reference, batch, lengths[key]), args.repeats)
    for size in (3, 6):
        if args.case in ("all", f"larger{size}"):
            benchmark(make_larger_problem(size), args.repeats)
    emit("COMPLETE", case=args.case, remat=False, noisy_optimization=False)


if __name__ == "__main__":
    main()
