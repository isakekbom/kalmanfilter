"""Host orchestration for checked float64 JAX objectives (issues #10 and #25).

Compile once, then reuse across explicit starts and methods. NumPy conversions
are confined to this host/SciPy boundary; the mathematical objective stays in
JAX. Checked failures propagate as exceptions and never become finite penalties.
See docs/baseline_optimization.md for timing and evaluation-count conventions.
Exact HVPs and curvature methods are documented in docs/curvature_optimization.md.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import checkify
from jax.typing import ArrayLike
from scipy.optimize import minimize

Method = Literal["BFGS", "L-BFGS-B", "GD", "Newton-CG", "trust-krylov"]
_CURVATURE_METHODS = ("Newton-CG", "trust-krylov")


@dataclass(frozen=True, slots=True)
class IterationRecord:
    iteration: int
    raw: Array
    objective: float
    gradient_norm: float
    elapsed_seconds: float
    step_norm: float


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    method: Method
    initial_raw: Array
    final_raw: Array
    initial_objective: float
    final_objective: float
    final_gradient: Array
    gradient_norm: float
    history: tuple[IterationRecord, ...]
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    solver_function_evaluations: int | None
    solver_gradient_evaluations: int | None
    success: bool
    status: int
    message: str
    optimization_seconds: float
    compilation_seconds: float
    hvp_evaluations: int = 0
    solver_hessian_evaluations: int | None = None
    hvp_compilation_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class MultistartResult:
    """All runs, with one shared first-call timing (also referenced by each run)."""

    runs: tuple[OptimizationResult, ...]
    compilation_seconds: float
    hvp_compilation_seconds: float = 0.0

    @property
    def best_index(self) -> int:
        """Lowest final objective, regardless of solver success; inspect status."""
        return min(range(len(self.runs)), key=lambda i: self.runs[i].final_objective)

    @property
    def best(self) -> OptimizationResult:
        return self.runs[self.best_index]


def _raw_vector(value: ArrayLike, shape: tuple[int, ...] | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("raw optimizer vector must be a nonempty rank-one array")
    if not (np.issubdtype(raw.dtype, np.floating) or np.issubdtype(raw.dtype, np.integer)):
        raise TypeError("raw optimizer vector must contain real numbers")
    raw = raw.astype(np.float64, copy=True)
    if shape is not None and raw.shape != shape:
        raise ValueError("raw shape must match the compiled objective")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw optimizer vector must be finite")
    return raw


class CompiledObjective:
    """One checked JIT value/gradient, warmed once for a fixed float64 shape.

    Close over the ParameterLayout and problem builder in the supplied scalar
    callable. Construction times and counts one synchronized first evaluation,
    including tracing/compilation and dispatch; it is not pure compiler time.
    evaluate() counts actual subsequent calls, including failures. This mutable
    host handle is intended for serial reuse; result records are immutable.
    HVP compilation is lazy and separately timed/counted. First-order methods
    do not trace or execute the second derivative.
    """

    def __init__(self, objective: Callable[[Array], ArrayLike], example_raw: ArrayLike):
        raw = _raw_vector(example_raw)
        self.shape = raw.shape

        def scalar(point):
            value = jnp.asarray(objective(point))
            if value.shape != ():
                raise ValueError("optimizer objective must return a scalar ()")
            if value.dtype != jnp.float64:
                raise TypeError("optimizer objective must return float64")
            checkify.check(jnp.isfinite(value), "optimizer objective must be finite")
            return value

        differentiated = jax.value_and_grad(scalar)

        def value_and_gradient(point):
            value, gradient = differentiated(point)
            checkify.check(jnp.all(jnp.isfinite(gradient)), "optimizer gradient must be finite")
            return value, gradient

        def checked_gradient(point):
            return value_and_gradient(point)[1]

        def hessian_product(point, vector):
            # Forward-over-reverse AD; no dense Hessian or finite differences.
            _, product = jax.jvp(checked_gradient, (point,), (vector,))
            checkify.check(jnp.all(jnp.isfinite(product)), "optimizer HVP must be finite")
            return product

        self._hvp_function = hessian_product
        self._compiled_hvp = None
        self.total_hvp_evaluations = 0
        self.hvp_compilation_seconds = 0.0
        self._compiled = jax.jit(checkify.checkify(value_and_gradient))
        self.total_evaluations = 0
        started = perf_counter()
        self.evaluate(raw)
        self.compilation_seconds = perf_counter() - started

    def evaluate(self, raw_vector: ArrayLike) -> tuple[float, np.ndarray]:
        """Synchronized combined evaluation; host conversion follows error.throw()."""
        raw = _raw_vector(raw_vector, self.shape)
        point = jnp.asarray(raw, dtype=jnp.float64)
        self.total_evaluations += 1
        error, (value, gradient) = self._compiled(point)
        jax.block_until_ready((error, value, gradient))
        error.throw()
        return float(value), np.asarray(gradient, dtype=np.float64).copy()

    def hessian_vector_product(self, raw_vector: ArrayLike, vector: ArrayLike) -> np.ndarray:
        """Evaluate exact H(raw) @ vector, checking value, gradient and product.

        The finite real direction must have the compiled raw shape; it is not
        normalized or clipped. Count every dispatch (also numerical failures),
        separately from evaluate(), even though AD computes a primal gradient.
        Shape/type failures before dispatch do not increment either counter.
        The first synchronized call includes compilation and records its cost.
        """
        raw = _raw_vector(raw_vector, self.shape)
        direction = _raw_vector(vector, self.shape)
        point, tangent = jnp.asarray(raw), jnp.asarray(direction)
        first_call = self._compiled_hvp is None
        if first_call:
            started = perf_counter()
            self._compiled_hvp = jax.jit(checkify.checkify(self._hvp_function))
        self.total_hvp_evaluations += 1
        error, product = self._compiled_hvp(point, tangent)
        jax.block_until_ready((error, product))
        try:
            error.throw()
            return np.asarray(product, dtype=np.float64).copy()
        finally:
            if first_call:
                self.hvp_compilation_seconds = perf_counter() - started


def _positive_option(value: float, name: str) -> float:
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def run_optimization(
    objective: CompiledObjective,
    initial_raw: ArrayLike,
    *,
    method: Method = "BFGS",
    max_iterations: int = 200,
    gradient_tolerance: float = 1e-6,
    function_tolerance: float = 1e-12,
    learning_rate: float = 0.01,
    step_tolerance: float = 1e-8,
) -> OptimizationResult:
    """Optimize unconstrained raw coordinates with supplied JAX gradients.

    BFGS stops on the gradient 2-norm; unbounded L-BFGS-B uses its infinity-norm
    criterion and relative objective tolerance. Reported norms are always 2-norms.
    GD uses a fixed learning rate and 2-norm stopping only. max_iterations=0
    returns the evaluated initial point. Failures raise without repair/retry.
    Newton-CG uses step_tolerance as native xtol, not a gradient criterion;
    trust-krylov uses gtol on the 2-norm. Native status/messages are preserved.

    A run-local last-point cache serves solver requests, callbacks, and final
    reporting. Cache misses invoke and count the combined compiled function.
    First-call time belongs to objective.compilation_seconds and is excluded
    from every run's optimization_seconds. Each run includes its initial call.
    Curvature methods warm a not-yet-compiled HVP before the run timer/counters,
    using the initial point and an all-ones direction. Its first-call cost is
    separately referenced by hvp_compilation_seconds; warmup is a global HVP
    dispatch, not a run evaluation. No HVP is needed when max_iterations=0.
    """
    if not isinstance(objective, CompiledObjective):
        raise TypeError("objective must be a CompiledObjective")
    if method not in ("BFGS", "L-BFGS-B", "GD", *_CURVATURE_METHODS):
        raise ValueError("method must be BFGS, L-BFGS-B, GD, Newton-CG, or trust-krylov")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    gtol = _positive_option(gradient_tolerance, "gradient_tolerance")
    ftol = _positive_option(function_tolerance, "function_tolerance")
    rate = _positive_option(learning_rate, "learning_rate")
    xtol = _positive_option(step_tolerance, "step_tolerance")
    initial = _raw_vector(initial_raw, objective.shape)
    uses_hvp = method in _CURVATURE_METHODS and max_iterations > 0
    if uses_hvp and objective._compiled_hvp is None:
        objective.hessian_vector_product(initial, np.ones_like(initial))
    evaluations_before = objective.total_evaluations
    hvps_before = objective.total_hvp_evaluations
    cached_raw, cached_value, cached_gradient = None, None, None
    history = []
    started = perf_counter()

    def evaluate(point):
        nonlocal cached_raw, cached_value, cached_gradient
        point = _raw_vector(point, objective.shape)
        if cached_raw is None or not np.array_equal(point, cached_raw):
            cached_value, cached_gradient = objective.evaluate(point)
            cached_raw = point.copy()
        # Give SciPy its own writable gradient; it cannot change cached values.
        return cached_value, cached_gradient.copy()

    def record(point):
        value, gradient = evaluate(point)
        step_norm = 0.0 if not history else float(np.linalg.norm(point - np.asarray(history[-1].raw)))
        history.append(IterationRecord(
            len(history), jnp.asarray(point, dtype=jnp.float64), value,
            float(np.linalg.norm(gradient)), perf_counter() - started, step_norm,
        ))

    record(initial)
    solver_nfev = solver_njev = None
    solver_nhev = None
    final = initial.copy()
    if method == "GD" or max_iterations == 0:
        success = history[-1].gradient_norm <= gtol
        if method == "GD":
            for _ in range(max_iterations):
                if success:
                    break
                _, gradient = evaluate(final)
                final = final - rate * gradient
                record(final)
                success = history[-1].gradient_norm <= gtol
        iterations = len(history) - 1
        status = 0 if success else 1
        message = "Gradient tolerance reached." if success else "Maximum iterations reached."
    else:
        options = {"maxiter": max_iterations, "gtol": gtol}
        if method == "BFGS":
            options.update(norm=2, xrtol=0, c1=1e-4, c2=0.9)
        elif method == "L-BFGS-B":
            options.update(ftol=ftol, maxcor=10, maxls=40, maxfun=20000)
        elif method == "Newton-CG":
            options = {"maxiter": max_iterations, "xtol": xtol, "c1": 1e-4, "c2": 0.9}
        else:
            options.update(initial_trust_radius=1.0, max_trust_radius=1000.0,
                           eta=0.15, inexact=True)
        curvature = {"hessp": objective.hessian_vector_product} if uses_hvp else {}
        solved = minimize(evaluate, initial, method=method, jac=True, bounds=None,
                          callback=record, options=options, **curvature)
        final = solved.x
        iterations, success, status = int(solved.nit), bool(solved.success), int(solved.status)
        message = str(solved.message)
        solver_nfev, solver_njev = int(solved.nfev), int(solved.njev)
        if uses_hvp and getattr(solved, "nhev", None) is not None:
            solver_nhev = int(solved.nhev)

    value, gradient = evaluate(final)
    elapsed = perf_counter() - started  # Every compiled call above synchronized.
    evaluations = objective.total_evaluations - evaluations_before
    return OptimizationResult(
        method, jnp.asarray(initial), jnp.asarray(final), history[0].objective, value,
        jnp.asarray(gradient), float(np.linalg.norm(gradient)), tuple(history), iterations,
        evaluations, evaluations, solver_nfev, solver_njev, success, status, message,
        elapsed, objective.compilation_seconds,
        objective.total_hvp_evaluations - hvps_before, solver_nhev,
        objective.hvp_compilation_seconds if uses_hvp else 0.0,
    )


def run_multistart(
    objective: CompiledObjective, starts: Sequence[ArrayLike], **options,
) -> MultistartResult:
    """Retain all explicit starts' runs, sharing compilation; fail fast on errors.

    Solver non-success statuses are retained normally. A checked numerical
    failure propagates as an exception; it is not ranked as a finite objective.
    Options are forwarded unchanged to run_optimization.
    """
    starts = tuple(starts)
    if not starts:
        raise ValueError("multistart requires at least one starting vector")
    runs = tuple(run_optimization(objective, start, **options) for start in starts)
    return MultistartResult(runs, objective.compilation_seconds,
                           max(run.hvp_compilation_seconds for run in runs))
