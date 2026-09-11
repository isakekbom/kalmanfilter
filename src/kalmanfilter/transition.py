"""Named structural maps for PDF (12), (13), (20)-(21), and (23)-(30).

No lifecycle, initialization, or filtering algorithm is implemented here.
See docs/transition.md for the coordinate contract and checked JIT protocol.
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.typing import ArrayLike

from .model import ModelDimensions


def _identities(ids: tuple[str, ...], name: str) -> None:
    if not isinstance(ids, tuple) or any(not isinstance(i, str) or not i for i in ids):
        raise TypeError(f"{name} must be a tuple of nonempty string identities")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} must contain unique identities")


@dataclass(frozen=True, slots=True)
class StateCoordinates:
    """Explicit identities in PDF block order; within-block order is supplied."""

    pca: tuple[str, ...]
    steps: tuple[str, ...]
    unsystematic: tuple[str, ...]

    def __post_init__(self):
        for name in ("pca", "steps", "unsystematic"):
            _identities(getattr(self, name), name)
        _identities(self.all, "state coordinates")

    @property
    def systematic(self) -> tuple[str, ...]:
        return self.pca + self.steps

    @property
    def all(self) -> tuple[str, ...]:
        return self.systematic + self.unsystematic

    def dimensions(self, n_z_t: int) -> ModelDimensions:
        return ModelDimensions(
            n_p_t=len(self.pca), n_c_t=len(self.steps),
            n_u_t=len(self.unsystematic), n_z_t=n_z_t,
        )


@dataclass(frozen=True, slots=True)
class StateChange:
    """Identity bookkeeping only: no transition coefficients or initial values."""

    previous: StateCoordinates
    current: StateCoordinates

    def __post_init__(self):
        if not isinstance(self.previous, StateCoordinates) or not isinstance(
            self.current, StateCoordinates
        ):
            raise TypeError("previous and current must be StateCoordinates")
        previous_blocks = {
            identity: block
            for block in ("pca", "steps", "unsystematic")
            for identity in getattr(self.previous, block)
        }
        for block in ("pca", "steps", "unsystematic"):
            for identity in getattr(self.current, block):
                if identity in previous_blocks and previous_blocks[identity] != block:
                    raise ValueError(f"state identity {identity!r} changed factor block")

    @property
    def surviving(self) -> tuple[tuple[str, int, int], ...]:
        """(identity, previous index, current index), ordered by current state."""
        previous = {identity: i for i, identity in enumerate(self.previous.all)}
        return tuple(
            (identity, previous[identity], i)
            for i, identity in enumerate(self.current.all) if identity in previous
        )

    @property
    def introduced(self) -> tuple[str, ...]:
        previous = set(self.previous.all)
        return tuple(i for i in self.current.all if i not in previous)

    @property
    def removed(self) -> tuple[str, ...]:
        current = set(self.current.all)
        return tuple(i for i in self.previous.all if i not in current)


@partial(jax.tree_util.register_dataclass, data_fields=["weights"],
         meta_fields=["rows", "columns", "sources"])
@dataclass(frozen=True, slots=True)
class CoordinateMap:
    """At most one source per row, with optional numerical weights.

    sources names columns, in row order. An explicit None source means a zero
    coefficient row. weights=None denotes unit selection, not a fitted default.
    Names are static JAX metadata; supplied JAX weight arrays are dynamic leaves.
    """

    rows: tuple[str, ...]
    columns: tuple[str, ...]
    sources: tuple[str | None, ...]
    weights: Array | None

    def __post_init__(self):
        _identities(self.rows, "map rows")
        _identities(self.columns, "map columns")
        if not isinstance(self.sources, tuple) or len(self.sources) != len(self.rows):
            raise ValueError("sources must be a tuple with one entry per map row")
        columns = set(self.columns)
        for source in self.sources:
            if source is not None and (not isinstance(source, str) or source not in columns):
                raise ValueError(f"unknown source identity {source!r}")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.columns)


@partial(jax.tree_util.register_dataclass, data_fields=["values"],
         meta_fields=["rows", "columns"])
@dataclass(frozen=True, slots=True)
class DenseMap:
    """Caller-supplied general linear map; axes retain their named ordering."""

    rows: tuple[str, ...]
    columns: tuple[str, ...]
    values: Array

    def __post_init__(self):
        _identities(self.rows, "map rows")
        _identities(self.columns, "map columns")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.columns)


LinearMap = CoordinateMap | DenseMap


def selection_map(rows, columns, sources) -> CoordinateMap:
    """Construct a unit selector from explicit names, never positional matching."""
    return CoordinateMap(rows, columns, sources, weights=None)


class DiagonalMatrix(NamedTuple):
    """Compact diagonal entries (variances for covariance inputs), shape (n,n)."""

    diagonal: Array

    @property
    def shape(self) -> tuple[int, int]:
        return (self.diagonal.shape[0],) * 2


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["transition_post_map", "transition_pre_map", "process_noise_map",
                 "observation_selector", "observation_noise_map"],
    meta_fields=["previous", "current"],
)
@dataclass(frozen=True, slots=True)
class StructuralStep:
    """One step's A, B, D, I^z, G with exact coordinate-axis contracts.

    Parameter order is A.columns == B.rows. Base noise order is D.columns and
    G.columns. These are local supplied spaces, not a generated master state.
    """

    previous: StateCoordinates
    current: StateCoordinates
    transition_post_map: LinearMap
    transition_pre_map: LinearMap
    process_noise_map: LinearMap
    observation_selector: CoordinateMap
    observation_noise_map: LinearMap

    def __post_init__(self):
        StateChange(self.previous, self.current)
        for mapping in (self.transition_post_map, self.transition_pre_map,
                        self.process_noise_map, self.observation_selector,
                        self.observation_noise_map):
            if not isinstance(mapping, (CoordinateMap, DenseMap)):
                raise TypeError("structural maps must be CoordinateMap or DenseMap")
        selector = self.observation_selector
        if not isinstance(selector, CoordinateMap) or selector.weights is not None:
            raise TypeError("observation_selector must be an unweighted CoordinateMap")
        contracts = (
            (self.transition_post_map.rows, self.current.all, "A rows / current state"),
            (self.transition_pre_map.columns, self.previous.all, "B columns / previous state"),
            (self.transition_post_map.columns, self.transition_pre_map.rows, "A columns / B rows"),
            (self.process_noise_map.rows, self.current.all, "D rows / current state"),
            (selector.columns, self.current.unsystematic, "I^z columns / unsystematic state"),
            (self.observation_noise_map.rows, selector.rows, "G rows / active observations"),
        )
        for actual, expected, name in contracts:
            if actual != expected:
                raise ValueError(f"coordinate order mismatch: {name}")

    @property
    def active_observations(self) -> tuple[str, ...]:
        return self.observation_selector.rows

    @property
    def dimensions(self) -> ModelDimensions:
        return self.current.dimensions(len(self.active_observations))

    @property
    def state_change(self) -> StateChange:
        return StateChange(self.previous, self.current)


def _real64(value: ArrayLike, name: str, ranks: tuple[int, ...]) -> Array:
    array = jnp.asarray(value)
    if array.ndim not in ranks:
        raise ValueError(f"{name} must have rank in {ranks}; got {array.shape}")
    if not (jnp.issubdtype(array.dtype, jnp.floating)
            or jnp.issubdtype(array.dtype, jnp.integer)):
        raise TypeError(f"{name} must contain real numbers")
    array = array.astype(jnp.float64)
    return _finite(array, name)


def _finite(array: Array, name: str) -> Array:
    checkify.check(jnp.all(jnp.isfinite(array)), f"{name} must be finite")
    return array


def _map_data(mapping: LinearMap) -> Array:
    if isinstance(mapping, CoordinateMap):
        data = (jnp.ones(len(mapping.rows), dtype=jnp.float64) if mapping.weights is None
                else _real64(mapping.weights, "map weights", (1,)))
        expected = (len(mapping.rows),)
    elif isinstance(mapping, DenseMap):
        data = _real64(mapping.values, "map values", (2,))
        expected = mapping.shape
    else:
        raise TypeError("mapping must be CoordinateMap or DenseMap")
    if data.shape != expected:
        raise ValueError(f"map data shape must be {expected}; got {data.shape}")
    return data


def validate_step(step: StructuralStep) -> StructuralStep:
    """Check every map's numerical shape/type/finiteness at the input boundary.

    Identity contracts are checked when constructing the immutable metadata.
    Numeric checks are deferred to this function (and the individual kernels)
    so JAX can reconstruct pytrees containing tracers during transformations.
    """
    if not isinstance(step, StructuralStep):
        raise TypeError("step must be StructuralStep")
    for mapping in (step.transition_post_map, step.transition_pre_map,
                    step.process_noise_map, step.observation_selector,
                    step.observation_noise_map):
        _map_data(mapping)
    return step


def _indices(mapping: CoordinateMap) -> tuple[Array, Array]:
    """The single translation from named sources to valid gather/scatter indices."""
    columns = {identity: i for i, identity in enumerate(mapping.columns)}
    rows = tuple(i for i, source in enumerate(mapping.sources) if source is not None)
    sources = tuple(columns[mapping.sources[i]] for i in rows)
    return jnp.array(rows, dtype=jnp.int64), jnp.array(sources, dtype=jnp.int64)


def apply_map(mapping: LinearMap, operand: ArrayLike) -> Array:
    """Apply a named map to a vector or matrix of columns, without expanding selectors."""
    data = _map_data(mapping)
    operand = _real64(operand, "operand", (1, 2))
    if operand.shape[0] != mapping.shape[1]:
        raise ValueError("operand leading dimension must equal map column count")
    if isinstance(mapping, DenseMap):
        result = data @ operand
    else:
        rows, sources = _indices(mapping)
        scale = data[rows] if operand.ndim == 1 else data[rows, None]
        result = jnp.zeros((mapping.shape[0],) + operand.shape[1:], dtype=jnp.float64)
        result = result.at[rows].set(scale * operand[sources])
    return _finite(result, "map result")


def materialize_map(mapping: LinearMap) -> Array:
    """Explicit opt-in to a dense matrix; never used to expand selector inputs."""
    data = _map_data(mapping)
    if isinstance(mapping, DenseMap):
        return data
    rows, sources = _indices(mapping)
    return jnp.zeros(mapping.shape, dtype=jnp.float64).at[rows, sources].set(data[rows])


def split_state(coordinates: StateCoordinates, state: ArrayLike) -> tuple[Array, Array]:
    """Centralized PDF block split, returning systematic and unsystematic arrays."""
    state = _real64(state, "state", (1,))
    if state.shape != (len(coordinates.all),):
        raise ValueError("state shape must match its supplied coordinates")
    boundary = len(coordinates.systematic)
    return state[:boundary], state[boundary:]


def select_observations(step: StructuralStep, available_ids, values: ArrayLike) -> Array:
    """Select supplied active IDs in step order; zero is a valid observed value."""
    selector = selection_map(step.active_observations, available_ids, step.active_observations)
    values = _real64(values, "observation values", (1,))
    return apply_map(selector, values)


def transition_matrix(theta_f: ArrayLike, step: StructuralStep) -> LinearMap:
    """F = A diag(theta_f) B, PDF (12)/(30), as a named compact or dense map.

    theta_f follows A.columns/B.rows. No diagonal parameter matrix is allocated.
    Coordinate-map composition retains one source and weight per output row.
    """
    a, b = step.transition_post_map, step.transition_pre_map
    av, bv = _map_data(a), _map_data(b)
    theta_f = _real64(theta_f, "theta_f", (1,))
    if theta_f.shape != (a.shape[1],):
        raise ValueError("theta_f shape must match A columns / B rows")
    if isinstance(a, CoordinateMap) and isinstance(b, CoordinateMap):
        row_by_id = {identity: i for i, identity in enumerate(b.rows)}
        sources = tuple(None if s is None else b.sources[row_by_id[s]] for s in a.sources)
        rows, inner = _indices(a)
        weights = jnp.zeros(a.shape[0], dtype=jnp.float64)
        weights = weights.at[rows].set(av[rows] * theta_f[inner] * bv[inner])
        return CoordinateMap(a.rows, b.columns, sources, _finite(weights, "F weights"))
    if isinstance(b, DenseMap):
        values = apply_map(a, theta_f[:, None] * bv)
    else:
        # A is dense, B selects/repeats columns. Accumulate directly into F;
        # repeated sources must add, not overwrite, their parameter contributions.
        rows, columns = _indices(b)
        values = jnp.zeros((a.shape[0], b.shape[1]), dtype=jnp.float64)
        values = values.at[:, columns].add(av[:, rows] * (theta_f[rows] * bv[rows]))
    return DenseMap(a.rows, b.columns, _finite(values, "F values"))


def covariance_dense(covariance: DiagonalMatrix | Array) -> Array:
    """Explicit dense output when a consumer requires one; no stabilization."""
    if isinstance(covariance, DiagonalMatrix):
        return jnp.diag(_real64(covariance.diagonal, "covariance diagonal", (1,)))
    return _real64(covariance, "covariance", (2,))


def mapped_covariance(
    mapping: LinearMap, base_covariance: DiagonalMatrix | Array
) -> DiagonalMatrix | Array:
    """M Sigma M.T, PDF (20)/(21), with Sigma in mapping.columns order.

    Diagonal base entries are checked nonnegative. Dense base covariances must
    be supplied symmetric PSD: symmetry is checked exactly; PSD is a caller
    precondition (no factorization or eigenvalue tolerance is selected here).
    """
    data = _map_data(mapping)
    size = mapping.shape[1]
    if isinstance(base_covariance, DiagonalMatrix):
        variances = _real64(base_covariance.diagonal, "base variances", (1,))
        if variances.shape != (size,):
            raise ValueError("base variances shape must match map columns")
        checkify.check(jnp.all(variances >= 0), "base variances must be nonnegative")
        if isinstance(mapping, CoordinateMap):
            rows, sources = _indices(mapping)
            diagonal = jnp.zeros(mapping.shape[0], dtype=jnp.float64)
            diagonal = diagonal.at[rows].set(data[rows] ** 2 * variances[sources])
            named_sources = tuple(s for s in mapping.sources if s is not None)
            if len(set(named_sources)) == len(named_sources):
                return DiagonalMatrix(_finite(diagonal, "mapped variances"))
            # Repeated sources induce correlations even with diagonal base noise.
            shared = sources[:, None] == sources[None, :]
            block = data[rows, None] * data[None, rows] * variances[sources, None] * shared
            result = jnp.zeros((mapping.shape[0],) * 2, dtype=jnp.float64)
            result = result.at[rows[:, None], rows[None, :]].set(block)
        else:
            result = (data * variances[None, :]) @ data.T
    else:
        covariance = _real64(base_covariance, "base covariance", (2,))
        if covariance.shape != (size, size):
            raise ValueError("base covariance shape must match map columns")
        checkify.check(jnp.all(covariance == covariance.T), "base covariance must be symmetric")
        result = apply_map(mapping, apply_map(mapping, covariance).T).T
    return _finite(result, "mapped covariance")


def process_covariance(step: StructuralStep, sigma_w: DiagonalMatrix | Array):
    """Q_t = D_t Sigma_w D_t.T; output axes are step.current.all."""
    return mapped_covariance(step.process_noise_map, sigma_w)


def observation_covariance(step: StructuralStep, sigma_v: DiagonalMatrix | Array):
    """R_t = G_t Sigma_v G_t.T; output axes are step.active_observations."""
    return mapped_covariance(step.observation_noise_map, sigma_v)
