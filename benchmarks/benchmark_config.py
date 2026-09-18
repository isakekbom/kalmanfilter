"""Shared, serializable configuration for the synthetic calibration benchmarks.

Two documented synthetic families are configurable here: ``reference`` (the
#10 anchor: one PCA state, two quotes, p=4) and ``larger`` (the #25 family:
n_x>=2 coupled states, two quotes per state, p=4*n_x). Everything in this
module is a benchmark convention, not a model semantic: process variance and
the initial law are fixed, phi/sigma_v/theta_g are shared across dates and
persistence uses the optional unit-interval transform. Production EKF, OIS,
transition and parameter code is used unchanged; no parameter tying, market
convention or new quote topology is introduced.

Dimensions: T = n_dates, n_x = n_states, n_z = quotes per date and
p = n_x + n_z + n_x free raw parameters (theta_f; sigma_v; theta_g).
Noise fields hold variances; ``*_variance_scale`` fields multiply variances,
so standard deviations scale by the square root. Validation is plain Python
and finishes before any data generation or JAX compilation.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
import json
from numbers import Real
from pathlib import Path
from typing import NamedTuple

import kalmanfilter  # Enable float64 before creating JAX arrays.
import jax
import jax.numpy as jnp
import numpy as np

from kalmanfilter.fixed_scan import FixedScanInputs, run_fixed_scan_likelihood
from kalmanfilter.gradient_validation import raw_negative_log_likelihood
from kalmanfilter.ois import OISInstrument
from kalmanfilter.params import ModelParameters, ParameterLayout, pack_parameters, unpack_parameters
from kalmanfilter.synthetic import SyntheticDataset, SyntheticStepInputs, generate_synthetic_dataset
from kalmanfilter.transition import DiagonalMatrix, StateCoordinates, StructuralStep, selection_map


FAMILIES = ("reference", "larger")
METHODS = ("BFGS", "L-BFGS-B", "GD", "Newton-CG", "trust-krylov")
QUOTES_PER_STATE = 2
INITIAL_VARIANCE = 0.0025  # Both families fix Sigma_0 = INITIAL_VARIANCE * scale * I.
PROTOCOL_FIELDS = ("repeats", "methods")
_VECTOR_FIELDS = ("persistence", "loading", "process_variances", "observation_variances", "initial_mean")


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _scale(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive multiplier")
    return value


def _vector(value, name, size, *, positive=False, unit_interval=False):
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be a sequence of real numbers")
    items = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in items):
        raise TypeError(f"{name} must contain real numbers")
    vector = tuple(float(item) for item in items)
    if len(vector) != size:
        raise ValueError(f"{name} must have {size} entries; got {len(vector)}")
    if not all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    if positive and any(item <= 0 for item in vector):
        raise ValueError(f"{name} entries are variances and must be positive")
    if unit_interval and any(not 0 < item < 1 for item in vector):
        raise ValueError(f"{name} uses the unit_interval transform and must lie strictly inside (0, 1)")
    return vector


def _linspace(start, stop, count):
    # jnp.linspace reproduces the historical builders bitwise; np.linspace does not always.
    return tuple(np.asarray(jnp.linspace(start, stop, count)).tolist())


def _family_defaults(family, n_states):
    """Generating values and raw start offsets of each documented family."""
    if family == "reference":
        return dict(persistence=(0.9,), loading=(0.8,), process_variances=(0.0025,),
                    observation_variances=(0.0004, 0.0009), initial_mean=(0.35,),
                    start_offsets=((0.25, 0.2, -0.2, 0.1), (-0.8, 0.8, -0.6, -0.2), (1.0, -1.0, 1.0, 0.3)))
    n = n_states
    return dict(persistence=_linspace(0.8, 0.94, n), loading=_linspace(0.65, 0.9, n),
                process_variances=_linspace(0.0015, 0.003, n),
                observation_variances=_linspace(0.0004, 0.001, 2 * n), initial_mean=_linspace(0.2, 0.35, n),
                start_offsets=((0.2,) * n + _linspace(-0.2, 0.2, 2 * n) + (0.05,) * n,
                               (-0.3,) * n + _linspace(0.3, -0.3, 2 * n) + (-0.08,) * n))


@dataclass(frozen=True, kw_only=True)
class BenchmarkConfig:
    """Immutable, validated description of one synthetic benchmark problem.

    ``None`` in a generating field selects the family default; ``resolve()``
    makes every such value explicit. ``repeats`` and ``methods`` are protocol
    fields whose defaults belong to the individual benchmark script; use
    ``defaulted()`` to fill them. Sequences are normalized to tuples of floats.
    """

    family: str = "reference"
    n_dates: int = 24
    n_states: int = 1
    quotes_per_state: int = QUOTES_PER_STATE
    seed: int = 20261010
    persistence: tuple[float, ...] | None = None
    loading: tuple[float, ...] | None = None
    process_variances: tuple[float, ...] | None = None
    observation_variances: tuple[float, ...] | None = None
    process_variance_scale: float = 1.0
    observation_variance_scale: float = 1.0
    initial_mean: tuple[float, ...] | None = None
    initial_covariance_scale: float = 1.0
    n_starts: int | None = None
    start_offsets: tuple[tuple[float, ...], ...] | None = None
    repeats: int | None = None
    methods: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        _integer(self.n_dates, "n_dates", 1)
        _integer(self.n_states, "n_states", 1)
        _integer(self.quotes_per_state, "quotes_per_state", 1)
        _integer(self.seed, "seed", 0)
        if self.family == "reference" and self.n_states != 1:
            raise ValueError("the reference family has exactly one latent state; use family='larger' for n_x >= 2")
        if self.family == "larger" and self.n_states < 2:
            raise ValueError("the larger family requires at least two latent states")
        if self.quotes_per_state != QUOTES_PER_STATE:
            raise ValueError("unsupported quote topology: both families define exactly two quotes per state")
        set_ = lambda name, value: object.__setattr__(self, name, value)
        set_("process_variance_scale", _scale(self.process_variance_scale, "process_variance_scale"))
        set_("observation_variance_scale", _scale(self.observation_variance_scale, "observation_variance_scale"))
        set_("initial_covariance_scale", _scale(self.initial_covariance_scale, "initial_covariance_scale"))
        sizes = dict(persistence=(self.n_x, dict(unit_interval=True)), loading=(self.n_x, {}),
                     process_variances=(self.n_x, dict(positive=True)),
                     observation_variances=(self.n_z, dict(positive=True)), initial_mean=(self.n_x, {}))
        for name, (size, options) in sizes.items():
            if getattr(self, name) is not None:
                set_(name, _vector(getattr(self, name), name, size, **options))
        if self.n_starts is not None:
            _integer(self.n_starts, "n_starts", 1)
        if self.start_offsets is not None:
            if isinstance(self.start_offsets, (str, bytes)) or not isinstance(self.start_offsets, Iterable):
                raise TypeError("start_offsets must be a sequence of raw offset vectors")
            offsets = tuple(_vector(row, "start offset", self.p) for row in self.start_offsets)
            if not offsets:
                raise ValueError("start_offsets must contain at least one start")
            if self.n_starts is not None and self.n_starts != len(offsets):
                raise ValueError("n_starts must equal the number of explicit start_offsets")
            set_("start_offsets", offsets)
        elif self.n_starts is not None and self.n_starts > self.n_preset_starts:
            raise ValueError(f"the {self.family} family defines {self.n_preset_starts} preset starts; "
                             "supply start_offsets for more")
        if self.repeats is not None:
            _integer(self.repeats, "repeats", 1)
        if self.methods is not None:
            if isinstance(self.methods, (str, bytes)) or not isinstance(self.methods, Iterable):
                raise TypeError("methods must be a sequence of solver names")
            methods = tuple(self.methods)
            if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
                raise ValueError(f"methods must be a nonempty subset of {METHODS} without repeats")
            set_("methods", methods)

    @property
    def n_x(self) -> int:
        return self.n_states

    @property
    def n_z(self) -> int:
        return self.n_states * self.quotes_per_state

    @property
    def p(self) -> int:
        """Free raw parameters: n_x persistence, n_z observation variances, n_x loadings."""
        return self.layout.n_parameters

    @property
    def layout(self) -> ParameterLayout:
        return ParameterLayout(n_f=self.n_x, n_w=0, n_v=self.n_z, n_x0=0, n_g=self.n_x,
                               theta_f_transform="unit_interval")

    @property
    def n_preset_starts(self) -> int:
        return len(_family_defaults(self.family, self.n_states)["start_offsets"])

    def resolve(self) -> "BenchmarkConfig":
        """Make family defaults explicit; protocol fields are left untouched."""
        defaults = _family_defaults(self.family, self.n_states)
        values = {name: defaults[name] for name in _VECTOR_FIELDS if getattr(self, name) is None}
        if self.start_offsets is None:
            count = self.n_preset_starts if self.n_starts is None else self.n_starts
            values["start_offsets"] = defaults["start_offsets"][:count]
            values["n_starts"] = count
        elif self.n_starts is None:
            values["n_starts"] = len(self.start_offsets)
        return replace(self, **values)

    def defaulted(self, **protocol) -> "BenchmarkConfig":
        """Fill protocol fields that are still None with a benchmark's own defaults."""
        unknown = set(protocol) - set(PROTOCOL_FIELDS)
        if unknown:
            raise ValueError(f"only protocol fields {PROTOCOL_FIELDS} take benchmark defaults; got {sorted(unknown)}")
        return replace(self, **{name: value for name, value in protocol.items() if getattr(self, name) is None})

    def generation(self, *, exclude: Iterable[str] = ()) -> dict:
        """Resolved fields that determine the generated problem (protocol fields excluded)."""
        skipped = set(PROTOCOL_FIELDS) | set(exclude)
        return {name: value for name, value in self.resolve().to_dict().items() if name not in skipped}

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, values: Mapping, base: "BenchmarkConfig | None" = None) -> "BenchmarkConfig":
        """Override ``base`` (or the defaults) with the mapping; unknown keys are rejected."""
        if not isinstance(values, Mapping):
            raise TypeError("configuration values must be a mapping")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        return replace(cls() if base is None else base, **values)

    @classmethod
    def from_json(cls, text: str, base: "BenchmarkConfig | None" = None) -> "BenchmarkConfig":
        return cls.from_dict(json.loads(text), base)


PRESETS = {
    # The documented regression anchors; resolve() makes their family defaults explicit.
    "reference": BenchmarkConfig(),
    "long100": BenchmarkConfig(n_dates=100, n_starts=2),
    "long1000": BenchmarkConfig(n_dates=1000, n_starts=2),
    "long5000": BenchmarkConfig(n_dates=5000, n_starts=1),
    "larger3": BenchmarkConfig(family="larger", n_states=3, n_dates=100, seed=202625),
    "larger6": BenchmarkConfig(family="larger", n_states=6, n_dates=100, seed=202625),
}


def preset_name(config: BenchmarkConfig) -> str | None:
    """Name of the preset generating the same problem, ignoring protocol fields."""
    generation = config.generation()
    return next((name for name, preset in PRESETS.items() if preset.generation() == generation), None)


def reject_unsupported(config: BenchmarkConfig, benchmark: str, *names: str) -> None:
    """A benchmark that cannot honour a field must refuse it rather than ignore it."""
    defaults = {item.name: item.default for item in fields(BenchmarkConfig)}
    for name in names:
        if getattr(config, name) != defaults[name]:
            raise ValueError(f"{benchmark} does not support the '{name}' configuration field")


class BenchmarkProblem(NamedTuple):
    """Generated synthetic problem with the ragged Python-driver objective."""

    layout: ParameterLayout
    true_parameters: ModelParameters
    true_raw: jax.Array
    dataset: SyntheticDataset
    objective: Callable[[jax.Array], jax.Array]
    starts: tuple[jax.Array, ...]
    fixed_process_variance: jax.Array
    config: BenchmarkConfig


def _reference_structure():
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
    return coordinates, step, instruments


def _larger_structure(n_states):
    """Two quotes per state loading on that factor and its cyclic neighbour."""
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
    return coordinates, step, tuple(instruments)


def _assemble(config, layout, truth, dataset, fixed_process_variance):
    true_raw = pack_parameters(truth, layout)

    def build(parameters):
        inputs = tuple(item._replace(theta_f=parameters.theta_f, sigma_v=parameters.sigma_v,
                                     theta_g=parameters.theta_g) for item in dataset.inputs)
        return dataset.initial_filter, inputs

    def objective(raw):
        return raw_negative_log_likelihood(raw, layout, build)

    starts = tuple(true_raw + jnp.array(offset) for offset in config.start_offsets)
    return BenchmarkProblem(layout, truth, true_raw, dataset, objective, starts, fixed_process_variance, config)


def build_problem(config: BenchmarkConfig) -> BenchmarkProblem:
    """Generate the configured dataset eagerly; nothing here is JIT-compiled.

    Estimated parameters are phi, the n_z observation variances and theta_g.
    The known process variance and initial distribution are supplied by the
    builder, exactly as in the historical benchmark builders.
    """
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig")
    config = config.resolve()
    coordinates, step, instruments = (_reference_structure() if config.family == "reference"
                                      else _larger_structure(config.n_states))
    layout = config.layout
    truth = ModelParameters(
        jnp.array(config.persistence), DiagonalMatrix(jnp.empty((0,))),
        DiagonalMatrix(jnp.array(config.observation_variances) * config.observation_variance_scale),
        jnp.empty((0,)), jnp.empty((0, 0)), jnp.array(config.loading),
    )
    fixed_process_variance = jnp.array(config.process_variances) * config.process_variance_scale
    date = SyntheticStepInputs(step, truth.theta_f, DiagonalMatrix(fixed_process_variance),
                               truth.theta_g, truth.sigma_v, instruments)
    sigma_0 = jnp.eye(config.n_x) * (INITIAL_VARIANCE * config.initial_covariance_scale)
    dataset = generate_synthetic_dataset(jax.random.key(config.seed), coordinates,
                                         jnp.array(config.initial_mean), sigma_0, (date,) * config.n_dates)
    return _assemble(config, layout, truth, dataset, fixed_process_variance)


def prefix_problem(problem: BenchmarkProblem, config: BenchmarkConfig) -> BenchmarkProblem:
    """Reuse generated dates for a shorter/other-start configuration.

    Generation splits one key per date, so a prefix is bitwise identical to a
    shorter generation with the same seed and generating values.
    """
    config = config.resolve()
    excluded = ("n_dates", "n_starts", "start_offsets")
    if config.generation(exclude=excluded) != problem.config.generation(exclude=excluded):
        raise ValueError("a prefix must share every generating field except n_dates and the starts")
    if config.n_dates > problem.config.n_dates:
        raise ValueError("a prefix cannot exceed the generated number of dates")
    shared = ("initial_filter", "true_initial_state")
    dataset = problem.dataset._replace(**{name: getattr(problem.dataset, name)[:config.n_dates]
                                          for name in SyntheticDataset._fields if name not in shared})
    return _assemble(config, problem.layout, problem.true_parameters, dataset, problem.fixed_process_variance)


def make_scan_objective(initial, batched: FixedScanInputs, layout: ParameterLayout):
    """Issue #10's explicit shared-parameter convention on stacked inputs.

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


def problem_record(problem: BenchmarkProblem) -> dict:
    """Self-describing record: dimensions, seed, noise and generating values."""
    config, truth, initial = problem.config, problem.true_parameters, problem.dataset.initial_filter
    return dict(
        config=config.to_dict(), preset=preset_name(config), T=config.n_dates, n_x=config.n_x, n_z=config.n_z,
        p=config.p, seed=config.seed, generating_raw=problem.true_raw,
        generating_parameters=dict(theta_f=truth.theta_f, sigma_v=truth.sigma_v.diagonal, theta_g=truth.theta_g),
        fixed_process_variance=problem.fixed_process_variance,
        fixed_initial_mean=initial.state, fixed_initial_covariance=initial.covariance, starts=problem.starts,
        raw_order="theta_f; sigma_v; theta_g", theta_f_transform=problem.layout.theta_f_transform,
        noise_convention="sigma_w/sigma_v entries are variances; *_variance_scale multiplies variances",
    )


def serialize(values) -> str:
    """Sorted JSON at full float precision; JAX/NumPy arrays become nested lists."""
    def convert(value):
        if isinstance(value, (jax.Array, np.ndarray, np.generic)):
            return np.asarray(value).tolist()
        raise TypeError(type(value).__name__)
    return json.dumps(values, default=convert, allow_nan=False, sort_keys=True)


_ARGUMENTS = (
    ("--family", dict(choices=FAMILIES, help="synthetic family")),
    ("--dates", dict(type=int, dest="n_dates", metavar="T", help="number of dates T")),
    ("--states", dict(type=int, dest="n_states", metavar="N_X", help="latent states n_x (reference: 1; larger: >= 2)")),
    ("--quotes-per-state", dict(type=int, metavar="N", help="quotes per state; only 2 is supported")),
    ("--seed", dict(type=int, help="generator seed")),
    ("--persistence", dict(type=float, nargs="+", metavar="PHI", help="generating theta_f (n_x values in (0,1))")),
    ("--loading", dict(type=float, nargs="+", metavar="THETA_G", help="generating theta_g (n_x values)")),
    ("--process-variances", dict(type=float, nargs="+", metavar="VAR", help="fixed Sigma_w variances (n_x values)")),
    ("--observation-variances", dict(type=float, nargs="+", metavar="VAR", help="generating Sigma_v variances (n_z values)")),
    ("--process-variance-scale", dict(type=float, metavar="SCALE", help="multiplies the process variances")),
    ("--observation-variance-scale", dict(type=float, metavar="SCALE", help="multiplies the observation variances")),
    ("--initial-mean", dict(type=float, nargs="+", metavar="MEAN", help="fixed initial mean a_x (n_x values)")),
    ("--initial-covariance-scale", dict(type=float, metavar="SCALE", help="multiplies Sigma_0 = 0.0025 I")),
    ("--starts", dict(type=int, dest="n_starts", metavar="N", help="number of preset starts to use")),
    ("--start-offset", dict(type=float, nargs="+", action="append", dest="start_offsets", metavar="RAW",
                            help="explicit raw offset from the generating vector (p values); repeatable")),
    ("--repeats", dict(type=int, help="timing repetitions")),
    ("--methods", dict(nargs="+", choices=METHODS, help="solver subset to run")),
)
CONFIG_ARGUMENTS = tuple(options.get("dest", flag[2:].replace("-", "_")) for flag, options in _ARGUMENTS)


def add_config_arguments(parser, *, preset: bool = True) -> None:
    """Every flag defaults to None so only explicit values override the base."""
    group = parser.add_argument_group("synthetic benchmark configuration",
                                      "explicit flags override --preset/--config values; see docs/benchmark_config.md")
    if preset:
        group.add_argument("--preset", choices=tuple(PRESETS), help="named reference configuration")
    group.add_argument("--config", type=Path, metavar="FILE",
                       help="JSON object of BenchmarkConfig fields, optionally with a 'preset' key")
    for flag, options in _ARGUMENTS:
        group.add_argument(flag, **options)


def argument_overrides(args) -> dict:
    """Configuration fields given explicitly on the command line."""
    return {name: getattr(args, name) for name in CONFIG_ARGUMENTS if getattr(args, name, None) is not None}


def load_config_file(path) -> dict:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("a configuration file must hold one JSON object")
    return values


def config_from_arguments(args, base: BenchmarkConfig | None = None) -> BenchmarkConfig:
    """Precedence: benchmark default < --preset < --config file < explicit flags."""
    config = BenchmarkConfig() if base is None else base
    preset = getattr(args, "preset", None)
    if preset is not None:
        config = PRESETS[preset]
    if getattr(args, "config", None) is not None:
        values = load_config_file(args.config)
        file_preset = values.pop("preset", None)
        if file_preset is not None:
            if preset is not None:
                raise ValueError("choose the preset either on the command line or in the file, not both")
            if file_preset not in PRESETS:
                raise ValueError(f"unknown preset {file_preset!r}; choose from {tuple(PRESETS)}")
            config = PRESETS[file_preset]
        config = BenchmarkConfig.from_dict(values, config)
    return replace(config, **argument_overrides(args))
