# Forward Extended Kalman Filter

This implements [issue #5](https://github.com/isakekbom/kalmanfilter/issues/5),
equations (38)–(56), pp. 4–5 of [kalmanRante.pdf](../kalmanRante.pdf), in
[`ekf.py`](../src/kalmanfilter/ekf.py). It follows
[model_spec.md §§4, 6–7](model_spec.md), the existing [OIS API](ois.md), and the
[named structural maps](transition.md). One observation linearization is made
at each predicted state. Equation (57), likelihood contributions, parameter
transforms, optimization, smoothing, and MATLAB parity are outside this issue.

## Functions and equations

| Function | Equations | Returned object / operation |
| --- | --- | --- |
| `initialize_filter(coordinates, a_x, sigma_0)` | (39)–(40) | `FilterState`: supplied mean and covariance |
| `predict(previous, step, theta_f, sigma_w)` | (41)–(43) | `Prediction`: structured F, Q, predicted mean/covariance, asymmetry |
| `build_observation_linearization(step, theta_g, predicted_state, instruments)` | (38), (44)–(45) | `ObservationLinearization`: split state, g, J, H, u, direct predicted observation |
| `measurement_update(x_pred, P_pred, observations, predicted_observation, observation_jacobian, noise_covariance)` | (46)–(56) | `MeasurementUpdate`: innovation, R, S, L, K, filtered mean/covariance, asymmetries |
| `ekf_step(previous, inputs)` | (38), (41)–(56) | `EKFStepResult`: production OIS/structural step and complete trace |
| `run_filter(initial, inputs, return_trace=False)` | Forward application of (38)–(56) | `FilterResult`: initial values, tuple of filtered states, optional tuple of traces |

`measurement_update` accepts an explicit predicted observation and Jacobian,
so a linear-Gaussian model can be tested directly without OIS instruments.
The production `ekf_step` always calls the existing OIS functions and structural
APIs. There is no alternate pricing or state-coordinate implementation.

## Inputs, ordering, and shapes

`FilterState` is a frozen JAX dataclass containing `coordinates`, `state`, and
`covariance`. Coordinates are static metadata; numerical arrays are dynamic
leaves. All other EKF containers are immutable named tuples and JAX pytrees.
Supply immutable JAX arrays inside containers. Numeric validation occurs at
function boundaries, allowing JAX to reconstruct containers with tracers.

`EKFInputs` contains one date's `step`, `theta_f`, `sigma_w`, `theta_g`, `sigma_v`,
`observations`, and `instruments`. Numerical parameter inputs follow exactly
the axes declared by the structural step:

| Input | Shape | Coordinate contract |
| --- | --- | --- |
| `a_x`, `sigma_0` | `(n_x_0,)`, `(n_x_0,n_x_0)` | Initial `StateCoordinates.all` |
| Previous filtered state, covariance | `(n_x_prev,)`, `(n_x_prev,n_x_prev)` | `step.previous.all` |
| `theta_f` | `(n_f,)` | `step.transition_post_map.columns == step.transition_pre_map.rows` |
| `sigma_w` | Dense `(n_w,n_w)` or `DiagonalMatrix` with `(n_w,)` entries | `step.process_noise_map.columns` |
| `theta_g` | `(n_g,)` | Supplied OIS parameter order |
| `sigma_v` | Dense `(n_v,n_v)` or `DiagonalMatrix` with `(n_v,)` entries | `step.observation_noise_map.columns` |
| `observations` | `(n_z_t,)` | Exactly `step.active_observations` |
| `instruments` | Ordered tuple of length `n_z_t` | Exactly `step.active_observations`; OIS loadings follow `step.current.pca` / `.steps` |

Use the issue #4 helper to select observations from an identified vector:

```python
observations = select_observations(step, available_ids, available_values)
instruments = tuple(instruments_by_id[i] for i in step.active_observations)
```

No activity is inferred from values: numeric zero is a valid measurement.
Inactive quotes have no rows. The EKF validates counts and separate PCA/step
block dimensions; unlabelled OIS arrays cannot prove semantic coordinate or
instrument identities. Their meaning and order remain the caller's contract.

## Initialization and prediction

Equations (39)–(40) are used directly:

```text
x_0 = a_x
P_0 = sigma_0
```

Real integer/floating inputs are promoted to float64, matching issues #3–#4.
Boolean and complex numerical inputs are rejected. Promotion cannot recover
precision previously lost in float32. Inputs must be finite, the state length
must match its coordinates, and the covariance must have the matching square
shape. Supplied dense covariances are checked for **exact symmetry**, consistent
with the structural layer; no tolerance is silently chosen. Positive
semidefiniteness of dense supplied covariances is a caller precondition.
Singular PSD initial covariance is allowed; there is no initial Cholesky check,
stationary initialization, estimation of Sigma_0, floor, or jitter.

For (41)–(43), the structural layer constructs
`F = transition_matrix(theta_f, step)` and
`Q = process_covariance(step, sigma_w)`. The computation is:

```text
x_pred = apply_map(F, x_prev)
FP = apply_map(F, P_prev)
P_raw = apply_map(F, FP.T).T + Q
P_pred = (P_raw + P_raw.T) / 2
```

The second application equals `F P_prev F.T` by transposition algebra. Its
operand is `(n_x_prev,n_x_t)`, so the product works when F is rectangular
`(n_x_t,n_x_prev)`. It does not assume equal state counts or position-based
continuity. The consumer never materializes F: a compact `CoordinateMap`
remains compact, while genuine mixing may produce `DenseMap` in the existing
structural layer. No dense `diag(theta_f)` is constructed.

Q has shape `(n_x_t,n_x_t)`. When represented by `DiagonalMatrix`, its entries
are added directly to the dense propagated covariance diagonal. General dense
Q also works. `Prediction.covariance_asymmetry` records P_raw's drift before
averaging; `Prediction.process_covariance` retains Q as supplied by the mapping.

## Linearization and innovation

The existing `split_state(step.current, x_pred)` provides systematic and
unsystematic blocks, of shapes `(n_s_t,)` and `(n_u_t,)`. The OIS layer evaluates
`observation_quotes(theta_g, x_s, instruments)` and
`quote_state_jacobian(theta_g, x_s, instruments)` at the predicted state.

```text
g = g_t(theta_g, x_s)          # (n_z_t,)
J = d g_t / d x_s             # (n_z_t,n_s_t), PDF (38)
H = [J, I_z]                 # (n_z_t,n_x_t), (44)
u = g - J @ x_s              # (n_z_t,), (45)
z_hat = g + apply_map(I_z, x_u)
epsilon = observations - z_hat  # (49)
```

`I_z` is materialized only for concatenation into the generally dense H.
The direct observation uses `apply_map`. Equations (46)–(48) give the equivalent
`observations - H @ x_pred - u`; cancellation of the two `J @ x_s` terms yields
(49). Tests exercise the nonzero affine offset and the equivalence explicitly.

## Innovation covariance, factorization, and update

The structural layer computes `R = observation_covariance(step, sigma_v)`,
equal to `G Sigma_v G.T`, with shape `(n_z_t,n_z_t)`. For general dense mapped R,
the OIS step measures and records its asymmetry, then averages its transpose
pair before passing R into `measurement_update`. The base covariance was
already checked for exact symmetry by issue #4. This records floating-point
drift in a derived product; it does not relax the generic update's exact
symmetry requirement for directly supplied R. Compact diagonal R stays compact.

The generic update then computes:

```text
HP = H @ P_pred               # (n_z_t,n_x_t)
S_raw = HP @ H.T + R          # (n_z_t,n_z_t), (50)
S = (S_raw + S_raw.T) / 2
L = cholesky(S)               # (n_z_t,n_z_t), lower triangular
Y = solve_triangular(L, HP)
K = solve_triangular(L.T, Y).T  # (n_x_t,n_z_t), (51)–(52)
x_filt = x_pred + K @ epsilon # (n_x_t,), (53)
P_raw = P_pred - K @ HP       # (n_x_t,n_x_t), (55)
P_filt = (P_raw + P_raw.T) / 2
```

The two triangular solves solve `S K.T = HP`. They never form an inverse or
pseudo-inverse. The **same lower factor L is retained in the trace for issue
#6**. No likelihood, log determinant, or quadratic likelihood term is computed.
Diagonal R entries are added directly to S_raw's diagonal. S itself is dense.

Equation (55) is the baseline covariance update; tests independently evaluate
(54) and (56) as references. There is no Joseph-form baseline or square-root
filter. Small asymmetry due to ordinary rounding is measured before correction.

`CovarianceAsymmetry` records two float64 scalars for a raw covariance C:

```text
max_absolute = max(abs(C - C.T))
relative = max_absolute / max(abs(C))
```

For empty or identically zero C, both diagnostics are zero. The denominator is
not floored for nonzero C. Diagnostics are retained for predicted covariance,
innovation covariance, filtered covariance, and mapped observation noise.
No hard asymmetry acceptance threshold is imposed. Tests bound these quantities
tightly for their deterministic, moderately scaled valid examples.

The implementation averages as `0.5*C + 0.5*C.T`, algebraically the requested
`0.5*(C+C.T)`, to avoid overflowing the intermediate sum of large finite values.
Symmetry does **not** establish PSD or repair an invalid covariance. No
eigenvalue clipping, jitter, variance floor, or fallback hides instability.
The subtractive PDF covariance update may lose PSD in difficult finite-precision
cases; symmetry and a successful innovation factorization do not certify that
every filtered covariance is PSD. Broader numerical policy remains Q10.

## Failure reporting and JAX

For a nonempty observed space, S must be finite and positive definite. The
kernel checks S before factorization, then explicitly checks that the Cholesky
factor is finite and that every diagonal entry is finite and strictly positive.
This handles JAX returning NaNs on factorization failure. It does not depend on
Cholesky raising a Python exception and it does not choose a near-zero cutoff.
A tiny positive definite S is used unchanged.

Static rank, shape, and coordinate errors raise `ValueError` or `TypeError`
during construction/tracing. Numerical failures use `checkify.check`;
eager calls raise `checkify.JaxRuntimeError`. For fixed shapes:

```python
compiled_step = jax.jit(checkify.checkify(ekf_step))
error, trace = compiled_step(initial, inputs)
error.throw()  # Outside JIT, before reading numerical results.
filtered = trace.filtered
```

**After any checkified error, all returned numerical values are invalid and
must be discarded.** Computation may continue after recording the error.
Direct `jax.jit` of checked kernels without first applying checkify is
unsupported. Kernels contain no NumPy conversions, callbacks, I/O, hidden
mutable state, or input mutations. Autodiff propagates through the OIS quote
Jacobian, covariance propagation, Cholesky, solves, and update on valid smooth
inputs. Tests compare step derivatives to independent finite differences;
full likelihood gradient validation belongs to issue #8.

## Missing observations and changing dimensions

When `n_z_t == 0`, a step performs prediction only:
`x_filt == x_pred` and `P_filt == P_pred`, with exact unchanged numerical arrays
at the update boundary. It returns g, u, z_hat, and epsilon as `(0,)`; J as
`(0,n_s_t)`; H as `(0,n_x_t)`; S and R as `(0,0)`; and K as `(n_x_t,0)`.
`innovation_cholesky` is **None**, and no Cholesky decomposition is attempted.
The update asymmetry is zero because no covariance subtraction occurs.

This prediction-only rule is an **explicit implementation convention** to
support issue #4's active-only representation. It is not a claim that the PDF
specifies all-missing dates. No likelihood contribution is assigned here.

`run_filter` validates the entire coordinate chain before running steps:
the initial coordinates equal the first `step.previous`, and every current
coordinate tuple equals the next previous tuple **including order**. Each output
`FilterState.coordinates` equals that date's `step.current`. Individual
`ekf_step` calls also check previous coordinate identity. Removal, introduction,
replacement, and reordering use the caller's structural maps without padding,
a fixed master state, positional inference, or automatic lifecycle rules.

Each date supplies explicit numerical inputs in its locally declared parameter
and noise spaces. To reuse the same parameter objects, the caller must maintain
their named axes and order across dates. Per-date input containers implement
neither parameter extraction nor parameter tying, and do not settle Q4.

The driver is an ordinary Python loop and stores ragged tuples. It does not use
`lax.scan`: state sizes, observation sizes, and OIS tuple/payment structures can
change. A compiled individual step is specialized to shapes and static
metadata; changes can require recompilation. One compiled function is not
promised to handle arbitrary changing structures. There is no time-zero
observation. An empty sequence returns the validated initial state as `.final`.

## Trace access and a complete example

`EKFStepResult` groups intermediate values by mathematical stage:

| Trace field | Contents |
| --- | --- |
| `structural_step`, `observations` | All coordinate identities, A/B/D/I_z/G, exact active measurements |
| `prediction` | `transition` F, `process_covariance` Q, `predicted_state`, `predicted_covariance`, `covariance_asymmetry` |
| `linearization` | `systematic_state`, `unsystematic_state`, `modeled_quotes` g, `quote_jacobian` J, `observation_jacobian` H, `linearization_offset` u, `predicted_observation` z_hat |
| `update` | `innovation`, `observation_covariance` R, `innovation_covariance` S, `innovation_cholesky` L or None, `kalman_gain` K, `filtered_state`, `filtered_covariance`, innovation/filtered covariance asymmetries |
| `observation_noise_asymmetry` | Drift measured before averaging dense mapped R |
| `.filtered` property | `FilterState` in `structural_step.current` coordinates, for the next step |

`run_filter(..., return_trace=True)` retains every full trace. By default it
retains only initial and per-date filtered states/covariances, with `.trace`
equal to None; intermediate step objects can be released between iterations.
`.final` returns the last filtered state or the initial state for an empty
sequence. These objects support future equation-level MATLAB comparison;
no MATLAB agreement is asserted yet.

This deterministic one-date example is algebraic, without financial defaults:

```python
import jax
import jax.numpy as jnp
from jax.experimental import checkify
from kalmanfilter.ekf import EKFInputs, initialize_filter, ekf_step, run_filter
from kalmanfilter.ois import OISInstrument
from kalmanfilter.transition import (
    StateCoordinates, StructuralStep, DiagonalMatrix, selection_map,
)

coordinates = StateCoordinates(("level",), (), ("deviation",))
identity = selection_map(coordinates.all, coordinates.all, coordinates.all)
step = StructuralStep(
    coordinates, coordinates, identity, identity, identity,
    selection_map(("quote",), coordinates.unsystematic, ("deviation",)),
    selection_map(("quote",), ("quote-noise",), ("quote-noise",)),
)
initial = initialize_filter(coordinates, [0.02, 0.001], [[0.01, 0], [0, 0.002]])
instrument = OISInstrument(
    jnp.array([0.5]), jnp.array([[[0.1]], [[-0.5]]]), jnp.empty((2, 0)),
)
inputs = EKFInputs(
    step=step,
    theta_f=jnp.array([0.9, 0.8]),
    sigma_w=DiagonalMatrix(jnp.array([0.001, 0.0001])),
    theta_g=jnp.array([0.7]),
    sigma_v=DiagonalMatrix(jnp.array([0.0002])),
    observations=jnp.array([0.018]),
    instruments=(instrument,),
)
result = run_filter(initial, (inputs,), return_trace=True)
assert result.final.coordinates == coordinates
assert result.trace[0].update.kalman_gain.shape == (2, 1)

error, trace = jax.jit(checkify.checkify(ekf_step))(initial, inputs)
error.throw()
assert trace.update.innovation_cholesky.shape == (1, 1)
```

## Source status and unresolved questions

| Status | Choices implemented or still open |
| --- | --- |
| PDF | Initialization, transition/covariance prediction, observation linearization and affine offset, innovation, S, gain equations, mean update, and covariance update (38)–(56). |
| Derived algebra | Structured covariance application; cancellation in (46)–(49); triangular solves equivalent to (51)–(52); covariance equivalence (54)–(56). The PDF's Cholesky identity following (57) also motivates retaining L for issue #6. |
| Implementation conventions | Immutable pytrees, exact named-axis contracts, exact symmetry checks for supplied covariances, float64 promotion, covariance asymmetry diagnostics and averaging, prediction-only empty observations, None for empty L, checked failure without repair, and a Python forward loop. These are not all prescribed by the PDF. |
| Q6, partly addressed | Active rows/order come from issue #4, and prediction-only all-missing handling is now explicit. Quote availability rules, state persistence while quotes are absent, and future all-missing likelihood behavior remain open. |
| **Q7 unresolved** | The printed recursions omit cross-noise terms. They require the relevant zero cross-covariances/independence conditions, but the PDF's marginal laws do not establish a joint noise model. No correlations or independence policy are invented here. |
| **Q9 unresolved** | Initial a_x and Sigma_0 are supplied exactly. Which parameters are fixed, estimated, tied, constrained, or stationary; transforms, burn-in, and a time-zero observation remain unspecified. |
| **Q10 unresolved beyond this baseline policy** | Clear failure, no repair, and recorded symmetrization are chosen for issue #5. Admissible parameter domains, conditioning criteria, broader covariance stabilization, and future optimization responses to invalid evaluations remain undecided. |

Q1–Q5 concerning instruments, factor construction, parameter spaces, and state
lifecycle remain as documented by the existing layers. This implementation does
not claim to resolve them through deterministic test data.

## Validation

[`tests/test_ekf.py`](../tests/test_ekf.py) covers a hand-computable scalar filter,
a two-dimensional linear update, independent NumPy OIS and filter references,
equation equivalences, declared observation subsets including zero, empty
observations without any Cholesky call, rectangular removal/introduction, a
four-date changing sequence with reordering, immutable traces, float64,
checked JIT, independent step finite differences, strict input validation,
symmetry diagnostics, singular/indefinite/nonfinite S, invalid Cholesky diagonal
reporting, tiny positive S without floors, and compact F/Q/R use. Source checks
guard against explicit inverses and callbacks in production, and NumPy imports
in mathematical model modules. NumPy is permitted at the separate
[optimizer host boundary](baseline_optimization.md) and in independent tests.

Run the complete suite:

```text
uv run --locked --extra test pytest
```

For the known Windows pytest temporary-directory permission problem:

```text
uv run --locked --extra test pytest --basetemp .pytest_tmp -p no:cacheprovider
```
