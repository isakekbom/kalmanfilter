# Structural matrices for one time step

This implements [issue #4](https://github.com/isakekbom/kalmanfilter/issues/4) in
[`transition.py`](../src/kalmanfilter/transition.py), following
[`model_spec.md`](model_spec.md) §§2, 4–5, and 6.5. The source is
[`kalmanRante.pdf`](../kalmanRante.pdf), pp. 3–4, equations (12), (13), (20)–(21),
and (23)–(30). It supplies structural maps and covariance products for later
filtering. It does not initialize states or implement an EKF recursion.

## Identities, order, and dimensions

`StateCoordinates(pca, steps, unsystematic)` contains three required tuples of
nonempty string identities. Identities are unique across the full state. The
full order is `pca + steps + unsystematic`, and the systematic order is
`pca + steps`, matching the PDF. Within-block ordering is supplied by the caller.
`dimensions(n_z_t)` produces the existing `ModelDimensions` metadata.

Names are opaque identifiers, not dates, positions, or instructions to construct
a financial factor. Reusing an identity asserts the same coordinate meaning;
the code cannot establish that semantic claim from a string. Reordering an ID
does not create a new state. Reusing an ID in a different factor block is rejected
as inconsistent metadata; a different meaning requires a different identity.
These naming rules are implementation conventions, not PDF prescriptions.

`StateChange(previous, current)` records:

- `surviving`: `(identity, previous_index, current_index)` triples, in current order;
- `introduced`: IDs present only in the current state, in current order;
- `removed`: IDs present only in the previous state, in previous order.

This comparison does **not** generate transition coefficients. A removed state
may still contribute to a current state through an explicitly supplied map.
An introduced state must have an explicit row in `A_t` and `D_t`. Its row may
depend on previous coordinates, or the caller can explicitly supply a zero row.
A zero deterministic row does not specify an initial distribution, independence,
an initial covariance, or a financial lifecycle rule.

Each `StructuralStep` contains previous/current coordinates and all five maps.
It verifies exact axis identities **and their ordering**, not just dimensions.

| Step field / object | PDF object | Shape | Row identities | Column identities |
| --- | --- | --- | --- | --- |
| `transition_post_map` | $A_t$ | `(n_x_t,n_f)` | `current.all` | Caller-supplied transition parameter axis |
| `transition_pre_map` | $B_t$ | `(n_f,n_x_prev)` | Same parameter axis as `A.columns` | `previous.all` |
| `process_noise_map` | $D_t$ | `(n_x_t,n_w)` | `current.all` | Caller-supplied base process noise axis |
| `observation_selector` | $I_t^z$ | `(n_z_t,n_u_t)` | Active observation IDs | `current.unsystematic` |
| `observation_noise_map` | $G_t$ | `(n_z_t,n_v)` | Same ordered active IDs as $I_t^z$ | Caller-supplied base observation noise axis |
| `transition_matrix(theta_f, step)` | $F_t$ | `(n_x_t,n_x_prev)` | `current.all` | `previous.all` |
| `process_covariance(step, sigma_w)` | $Q_t$ | `(n_x_t,n_x_t)` | `current.all` | `current.all` |
| `observation_covariance(step, sigma_v)` | $R_t$ | `(n_z_t,n_z_t)` | Active observation IDs | Active observation IDs |

`theta_f` has shape `(n_f,)` in `A.columns == B.rows` order. `sigma_w` and
`sigma_v` follow `D.columns` and `G.columns`, respectively. For diagonal inputs,
their stored shapes are `(n_w,)` and `(n_v,)`; dense inputs use square matrices.
These parameter/noise axes are explicit local spaces, not padded state vectors.
If one parameter vector is reused over time, its IDs and ordering must remain
consistent. The package performs no parameter extraction, sharing, resizing,
or tying on the caller's behalf. Equal lengths alone do not establish parameter
identity; unlabelled numerical vectors must follow the declared named axis.

## Representations and numerical API

All containers are immutable. Supply immutable JAX arrays for numerical fields.
Naming/type/order contracts are checked during metadata construction. Call
`validate_step(step)` to check every map's array shape, real dtype, and finite
values before use. Individual kernels also validate the maps they consume.
Numerical checks occur at these boundaries so JAX can reconstruct pytrees during
tracing and autodiff. Kernels promote real numerical inputs to float64; promotion
does not recover precision previously lost in float32. Booleans and complex
numerical arrays are rejected.

- `CoordinateMap(rows, columns, sources, weights)` stores at most one source per
  output row. Every entry of `sources` names a column or is explicitly `None`
  for a zero coefficient row. `weights` is `(len(rows),)`; weights for `None`
  rows are ignored. `weights=None` denotes unit selection. Sources may repeat:
  this explicitly connects multiple rows to the same coordinate.
- `selection_map(rows, columns, sources)` constructs an unweighted
  `CoordinateMap`. Identity, permutation, removal, and explicitly zero rows need
  only linear storage. `I_t^z` must use this unweighted representation; a `None`
  source means an explicitly absent deviation contribution to an **active**
  observation, not a missing observation or a fabricated zero measurement.
- `DenseMap(rows, columns, values)` accepts arbitrary supplied coefficients,
  with `values.shape == (len(rows),len(columns))`. It supports genuine mixing
  and rectangular maps; it does not assume $A_t$, $B_t$, or $D_t$ are identities.
- `DiagonalMatrix(diagonal)` stores just diagonal entries. Covariance entries
  are variances, not standard deviations. `.shape` reports the mathematical
  square shape, while `.diagonal` retains linear storage.

`apply_map(mapping, operand)` handles a vector `(n_columns,)` or a matrix of
columns `(n_columns,r)`. Coordinate maps use indexed gathers and row scaling.
The named-to-integer conversion lives in one helper; consumers do not reconstruct
lifecycle slices. `split_state(step.current, state)` centralizes the PDF block
split and returns `(x_s, x_u)` with shapes `(n_s_t,)` and `(n_u_t,)`.

`transition_matrix` evaluates

$$F_t(\theta^F)=A_t\operatorname{diag}(\theta^F)B_t \qquad\text{(12), (30)}.$$

It returns a **named map**, not necessarily a dense array. It never constructs
`diag(theta_f)`. Composing two coordinate maps produces another coordinate map;
weights incorporate the appropriate parameters. General dense cases scale and
multiply the supplied arrays. With dense $A_t$ and coordinate $B_t$, contributions
are accumulated into the output columns, including repeated sources, without
expanding $B_t$. `materialize_map(mapping)` is the explicit opt-in for a consumer
that needs a dense matrix. Numerical kernels never call it to expand selectors.

`mapped_covariance(mapping, base_covariance)` implements $M\Sigma M^T$.
The process and observation wrappers select $M=D_t$ and $M=G_t$:

$$Q_t=D_t\Sigma^wD_t^T,\qquad R_t=G_t\Sigma^vG_t^T \qquad\text{(21), (20)}.$$

For a `DiagonalMatrix` base, unique coordinate sources preserve a diagonal
result, including explicit zero rows. Repeated sources induce cross-covariances
and produce a dense result; they are not silently treated as independent noise.
A dense map computes `(M * variances[None,:]) @ M.T`, without expanding the base
diagonal. Full dense base covariances are also supported. The returned API is
`DiagonalMatrix | jax.Array`; `.shape` is available on both.
`covariance_dense(result)` explicitly converts a compact result when needed.

Diagonal base variances must be finite and nonnegative; zero is permitted.
Dense base inputs must be finite, square, symmetric, and positive semidefinite.
The kernel checks exact symmetry but **does not verify positive semidefiniteness
of full dense inputs**. That is a caller precondition, not a claim that symmetry
alone establishes validity. No factorization, eigenvalue tolerance, repair,
symmetrization, jitter, or positive variance floor is introduced. Finite-input
arithmetic overflow is reported. Dense mapped outputs can exhibit ordinary
floating-point asymmetry; tests compare symmetry with float64 tolerances.

## Removal and introduction example

This is supplied algebra, with no proposed calendar or initialization policy:

```python
import jax.numpy as jnp
from kalmanfilter.transition import (
    StateCoordinates, StructuralStep, DenseMap, DiagonalMatrix,
    selection_map, validate_step, transition_matrix, materialize_map,
    process_covariance, observation_covariance,
)

previous = StateCoordinates(pca=("level",), steps=("old-step",), unsystematic=("u",))
current = StateCoordinates(pca=("level",), steps=("new-step",), unsystematic=("u",))
parameter_ids = ("f-level", "f-old", "f-u")
step = StructuralStep(
    previous=previous,
    current=current,
    transition_post_map=DenseMap(
        current.all, parameter_ids,
        jnp.array([[1., 0., 0.], [0.25, 0.5, 0.], [0., 0., 1.]]),
    ),
    transition_pre_map=selection_map(parameter_ids, previous.all, previous.all),
    process_noise_map=selection_map(
        current.all, ("w-level", "w-step", "w-u"), ("w-level", "w-step", "w-u"),
    ),
    observation_selector=selection_map(("quote-u",), current.unsystematic, ("u",)),
    observation_noise_map=selection_map(("quote-u",), ("v-u",), ("v-u",)),
)
validate_step(step)
assert step.state_change.removed == ("old-step",)
assert step.state_change.introduced == ("new-step",)
theta_f = jnp.array([0.8, 0.6, 0.9], dtype=jnp.float64)
f = transition_matrix(theta_f, step)  # rows=current, columns=previous
f_dense = materialize_map(f)         # [[.8,0,0], [.2,.3,0], [0,0,.9]]
q = process_covariance(step, DiagonalMatrix(jnp.array([1., 2., 3.])))
r = observation_covariance(step, DiagonalMatrix(jnp.array([4.])))
assert f.shape == q.shape == (3, 3)
assert r.shape == (1, 1)
```

The new row's coefficients above are explicitly chosen example data. Changing
them changes the result. The package supplies no mean/covariance for a newly
introduced state and no affine intercept. Rectangular examples in the tests use
`F.shape == (3,5)` for removal and `(3,2)` for introduction.

## Active observations

The issue's active-only requirement selects an implementation convention for the
row-handling part of Q6: `step.active_observations == I.rows == G.rows`. That tuple
defines order; the package does not sort IDs or infer activity from values.
`select_observations(step, available_ids, values)` selects those IDs from an
explicitly identified finite input vector. An absent requested ID is an error.
An observed numeric zero remains a measurement. Inactive IDs contribute no row
to the selected vector, $I_t^z$, or $G_t$.

The caller must assemble the OIS input tuple in this same order, for example
`tuple(instruments_by_id[i] for i in step.active_observations)`. OIS loadings must
use `step.current.systematic` coordinates. The existing pricing interface then
returns `g.shape == (n_z_t,)` and `J.shape == (n_z_t,n_s_t)`. This layer does not
evaluate a complete state-space observation or perform a measurement update.

An empty active tuple is representable, with `I.shape == (0,n_u_t)`,
`G.shape == (0,n_v)`, and `R.shape == (0,0)`. This does not define the later
filter's all-missing-date behavior or a likelihood contribution. Empty state and
parameter spaces likewise remain algebraic shapes, with no artificial coordinates.

## JAX execution and changing shapes

Numerical functions use JAX only. Map arrays are dynamic pytree leaves; identity
tuples and `StateCoordinates` in a structural step are immutable static metadata.
This follows [JAX's dataclass registration contract](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_dataclass.html).
Use the same checked compilation protocol as the merged OIS implementation:

```python
import jax
from jax.experimental import checkify

compiled_transition = jax.jit(checkify.checkify(transition_matrix))
error, f = compiled_transition(theta_f, step)
error.throw()  # Outside JIT, before using f.
compiled_process_covariance = jax.jit(checkify.checkify(process_covariance))
error, q = compiled_process_covariance(step, DiagonalMatrix(jnp.array([1., 2., 3.])))
error.throw()
```

[Checkify](https://docs.jax.dev/en/latest/debugging/checkify_guide.html) represents
runtime validation errors as data. Direct `jax.jit` of functions containing these
checks is unsupported: apply `checkify` first, and discard numerical results if
the returned error reports failure. Static identity/shape errors raise Python
exceptions during construction/tracing. Eager numeric errors raise
`checkify.JaxRuntimeError`. There are no callbacks, host array conversions, input
mutations, or hidden numerical state. For helpers taking identity tuples directly,
close over those tuples or mark them static in the surrounding compiled function.

The PDF permits different state and observation sizes at successive times. JAX
specializes computation to shapes and static metadata: different step shapes,
coordinate IDs, or mappings may trigger recompilation. Keep a Python sequence of
individual structural steps for now. Require `steps[t].current ==
steps[t+1].previous` when connecting them, including identity order. There is no
`lax.scan`, padding, fixed master state, or promise of one compilation for every
shape. A future execution strategy must preserve the mathematical coordinate
contracts before addressing performance.

## Source statements, conventions, and open questions

| Status | Behavior |
| --- | --- |
| PDF / dimensional algebra | State blocks `(p,c,u)`; rectangular $F=A\operatorname{diag}(\theta^F)B$; $I^z$ maps deviations to observations; $D\Sigma^wD^T$ and $G\Sigma^vG^T$ map noise covariances. |
| PDF specialization, opt-in | (23)–(29) give identity-like $A,B,D,I^z$, $G=I^z$, and diagonal base noises. Callers may explicitly supply these compatible maps. None is imposed universally. |
| Implementation convention | Unique named axes with exact order checking; stable IDs cannot switch factor blocks; compact coordinate/diagonal storage; explicit `None` means a zero coefficient row. |
| User-directed implementation convention | One row per supplied active observation; no fake zero measurements or inactive rows. This clarifies row handling without claiming a complete answer to Q6. |
| Q4 remains open | Global parameter dimensions, coefficient sharing/extraction, and noise parameter identities across dimension changes. The package checks supplied local spaces but constructs none automatically. |
| Q5 remains open | Event timing, removal/introduction rules, new-state means/covariances, and elapsed-time coefficient interpretation. Tests use declared algebra, not lifecycle defaults. |
| Q6 remains partly open | Which quotes are active, their deviation/noise associations, state persistence when quotes are absent, and all-missing filter behavior. These are supplied metadata or later decisions. |
| Q7, Q9–Q10 remain open | Joint noise assumptions, estimation constraints, initial conditions, innovation definiteness, and later numerical policy. No filtering, optimization, or covariance stabilization is added. |

Q1–Q3 about financial instruments, pricing conventions, and factor construction
remain as documented in the model specification and the OIS implementation.

## Acceptance review and validation

`tests/test_transition.py` covers constant diagonal transitions, removal,
introduction, simultaneous replacement, reordered parameters, general mixing,
repeated selector sources, missing quotes, covariance references, float64,
checked JIT, and derivatives of structural operations. Independent small NumPy
matrix products are test oracles; production code contains no NumPy conversions.
A traced compact case verifies that no `(n,n)` intermediate is allocated for
coordinate transitions and independent diagonal noise.

The four-step structural sequence has `(n_x_prev,n_x_t,n_z_t)` equal to
`(3,2,1)`, `(2,4,2)`, `(4,3,0)`, `(3,3,1)`. Adjacent coordinate identities match.
Its shape audit checks every matrix product in (30), (41)–(43), and (44)–(56),
including $H=(J,I^z)$ and the gain-system right-hand side. This is a dimension
audit only: it computes no predicted/filtered states, Kalman gains, covariance
updates, or likelihood. Dimensional validity does not assert invertibility or
settle the statistical questions needed for a future EKF.

Run the entire suite, including the merged OIS tests:

```text
uv run --locked --extra test pytest
```
