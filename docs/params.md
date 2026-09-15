# Parameter coordinates and transforms

[Issue #7](https://github.com/isakekbom/kalmanfilter/issues/7) implements the
parameter layer in [`params.py`](../src/kalmanfilter/params.py), following the
tuple in [model_spec.md §5.2](model_spec.md#52-parameter-vector-and-estimation-objective):

$$
\theta=(\theta^F,\Sigma^w,\Sigma^v,a^x,\Sigma^0,\theta^g).
$$

`unpack_parameters(raw_vector, layout)` maps one flat optimizer-coordinate
vector to `ModelParameters`. `pack_parameters(parameters, layout)` provides
the inverse for initialization and debugging. This module implements transforms;
[full likelihood gradient validation](gradient_validation.md) is separate.
[Baseline optimization](baseline_optimization.md) operates on these raw coordinates.

## Source status

| Status | Content |
| --- | --- |
| Directly specified by PDF | The six-block mathematical tuple; diagonal base covariances in (28)–(29); initial mean and covariance in (31)–(32); the EKF innovation likelihood in (57). |
| Derived mathematical requirements | Covariances are symmetric PSD, diagonal variances are nonnegative, and a nonsingular lower factor with positive diagonal gives an SPD Gram matrix. |
| Implementation conventions | Explicit flat slices in tuple order; row-major lower-triangle packing; softplus variances; a dense initial covariance built from a softplus-diagonal Cholesky factor; identity `theta_f` by default; an optional, explicitly selected logistic transform. |
| Still unresolved | Q9's admissible transition/loadings domains, which parameters are fixed or estimated, additional covariance structure, stationary initialization, identification, tying, burn-in, and time-zero observations; Q4's extraction/sharing across changing structural parameter axes. |

The PDF does **not** establish a universal bound on `theta_f`. Identity preserves
that uncertainty. The implementation's flattening and positive-interior choices
are explicit conventions; they do not resolve all of Q9's estimation choices.

## Static layout and exact flat order

`ParameterLayout` is a frozen dataclass with required keyword arguments
`n_f`, `n_w`, `n_v`, `n_x0`, `n_g`, and optional
`theta_f_transform="identity"`. Counts must be nonnegative Python integers
(booleans are rejected). All fields are static JAX pytree metadata; the layout
has no numerical leaves or mutable state.

For $n=n_{x0}$, its total count is

$$
n_{\rm parameters}=n_f+n_w+n_v+n+\frac{n(n+1)}2+n_g.
$$

`n_parameters`, `n_cholesky`, `block_sizes`, and `slices` expose this contract.
Each block starts immediately after the preceding block, in this exact order:

| Slice name | Raw length | Mathematical output |
| --- | ---: | --- |
| `theta_f` | `n_f` | Vector `(n_f,)`, identity or explicitly selected sigmoid |
| `sigma_w` | `n_w` | `DiagonalMatrix` with `(n_w,)` softplus **variances** |
| `sigma_v` | `n_v` | `DiagonalMatrix` with `(n_v,)` softplus **variances** |
| `a_x` | `n_x0` | Initial mean `(n_x0,)`, identity |
| `sigma_0` | `n_x0*(n_x0+1)//2` | Dense initial covariance `(n_x0,n_x0)`, via a lower factor |
| `theta_g` | `n_g` | Vector `(n_g,)`, identity |

For example, counts `(2,2,1,2,2)` give 12 raw coordinates, with slices
`[0:2]`, `[2:4]`, `[4:5]`, `[5:7]`, `[7:10]`, `[10:12]` respectively.
These semantics are defined by explicit slicing and concatenation, independent
of dictionary iteration or pytree flattening order.

`ModelParameters` is an immutable named tuple with the six mathematical fields
above. The two `DiagonalMatrix` objects each hold only their variance vector;
no dense process or observation diagonal matrix is materialized. All six
numerical leaves returned by the transform are float64 JAX arrays. Constructing
the container alone does not validate inputs; pack/unpack are the boundaries.

## Positive diagonal variances

For each raw variance coordinate $r$, the mathematical variance is

$$
v=\operatorname{softplus}(r)=\log(1+\exp(r)).
$$

The implementation uses stable `jax.nn.softplus`, without directly evaluating
that exponential formula. This is a variance, **not a standard deviation**:
there is no extra square. Neither an absolute value nor an epsilon floor is
used. In exact arithmetic, every finite $r$ produces $v>0$; zero variance is
permitted by the mathematical PSD model but lies on the boundary outside this
finite optimizer parameterization.

The stable inverse is

$$
r=v+\log(-\operatorname{expm1}(-v)),\qquad v>0.
$$

`inverse_softplus` requires real, finite, strictly positive input and works
elementwise on scalars or arrays. Both `sigma_w` and `sigma_v` must be
`DiagonalMatrix` objects when packing. Zero and negative variances fail clearly;
they are never replaced by a small positive value.

## Initial covariance and lower-triangle order

The `sigma_0` raw block holds exactly the independent entries of a lower factor
$L_0$. `layout.lower_triangle_order` lists the pairs in **row-major** order:

```text
(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...
```

For a three-state initial space and raw block $(r_0,\ldots,r_5)$,

$$
L_0=\begin{bmatrix}
\operatorname{softplus}(r_0)&0&0\\
r_1&\operatorname{softplus}(r_2)&0\\
r_3&r_4&\operatorname{softplus}(r_5)
\end{bmatrix},\qquad
\Sigma^0=L_0L_0^T.
$$

Only diagonals are transformed; off-diagonals remain raw. Positive diagonals
make $L_0$ nonsingular and $\Sigma^0$ SPD in exact arithmetic. This selects a
dense SPD interior of the mathematical PSD domain without optimizing covariance
entries directly or imposing a stationary initial distribution.

The computed Gram product is averaged with its transpose to make floating-point
symmetry exact for the existing EKF input contract. This averages only symmetry
roundoff; it does not establish or repair positive definiteness. The forward
transform checks finite raw coordinates, strictly positive softplus diagonals,
and a finite Gram product. It relies on the factor construction for positive
definiteness in exact arithmetic and performs no Cholesky factorization. This
avoids a redundant $O(n_{x0}^3)$ factorization in each future likelihood and
gradient evaluation. Cholesky validation belongs to the inverse path, where
the covariance is supplied externally.

To pack a supplied initial covariance:

1. Require real finite values, shape `(n_x0,n_x0)`, and exact symmetry, consistent
   with the existing supplied-covariance contract.
2. Compute its lower JAX Cholesky factor, requiring finite entries and positive
   diagonals. An indefinite or singular input fails.
3. Apply inverse softplus to the diagonal, retain off-diagonals, and gather in
   the same row-major order.

There is no jitter, eigenvalue clipping, nearest-SPD repair, or epsilon floor.
An empty initial space has empty mean/factor coordinates and a `(0,0)` covariance;
neither direction calls Cholesky in that case.

## Transition, mean, and loading coordinates

The default `theta_f_transform="identity"` gives `theta_f = raw_theta_f` and
preserves arbitrary finite values, including negative values, one, and values
above one. It adds no stationarity or eigenvalue constraint on $F_t$.

The explicitly optional `theta_f_transform="unit_interval"` gives

$$
\theta^F_i=\operatorname{sigmoid}(r_i),\qquad
r_i=\log(\theta^F_i)-\log1p(-\theta^F_i).
$$

This is an implementation option for future use **if a persistence interpretation
and domain are confirmed**, not a conclusion from the PDF. In exact arithmetic,
its range for finite raw values is strictly $(0,1)$; it cannot exactly represent
$\theta^F_i=1$ or zero. `inverse_logit` and the inverse parameter transform require
strictly interior, finite values, without clipping before the logarithms.

`a_x` and `theta_g` always use identity transforms. This adds no bounds, priors,
regularization, normalization, or scaling. Whether future estimation should fix
some of their entries is still a separate Q9 decision.

## Float64 checks and differentiation

Production numerical operations use only JAX. Real integer and floating inputs
are promoted to float64; boolean and complex arrays are rejected. Raw inputs must
have rank one and the exact layout length. The inverse also checks every
mathematical block's shape. Static layout/type/shape mistakes raise
`TypeError`/`ValueError`; numerical validity uses the project's `checkify` protocol.

Mathematical positivity does not imply unrestricted machine representability.
Very negative softplus arguments can underflow to zero, sigmoid can round to
zero or one, and huge factors can overflow the covariance. These failures are
reported, not clipped or repaired. Squaring tiny positive factor diagonals can
also underflow to a finite zero, and rounding an ill-conditioned Gram product
can lose positive definiteness. The forward transform does not separately test
the resulting covariance for positive definiteness; a singular or indefinite
covariance will fail Cholesky if subsequently supplied to the inverse transform.
Round-trip accuracy is therefore claimed for representable, suitably conditioned
interior values, not after information has been lost to underflow or saturation.
Inverse logit is also sensitive near endpoints. No impossible exact recovery
claim is made for such limits.

For a fixed layout, compile the checked transform as follows:

```python
import kalmanfilter  # Enables the package's central JAX float64 policy.
import jax
import jax.numpy as jnp
from jax.experimental import checkify
from kalmanfilter.params import ParameterLayout, pack_parameters, unpack_parameters

layout = ParameterLayout(n_f=2, n_w=2, n_v=1, n_x0=2, n_g=1)
raw = jnp.array([0.9, 0.8, -6, -8, -7, 0.03, 0.002, -2.5, 0.01, -3, 0.7],
                dtype=jnp.float64)
error, parameters = jax.jit(checkify.checkify(unpack_parameters))(raw, layout)
error.throw()
error, recovered = jax.jit(checkify.checkify(pack_parameters))(parameters, layout)
error.throw()
```

Check the error outside JIT **before consuming outputs**. After any reported
failure, discard all numerical results, including gradients. There are no host
callbacks, NumPy conversions, or Python scalar conversions in the kernels.
Valid interior transforms support reverse-mode `jax.grad` and
`jax.value_and_grad`; wrap a compiled differentiable objective with
`jax.jit(checkify.checkify(jax.value_and_grad(objective)))`.

## EKF/likelihood integration and changing dimensions

The parameter object feeds existing APIs directly. Continuing the example with
caller-supplied coordinate identities, structural maps, and OIS loadings:

```python
from kalmanfilter.ekf import EKFInputs, initialize_filter
from kalmanfilter.likelihood import run_likelihood
from kalmanfilter.ois import OISInstrument
from kalmanfilter.transition import StateCoordinates, StructuralStep, selection_map

coordinates = StateCoordinates(("level",), (), ("u",))
f_axes, w_axes = ("f-level", "f-u"), ("w-level", "w-u")
step = StructuralStep(
    coordinates, coordinates,
    selection_map(coordinates.all, f_axes, f_axes),
    selection_map(f_axes, coordinates.all, coordinates.all),
    selection_map(coordinates.all, w_axes, w_axes),
    selection_map(("quote",), coordinates.unsystematic, ("u",)),
    selection_map(("quote",), ("v",), ("v",)),
)
instrument = OISInstrument(
    jnp.array([0.5]), jnp.array([[[0.1]], [[-0.5]]]), jnp.empty((2, 0)),
)

def objective(raw_vector):
    p = unpack_parameters(raw_vector, layout)
    initial = initialize_filter(coordinates, p.a_x, p.sigma_0)
    inputs = EKFInputs(step, p.theta_f, p.sigma_w, p.theta_g, p.sigma_v,
                       jnp.array([0.025]), (instrument,))
    return run_likelihood(initial, (inputs,)).total_log_likelihood

error, (value, gradient) = jax.jit(checkify.checkify(jax.value_and_grad(objective)))(raw)
error.throw()
```

These numbers are a small algebraic smoke example, not financial conventions or
an estimation run. No covariance repair is needed between APIs. `params.py`
itself evaluates no likelihood and implements no optimizer.

`n_x0` belongs **only to the initial coordinate space**. Later states may change
dimension or identity, using explicit `StructuralStep` maps as described in
[transition.md](transition.md) and [ekf.md](ekf.md). No padded/master state is
created. The tests also exercise a two-state initialization followed by a
one-state update with caller-supplied rectangular maps.

The layout contains counts and packing conventions, not structural coordinate
names. Callers must align `theta_f` with `A.columns == B.rows`, process variances
with `D.columns`, observation variances with `G.columns`, and `theta_g` with the
supplied OIS loading maps. A canonical vector does not determine how values
should be extracted or shared if those axes change across dates. Q4 remains
unresolved; no automatic parameter tying or lifecycle rules are introduced.

Tests cover both inverse directions, explicit ordering, independent small
NumPy covariance references/eigenvalues, compact variances, checked failures,
float64 limits, static pytrees, JIT, and gradients through every transform block.
The raw-vector-to-likelihood derivative test here is a small smoke test only.
[Issue #8](gradient_validation.md) separately validates full likelihood gradients;
[issue #9](synthetic.md) provides synthetic series at the mathematical-parameter
level. [Issue #10](baseline_optimization.md) adds baseline synthetic estimation;
[Issue #25](curvature_optimization.md) studies raw-coordinate curvature;
noisy optimization remains later work.
