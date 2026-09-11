# OIS pricing kernel

This implements [issue #3](https://github.com/isakekbom/kalmanfilter/issues/3),
equations (8)–(10) on page 2 of [kalmanRante.pdf](../kalmanRante.pdf), using
[model_spec.md §3](model_spec.md) as the
mathematical specification. The implementation is in
[`src/kalmanfilter/ois.py`](../src/kalmanfilter/ois.py).

The kernel evaluates the telescoping quote for caller-supplied data. It does not
generate dates, accrual factors, PCA bases, central-bank steps, observations, or
a state time series. It does not add instrument deviations, transition matrices,
an EKF, a likelihood, or an optimizer.

## Inputs and shapes

An immutable `OISInstrument` named tuple contains three arrays. Supply immutable
JAX arrays, using the caller's existing payment order and systematic coordinates.
Validation occurs when calling a pricing function, rather than during pytree
construction. All numerical inputs are checked as real and promoted to float64;
complex and Boolean arrays are rejected. Converting existing float32 inputs
cannot recover precision already lost before the call.

| Input | PDF notation | Shape and meaning |
| --- | --- | --- |
| `theta_g` | $\theta^g$ | `(n_g,)`; pricing/loading parameters |
| `x_s` | $x_t^s=(x_t^p;x_t^c)$ | `(n_s_t,)`, with `n_s_t = n_p_t + n_c_t` |
| `instrument.accrual_factors` | $t_k^c$ | `(K,)`; supplied factors for payments `k=1..K`, with `K >= 1` |
| `instrument.pca_loading_map` | $O_{t,i,k}$ | `(K+1,n_p_t,n_g)`; includes the start row `k=0` |
| `instrument.step_loading` | $o_{t,i,k}^{(2)}$ | `(K+1,n_c_t)`; supplied fixed block, also including the start row |
| `instruments` | Active instrument list at $t$ | Ordered tuple of `OISInstrument`; its length is `n_z_t` |

No input has a financial default. The first loading row is always used as
supplied, including its effect on the numerator through $d_0$. The final row
`K` is used in both the numerator and annuity; the start row is excluded from
the annuity. There is no assumption that $d_0=1$ and no added minus sign in the
exponential. All instruments in one call must share `n_p_t`, `n_c_t`, `n_g`, and
the meaning of their coordinates. Only dimension compatibility can be checked
by this module; coordinate identities are the caller's responsibility.

Payment counts may differ across instruments. The vector function stacks the
scalar kernels in tuple order. Array operations cover all payments within each
instrument; JAX traces the finite tuple iteration when compiling. It neither
pads cash flows nor generates a mask. A later call can supply another tuple or
different payment counts. Different tuple structures/array shapes can require
recompilation; this is not a fixed instrument universe or a time-series model.

An empty tuple returns a `(0,)` quote vector and a `(0,n_s_t)` Jacobian. These
are mathematical output shapes only; they do not prescribe the future EKF's
behavior on an all-missing date. Empty PCA or step blocks have zero-width array
axes, not omitted inputs.

## Functions and equations

| Function | Equation / implementation | Output shape |
| --- | --- | --- |
| `discount_loadings(theta_g, pca_loading_map, step_loading)` | Text following (10): concatenate `pca_loading_map @ theta_g` and `step_loading` along the state axis | `(K+1,n_s_t)` for one instrument |
| `discount_factors(loadings, x_s)` | (8): $d_k=\exp(o_{t,i,k}^T x_t^s)$ | `(K+1,)` |
| `ois_quote(theta_g, x_s, instrument)` | (9): $(d_0-d_K)/\sum_{k=1}^K t_k^c d_k$ | `()` |
| `observation_quotes(theta_g, x_s, instruments)` | Vector $g_t(\theta^g,x_t^s)$ in active observation order | `(n_z_t,)` |
| `quote_state_gradient(theta_g, x_s, instrument)` | Independent analytical expression (10), with parameters fixed | `(n_s_t,)` |
| `quote_state_jacobian(theta_g, x_s, instruments)` | Production `jax.jacfwd(observation_quotes, argnums=1)` | `(n_z_t,n_s_t)` |

The two lower-level loading/discount functions also accept a general
`n_dates` row count. The scalar quote validates the OIS-specific `K+1` relation.
The analytical function shares the primal loading and discount calculations but
never calls autodiff. Writing $b=d_0-d_K$ and $a=\sum_k t_k^c d_k$, it evaluates

$$
\nabla_{x^s}g_{t,i}
=\frac{d_0o_{t,i,0}-d_Ko_{t,i,K}}{a}
-\frac{b}{a}\frac{\sum_{k=1}^K t_k^c d_k o_{t,i,k}}{a}.
$$

This is equation (10), with the second term regrouped to avoid explicitly
squaring the denominator. It does not change, bound, or replace the denominator.
Row `i` of the production Jacobian is the transpose of this scalar column
gradient. The pricing functions also support differentiating with respect to
`theta_g`; those derivatives flow through the explicit PCA loading maps.

## Eager use

This is an arbitrary algebraic example, not a choice of financial conventions:

```python
import jax.numpy as jnp
from kalmanfilter.ois import (
    OISInstrument,
    observation_quotes,
    quote_state_gradient,
    quote_state_jacobian,
)

theta_g = jnp.array([0.7], dtype=jnp.float64)
x_s = jnp.array([0.02, -0.01], dtype=jnp.float64)
instruments = (
    OISInstrument(
        accrual_factors=jnp.array([0.4], dtype=jnp.float64),
        pca_loading_map=jnp.array([[[0.1]], [[-0.5]]], dtype=jnp.float64),
        step_loading=jnp.array([[0.2], [-0.1]], dtype=jnp.float64),
    ),
    OISInstrument(
        accrual_factors=jnp.array([0.2, 0.3], dtype=jnp.float64),
        pca_loading_map=jnp.array([[[0.2]], [[-0.3]], [[-0.8]]], dtype=jnp.float64),
        step_loading=jnp.array([[0.1], [-0.1], [-0.2]], dtype=jnp.float64),
    ),
)
quotes = observation_quotes(theta_g, x_s, instruments)       # (2,)
jacobian = quote_state_jacobian(theta_g, x_s, instruments)   # (2, 2)
reference_row = quote_state_gradient(theta_g, x_s, instruments[0])  # (2,)
```

## Validation and compiled execution

Static shape/container errors raise `ValueError` or `TypeError`. Numerical
checks reject non-finite inputs, non-finite computed loadings, zero/non-finite
discounts caused by exponential underflow/overflow, zero/non-finite annuities,
and non-finite returned quotes or state derivatives. In eager execution these
raise `jax.experimental.checkify.JaxRuntimeError` with the relevant quantity in
the message.

No near-zero cutoff is selected: a small nonzero annuity is used exactly as
computed. Finite negative or individual zero accrual factors are not rejected on
financial grounds; only the defined algebra and numerical preconditions are
checked. Financial admissibility is outside this kernel. There is no clipping,
jitter, discount rescaling, denominator replacement, or invalid-result fallback.
Extremely scaled inputs may still exceed float64's representable range, including
in autodiff intermediates. Reported numerical failure is not an invitation to
interpret a repaired value as the PDF's quote.

To retain dynamic checks while compiling, functionalize them first:

```python
import jax
from jax.experimental import checkify

compiled_quotes = jax.jit(checkify.checkify(observation_quotes))
compiled_jacobian = jax.jit(checkify.checkify(quote_state_jacobian))

error, quotes = compiled_quotes(theta_g, x_s, instruments)
error.throw()  # Outside JIT; do this before using the result.
error, jacobian = compiled_jacobian(theta_g, x_s, instruments)
error.throw()
```

As described by [JAX's checkify guide](https://docs.jax.dev/en/latest/debugging/checkify_guide.html),
the transformed computation returns the error as data, so it can be compiled
without Python callbacks. Calling `jax.jit` directly on a function containing
these checks is unsupported: use the explicit `checkify` composition above.
Always inspect the returned error before consuming results; compiled computation
may continue after recording an error and its returned numbers are then invalid.
Discard those numbers when an error is reported. There are no host
callbacks, file operations, hidden mutable model state, or input mutations in
the numerical functions. Eager exception reporting and host-side `error.throw()`
are outside the pure compiled computation.

## Ambiguities left open

- **Q1 — Schedule and quote conventions:** callers supply the accrual factors
  and instruments to which the telescoping condition applies. The kernel cannot
  infer or verify calendars, day counts, payment lags, or quote units from arrays.
- **Q2 — Discount origin:** callers supply row 0 and all other loading rows.
  The kernel does not normalize the start discount, integrate a curve, or add an
  intercept.
- **Q3 — Factor construction:** callers supply `pca_loading_map` and the fixed
  `step_loading` component. Their construction, units, ordering, and identification
  constraints remain open. The lower block is held fixed with respect to
  `theta_g` as explicitly required for this issue.
- **Q10 — Numerical policy:** this issue requires clear failure without repair.
  It supplies no threshold defining “near zero,” so no such threshold is invented.
  Exact zero and non-finite values fail; small nonzero values remain unchanged.
  Broader numerical policy for the future EKF is still open.

## Acceptance checks

`tests/test_ois.py` checks one scalar quote per active tuple entry and Jacobian
shape `(n_z_t,n_s_t)`, including different payment counts and changed active sets.
The reference valuation for finite differences uses separate NumPy/scalar code,
not the implementation's quote function.

Across four dimension/payment configurations and three deterministic states per
configuration, analytical and autodiff state derivatives must agree at
`rtol=2e-12, atol=2e-13`. Central differences with
`h_j=1e-5 * max(1, abs(x_s[j]))` must agree at `rtol=2e-8, atol=2e-10`.
Additional checks cover parameter differentiation, compiled dynamic inputs,
known scalar values, both factor blocks, zero blocks, float64 output, numerical
failure, and the absence of small-denominator replacement.

Run the complete suite, including the issue #2 tests, from the repository root:

```text
uv run --locked --extra test pytest
```
