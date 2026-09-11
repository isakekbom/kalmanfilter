"""OIS pricing equations (8)-(10), with explicit per-instrument inputs.

See docs/model_spec.md section 3 and docs/ois.md for shapes and preconditions.
Value checks raise in eager calls. For compiled execution, functionalize them
with ``jax.jit(checkify.checkify(function))`` and inspect/throw the returned
error outside JIT. No host callbacks, input mutation, or financial defaults are
used. State and parameter differentiation are supported.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jax.typing import ArrayLike


class OISInstrument(NamedTuple):
    """Inputs for one active instrument, in the caller's payment order.

    Supply immutable JAX arrays. With K payments, the field shapes are
    ``(K,)``, ``(K+1, n_p_t, n_g)``, and ``(K+1, n_c_t)`` respectively.
    Loading row 0 is the explicit start; rows 1..K are the payment rows.
    The container is a JAX pytree. Validation occurs at the pricing boundary.
    No dates, accrual conventions, or loading components are generated here.
    """

    accrual_factors: Array
    pca_loading_map: Array
    step_loading: Array


def _real64(value: ArrayLike, name: str, ndim: int) -> Array:
    """Validate static rank/real type and stage a finite-value check."""
    array = jnp.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; got shape {array.shape}")
    if not (
        jnp.issubdtype(array.dtype, jnp.floating)
        or jnp.issubdtype(array.dtype, jnp.integer)
    ):
        raise TypeError(f"{name} must contain real numbers; got {array.dtype}")
    array = array.astype(jnp.float64)
    checkify.check(jnp.all(jnp.isfinite(array)), f"{name} must be finite")
    return array


def discount_loadings(
    theta_g: ArrayLike, pca_loading_map: ArrayLike, step_loading: ArrayLike
) -> Array:
    """Construct o[k] = (O[k] theta_g; o[k]^(2)), text after PDF (10).

    Shapes: theta_g ``(n_g,)``, O ``(n_dates,n_p_t,n_g)``, fixed block
    ``(n_dates,n_c_t)``; result ``(n_dates,n_s_t)`` in PCA/step order.
    """
    theta_g = _real64(theta_g, "theta_g", 1)
    pca_loading_map = _real64(pca_loading_map, "pca_loading_map", 3)
    step_loading = _real64(step_loading, "step_loading", 2)
    if pca_loading_map.shape[2] != theta_g.shape[0]:
        raise ValueError("pca_loading_map last dimension must equal len(theta_g)")
    if pca_loading_map.shape[0] != step_loading.shape[0]:
        raise ValueError("pca_loading_map and step_loading must have equal row counts")
    loadings = jnp.concatenate((pca_loading_map @ theta_g, step_loading), axis=1)
    checkify.check(jnp.all(jnp.isfinite(loadings)), "discount loadings must be finite")
    return loadings


def discount_factors(loadings: ArrayLike, x_s: ArrayLike) -> Array:
    """PDF (8): exp(o[k].T x_s), including the supplied start row.

    Shapes: loadings ``(n_dates,n_s_t)``, x_s ``(n_s_t,)``;
    result ``(n_dates,)``. No minus sign or normalization is inserted.
    """
    loadings = _real64(loadings, "loadings", 2)
    x_s = _real64(x_s, "x_s", 1)
    if loadings.shape[1] != x_s.shape[0]:
        raise ValueError("loadings columns must equal len(x_s)")
    discounts = jnp.exp(loadings @ x_s)
    checkify.check(
        jnp.all(jnp.isfinite(discounts) & (discounts > 0)),
        "discount factors must be finite and nonzero (exponential overflow/underflow)",
    )
    return discounts


def _pricing_terms(theta_g: ArrayLike, x_s: ArrayLike, instrument: OISInstrument):
    if not isinstance(instrument, OISInstrument):
        raise TypeError("instrument must be an OISInstrument")
    accrual = _real64(instrument.accrual_factors, "accrual_factors", 1)
    if accrual.shape[0] == 0:
        raise ValueError("an OIS instrument must have at least one payment")
    loadings = discount_loadings(
        theta_g, instrument.pca_loading_map, instrument.step_loading
    )
    if loadings.shape[0] != accrual.shape[0] + 1:
        raise ValueError("loadings must have K+1 rows for K accrual factors")
    discounts = discount_factors(loadings, x_s)
    annuity = jnp.dot(accrual, discounts[1:])
    checkify.check(jnp.isfinite(annuity), "annuity must be finite")
    checkify.check(annuity != 0, "annuity must be nonzero")
    return accrual, loadings, discounts, annuity


def ois_quote(theta_g: ArrayLike, x_s: ArrayLike, instrument: OISInstrument) -> Array:
    """Scalar g[t,i] from PDF (9): (d[0]-d[K]) / sum(accrual[k] d[k]).

    The caller supplies data for the telescoping schedule class in PDF (7).
    All nonzero finite annuities are used unchanged, including small or negative
    ones; this kernel does not decide financial eligibility or quote units.
    """
    _, _, discounts, annuity = _pricing_terms(theta_g, x_s, instrument)
    quote = (discounts[0] - discounts[-1]) / annuity
    checkify.check(jnp.isfinite(quote), "OIS quote must be finite")
    return quote


def quote_state_gradient(
    theta_g: ArrayLike, x_s: ArrayLike, instrument: OISInstrument
) -> Array:
    """Independent analytical column-gradient reference from PDF (10).

    Returns ``(n_s_t,)`` with theta_g fixed. This uses the quotient-rule formula,
    not autodiff or the production Jacobian. The second term is evaluated as
    (numerator / annuity) * (annuity_gradient / annuity), algebraically equal to
    numerator * annuity_gradient / annuity**2 without squaring the denominator.
    """
    accrual, loadings, discounts, annuity = _pricing_terms(theta_g, x_s, instrument)
    numerator = discounts[0] - discounts[-1]
    numerator_gradient = discounts[0] * loadings[0] - discounts[-1] * loadings[-1]
    annuity_gradient = (accrual * discounts[1:]) @ loadings[1:]
    gradient = numerator_gradient / annuity - (numerator / annuity) * (
        annuity_gradient / annuity
    )
    checkify.check(jnp.all(jnp.isfinite(gradient)), "analytical state gradient must be finite")
    return gradient


def observation_quotes(
    theta_g: ArrayLike, x_s: ArrayLike, instruments: tuple[OISInstrument, ...]
) -> Array:
    """Vector g_t(theta_g, x_s), shape ``(n_z_t,)``, in input tuple order.

    Each tuple entry is one active observation. Payment counts may differ;
    no cash flows are padded. A tuple with a different structure describes a
    different time-specific instrument set. All entries share the same PCA,
    step, and parameter coordinate dimensions.
    """
    theta_g = _real64(theta_g, "theta_g", 1)
    x_s = _real64(x_s, "x_s", 1)
    if not isinstance(instruments, tuple):
        raise TypeError("instruments must be an ordered tuple of OISInstrument inputs")
    if not instruments:
        return jnp.empty((0,), dtype=jnp.float64)
    quotes = []
    block_shape = None
    for instrument in instruments:
        # Validate each entry before looking at its rank-dependent metadata.
        quotes.append(ois_quote(theta_g, x_s, instrument))
        shape = (
            jnp.shape(instrument.pca_loading_map)[1],
            jnp.shape(instrument.step_loading)[1],
        )
        if block_shape is not None and shape != block_shape:
            raise ValueError("instruments must share PCA and step block dimensions")
        block_shape = shape
    return jnp.stack(quotes)


def quote_state_jacobian(
    theta_g: ArrayLike, x_s: ArrayLike, instruments: tuple[OISInstrument, ...]
) -> Array:
    """Production JAX forward-mode state Jacobian, shape ``(n_z_t,n_s_t)``.

    Row i is the transpose of the scalar column gradient in PDF (10).
    Use checkify before JIT to preserve the runtime validation checks.
    """
    theta_g = _real64(theta_g, "theta_g", 1)
    x_s = _real64(x_s, "x_s", 1)
    jacobian = jax.jacfwd(observation_quotes, argnums=1)(theta_g, x_s, instruments)
    checkify.check(jnp.all(jnp.isfinite(jacobian)), "state Jacobian must be finite")
    return jacobian
