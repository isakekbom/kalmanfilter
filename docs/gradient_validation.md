# Full-likelihood gradient validation

[Issue #8](https://github.com/isakekbom/kalmanfilter/issues/8) validates derivatives
of the complete raw-parameter likelihood in
[`gradient_validation.py`](../src/kalmanfilter/gradient_validation.py) and
[`test_gradient_validation.py`](../tests/test_gradient_validation.py).
A finite automatic derivative alone does not establish correctness. Independent
function-value differences check the reverse-mode path before optimization.

## Objective and architecture

The scalar objective is

$$
\operatorname{NLL}(r)=-\ell(\theta(r)).
$$

The dependency chain is raw coordinates → `unpack_parameters` → initial
distribution, transition/noise parameters and OIS loadings → the complete
multi-date EKF → equation (57) innovation log-likelihood → negative log-likelihood.
The result has shape `()` and dtype float64; its raw-vector gradient has shape
`(layout.n_parameters,)` and dtype float64.

`raw_negative_log_likelihood(raw_vector, layout, build_problem)` delegates to
the existing parameter transforms and `run_likelihood(..., return_trace=False)`.
It introduces no alternate filtering or likelihood mathematics and retains no
full trace. Autodiff may retain intermediates needed for the backward pass.
The likelihood remains the existing EKF Gaussian approximation to the nonlinear
observation likelihood, rather than an exact nonlinear marginal likelihood.

`build_problem(parameters)` is a caller-supplied, pure, fixed-structure callable
returning an initial `FilterState` and a sequence of `EKFInputs`. It explicitly
supplies coordinate maps, observations, and instruments. The parameter layer
does not learn coordinate identities, select parameter sharing, or resolve Q4.
No global market-data object, automatic tying, master-state padding, optimizer,
Hessian, or synthetic-series generator is introduced by the gradient module.
[Issue #9](synthetic.md) separately provides synthetic generation from explicit
mathematical parameters per date.

## Independent numerical methodology

For each raw coordinate, `central_difference_gradient` computes

$$
g_i^{FD}(h)=\frac{f(r+h_i e_i)-f(r-h_i e_i)}{2h_i},
\qquad h_i=h\max(1,|r_i|).
$$

The returned `CentralDifference` contains the gradient and actual coordinate
steps. The utility evaluates the objective twice per coordinate in a Python
loop. It uses no AD derivative, and its steps never depend on the AD gradient.
The objective still runs the normal EKF, including its OIS state Jacobian;
independence here concerns the numerical derivative of the scalar NLL.

We compare against both `jax.grad(objective)` and
`jax.value_and_grad(objective)`. Those two reverse-mode paths must also agree
with `rtol=2e-13`, `atol=2e-13`. A separate smooth scalar function with a known
analytical gradient validates the finite-difference utilities while JAX AD
entry points are disabled by the test.

The nonlinear likelihood is evaluated at base steps **`1e-4`, `1e-5`, `1e-6`,
and `1e-7`**. For a smooth function, central differences have $O(h_i^2)$
truncation error; subtracting nearly equal floating-point values contributes
cancellation/roundoff that grows as the step shrinks. Multiple steps reveal a
useful accuracy region. The smallest step is not assumed best, and we do not
require accuracy to improve monotonically. Formal acceptance uses **`h=1e-5`**,
selected from the measured stable region; the other steps remain diagnostics.

For each component `gradient_errors` exposes

$$
E_i^{abs}=|g_i^{AD}-g_i^{FD}|,\qquad
E_i^{rel}=\frac{E_i^{abs}}{\max(|g_i^{AD}|,|g_i^{FD}|)}.
$$

When both derivatives are zero, the relative error is defined to be zero.
There is no denominator floor for nonzero derivatives. Maxima and the indices
of the worst absolute and relative errors are available separately; ties use
the first index. Empty vectors have zero maxima and no worst index. Tests print
the offending component, block, AD/FD values, and base step, plus summaries
for every explicit `ParameterLayout` slice.

Relative error is unstable near a true zero derivative, so acceptance uses an
absolute-plus-relative rule, not a relative error alone. The tests use AD as
the `assert_allclose` reference:

$$
|g_i^{FD}-g_i^{AD}|\leq \mathrm{atol}+\mathrm{rtol}|g_i^{AD}|.
$$

## Directional checks

`directional_central_difference` normalizes a supplied nonzero direction and
returns the scalar derivative, unit direction actually used, and scalar step:

$$
D^{FD}(h)=\frac{f(r+h d)-f(r-h d)}{2h},\quad \|d\|_2=1,
\qquad D^{AD}=\nabla f(r)^T d.
$$

Here $h$ is an absolute displacement along the unit direction, without the
coordinate scaling used above. All four base steps are reported. Every parameter
case uses three dense directions: `(1,-2,3,-4,...,11)` and two standard-normal
vectors from NumPy `default_rng(17)`. Normalization occurs inside the utility;
the AD dot product uses its returned direction. Every direction touches all
11 raw coordinates. Acceptance is required at `h=1e-5` for all 12 case/direction
pairs.

## Controlled problem and implementation conventions

These are fixed algebraic test inputs, not inferred market conventions or
simulated observations from issue #9. The state is `(level,u)`: one systematic
PCA factor and one unsystematic quote-deviation state, with no central-bank
step factor. There are three dates, one OIS quote per date, and two payments
per instrument with accruals `(0.5,0.5)`.

| Date | Observed quote | PCA loading map rows (start, payment 1, payment 2) |
| --- | ---: | --- |
| 1 | 0.04 | `(0.1,-0.5,-1.1)` |
| 2 | 0.025 | `(0.08,-0.45,-1.05)` |
| 3 | 0.055 | `(0.12,-0.55,-1.2)` |

Each loading map has shape `(3,1,1)` and is multiplied by `theta_g`; its fixed
step-loading block has shape `(3,0)`. Existing nonlinear OIS pricing evaluates
the exponential discount ratio and its state Jacobian at each predicted state.
The supplied maps make $F_t=\operatorname{diag}(\theta_f)$ and
$Q_t=\operatorname{diag}(\sigma_w)$, select `u` into the quote, and map the single
observation variance to that quote. State, transition, and noise axes are named
explicitly. Reusing these fixed axes across the three dates is a declared test
setup, not a general parameter-tying rule.

The layout is `(n_f,n_w,n_v,n_x0,n_g)=(2,2,1,2,1)`, with 11 raw coordinates:

| Block | Slice | Role in the objective |
| --- | --- | --- |
| `theta_f` | `[0:2]` | Both state transitions |
| `sigma_w` | `[2:4]` | Positive softplus process variances |
| `sigma_v` | `[4:5]` | Positive softplus observation variance |
| `a_x` | `[5:7]` | Both initial means |
| `sigma_0` | `[7:10]` | Initial factor entries `(0,0),(1,0),(1,1)` |
| `theta_g` | `[10:11]` | OIS loading coefficient |

The initial covariance is dense, with a nonzero off-diagonal factor coordinate,
constructed by issue #7's existing transform without a forward refactorization.
The tests require every individual raw derivative in every block to have
magnitude greater than `1e-6` for these controlled cases. This is a coverage
check, not a restriction on future model gradients. All 11 coordinates enter
the full-vector acceptance assertion and all dates enter the objective.

Three identity-mode cases are modest fixed-seed perturbations of this center:

```text
center = [0.85,0.6,-5,-6,-5.5,0.03,-0.01,-2.3,0.025,-2.8,0.75]
scales = [0.03,0.03,0.15,0.15,0.15,0.005,0.004,0.1,0.005,0.1,0.03]
raw = center + default_rng(20260915).normal(size=(3,11)) * scales
```

Identity is the source-faithful default; the PDF does not establish a universal
transition-parameter domain. A fourth, focused `unit_interval` case reuses the
first perturbed vector with its first two raw entries replaced by `(1.2,0.4)`.
It exercises the optional sigmoid away from saturation and does not attribute
that constraint to the PDF. The primary variance coordinates are also far from
severe softplus underflow.

## Actual numerical results and acceptance

The controlled float64 Windows CPU run uses the locked environment: Python
3.12.14 and JAX 0.11.1. At the formal acceptance step `h=1e-5`:

| Case | Maximum absolute component error | Maximum relative component error |
| --- | ---: | ---: |
| Identity 0 | `2.600632154e-9` | `1.098425706e-9` |
| Identity 1 | `1.783658110e-9` | `7.216710473e-10` |
| Identity 2 | `1.485825241e-9` | `7.068491316e-10` |
| Optional unit interval 0 | `2.427772650e-9` | `1.093064615e-9` |

All four cases' worst component at this step is raw index **8**, the `sigma_0`
off-diagonal factor entry $L_{1,0}$. The overall worst case is identity 0:
AD `2.367599501921`, FD `2.367599499320`. This discrepancy decreases by about
100 times from `h=1e-4` to `h=1e-5`, consistent with central-difference truncation.

Across all four cases, the step-size pattern is:

| Base step | Maximum absolute component error | Maximum relative component error |
| --- | ---: | ---: |
| `1e-4` | `2.624305675e-7` | `1.108424661e-7` |
| `1e-5` | `2.600632154e-9` | `1.098425706e-9` |
| `1e-6` | `4.810116749e-10` | `4.227632041e-9` |
| `1e-7` | `6.350400317e-9` | `2.023364870e-8` |

Both `1e-5` and `1e-6` are useful here; `1e-7` shows increased roundoff error.
The largest error over the entire diagnostic grid is thus `2.624305675e-7`,
distinct from the accepted-step maximum `2.600632154e-9`.

At `h=1e-5`, the maximum directional absolute error is `2.825177869e-10`,
with maximum relative error `1.468728313e-10`. Both occur for identity case 0,
deterministic direction 0: AD `1.923553759847`, FD `1.923553759564`.
Across all diagnostic steps, the maximum directional absolute error is
`2.468077875e-8` at `h=1e-4`; the maximum relative error is `1.540337066e-8`
at that step (identity case 0, random direction 2).

Formal acceptance, only at `h=1e-5`, uses:

| Comparison | `rtol` | `atol` |
| --- | ---: | ---: |
| Every component of every raw-vector gradient | `1e-8` | `5e-9` |
| Every dense directional derivative | `1e-8` | `1e-9` |

These tolerances leave a reasonable float64 margin above the observed errors,
while remaining several orders stricter in relative terms than discrepancies
of `1e-4` or `1e-2` for ordinary non-negligible components. Absolute tolerance
handles derivatives near zero without hiding their reported relative errors.
The other step sizes are retained for diagnosis and do not use this acceptance
assertion. No tolerance was loosened to suppress a persistent discrepancy;
no prerequisite mathematical bug was found or corrected.

## JAX execution and failures

Fix the layout and builder in a closure before differentiation/JIT. Arbitrary
Python callables should not be passed as dynamic array arguments:

```python
def objective(raw):
    return raw_negative_log_likelihood(raw, layout, build_problem)

checked_value = jax.jit(checkify.checkify(objective))
checked_gradient = jax.jit(checkify.checkify(jax.grad(objective)))
checked_both = jax.jit(checkify.checkify(jax.value_and_grad(objective)))
error, (value, gradient) = checked_both(raw)
error.throw()  # Before consuming either numerical result.

def numerical_value(point):
    error, value = checked_value(point)
    error.throw()  # Also check every finite-difference perturbation.
    return value

estimate = central_difference_gradient(numerical_value, raw, 1e-5)
errors = gradient_errors(gradient, estimate.gradient)
```

This usage assumes imports from JAX/checkify and this module, plus the caller's
fixed builder/layout/raw vector. Numerical kernels use JAX float64 without
host conversions or callbacks. Real numerical inputs are promoted to float64;
the finite-difference utility rejects a float32 objective result because casting
it cannot recover lost precision.

The dedicated failure test replaces one raw process-variance coordinate with
`-1000` and runs checked JIT `value_and_grad` through the full objective.
`error.throw()` raises `checkify.JaxRuntimeError` for softplus underflow.
All returned numerical values and gradients after failure are invalid and must
be discarded, even if some happen to be finite. No finite penalty, zero-gradient
fallback, clipping, jitter, or error suppression is introduced.

Known failure modes include sigmoid saturation, softplus underflow, covariance
overflow or invalid innovation Cholesky factors, numerically ill-conditioned
points, unsuitable finite-difference scales, and cancellation near zero
derivatives. A step too small to change a perturbed coordinate is reported.
An error at either finite-difference perturbation invalidates that estimate;
it is not evidence of an AD bug. Persistent discrepancies in a usable step-size
region require isolating the offending block, then transforms/OIS/EKF/likelihood.

## Reproduction and limits

Run the diagnostics and assertions, then the complete suite:

```text
uv run --locked --extra test pytest tests/test_gradient_validation.py -s -q --basetemp .pytest_tmp -p no:cacheprovider
uv run --locked --extra test pytest --basetemp .pytest_tmp -p no:cacheprovider
```

The measured agreement strongly supports correctness of the reverse-mode
derivative path on these controlled nonlinear cases. It does **not** establish
identifiability, good conditioning of a 20-year real-data likelihood, or easy
optimization. Synthetic-series generation/recovery (#9), optimization (#10),
and MATLAB/reference parity (#12) remain later work.
