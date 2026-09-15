# Synthetic data and end-to-end validation

[Issue #9](https://github.com/isakekbom/kalmanfilter/issues/9) provides known
latent truth and explicit mathematical parameters for validating the existing
OIS, transition, EKF, and innovation-likelihood path before optimization.
The implementation is in [`synthetic.py`](../src/kalmanfilter/synthetic.py);
the three scenarios and independent references are in
[`test_synthetic.py`](../tests/test_synthetic.py).

## Generative model and API

The generator reuses the model equations documented in
[`model_spec.md`](model_spec.md):

$$
x_0\sim\mathcal N(a_x,\Sigma_0),\qquad
F_t=A_t\operatorname{diag}(\theta_t^F)B_t,\qquad
x_t=F_tx_{t-1}+w_t,
$$

$$
\eta_t^w\sim\mathcal N(0,\Sigma_t^w),\quad w_t=D_t\eta_t^w,
\qquad
\eta_t^v\sim\mathcal N(0,\Sigma_t^v),\quad v_t=G_t\eta_t^v,
$$

$$
\bar z_t=g_t(\theta_t^g,x_t^s)+I_t^z x_t^u,\qquad z_t=\bar z_t+v_t.
$$

`transition_matrix`, `apply_map`, `split_state`, and `observation_quotes` perform
the transition, mapping, state split, and exact nonlinear pricing. The generator
does not duplicate their algebra or linearize the generated observations.
The EKF subsequently linearizes at its own predicted state, using the existing
filter implementation.

```python
dataset = generate_synthetic_dataset(key, initial_coordinates, a_x, sigma_0, steps)
result = run_likelihood(dataset.initial_filter, dataset.inputs, return_trace=True)
```

Each `SyntheticStepInputs(step, theta_f, sigma_w, theta_g, sigma_v, instruments)`
supplies one date's mathematical parameters. The transition coordinates follow
`A.columns == B.rows`, process covariance follows `D.columns`, observation
covariance follows `G.columns`, and instruments follow `step.active_observations`.
OIS loading blocks follow the current PCA and central-bank coordinates.
The time subscripts above make this explicit input interface visible; they do
not resolve the PDF's fixed-parameter interpretation across changing dimensions
(Q4). There is no dependency on `ParameterLayout`, raw parameters, automatic
extraction, or parameter tying.

`SyntheticDataset` is an immutable named tuple containing:

| Field | Meaning |
| --- | --- |
| `initial_filter` | Distribution with mean `a_x` and covariance `sigma_0` |
| `true_initial_state` | One sampled realization from that distribution |
| `true_states` | Current true state at every date |
| `base_process_noise`, `process_noise` | Base draws `eta_w` and mapped `w` |
| `noiseless_observations` | Exact nonlinear `g + I_z x_u`, evaluated at truth |
| `base_observation_noise`, `observation_noise` | Base draws `eta_v` and mapped `v` |
| `observations` | Generated `z` |
| `inputs` | Ordinary `EKFInputs`, referencing the supplied steps and instruments |

All per-date fields are tuples of arrays in their natural shapes. Coordinate
identities are retained through each `inputs[t].step`. The filter knows the
initial distribution, not the sampled initial state. Supply immutable JAX
arrays in the input containers, as for the existing EKF interfaces.

## Sampling and numerical contract

The generator accepts an explicit typed `jax.random.key` or legacy `PRNGKey`.
It splits the supplied key into `(carry, initial_key)`, then splits `carry` into
`(carry, w_key, v_key)` once per date. This schedule includes dates with zero
observations. Empty mapped observations do not shift subsequent process draws;
base observation draws still follow the declared `G.columns`. Each draw key is
used once. Identical keys and inputs reproduce bitwise within the locked
environment; cross-version or cross-backend bitwise equality is not promised.

Mutual independence of the initial state, process draws, and observation draws
across dates is an **explicit synthetic convention**. The PDF's marginal
Gaussian laws do not establish their full joint distribution (Q7). This
convention matches the absence of cross terms in the implemented recursions.

Diagonal base noise is sampled as `sqrt(variance) * standard_normal`, including
exact zero variances. Dense initial and base covariances must be finite, exactly
symmetric, and admit a finite Cholesky factor with positive diagonal. Singular
nonempty dense covariances are unsupported by this sampling interface; a
`DiagonalMatrix` can represent zero base variances, and maps can represent
shared sources and zero rows. Empty dense covariances are supported without
factorization. The initial covariance input is dense, consistent with
`initialize_filter`.

Sampling in base coordinates avoids factoring mapped covariances, which may
legitimately be singular. Repeated sources, selector weights, zero rows, and
dense maps are preserved. There is no jitter, eigenvalue clipping, variance
floor, silent repair, or finite failure penalty. A valid generated dataset can
still describe an EKF with singular innovation covariance; the existing filter
correctly fails in that case. Generator validity does not guarantee an
admissible likelihood for arbitrary supplied inputs.

All numerical work uses JAX float64. Runtime checks raise eagerly. For fixed
structures, apply `checkify.checkify` before `jax.jit`, check the returned error
outside JIT, and discard **all** outputs after any failure. The sequence uses
an ordinary Python loop; no padding or `lax.scan` is introduced.

## Three controlled validation scenarios

### Exact scalar linear Gaussian case

Seed **42**, 12 dates, one unsystematic coordinate `u`, no systematic states,
and one quote give

$$x_t=0.8x_{t-1}+w_t,\quad z_t=x_t+v_t,\quad
\Sigma^w=0.04,\quad\Sigma^v=0.09,\quad a_x=0.3,\quad\Sigma_0=0.25.$$

The OIS instrument has one accrual payment of 1 and empty PCA/step loadings.
Every discount is exactly 1, so `g=0`; the selector supplies `u`. Thus the
normal production EKF is exactly the scalar Kalman filter, and equation (57)
is the exact Gaussian likelihood for this case.

A separate scalar NumPy recursion in the test computes every predicted and
filtered mean/variance, innovation, innovation variance, and total likelihood.
Observed maximum absolute differences in the locked CPU environment are:

| Quantity | Maximum absolute difference |
| --- | ---: |
| Predicted mean | 1.11022302463e-16 |
| Filtered mean | 1.11022302463e-16 |
| Predicted variance | 1.38777878078e-17 |
| Filtered variance | 1.38777878078e-17 |
| Innovation | 1.11022302463e-16 |
| Innovation variance | 2.77555756156e-17 |
| Total log-likelihood | 0 |

The total is **-12.920097839191344**. Recursion comparisons use `rtol=2e-14`,
`atol=2e-15`; the total uses `rtol=atol=2e-14`.

### Nonlinear OIS with fixed dimensions

Seed **20260909**, 24 dates, coordinates `(p, ua, ub)`, and two active quotes:

| Input | Explicit test value |
| --- | --- |
| `theta_f` | `(0.95, 0.65, 0.55)` |
| Process variances | `(2e-4, 1e-5, 1e-5)` |
| Observation variances | `(1e-5, 2e-5)` |
| `theta_g` | `(0.8,)` |
| `a_x` | `(0.03, 0, 0)` |
| Initial covariance | `L @ L.T`, with `L = [[.02,0,0],[.003,.005,0],[-.002,.001,.006]]` |
| Quote `qa` | Accruals `(.5,.5)`, PCA loading-map rows `(.1,-.5,-1.1)` |
| Quote `qb` | Accruals `(.5,1)`, PCA loading-map rows `(.15,-1,-2)` |

Transitions and noise maps select corresponding named coordinates. `qa` selects
`ua` and `qb` selects `ub`. Each PCA row has shape `(1,1)` and is multiplied by
`theta_g`; there are no central-bank loading columns in this case. The quotes
have nonzero curvature, which is checked explicitly.

Tracking RMSE gives equal weight to each state-coordinate error over all dates:

$$\operatorname{RMSE}=\sqrt{\frac{\sum_t\|\hat x_t-x_t^{true}\|^2}{\sum_t n_t^x}}.$$

The filtered RMSE is **0.00352451223601**, versus **0.00754532006207** for
pre-measurement predictions from the same filter. The test requires aggregate
improvement; it does not require every individual measurement update to improve
realized state error.

Using the **same observations and initial filtering distribution**, two
prespecified, interpretable poor models give:

| Model | Innovation log-likelihood |
| --- | ---: |
| True generating parameters | 150.548807705 |
| Persistence replaced by `(0.2, 0.1, 0.1)` | 135.152828144 |
| Both process and observation variances multiplied by 25 | 95.3677686557 |

These values demonstrate sensible separation for this controlled sample.
The true generating parameters need not be the finite-sample MLE, and no test
claims they beat every perturbation. Neither alternative is optimized. For
nonlinear observations the reported objective is the existing EKF Gaussian
innovation approximation, not the exact nonlinear marginal likelihood.

### Changing dimensions and missing observations

Seed **314** uses the same initial distribution and two quotes, with this
explicit six-date sequence after initialization:

| Date | Current state coordinates | Active observations | Transition parameter count |
| --- | --- | --- | ---: |
| Initial | `(p, ua, ub)` | No time-zero observation | — |
| 1 | `(p, ua, ub)` | `(qa, qb)` | 3 |
| 2 | `(p, policy, ua, ub)` | `(qa, qb)` | 3 |
| 3 | `(p, policy, ua, ub)` | `(qb,)` | 4 |
| 4 | `(p, ua, ub)` | `(qa, qb)` | 4 |
| 5 | `(p, ua, ub)` | `()` | 3 |
| 6 | `(p, ua, ub)` | `(qb, qa)` | 3 |

On introduction, the `policy` transition row is explicitly zero, so its true
state equals its process-noise realization. Its variance is `5e-5`; its
persistence on the next date is `0.8`. The central-bank loading rows are
`(0,-.2,-.4)` for `qa` and `(.1,-.4,-.9)` for `qb`. Removal uses the explicitly
rectangular transition. The other parameters retain the values above. The
test supplies each local parameter/noise axis; no lifecycle is inferred.

State shapes are `(3,), (4,), (4,), (3,), (3,), (3,)`; observation shapes are
`(2,), (2,), (1,), (2,), (0,), (2,)`. Base observation noise retains two named
coordinates at every date, while its mapped shape follows the active rows.
The partial date selects only `qb`; the last date explicitly reverses quote,
selector, noise-map, and instrument order. Missing observations are represented
by active structural rows, never NaNs. The empty date has no instruments,
performs prediction only, and contributes exactly zero likelihood.

Coordinate-aligned aggregate RMSE is **0.00341476221463** after filtering versus
**0.00680373223553** before measurement updates. Total log-likelihood is
**25.5134688944**. Tests check every adjacent coordinate identity, natural
shape, finite truth/noise/innovation, and agreement of `run_filter` with the
filter steps retained by `run_likelihood`.

## Reproduction and scope

Run a complete generation → EKF → likelihood example from the repository root:

```text
uv run --locked python examples/synthetic_validation.py
```

The [example](../examples/synthetic_validation.py) constructs the scalar case
above and prints its likelihood and final true/filtered states. Run all three
scenarios and the boundary tests with:

```text
uv run --locked --extra test pytest tests/test_synthetic.py -v --basetemp .pytest_tmp -p no:cacheprovider
```

Add `-s` to print the linear errors, RMSEs, and likelihood comparisons. Tests
also reconstruct transitions, noise mappings, and nonlinear quotes independently
with NumPy; check exact observation addition and reproducibility; and cover
dense covariances, shared noise, zero variance, empty spaces, invalid inputs,
checked JIT, immutability, and the absence of optimizer functionality.

The model equations come from the PDF. Seeds, loading maps, parameter values,
independence assumptions, compressed observations, and event timings here are
explicit synthetic/implementation conventions. Real-data calendars, loading
construction, global parameter tying (Q4), lifecycle timing (Q5), missing-data
interpretation (Q6), and joint-noise assumptions (Q7) remain unresolved.
Synthetic agreement does **not** establish the supervisor's MATLAB interpretation
or reference parity (#12), which remains blocked by missing reference material.
Full gradient validation (#8) remains implemented. Optimization and parameter
recovery by estimation (#10) are still not implemented.
