# %% [markdown]
# # Week 8 — Homework (do BEFORE the Thursday exercise)
#
# This week braids three lectures' Week 8 content onto a single dataset:
#
# 1. **MFML Unit 8** — Probabilistic view of learning. Aleatoric vs
#    epistemic uncertainty; Gaussian noise; MLE = MSE; MAP = regularised
#    MLE; Bayes for predictive distributions; calibration.
# 2. **MG Week 8** — Regression and generalisation in materials data.
#    Bias-variance trade-off, dataset-size vs model complexity, why split
#    design matters more than a small accuracy gain.
# 3. **ML-PC Unit 8** — Generalisation, robustness, and process windows.
#    Out-of-condition generalisation; sensitivity to noise and parameter
#    drift; the *process window* as a region of input space where the
#    model can still be trusted.
#
# **Red thread.** *Fitting a regression model is making a probabilistic
# claim: "given $\mathbf{x}$, my best guess is $\hat{y}$ with spread
# $\hat{\sigma}$." Today's homework does the algebra that links MSE to a
# Gaussian likelihood, sweeps complexity to see the bias-variance U-curve
# on real lab data, and quantifies what happens when you forget that test
# data should come from a different process condition than training data.
# Thursday will then ask: how much of that spread is irreducible noise vs
# lack of data, and where in input space is the model still trustworthy?*
#
# **Time:** ~75 minutes.
#
# ## What this homework is
#
# Four short workouts on the same dataset (`TensileTestDataset` — strain →
# stress at three process temperatures), each anchored on one core idea:
#
# | Part | Min | Topic | Lecture anchor |
# |---|---:|---|---|
# | A | 20 | MLE for a Gaussian regression: log-likelihood ⇔ MSE; recover $\hat{\sigma}^2$ from residuals | MFML §"Likelihood and MLE" |
# | B | 20 | Bias-variance U-curve: polynomial degree sweep with and without ridge | MG §"Dataset size vs model complexity" + MFML §"Bias-variance" |
# | C | 25 | Process-condition split: random vs leave-T-out on the 3-temperature dataset; quantify the leakage gap | ML-PC §"Generalisation to factory-floor data" + MG §"Why split design matters" |
# | D | 10 | Reflection: when does Part B's U-curve agree with Part C's leakage gap, and when do they disagree? | bridge to Thursday |
#
# ## What you must hand in (or be able to show on Thursday)
#
# 1. **Part A:** residual histogram + a printed line showing
#    $\hat{\sigma}^2 \approx \mathrm{MSE}_{\text{train}}$ to 3 decimal
#    places.
# 2. **Part B:** train vs test RMSE plotted against polynomial degree for
#    OLS *and* for ridge ($\alpha = 1.0$). Show the U.
# 3. **Part C:** a small table of test RMSE under (i) random 80/20 split
#    over all three temperatures, (ii) leave-$T=600$-out, (iii)
#    leave-$T=0$-out. Comment on why (i) is misleading.
# 4. **Part D:** your reflection paragraph (4–6 sentences).

# %%
# Standard imports for the whole homework. Same idiom as weeks 2-6.
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from ai4mat.datasets import TensileTestDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# ## Helper: load one temperature as numpy arrays
#
# `TensileTestDataset` returns torch tensors; for the linear-algebra parts
# of this homework, plain numpy is easier to read and debug. We keep the
# helper here because every Part needs it.

# %%
def load_tensile_np(temperature: int):
    """Return (strain, stress) as 1-D numpy arrays at the given temperature."""
    ds = TensileTestDataset(temperature=temperature)
    X = ds.X.numpy().reshape(-1)   # (350,)  strain
    y = ds.y.numpy().reshape(-1)   # (350,)  stress
    return X, y


# %% [markdown]
# # Part A — MLE for a Gaussian regression
#
# We model the stress at strain $\varepsilon$ as
# $$ y \mid \varepsilon \sim \mathcal{N}(f_\theta(\varepsilon),\ \sigma^2), $$
# where $f_\theta$ is whatever function class we are fitting (today: a
# polynomial in $\varepsilon$). Maximising the log-likelihood over $\theta$
# gives
# $$
# \log p(y \mid \varepsilon, \theta, \sigma)
# = -\frac{1}{2\sigma^2}\sum_i (y_i - f_\theta(\varepsilon_i))^2 + \text{const}.
# $$
# **The first term is $-\frac{1}{2\sigma^2} \cdot N \cdot \mathrm{MSE}$.**
# Maximising it over $\theta$ is *exactly* minimising the MSE. That is why
# every regression in this course can be read either as least squares or
# as Gaussian MLE — the two procedures find the same optimum.
#
# The MLE for the noise variance $\sigma^2$, computed *after* the
# coefficient MLE, is
# $$ \hat{\sigma}^2 = \frac{1}{N}\sum_i (y_i - f_{\hat\theta}(\varepsilon_i))^2 = \mathrm{MSE}_\text{train}. $$
#
# **Your task.** Fit a degree-3 polynomial in strain to the $T=600\,°\mathrm{C}$
# tensile curve, compute the MLE residual variance, and verify that it
# equals the training MSE.

# %%
strain, stress = load_tensile_np(temperature=600)

# Build the polynomial design matrix Phi: columns are 1, eps, eps^2, eps^3.
def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Return the (N, degree+1) Vandermonde matrix [1, x, x^2, ..., x^d]."""
    return np.vander(x, degree + 1, increasing=True)

Phi = poly_features(strain, degree=3)
ols = LinearRegression(fit_intercept=False).fit(Phi, stress)
y_hat = ols.predict(Phi)

residuals = stress - y_hat
sigma2_mle = np.mean(residuals ** 2)
mse_train = mean_squared_error(stress, y_hat)

print(f"Part A:")
print(f"  Number of samples N = {len(strain)}")
print(f"  Fitted polynomial coefficients (degree 0..3): {ols.coef_}")
print(f"  MLE residual variance sigma_hat^2 = {sigma2_mle:.4f}")
print(f"  Training MSE                       = {mse_train:.4f}")
print(f"  Difference (should be ~ 0)         = {abs(sigma2_mle - mse_train):.2e}")


# %%
# Diagnostic: residual histogram with a Gaussian overlay at sigma_hat.
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(residuals, bins=30, density=True, alpha=0.6, label="residuals")
xs = np.linspace(residuals.min(), residuals.max(), 200)
gauss = (1.0 / np.sqrt(2 * np.pi * sigma2_mle)) * np.exp(-(xs ** 2) / (2 * sigma2_mle))
ax.plot(xs, gauss, "r-", lw=2, label=fr"$\mathcal{{N}}(0,\hat\sigma^2={sigma2_mle:.2f})$")
ax.set_xlabel("residual y - y_hat (MPa)")
ax.set_ylabel("density")
ax.set_title("Part A — residuals vs MLE Gaussian")
ax.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# **Part A deliverable:** the residual histogram and the printed
# `sigma_hat^2 ≈ MSE_train` check. The Gaussian overlay should follow the
# histogram reasonably closely; any tail asymmetry is a hint that the
# Gaussian-noise assumption is approximate (later: Student-t robustness).


# %% [markdown]
# # Part B — Bias-variance: polynomial degree sweep
#
# The Gaussian likelihood from Part A says nothing about *which* function
# class to use. Take it too small and you under-fit (high bias); take it
# too large and you over-fit on the 350 training samples (high variance).
# The classical visual is the U-curve of test error against complexity.
#
# We sweep polynomial degree from 1 to 7 with two configurations:
# 1. **OLS** — closed-form least squares with no regularisation.
# 2. **Ridge** ($\alpha = 1.0$) — least squares with a Gaussian prior on
#    the coefficients (we will derive that this is the same as MAP under
#    a Gaussian prior in Thursday's Block 3).
#
# Standardising the polynomial features matters: $\varepsilon^7$ is many
# orders of magnitude larger than $\varepsilon$, and ridge would treat the
# raw columns very unequally.

# %%
strain_T, stress_T = load_tensile_np(temperature=600)

rng = np.random.default_rng(0)
n = len(strain_T)
idx = rng.permutation(n)
n_train = int(0.8 * n)
tr, te = idx[:n_train], idx[n_train:]

X_tr_raw, y_tr = strain_T[tr], stress_T[tr]
X_te_raw, y_te = strain_T[te], stress_T[te]

degrees = list(range(1, 8))
rmse_ols_tr, rmse_ols_te = [], []
rmse_rdg_tr, rmse_rdg_te = [], []

for d in degrees:
    Phi_tr = poly_features(X_tr_raw, d)
    Phi_te = poly_features(X_te_raw, d)
    # Standardise non-bias columns; leave the constant column at 1.
    scaler = StandardScaler().fit(Phi_tr[:, 1:])
    Phi_tr_std = np.hstack([Phi_tr[:, :1], scaler.transform(Phi_tr[:, 1:])])
    Phi_te_std = np.hstack([Phi_te[:, :1], scaler.transform(Phi_te[:, 1:])])

    # OLS
    m_ols = LinearRegression(fit_intercept=False).fit(Phi_tr_std, y_tr)
    rmse_ols_tr.append(np.sqrt(mean_squared_error(y_tr, m_ols.predict(Phi_tr_std))))
    rmse_ols_te.append(np.sqrt(mean_squared_error(y_te, m_ols.predict(Phi_te_std))))

    # Ridge
    m_rdg = Ridge(alpha=1.0, fit_intercept=False).fit(Phi_tr_std, y_tr)
    rmse_rdg_tr.append(np.sqrt(mean_squared_error(y_tr, m_rdg.predict(Phi_tr_std))))
    rmse_rdg_te.append(np.sqrt(mean_squared_error(y_te, m_rdg.predict(Phi_te_std))))


# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, tr_curve, te_curve, title in zip(
    axes,
    [rmse_ols_tr, rmse_rdg_tr],
    [rmse_ols_te, rmse_rdg_te],
    ["OLS (no regularisation)", r"Ridge ($\alpha = 1.0$)"],
):
    ax.plot(degrees, tr_curve, "o-", label="train RMSE")
    ax.plot(degrees, te_curve, "s-", label="test RMSE")
    ax.set_xlabel("polynomial degree")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("RMSE (MPa)")
fig.suptitle(f"Part B — bias-variance on T={600} °C tensile (N_train={n_train}, N_test={n-n_train})")
plt.tight_layout()
plt.show()

print(f"\nPart B test RMSE per degree:")
print(f"{'deg':>4} | {'OLS test':>10} | {'Ridge test':>10}")
for d, ot, rt in zip(degrees, rmse_ols_te, rmse_rdg_te):
    print(f"{d:>4} | {ot:>10.3f} | {rt:>10.3f}")


# %% [markdown]
# **Part B deliverable:** the two-panel figure and the table above.
#
# Things to notice (you will need them on Thursday):
#
# - OLS test error tends to climb steeply at high degrees while train
#   error keeps dropping — that is the variance side of the U.
# - Ridge keeps high-degree fits well-behaved; the U flattens. The penalty
#   acts like a Gaussian prior pulling coefficients toward 0.
# - At low degrees both curves are close, because regularisation does not
#   matter when there are no spare parameters to misuse.


# %% [markdown]
# # Part C — Process-condition split: leakage you cannot see in Part B
#
# Part B used a random 80/20 split *within one temperature*. Real
# process-monitoring data comes from runs at different conditions: temperatures,
# rates, machines, days. If you train on a mix of conditions and evaluate
# on a mix of conditions, you measure interpolation between conditions
# you have already seen. You do not measure what happens on a *new*
# condition you have not.
#
# The tensile dataset has three process conditions: $T \in \{0, 400, 600\}$ °C.
# Each curve is genuinely different — at $T=0$ the response is brittle and
# nearly linear, at $T=600$ it is highly nonlinear. We compare three
# splits:
#
# | Split | Train | Test |
# |---|---|---|
# | (i) random 80/20 over **all three** temperatures | random 80% | held-out 20% |
# | (ii) leave-$T=600$-out | T=0 + T=400 | T=600 |
# | (iii) leave-$T=0$-out | T=400 + T=600 | T=0 |
#
# We fit the same model (degree-5 ridge) under each split and compare the
# test RMSE. The gap between (i) and (ii)/(iii) is the **leakage gap**.

# %%
data_by_T = {T: load_tensile_np(T) for T in [0, 400, 600]}

def fit_eval(strain_tr, stress_tr, strain_te, stress_te, degree=5, alpha=1.0):
    """Standardise polynomial features on train, fit Ridge, return test RMSE."""
    Phi_tr = poly_features(strain_tr, degree)
    Phi_te = poly_features(strain_te, degree)
    scaler = StandardScaler().fit(Phi_tr[:, 1:])
    Phi_tr_s = np.hstack([Phi_tr[:, :1], scaler.transform(Phi_tr[:, 1:])])
    Phi_te_s = np.hstack([Phi_te[:, :1], scaler.transform(Phi_te[:, 1:])])
    m = Ridge(alpha=alpha, fit_intercept=False).fit(Phi_tr_s, stress_tr)
    return np.sqrt(mean_squared_error(stress_te, m.predict(Phi_te_s))), m, scaler

# (i) random over all three
all_strain = np.concatenate([data_by_T[T][0] for T in [0, 400, 600]])
all_stress = np.concatenate([data_by_T[T][1] for T in [0, 400, 600]])

rng = np.random.default_rng(1)
N = len(all_strain)
idx = rng.permutation(N)
tr_i, te_i = idx[: int(0.8 * N)], idx[int(0.8 * N) :]
rmse_random, _, _ = fit_eval(all_strain[tr_i], all_stress[tr_i],
                             all_strain[te_i], all_stress[te_i])

# (ii) leave-T=600-out
tr_strain = np.concatenate([data_by_T[T][0] for T in [0, 400]])
tr_stress = np.concatenate([data_by_T[T][1] for T in [0, 400]])
te_strain, te_stress = data_by_T[600]
rmse_loo600, _, _ = fit_eval(tr_strain, tr_stress, te_strain, te_stress)

# (iii) leave-T=0-out
tr_strain = np.concatenate([data_by_T[T][0] for T in [400, 600]])
tr_stress = np.concatenate([data_by_T[T][1] for T in [400, 600]])
te_strain, te_stress = data_by_T[0]
rmse_loo000, _, _ = fit_eval(tr_strain, tr_stress, te_strain, te_stress)

print("Part C — test RMSE under three split protocols (degree-5 ridge):")
print(f"  (i)   random 80/20 over T in {{0, 400, 600}}:  RMSE = {rmse_random:7.2f} MPa")
print(f"  (ii)  leave-T=600-out (train 0+400, test 600): RMSE = {rmse_loo600:7.2f} MPa")
print(f"  (iii) leave-T=0-out   (train 400+600, test 0): RMSE = {rmse_loo000:7.2f} MPa")
print()
print(f"  Leakage gap (ii)/(i) = {rmse_loo600 / rmse_random:.1f}x")
print(f"  Leakage gap (iii)/(i) = {rmse_loo000 / rmse_random:.1f}x")


# %%
# Visualise why the leakage exists: the three curves are different shapes.
fig, ax = plt.subplots(figsize=(7, 5))
for T in [0, 400, 600]:
    s, st = data_by_T[T]
    order = np.argsort(s)
    ax.plot(s[order], st[order], "o-", ms=3, alpha=0.7, label=fr"$T = {T}$ °C")
ax.set_xlabel("strain")
ax.set_ylabel("stress (MPa)")
ax.set_title("Three process conditions in the tensile dataset")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# %% [markdown]
# **Part C deliverable:** the printed table and the three-curve plot.
#
# Things to notice:
#
# - Split (i) lets stress observations at $T=600$ leak into training,
#   because the model memorises the $T=600$ shape from points adjacent to
#   the held-out 20% of $T=600$ samples. Test RMSE looks great.
# - Split (ii) and (iii) measure the *real* generalisation question: can
#   the model handle a process condition it has not seen at all? The RMSE
#   typically blows up by a factor of 5-50x.
# - This is the same lesson Week 3 taught for sample-vs-specimen splits,
#   now in the engineering-process flavour ML-PC emphasises.


# %% [markdown]
# # Part D — Reflection: when does the U-curve track the leakage gap?
#
# In Part B you saw a bias-variance U inside one process condition. In
# Part C you saw a leakage gap *across* process conditions. The two
# diagnostics measure different things, and they can disagree:
#
# - A model can sit perfectly at the bottom of its in-condition U (Part B
#   says "this is the right complexity!") and still fail catastrophically
#   under leave-condition-out (Part C says "this model has no idea about
#   $T=0$").
# - Conversely, a heavily under-fit model can have a small leakage gap
#   (because it predicts roughly the same wrong thing in every condition)
#   while still being useless.
#
# **Your task (~10 min, write 4-6 sentences):**
#
# Pick a single materials-science scenario you have worked on or know
# well — a microscopy classifier, a property predictor, a process monitor,
# whatever you like — and answer two questions:
#
# 1. What is the analogue of "process condition" in your scenario? (e.g.
#    sample, specimen, day, instrument, alloy family, temperature, dose…)
# 2. If you only had time to run *one* validation experiment before
#    deploying the model, would you run a Part-B-style bias-variance sweep
#    or a Part-C-style group-out test, and why?
#
# Bring this paragraph to Thursday; we will pick two volunteers to read
# theirs aloud at the start of Block 1.
#
# **Hand in:** your written paragraph (Markdown cell below).

# %% [markdown]
# > *(your reflection paragraph here)*
