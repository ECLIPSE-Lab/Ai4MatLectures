# %% [markdown]
# # Week 7 — Uncertainty and robustness on tensile data
#
# This week braids three lectures:
#
# 1. **MFML Unit 7** — Probabilistic view of learning. We turn the OLS
#    regression from the homework into an explicit Bayesian model and
#    decompose the predictive variance into *aleatoric* (irreducible noise)
#    and *epistemic* (lack of data) parts.
# 2. **MG Week 8** — Regression and generalisation in materials data. The
#    homework already showed the bias-variance U inside a single process
#    condition; today we keep the small-data discipline but build models
#    that *report* their own uncertainty.
# 3. **ML-PC Unit 7** — Generalisation, robustness, and process windows.
#    We end the day by computing input sensitivities and identifying the
#    region of the input space where the model is still trustworthy.
#
# **Red thread.** *Fitting a regression model is making a probabilistic
# claim: "given $\mathbf{x}$, my best guess is $\hat{y}$ with spread
# $\hat\sigma$." Today we make that spread real — first analytically with
# Bayesian linear regression, then empirically with an ensemble of small
# neural networks — and we end by drawing the **process window**: the
# region of $(\varepsilon, T)$ space where (i) the predicted stress is in
# spec and (ii) the model's own self-reported uncertainty is below a
# threshold. Outside that window, you do not get to use this model.*
#
# > **Pre-flight check.** This notebook **assumes** you have run
# > `notebooks/week7_uncertainty_and_robustness_homework.py`. Block 1 picks
# > up directly from your MLE = MSE result, Part B's polynomial sweep, and
# > Part C's leakage gap; we will not re-derive them.
#
# ## Agenda (90 min)
#
# | Block | Min | Topic |
# |------:|:---:|:------|
# | 1 |  6 | Recap from homework — MLE = MSE, the U-curve, the leakage gap |
# | 2 | 12 | Bayesian linear regression: closed-form posterior + predictive variance |
# | 3 | 10 | MAP = ridge: $\lambda = \sigma^2 / \tau^2$, numerically verified |
# | 4 | 14 | Aleatoric vs epistemic via a deep ensemble |
# | 5 | 12 | Calibration plot — does the predicted $\hat\sigma$ match reality? |
# | 6 | 12 | Sensitivity analysis on $(\varepsilon, T)$ |
# | 7 | 10 | Process windows: where is the model trustworthy? |
# | 8 | 14 | Student exercises (3 core + 1 stretch) |

# %%
# Standard imports. Same idiom as weeks 2-6: explicit seeds, no hidden state.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from ai4mat.datasets import TensileTestDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# ## Helpers used by every block

# %%
def load_tensile_np(temperature: int):
    """Return (strain, stress) at the given temperature as 1-D numpy arrays."""
    ds = TensileTestDataset(temperature=temperature)
    return ds.X.numpy().reshape(-1), ds.y.numpy().reshape(-1)


def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """(N, degree+1) Vandermonde matrix [1, x, x^2, ..., x^d]."""
    return np.vander(x, degree + 1, increasing=True)


def standardise_poly(Phi_train: np.ndarray, *Phis: np.ndarray):
    """Standardise non-bias columns; share the scaler with held-out matrices."""
    scaler = StandardScaler().fit(Phi_train[:, 1:])
    out = []
    for P in (Phi_train, *Phis):
        out.append(np.hstack([P[:, :1], scaler.transform(P[:, 1:])]))
    return tuple(out) if Phis else out[0]


# %% [markdown]
# ## Block 1 — Recap from the homework
#
# Three results travel into today's session:
#
# 1. **MLE = MSE.** Maximising the Gaussian log-likelihood over coefficients
#    is identical to minimising MSE. The MLE for $\sigma^2$ is the
#    training MSE. (Part A)
# 2. **Bias-variance U.** Polynomial degree sweep at $T = 600$ °C: train
#    error keeps falling, test error has a U with the minimum near
#    degree 4-5 for OLS and shifted right for ridge. (Part B)
# 3. **Leakage gap.** A random 80/20 split across $T \in \{0, 400, 600\}$
#    looks great; leave-condition-out test RMSE is 5-50× larger. The
#    in-condition U-curve cannot tell you that. (Part C)
#
# Today we make the *spread* of the prediction the first-class output of
# the model. We will use the same dataset and the same polynomial basis.

# %%
# Load all three temperatures; we will use them in different ways below.
data_by_T = {T: load_tensile_np(T) for T in [0, 400, 600]}
strain_600, stress_600 = data_by_T[600]
print("Block 1 — datasets loaded:")
for T in [0, 400, 600]:
    s, y = data_by_T[T]
    print(f"  T = {T:>3} °C:  N = {len(s)},  strain in [{s.min():.3f}, {s.max():.3f}],  stress in [{y.min():6.1f}, {y.max():6.1f}] MPa")


# %% [markdown]
# ## Block 2 — Bayesian linear regression
#
# A Bayesian linear regression treats the polynomial coefficients as
# random variables. With a Gaussian prior $\mathbf{w} \sim \mathcal{N}(0, \tau^2 \mathbf{I})$
# and a Gaussian likelihood $y \mid \mathbf{x} \sim \mathcal{N}(\boldsymbol\phi(\mathbf{x})^\top \mathbf{w}, \sigma^2)$,
# the posterior over $\mathbf{w}$ is also Gaussian:
# $$
# p(\mathbf{w} \mid \mathbf{X}, \mathbf{y})
# = \mathcal{N}(\boldsymbol\mu_N, \mathbf{S}_N), \qquad
# \mathbf{S}_N^{-1} = \frac{1}{\tau^2}\mathbf{I} + \frac{1}{\sigma^2}\boldsymbol\Phi^\top\boldsymbol\Phi, \qquad
# \boldsymbol\mu_N = \frac{1}{\sigma^2}\mathbf{S}_N \boldsymbol\Phi^\top \mathbf{y}.
# $$
# The **predictive distribution** at a new point $\mathbf{x}^\star$ is then
# $$
# p(y^\star \mid \mathbf{x}^\star) = \mathcal{N}\!\left(\boldsymbol\phi^\star{}^\top \boldsymbol\mu_N,\ \underbrace{\sigma^2}_\text{aleatoric} + \underbrace{\boldsymbol\phi^\star{}^\top \mathbf{S}_N \boldsymbol\phi^\star}_\text{epistemic}\right).
# $$
# The two terms are exactly the decomposition we will see again in Block 4
# from a deep ensemble. The first term is the irreducible noise we cannot
# remove with more data; the second term shrinks as $\boldsymbol\Phi^\top\boldsymbol\Phi$
# accumulates more samples in the direction of $\boldsymbol\phi^\star$.
#
# We fit the model on $T = 600$ °C with a degree-5 polynomial and plot
# the predictive band over the full strain range.

# %%
deg = 5
strain_grid = np.linspace(strain_600.min() - 0.1 * (strain_600.max() - strain_600.min()),
                          strain_600.max() + 0.1 * (strain_600.max() - strain_600.min()),
                          400)

Phi_tr = poly_features(strain_600, deg)
Phi_grid = poly_features(strain_grid, deg)
Phi_tr_s, Phi_grid_s = standardise_poly(Phi_tr, Phi_grid)

# Set hyperparameters: sigma^2 from the homework's MLE estimate; tau from
# a weak Gaussian prior over standardised coefficients.
ols_for_sigma = Ridge(alpha=1e-8, fit_intercept=False).fit(Phi_tr_s, stress_600)
sigma2 = float(np.mean((stress_600 - ols_for_sigma.predict(Phi_tr_s)) ** 2))
tau2 = 1e3   # weak prior; you can sweep it in the exercise

# Closed-form posterior over coefficients.
D = Phi_tr_s.shape[1]
S_N_inv = (1.0 / tau2) * np.eye(D) + (1.0 / sigma2) * (Phi_tr_s.T @ Phi_tr_s)
S_N = np.linalg.inv(S_N_inv)
mu_N = (1.0 / sigma2) * S_N @ Phi_tr_s.T @ stress_600

# Predictive mean and variance on the strain grid.
mean_pred = Phi_grid_s @ mu_N
var_epistemic = np.einsum("ij,jk,ik->i", Phi_grid_s, S_N, Phi_grid_s)
var_total = sigma2 + var_epistemic
std_pred = np.sqrt(var_total)

print(f"Block 2 — Bayesian linear regression:")
print(f"  sigma^2 (aleatoric, from training residuals) = {sigma2:.3f}")
print(f"  tau^2   (prior variance per std coeff)        = {tau2:g}")


# %%
# Plot data + predictive mean + ±2σ predictive band.
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(strain_600, stress_600, s=10, alpha=0.5, label="training data (T = 600 °C)")
ax.plot(strain_grid, mean_pred, "r-", lw=2, label="predictive mean")
ax.fill_between(strain_grid, mean_pred - 2 * std_pred, mean_pred + 2 * std_pred,
                color="red", alpha=0.15, label=r"predictive $\pm 2\sigma$")
ax.fill_between(strain_grid,
                mean_pred - 2 * np.sqrt(var_epistemic),
                mean_pred + 2 * np.sqrt(var_epistemic),
                color="orange", alpha=0.25, label=r"epistemic $\pm 2\sigma$ only")
ax.set_xlabel("strain")
ax.set_ylabel("stress (MPa)")
ax.set_title("Block 2 — Bayesian linear regression: predictive distribution")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# Notice the two bands. The wide outer band (red) is what the model
# predicts an *individual measurement* will look like — it includes the
# noise floor. The narrow inner band (orange) is the model's uncertainty
# about the *mean function* and shrinks where data is dense, fans out
# where data is sparse or extrapolated. Bishop Fig. 3.8 is exactly this
# picture, but on real lab data.


# %% [markdown]
# ## Block 3 — MAP = ridge with $\lambda = \sigma^2 / \tau^2$
#
# The MAP estimate is the posterior mode. For a Gaussian prior and
# Gaussian likelihood the posterior is Gaussian, so the MAP coincides
# with the posterior mean $\boldsymbol\mu_N$. Working out
# $\nabla_\mathbf{w}\log p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) = 0$ gives
# $$
# \mathbf{w}_\text{MAP} = \big(\boldsymbol\Phi^\top\boldsymbol\Phi + \tfrac{\sigma^2}{\tau^2}\mathbf{I}\big)^{-1} \boldsymbol\Phi^\top \mathbf{y}.
# $$
# **This is ridge regression** with $\lambda = \sigma^2 / \tau^2$.
# Regularisation is not an arbitrary trick to stabilise the fit — it is
# what falls out of the Bayesian posterior under a Gaussian prior. We
# verify this numerically.

# %%
lam = sigma2 / tau2
ridge = Ridge(alpha=lam, fit_intercept=False).fit(Phi_tr_s, stress_600)

print(f"Block 3 — MAP/ridge equivalence:")
print(f"  lambda = sigma^2 / tau^2 = {lam:.6f}")
print(f"  MAP coefficients   (closed form via Bayes posterior mean):")
print(f"    {np.round(mu_N, 3)}")
print(f"  ridge coefficients (sklearn Ridge with alpha = lambda):")
print(f"    {np.round(ridge.coef_, 3)}")
print(f"  max |MAP - ridge| = {np.max(np.abs(mu_N - ridge.coef_)):.2e}")


# %% [markdown]
# The two coefficient vectors agree to numerical precision. Whenever you
# have used `Ridge(alpha=...)` in this course you have implicitly assumed
# a Gaussian prior on the standardised coefficients with variance
# $\tau^2 = \sigma^2 / \alpha$. Increase $\alpha$ ↔ tighten the prior;
# decrease $\alpha$ ↔ loosen it. With $\alpha \to 0$ we recover OLS / MLE;
# with $\alpha \to \infty$ we shrink to zero.


# %% [markdown]
# ## Block 4 — Aleatoric vs epistemic via a deep ensemble
#
# The Bayesian polynomial in Block 2 had a closed form because we used a
# linear model with a Gaussian prior. For a neural network there is no
# closed form, but we can approximate the same decomposition with a
# **deep ensemble**: train $M$ independently initialised networks on the
# same data and treat their predictions as samples from an approximate
# predictive distribution.
#
# Given ensemble outputs $\hat\mu_m(\mathbf{x})$ for $m = 1, \dots, M$:
# $$
# \underbrace{\mathrm{Var}_\text{total}(\mathbf{x})}_{\text{predictive variance}}
# = \underbrace{\sigma_\text{ale}^2(\mathbf{x})}_{\text{aleatoric, irreducible noise}}
# + \underbrace{\frac{1}{M}\sum_m \big(\hat\mu_m(\mathbf{x}) - \bar\mu(\mathbf{x})\big)^2}_{\text{epistemic, ensemble disagreement}}.
# $$
# We use a small ensemble of MLPs that take the *combined* data
# $(\varepsilon, T_\text{norm}) \to \sigma_\text{stress}$, so we can later
# slice over both axes for the process window.

# %%
def make_combined(temps=(0, 400, 600)):
    """Return X = (strain, T_norm) and y = stress arrays over multiple temperatures."""
    Xs, ys = [], []
    for T in temps:
        s, st = load_tensile_np(T)
        T_norm = np.full_like(s, (T - 300.0) / 300.0)   # roughly [-1, +1]
        Xs.append(np.stack([s, T_norm], axis=1))
        ys.append(st)
    return np.concatenate(Xs), np.concatenate(ys)


X_all, y_all = make_combined()
mu_X = X_all.mean(0); sd_X = X_all.std(0) + 1e-12
mu_y = y_all.mean();  sd_y = y_all.std() + 1e-12

X_std = (X_all - mu_X) / sd_X
y_std = (y_all - mu_y) / sd_y

X_t = torch.tensor(X_std, dtype=torch.float32)
y_t = torch.tensor(y_std, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(seed: int, X_t, y_t, n_epochs=400, lr=1e-2, weight_decay=1e-4):
    """Train a single MLP from scratch with the given seed; returns the model."""
    torch.manual_seed(seed)
    m = MLP()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(n_epochs):
        opt.zero_grad()
        pred = m(X_t)
        loss = F.mse_loss(pred, y_t)
        loss.backward()
        opt.step()
    return m


M = 10
print(f"Block 4 — training a deep ensemble (M = {M}) on combined-temperature tensile data ...")
ensemble = [train_one(seed=s, X_t=X_t, y_t=y_t) for s in range(M)]
print(f"  done.")


# %%
# Predict on a 2-D grid in (strain, T_norm) for visualisation.
strain_grid = np.linspace(X_all[:, 0].min(), X_all[:, 0].max(), 80)
T_norm_grid = np.array([(T - 300.0) / 300.0 for T in [0, 400, 600]])
SS, TT = np.meshgrid(strain_grid, T_norm_grid)
Xq = np.stack([SS.ravel(), TT.ravel()], axis=1)
Xq_std = (Xq - mu_X) / sd_X
Xq_t = torch.tensor(Xq_std, dtype=torch.float32)

with torch.no_grad():
    preds_std = torch.stack([m(Xq_t) for m in ensemble], dim=0).numpy()  # (M, Ngrid)
preds = preds_std * sd_y + mu_y

mu_bar = preds.mean(axis=0)
sigma_epi = preds.std(axis=0)               # ensemble disagreement
sigma_ale = float(np.sqrt(np.mean(
    [F.mse_loss(m(X_t), y_t).item() for m in ensemble]
)) * sd_y)                                  # train residual std, mapped back to MPa
print(f"  aleatoric sigma estimate (homoscedastic, MPa): {sigma_ale:.2f}")
print(f"  epistemic std on grid:  min = {sigma_epi.min():.2f}, max = {sigma_epi.max():.2f} MPa")


# %%
# Plot predictions over strain at each of the three temperatures, with
# ±2σ_total bands. Aleatoric is the homoscedastic noise floor.
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, T_norm, T_label in zip(axes, T_norm_grid, [0, 400, 600]):
    sel = (TT.ravel() == T_norm)
    s_axis = SS.ravel()[sel]
    mu = mu_bar[sel]
    epi = sigma_epi[sel]
    total = np.sqrt(epi ** 2 + sigma_ale ** 2)

    s_data, st_data = data_by_T[T_label]
    ax.scatter(s_data, st_data, s=8, alpha=0.4, color="gray", label="data")
    ax.plot(s_axis, mu, "r-", lw=2, label="ensemble mean")
    ax.fill_between(s_axis, mu - 2 * total, mu + 2 * total, color="red", alpha=0.15,
                    label=r"$\pm 2\sigma_\mathrm{total}$")
    ax.fill_between(s_axis, mu - 2 * epi, mu + 2 * epi, color="orange", alpha=0.3,
                    label=r"$\pm 2\sigma_\mathrm{epi}$")
    ax.set_title(fr"T = {T_label} °C")
    ax.set_xlabel("strain")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("stress (MPa)")
axes[0].legend(loc="lower right", fontsize=8)
fig.suptitle("Block 4 — deep-ensemble predictions with aleatoric/epistemic decomposition")
plt.tight_layout()
plt.show()


# %% [markdown]
# Interpretation:
#
# - The orange band is the ensemble's disagreement at each $(\varepsilon, T)$.
#   It is small wherever many training points support the prediction and
#   *grows* whenever you move into a region the ensemble has not seen.
# - The red band adds the irreducible noise floor $\sigma_\text{ale}$.
# - The total band is what you would quote to a process engineer as
#   "stress predicted at $\hat\mu \pm 2\sigma_\text{total}$".


# %% [markdown]
# ## Block 5 — Calibration: does $\hat\sigma$ match reality?
#
# A model that predicts mean and variance is calibrated if the empirical
# error in each $\hat\sigma$ bin matches that bin's $\hat\sigma$. We
# evaluate this on a held-out 20% slice of the combined data.

# %%
rng = np.random.default_rng(2)
N = len(X_all)
idx = rng.permutation(N)
ntr = int(0.8 * N)
te_idx = idx[ntr:]

X_te = X_t[te_idx]
y_te = y_all[te_idx]

with torch.no_grad():
    preds_te_std = torch.stack([m(X_te) for m in ensemble], dim=0).numpy()
preds_te = preds_te_std * sd_y + mu_y
mu_te = preds_te.mean(axis=0)
sigma_epi_te = preds_te.std(axis=0)
sigma_total_te = np.sqrt(sigma_epi_te ** 2 + sigma_ale ** 2)

abs_err = np.abs(mu_te - y_te)

# Bin test points into 8 quantile bins of predicted sigma_total.
n_bins = 8
order = np.argsort(sigma_total_te)
chunks = np.array_split(order, n_bins)
mean_pred_sigma = np.array([sigma_total_te[c].mean() for c in chunks])
mean_emp_rmse = np.array([np.sqrt(np.mean((mu_te[c] - y_te[c]) ** 2)) for c in chunks])

print("Block 5 — calibration table:")
print(f"  {'bin':>4}  {'mean pred sigma':>16}  {'empirical RMSE':>16}")
for i, (ps, er) in enumerate(zip(mean_pred_sigma, mean_emp_rmse)):
    print(f"  {i+1:>4}  {ps:>16.3f}  {er:>16.3f}")


# %%
fig, ax = plt.subplots(figsize=(6, 5))
m_lim = max(mean_pred_sigma.max(), mean_emp_rmse.max()) * 1.05
ax.plot([0, m_lim], [0, m_lim], "k--", alpha=0.6, label="perfect calibration")
ax.plot(mean_pred_sigma, mean_emp_rmse, "o-", lw=2, ms=8, label="ensemble (Block 4)")
ax.set_xlabel(r"mean predicted $\hat\sigma_\mathrm{total}$ in bin (MPa)")
ax.set_ylabel("empirical RMSE in bin (MPa)")
ax.set_title("Block 5 — calibration plot (held-out 20% of combined data)")
ax.set_xlim(0, m_lim); ax.set_ylim(0, m_lim)
ax.set_aspect("equal")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()


# %% [markdown]
# The diagonal is perfect calibration: when you say $\hat\sigma = 50$ MPa
# the average error in that bin should be 50 MPa. *Above* the line means
# **under-confident** (you said 50, you got 80 — your model is more wrong
# than it admits). *Below* the line means **over-confident**. With a
# 10-member ensemble and homoscedastic aleatoric, expect to be
# under-confident in low-data regions and over-confident on the noise floor.
# Block 8 Exercise 3 asks you to fix the binning and revisit this.


# %% [markdown]
# ## Block 6 — Sensitivity analysis on $(\varepsilon, T)$
#
# A trained model might have high accuracy in expectation and still be
# *fragile*: a small drift in one input causes a large change in the
# prediction. ML-PC formalises this with the local sensitivity
# $\partial \hat y / \partial x_i$, evaluated numerically by central
# difference.

# %%
def grad_finite_diff(model, x_std: np.ndarray, eps_h=1e-3) -> np.ndarray:
    """Central-difference gradient of model w.r.t. each input feature."""
    N, D = x_std.shape
    grads = np.zeros_like(x_std)
    for j in range(D):
        x_plus = x_std.copy(); x_plus[:, j] += eps_h
        x_minus = x_std.copy(); x_minus[:, j] -= eps_h
        with torch.no_grad():
            f_plus = model(torch.tensor(x_plus, dtype=torch.float32)).numpy()
            f_minus = model(torch.tensor(x_minus, dtype=torch.float32)).numpy()
        grads[:, j] = (f_plus - f_minus) / (2 * eps_h)
    return grads


# Sample a regular (strain, T) grid to scan sensitivity over the design space.
strain_scan = np.linspace(X_all[:, 0].min(), X_all[:, 0].max(), 30)
T_scan = np.linspace(-1.0, 1.0, 9)   # T_norm
SS_s, TT_s = np.meshgrid(strain_scan, T_scan)
Xs_raw = np.stack([SS_s.ravel(), TT_s.ravel()], axis=1)
Xs_std = (Xs_raw - mu_X) / sd_X

# Use the ensemble mean as the sensitivity target.
gs = np.zeros((len(ensemble), Xs_std.shape[0], 2))
for i, m in enumerate(ensemble):
    gs[i] = grad_finite_diff(m, Xs_std)
g_mean = gs.mean(axis=0)
# Convert standardised-input gradient back to physical units:
#   d(stress) / d(strain) = (d_std f) * (sd_y / sd_X[strain])
g_strain_phys = g_mean[:, 0] * (sd_y / sd_X[0])
g_Tnorm_phys = g_mean[:, 1] * (sd_y / sd_X[1])
# T_norm = (T - 300)/300, so d(stress)/dT = g_Tnorm_phys / 300
g_T_phys = g_Tnorm_phys / 300.0

print("Block 6 — sensitivities at the centre of the (strain, T) grid:")
mid = len(strain_scan) // 2 + (len(T_scan) // 2) * len(strain_scan)
print(f"  mid-point (strain, T_norm) = ({SS_s.ravel()[mid]:.3f}, {TT_s.ravel()[mid]:.3f})")
print(f"  d(stress)/d(strain)        = {g_strain_phys[mid]:>9.1f} MPa per unit strain")
print(f"  d(stress)/dT               = {g_T_phys[mid]:>9.3f} MPa per °C")


# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, g_phys, label in zip(
    axes,
    [g_strain_phys.reshape(SS_s.shape), g_T_phys.reshape(SS_s.shape)],
    ["d(stress)/d(strain)  [MPa/strain]", "d(stress)/dT  [MPa/°C]"],
):
    im = ax.pcolormesh(strain_scan,
                       300.0 * T_scan + 300.0,    # convert T_norm back to °C
                       g_phys, cmap="RdBu_r", shading="auto")
    ax.set_xlabel("strain")
    ax.set_ylabel("T (°C)")
    ax.set_title(label)
    plt.colorbar(im, ax=ax)
fig.suptitle("Block 6 — input sensitivities of the ensemble mean")
plt.tight_layout()
plt.show()


# %% [markdown]
# Sensitivity is strongly heterogeneous over the input plane:
#
# - $\partial(\text{stress}) / \partial(\text{strain})$ is large at low
#   strain (the elastic regime, almost-vertical slope) and small near
#   peak stress (the plateau).
# - $\partial(\text{stress}) / \partial T$ jumps where the curve shape
#   changes between conditions — those points are *process-fragile*: a
#   real lab is unlikely to control $T$ to better than a few °C, and that
#   uncertainty should be propagated.


# %% [markdown]
# ## Block 7 — Process windows
#
# A **process window** is a region of input space where the model is
# *both*:
#
# 1. **In spec.** The predicted stress is within the engineer's
#    acceptable band.
# 2. **Trustworthy.** The predicted total uncertainty is below a
#    threshold, so the in-spec prediction is not just an overconfident
#    extrapolation.
#
# We compute both fields on a 2-D grid in $(\varepsilon, T)$, then plot
# the intersection.

# %%
strain_pw = np.linspace(X_all[:, 0].min(), X_all[:, 0].max(), 80)
T_pw = np.linspace(0, 600, 60)              # in °C
SS_p, TT_p = np.meshgrid(strain_pw, T_pw)
T_norm_p = (TT_p - 300.0) / 300.0

Xp_raw = np.stack([SS_p.ravel(), T_norm_p.ravel()], axis=1)
Xp_std = (Xp_raw - mu_X) / sd_X
Xp_t = torch.tensor(Xp_std, dtype=torch.float32)

with torch.no_grad():
    preds_p_std = torch.stack([m(Xp_t) for m in ensemble], dim=0).numpy()
preds_p = preds_p_std * sd_y + mu_y
mu_p = preds_p.mean(axis=0).reshape(SS_p.shape)
sigma_epi_p = preds_p.std(axis=0).reshape(SS_p.shape)
sigma_total_p = np.sqrt(sigma_epi_p ** 2 + sigma_ale ** 2)

# Spec: stress in [200, 500] MPa.  Trust: sigma_total below 0.10 * (max - min) of training stress.
spec_lo, spec_hi = 200.0, 500.0
trust_thresh = 0.10 * (y_all.max() - y_all.min())
in_spec = (mu_p >= spec_lo) & (mu_p <= spec_hi)
trustworthy = sigma_total_p <= trust_thresh
window = in_spec & trustworthy
print(f"Block 7 — process window summary:")
print(f"  spec band:       stress in [{spec_lo}, {spec_hi}] MPa")
print(f"  trust threshold: sigma_total <= {trust_thresh:.1f} MPa")
print(f"  in-spec fraction:        {in_spec.mean():.2%}")
print(f"  trustworthy fraction:    {trustworthy.mean():.2%}")
print(f"  process-window fraction: {window.mean():.2%}")


# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
im0 = axes[0].pcolormesh(strain_pw, T_pw, mu_p, cmap="viridis", shading="auto")
axes[0].set_title(r"predicted $\hat\mu$  (MPa)"); plt.colorbar(im0, ax=axes[0])
im1 = axes[1].pcolormesh(strain_pw, T_pw, sigma_total_p, cmap="magma", shading="auto")
axes[1].set_title(r"predicted $\hat\sigma_\mathrm{total}$  (MPa)"); plt.colorbar(im1, ax=axes[1])
im2 = axes[2].pcolormesh(strain_pw, T_pw, window.astype(float),
                         cmap="Greens", shading="auto", vmin=0, vmax=1)
axes[2].set_title("process window  (in spec ∧ trustworthy)"); plt.colorbar(im2, ax=axes[2])
for ax in axes:
    ax.set_xlabel("strain"); ax.set_ylabel("T (°C)")
fig.suptitle("Block 7 — process window over (strain, T)")
plt.tight_layout()
plt.show()


# %% [markdown]
# The right panel is the deliverable. The green region is where you
# *can* deploy this model. White regions outside it fail for one of two
# reasons — either the prediction lies outside the acceptable stress
# band, or the model is honestly uncertain about it. Both are valid
# reasons to refuse a prediction; both are visible only because we built
# uncertainty into the model from the start.


# %% [markdown]
# # Student exercises (Block 8 — ~14 min)

# %% [markdown]
# ## Exercise 1 (core) — Halve the ensemble
#
# Re-run Block 4 with `M = 5` instead of `M = 10`, keeping seeds 0..4.
# Re-plot the three-panel figure.
#
# **Your task:**
#
# 1. How does the *epistemic* band change visually? Where does it grow
#    most — at the edges of the strain range, in the middle, or
#    everywhere?
# 2. Compute the change in epistemic std at the highest-strain point of
#    $T = 600$ °C between $M = 10$ and $M = 5$. Is the change small
#    enough that 5 is "enough", or do you really need 10?
# 3. Bonus: at what $M$ does the epistemic estimate stabilise? (Try 2, 3,
#    5, 10, 20.)

# %%
# YOUR CODE for Exercise 1 below.


# %% [markdown]
# ## Exercise 2 (core) — Robustness under input perturbation
#
# Block 6 found $\partial(\text{stress}) / \partial T$ to be locally
# large in some regions. Pick the highest-magnitude $\partial / \partial T$
# point on the grid. At that point, perturb $T$ by $\pm 10$ °C and refit
# the Bayesian linear regression of Block 2 on the perturbed dataset.
# (Hint: you only need to refit if the perturbation is applied to the
# *training* data; if it is applied at *prediction* time, just evaluate
# the existing model at the perturbed input.)
#
# **Your task:**
#
# 1. Compare the predictive band before and after perturbation. Is the
#    in-spec region of Block 7 stable under this 10 °C drift?
# 2. Repeat with $\pm 1$ °C. ML-PC's "robust to factory-floor noise"
#    criterion is that the spec window should not shift visibly under
#    realistic process drift. Does your model pass?

# %%
# YOUR CODE for Exercise 2 below.


# %% [markdown]
# ## Exercise 3 (core) — Calibration under quantile vs equal-width binning
#
# In Block 5 we used 8 *quantile* bins of $\hat\sigma_\text{total}$ — each
# bin has the same number of samples. An alternative is **equal-width**
# binning, where each bin spans the same range of $\hat\sigma_\text{total}$
# but may contain very few samples in the tails.
#
# **Your task:**
#
# 1. Re-do the calibration plot with 8 equal-width bins.
# 2. The two curves usually disagree most in the tails. Why is that?
#    Which version do you trust more in a small-data setting like this
#    (350 samples per condition)?
# 3. Add error bars: in each bin, the empirical RMSE has a confidence
#    interval that depends on the number of points in the bin. Estimate
#    a bootstrap 90% interval per bin and overlay it on the plot.

# %%
# YOUR CODE for Exercise 3 below.


# %% [markdown]
# ## Exercise 4 (stretch) — Heteroscedastic likelihood
#
# In Blocks 2-5 we assumed a single, constant aleatoric noise level
# $\sigma_\text{ale}$. Real measurements often have noise that *grows*
# with the signal (Poisson-like), or that varies across process
# conditions (more spread at $T = 600$ °C than at $T = 0$ °C).
#
# **Your task:**
#
# 1. Modify the MLP class so it outputs *two* heads: $\hat\mu(\mathbf{x})$
#    and $\log\hat\sigma^2(\mathbf{x})$.
# 2. Replace the MSE loss with the *Gaussian negative log-likelihood*:
#    $$
#      \mathcal{L} = \tfrac{1}{2}\log\hat\sigma^2(\mathbf{x}) + \tfrac{(y - \hat\mu(\mathbf{x}))^2}{2\hat\sigma^2(\mathbf{x})} + \text{const}.
#    $$
# 3. Re-run Block 4. Now $\sigma_\text{ale}^2(\mathbf{x})$ is itself a
#    function of the input. Re-do the calibration plot in Block 5 and
#    the process window in Block 7. Where does the heteroscedastic
#    model differ most from the homoscedastic one?
#
# *Pedagogical pointer: this is the foundation of mixture density networks
# (MFML §"Stochastic enrichment and MDNs") and of the heteroscedastic-
# uncertainty losses widely used in scientific ML.*

# %%
# YOUR CODE for Exercise 4 below.


# %% [markdown]
# ## Exam-aligned must-know statements
#
# Re-read these after the exercises; today's blocks have given you the
# concrete intuition for every one of them.
#
# 1. Maximising the Gaussian log-likelihood over coefficients is the
#    same as minimising MSE (homework Part A).
# 2. The MLE for $\sigma^2$ given the coefficient MLE is the training
#    MSE (homework Part A).
# 3. With a Gaussian prior and Gaussian likelihood, the posterior over
#    coefficients is Gaussian; the predictive distribution is also
#    Gaussian (Block 2).
# 4. The predictive variance decomposes into an irreducible **aleatoric**
#    term ($\sigma^2$, the noise floor) and a data-dependent **epistemic**
#    term that shrinks with more data (Block 2 closed form, Block 4
#    deep ensemble).
# 5. MAP under a Gaussian prior is ridge regression with
#    $\lambda = \sigma^2 / \tau^2$ (Block 3, verified numerically).
# 6. A deep ensemble of $M$ networks gives a non-parametric estimate of
#    the epistemic variance via the variance of the ensemble means
#    (Block 4).
# 7. A model is **calibrated** if the bin-mean predicted $\hat\sigma$
#    matches the bin-empirical RMSE; the diagonal of the calibration
#    plot is the target (Block 5).
# 8. A *process window* is the set of inputs where (i) the predicted
#    output is in spec **and** (ii) the predicted uncertainty is below
#    threshold; outside that set, the model should refuse to answer
#    (Block 7).
# 9. In-condition test error (homework Part B U-curve) and
#    cross-condition test error (homework Part C leakage gap) measure
#    different generalisation problems; passing one says nothing about
#    passing the other.
# 10. Local input sensitivities $\partial \hat y / \partial x_i$ are
#     part of the deployment story, not just the academic story:
#     they identify which inputs need to be controlled in the lab to
#     make the model's process window meaningful (Block 6).
