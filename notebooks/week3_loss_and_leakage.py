# %% [markdown]
# # Week 3 — Loss minimization & leakage-safe regression
#
# This week we braid two lectures:
#
# 1. **MFML Unit 3**: Regression as loss minimization — gradient descent, Newton's method,
#    loss-as-noise-model (MSE / Huber / MAE), and basis functions (polynomial → Runge → splines).
# 2. **ML-PC Unit 3**: Data quality, validation, and leakage — preprocessing, group, and
#    temporal leakage; honest cross-validation.
#
# **Red thread:** *Optimization finds the minimum of whatever loss landscape you hand it;
# data quality and validation decide whether that minimum means anything.* The same
# TensileTest fit can look great or fall apart depending on which loss you minimise and
# how you split the data.
#
# > **Pre-flight check.** This notebook **assumes** you have run `notebooks/week3_homework.py`.
# > Block 1 escalates directly from that homework's PyTorch baseline.
#
# ## Agenda (90 min)
#
# | Block | Min | Topic |
# |------:|:---:|:------|
# | 1 | ~10 | Recap + closed-form OLS via `torch.linalg.lstsq` |
# | 2 | ~12 | Optimizer zoo: GD vs SGD vs minibatch vs Newton vs L-BFGS |
# | 3 | ~10 | The loss is a noise-model choice (MSE / Huber / MAE on contaminated data) |
# | 4 | ~12 | Basis functions: polynomial → Runge → cubic spline |
# | 5 | ~12 | Three flavours of leakage on the same TensileTest data |
# | 6 | ~35 | Student exercises (3 core + 1 stretch) |

# %%
# Standard imports. Same style as week 2: explicit seeds, no implicit globals.
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

from ai4mat.datasets import TensileTestDataset

np.random.seed(0)
torch.manual_seed(0)


def standardise(z):
    """Return (z - mean) / std using torch ops; works on (N,) and (N, D) tensors."""
    mu  = z.mean(dim=0, keepdim=True)
    std = z.std(dim=0,  keepdim=True)
    return (z - mu) / (std + 1e-12), mu, std


# %% [markdown]
# # Block 1 — Recap from homework, then closed-form OLS
#
# In the homework you trained `nn.Linear(1, 1)` with manual SGD on the 600 °C tensile
# curve. Here we (a) reproduce that in 3 lines, (b) compute the analytic minimum of the
# MSE — the **closed-form OLS estimator** — and (c) confirm that SGD-with-enough-epochs
# converges to the same answer. This is a mini sanity check of the whole supervised-learning
# loop.
#
# *(see MFML §"supervised learning framework"; ML-PC §47 "putting it together")*

# %%
ds = TensileTestDataset(temperature=600)
X = ds.X.squeeze(1)        # (350,) strain (raw units, ~1e-2)
y = ds.y                   # (350,) stress (raw units, MPa, ~10)

# Standardise both before we compare optimizers. Otherwise SGD's single learning
# rate would have to handle a feature with std ~0.005 and a target with std ~10
# in the same step — which doesn't work without per-parameter preconditioning.
# Block 2 fits in the same standardised space, so this also keeps the two blocks
# consistent.
Xn, mu_x, sd_x = standardise(X)
yn, mu_y, sd_y = standardise(y)


# %%
# Closed-form OLS in standardised space, using design matrix [Xn, 1].
A = torch.stack([Xn, torch.ones_like(Xn)], dim=1)      # (350, 2)
result = torch.linalg.lstsq(A, yn.unsqueeze(1))        # solves min || A w - yn ||^2
w_ols = result.solution.squeeze()                      # (2,) -> [slope_std, intercept_std]
print(f"OLS closed form (std): slope = {w_ols[0]:7.4f}    intercept = {w_ols[1]:7.4f}")


# %%
# SGD as in the homework, trained long enough on the standardised data to converge.
torch.manual_seed(0)
model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

Xn_col = Xn.unsqueeze(1)
for epoch in range(2000):
    opt.zero_grad()
    yhat = model(Xn_col).squeeze(1)
    loss = loss_fn(yhat, yn)
    loss.backward()
    opt.step()

w_sgd = torch.tensor([model.weight.item(), model.bias.item()])
print(f"SGD after 2000 ep    : slope = {w_sgd[0]:7.4f}    intercept = {w_sgd[1]:7.4f}")
print(f"||w_sgd - w_ols||_2 = {(w_sgd - w_ols).norm().item():.6f}")


# %% [markdown]
# The two should agree to ~5 decimal places. SGD is *just* an iterative way to find
# what the closed form gives in one matrix solve. We will use this fact to benchmark
# every other optimizer in Block 2 (which also fits in standardised coordinates).
#
# **Note on the printed numbers.** The intercept is exactly `0.0000` because we
# standardised `y` so its sample mean is zero — the slope carries all the signal,
# and any non-zero intercept would just mean we mis-centred. The slope value
# (~0.51) is the Pearson correlation between standardised strain and standardised
# stress: less than 1 because a straight line through this curved stress–strain
# data leaves substantial unexplained variance. Block 4 will fix that with a
# better basis.
#
# **Why standardise?** With the raw stress range of ~30 MPa and the raw strain range
# of ~0.02, the design matrix is staggeringly ill-conditioned — full-batch GD on the
# raw data with any single learning rate moves the slope and intercept at vastly
# different speeds. Standardisation puts both features on the same scale so a single
# `lr` works for both. This is the same "always standardise before fitting" lesson
# from ML-PC §13–§14 — the cost of skipping it is not bias but **trainability**.
# %% [markdown]
# # Block 2 — Optimizer zoo on the same loss landscape
#
# Same data, same MSE, four optimizers:
#
# - **Full-batch GD** — uses every sample to compute one gradient.
# - **SGD** — single sample per step.
# - **Minibatch SGD** — the workhorse of modern deep learning.
# - **Newton's method** — uses second derivatives. For a linear model + MSE the
#   Hessian is constant and Newton lands on the OLS solution **in one step**.
# - **L-BFGS** — a quasi-Newton method that approximates the Hessian from gradient
#   history. PyTorch ships it as `torch.optim.LBFGS`.
#
# *(see MFML §"gradient descent" through §"Newton's method")*

# %%
# Reuse the standardised tensors Xn, yn from Block 1 -- same coordinate system.
A_n = torch.stack([Xn, torch.ones_like(Xn)], dim=1)         # design matrix in standardised coords

# Closed-form OLS in standardised coords -- the target every iterative method must hit.
w_target = torch.linalg.lstsq(A_n, yn.unsqueeze(1)).solution.squeeze()


def mse(w):
    return ((A_n @ w - yn) ** 2).mean()

def grad(w):
    """Analytic gradient of (1/N) || A w - y ||^2 = (2/N) A^T (Aw - y)."""
    N = A_n.shape[0]
    return (2.0 / N) * A_n.t() @ (A_n @ w - yn)

# The Hessian is constant for linear+MSE: (2/N) A^T A.
H = (2.0 / A_n.shape[0]) * A_n.t() @ A_n
H_inv = torch.linalg.inv(H)


# %%
# Run each optimizer for up to 200 steps; record ||w - w_target|| each step.
def run_full_batch_gd(eta, n_steps=200):
    w = torch.zeros(2)
    hist = []
    for _ in range(n_steps):
        w = w - eta * grad(w)
        hist.append((w - w_target).norm().item())
    return hist

def run_sgd(eta, n_steps=200):
    """Pure stochastic: one random sample per step."""
    rng = np.random.default_rng(0)
    w = torch.zeros(2)
    hist = []
    N = A_n.shape[0]
    for _ in range(n_steps):
        i = int(rng.integers(N))
        a_i = A_n[i:i+1]
        y_i = yn[i:i+1]
        g_i = 2.0 * a_i.t() @ (a_i @ w - y_i)
        w = w - eta * g_i.squeeze()
        hist.append((w - w_target).norm().item())
    return hist

def run_minibatch_sgd(eta, batch=32, n_steps=200):
    rng = np.random.default_rng(0)
    w = torch.zeros(2)
    hist = []
    N = A_n.shape[0]
    for _ in range(n_steps):
        idx = rng.choice(N, size=batch, replace=False)
        Ab, yb = A_n[idx], yn[idx]
        g_b = (2.0 / batch) * Ab.t() @ (Ab @ w - yb)
        w = w - eta * g_b
        hist.append((w - w_target).norm().item())
    return hist

def run_newton(n_steps=5):
    w = torch.zeros(2)
    hist = []
    for _ in range(n_steps):
        w = w - H_inv @ grad(w)
        hist.append((w - w_target).norm().item())
    return hist

def run_lbfgs(n_steps=20):
    w = torch.zeros(2, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=1, line_search_fn="strong_wolfe")
    hist = []
    def closure():
        opt.zero_grad()
        loss = mse(w)
        loss.backward()
        return loss
    for _ in range(n_steps):
        opt.step(closure)
        hist.append((w.detach() - w_target).norm().item())
    return hist


hist_gd  = run_full_batch_gd(eta=0.5,    n_steps=200)
hist_sgd = run_sgd(           eta=0.05,   n_steps=200)
hist_mb  = run_minibatch_sgd( eta=0.5,    batch=32, n_steps=200)
hist_nt  = run_newton(        n_steps=5)
hist_lb  = run_lbfgs(         n_steps=20)

plt.figure(figsize=(8, 4))
plt.semilogy(hist_gd,  label='full-batch GD')
plt.semilogy(hist_sgd, label='SGD (1 sample/step)')
plt.semilogy(hist_mb,  label='minibatch SGD (b=32)')
plt.semilogy(hist_nt,  'o-', label="Newton (5 steps)")
plt.semilogy(hist_lb,  's-', label="L-BFGS (20 steps)")
plt.xlabel("step"); plt.ylabel(r'$\| w - w_{OLS} \|_2$')
plt.title("All five optimizers chasing the same OLS minimum")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# %% [markdown]
# Reading the plot:
#
# - **Newton lands in one step** because the loss is a quadratic, so the local
#   second-order Taylor model is exact globally.
# - **L-BFGS** also gets there fast, in ≤ 5 outer iterations.
# - **Full-batch GD** decays geometrically — the rate is set by the condition number
#   of the Hessian. With standardisation that condition number is ~1, so GD is fast too.
# - **SGD** bounces around the minimum: the noise floor scales with $\eta$.
# - **Minibatch SGD** is the practical compromise — averages out single-sample noise
#   while still being one matrix-vector multiply per step.
#
# *(Forward-pointer: full optimization deep dive — momentum, Adam, conditioning — is MFML Unit 6.)*

# %% [markdown]
# # Block 3 — The loss function is a noise-model choice
#
# So far we have used MSE because the homework used MSE. But MSE corresponds to
# the assumption that the residuals $y - \hat y$ are **Gaussian** with constant
# variance. If they are not, MSE gives a **biased** estimator.
#
# The fastest way to see this: contaminate the dataset with a few outliers (sensor
# failures, mis-recorded specimens) and refit with three different losses.
#
# *(see MFML §"loss as decision proxy", §"MSE/MAE/Huber"; ML-PC §28 "bias-variance")*

# %%
# Build a contaminated copy: take the original 600 °C data and corrupt 3% of points.
torch.manual_seed(0)
X_clean = X.clone()
y_clean = y.clone()

n_outliers = max(1, int(0.03 * len(X_clean)))
outlier_idx = torch.randperm(len(X_clean))[:n_outliers]
y_dirty = y_clean.clone()
y_dirty[outlier_idx] += torch.empty(n_outliers).uniform_(300.0, 600.0)


# %%
# Use Adam for all three losses: it adapts per-parameter step sizes so a single
# `lr` works on the raw stress / strain scale (the design matrix is hopelessly
# ill-conditioned in raw units; see Block 1's "Why standardise?" sidebar).
# Adam is forward-pointed in MFML §"beyond vanilla SGD".
def fit_with_loss(loss_module, n_iters=2000, lr=3.0):
    torch.manual_seed(0)
    m = nn.Linear(1, 1)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xc = X_clean.unsqueeze(1)
    for _ in range(n_iters):
        opt.zero_grad()
        loss = loss_module(m(Xc).squeeze(1), y_dirty)
        loss.backward()
        opt.step()
    return m

m_mse   = fit_with_loss(nn.MSELoss())
# delta=10 sits just above the typical inlier residual scale (~5 MPa) and well
# below the outlier injections (300+ MPa). So inliers stay in the quadratic
# regime and outliers in the linear regime — exactly the split Huber is built
# to make. Picking delta is data-dependent: too small acts like MAE, too large
# acts like MSE.
m_huber = fit_with_loss(nn.HuberLoss(delta=10.0))
m_mae   = fit_with_loss(nn.L1Loss())

xs_grid = torch.linspace(X_clean.min(), X_clean.max(), 200).unsqueeze(1)
with torch.no_grad():
    yh_mse   = m_mse(xs_grid).squeeze().numpy()
    yh_huber = m_huber(xs_grid).squeeze().numpy()
    yh_mae   = m_mae(xs_grid).squeeze().numpy()

# Reference: OLS on the *clean* data — the "what would an outlier-free MSE give?" line.
A_c = torch.stack([X_clean, torch.ones_like(X_clean)], dim=1)
w_ref = torch.linalg.lstsq(A_c, y_clean.unsqueeze(1)).solution.squeeze()
yh_ref = (w_ref[0] * xs_grid.squeeze() + w_ref[1]).numpy()

print(f"slope (clean OLS):   {w_ref[0]:7.2f}")
print(f"slope (MSE on dirty):  {m_mse.weight.item():7.2f}   <- pulled UP by outliers")
print(f"slope (Huber on dirty):{m_huber.weight.item():7.2f}   <- much closer to clean OLS than MSE; still slightly biased")
print(f"slope (MAE on dirty):  {m_mae.weight.item():7.2f}   <- median, not mean — different story (see take-away)")


# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X_clean.numpy(), y_dirty.numpy(), s=10, alpha=0.5, label='dirty data')
ax[0].plot(xs_grid.numpy().squeeze(), yh_ref,   'k--', lw=1.5, label='OLS on CLEAN data')
ax[0].plot(xs_grid.numpy().squeeze(), yh_mse,   label='MSE')
ax[0].plot(xs_grid.numpy().squeeze(), yh_huber, label='Huber')
ax[0].plot(xs_grid.numpy().squeeze(), yh_mae,   label='MAE (L1)')
ax[0].set_xlabel('strain'); ax[0].set_ylabel('stress'); ax[0].legend(); ax[0].set_title('three losses, same dirty data')

# Residual histograms — clip the x-axis to the inlier range so the outlier tail
# doesn't squash everything into one bar at zero. The shifts among MSE / Huber /
# MAE are visible only in the inlier band.
with torch.no_grad():
    r_mse   = (y_dirty - m_mse(X_clean.unsqueeze(1)).squeeze(1)).numpy()
    r_huber = (y_dirty - m_huber(X_clean.unsqueeze(1)).squeeze(1)).numpy()
    r_mae   = (y_dirty - m_mae(X_clean.unsqueeze(1)).squeeze(1)).numpy()
inlier_bins = np.linspace(-25, 25, 50)
ax[1].hist(r_mse,   bins=inlier_bins, alpha=0.5, label='MSE residuals')
ax[1].hist(r_huber, bins=inlier_bins, alpha=0.5, label='Huber residuals')
ax[1].hist(r_mae,   bins=inlier_bins, alpha=0.5, label='MAE residuals')
ax[1].axvline(0, color='k', lw=0.5, alpha=0.5)
ax[1].set_xlim(-25, 25)
ax[1].set_xlabel('residual (MPa, inlier band)'); ax[1].set_ylabel('count')
ax[1].legend(); ax[1].set_title(f'residual distributions ({n_outliers} outlier residuals at ~+300..+500 not shown)')
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the picture.** Three regimes:
#
# - **MSE drifts noticeably toward the outliers** — because it is the negative
#   log-likelihood of a Gaussian, and Gaussians have no heavy tail to absorb
#   them. Each outlier contributes a *quadratic* penalty that the optimizer pays
#   by tilting the line.
# - **Huber stays close to "OLS on clean data"** — the dashed line. Huber tells
#   you "treat small residuals as squared, large residuals as absolute" — which
#   is exactly what robust statistics has said since the 1960s. This is the
#   right move when you trust most of your data and want one or two specimens
#   to not dictate the fit.
# - **MAE goes somewhere different on its own.** L1 regression solves for the
#   *median residual*, not the mean — and on a curved stress–strain plot the
#   median line is dominated by the dense cloud of low-strain low-stress points,
#   giving a much shallower slope. So MAE is "robust to outliers" *and* "robust
#   to the curvature being wrong" — it solves a different problem from MSE.
#
# **Decision rule for the semester:** plot the residuals. Heavy tails → drop MSE
# in favour of Huber. Want central tendency, not mean? Use MAE. Want the
# Gaussian assumption to actually hold? Use MSE — and check the residuals.


# %% [markdown]
# # Block 4 — Basis functions: polynomial → Runge → cubic spline
#
# `nn.Linear(1, 1)` cannot fit the post-yield work-hardening tail. Time for the
# linearity-in-parameters trick from MFML: replace strain $x$ with a vector of
# basis functions $\phi(x) = [\phi_0(x), \phi_1(x), \dots, \phi_{p-1}(x)]^\top$,
# then fit $\hat y = w^\top \phi(x)$.
#
# Two bases:
# 1. **Polynomial:** $\phi_k(x) = x^k$. Easy to write, but high degrees suffer from
#    **Runge's phenomenon** — wild oscillations near data boundaries.
# 2. **Cubic B-spline:** piecewise cubics joined smoothly at *knots*. Locally
#    flexible, globally well-behaved.
#
# *(see MFML §"linearity principle", §"Runge's phenomenon", §"splines")*

# %%
def polynomial_basis(x, degree):
    """Returns the design matrix [1, x, x^2, ..., x^degree], shape (N, degree+1)."""
    cols = [torch.ones_like(x)]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    return torch.stack(cols, dim=1)

def cubic_bspline_basis(x_np, n_knots):
    """Returns a clamped cubic B-spline design matrix, shape (N, n_basis_functions).

    Uses scipy for the basis construction; the regression itself stays in torch.
    """
    k = 3   # cubic
    knots_inner = np.linspace(x_np.min(), x_np.max(), n_knots)[1:-1]
    knots = np.concatenate(([x_np.min()] * (k + 1), knots_inner, [x_np.max()] * (k + 1)))
    n_basis = len(knots) - k - 1
    cols = []
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spline = BSpline(knots, c, k, extrapolate=False)
        cols.append(np.nan_to_num(spline(x_np)))
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)

def fit_basis(Phi, y):
    """Closed-form least squares: w = (Phi^T Phi)^-1 Phi^T y."""
    return torch.linalg.lstsq(Phi, y.unsqueeze(1)).solution.squeeze()


# %%
# Sort once for clean line plots.
order = torch.argsort(X)
X_sorted = X[order]
y_sorted = y[order]
xs_grid = torch.linspace(X_sorted.min(), X_sorted.max(), 400)

degrees   = [1, 5, 15]
n_knots_s = [4, 8, 16]

fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharey=True)

for j, d in enumerate(degrees):
    Phi      = polynomial_basis(X_sorted, d)
    Phi_grid = polynomial_basis(xs_grid,  d)
    w        = fit_basis(Phi, y_sorted)
    yhat     = (Phi_grid @ w).numpy()
    ax[0, j].scatter(X_sorted.numpy(), y_sorted.numpy(), s=8, alpha=0.5)
    ax[0, j].plot(xs_grid.numpy(), yhat, 'r-', lw=2)
    ax[0, j].set_title(f"polynomial degree {d}")
    ax[0, j].set_xlabel("strain"); ax[0, 0].set_ylabel("stress")

for j, n in enumerate(n_knots_s):
    Phi      = cubic_bspline_basis(X_sorted.numpy(), n)
    Phi_grid = cubic_bspline_basis(xs_grid.numpy(),  n)
    w        = fit_basis(Phi, y_sorted)
    yhat     = (Phi_grid @ w).numpy()
    ax[1, j].scatter(X_sorted.numpy(), y_sorted.numpy(), s=8, alpha=0.5)
    ax[1, j].plot(xs_grid.numpy(), yhat, 'g-', lw=2)
    ax[1, j].set_title(f"cubic B-spline, {n} knots")
    ax[1, j].set_xlabel("strain"); ax[1, 0].set_ylabel("stress")

plt.suptitle("Polynomial bases (top) wiggle at high degree; spline bases (bottom) stay sane", y=1.02)
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the picture.** A polynomial of degree 15 over-fits the noise *and*
# develops the Runge oscillation at the boundaries. A cubic spline with 16 knots
# uses about the same number of parameters but stays locally smooth — because
# each knot only influences its neighbourhood, the model can be flexible *where
# the data needs it* and rigid elsewhere.
#
# *Take-away:* "more parameters" is a bad summary of model capacity. *Where* the
# parameters can flex matters more.
