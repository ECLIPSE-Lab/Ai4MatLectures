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
