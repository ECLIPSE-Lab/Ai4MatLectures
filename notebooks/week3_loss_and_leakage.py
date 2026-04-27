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
