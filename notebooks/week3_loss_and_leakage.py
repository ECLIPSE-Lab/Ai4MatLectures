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
X = ds.X.squeeze(1)        # (350,) strain
y = ds.y                   # (350,) stress

# Closed-form OLS using the design matrix [strain, 1].
A = torch.stack([X, torch.ones_like(X)], dim=1)        # (350, 2)
result = torch.linalg.lstsq(A, y.unsqueeze(1))         # solves min || A w - y ||^2
w_ols = result.solution.squeeze()                      # (2,) -> [slope, intercept]
print(f"OLS closed form:  slope = {w_ols[0]:8.3f}    intercept = {w_ols[1]:8.3f}")


# %%
# SGD as in the homework, but trained long enough to converge to OLS.
torch.manual_seed(0)
model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

X_col = X.unsqueeze(1)
for epoch in range(2000):
    opt.zero_grad()
    yhat = model(X_col).squeeze(1)
    loss = loss_fn(yhat, y)
    loss.backward()
    opt.step()

w_sgd = torch.tensor([model.weight.item(), model.bias.item()])
print(f"SGD after 2000 ep: slope = {w_sgd[0]:8.3f}    intercept = {w_sgd[1]:8.3f}")
print(f"||w_sgd - w_ols||_2 = {(w_sgd - w_ols).norm().item():.4f}")


# %% [markdown]
# The two should agree to a couple of decimal places. SGD is *just* an iterative way
# to find what the closed form gives in one matrix solve. We will use this fact to
# benchmark every other optimizer in Block 2.
