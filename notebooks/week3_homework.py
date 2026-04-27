# %% [markdown]
# # Week 3 — Homework (do BEFORE the Thursday exercise)
#
# This notebook is the **mandatory warm-up** for the Week 3 in-class exercise. Working
# through it gets the `ai4mat` install pain out of the way, gives everyone the same
# PyTorch baseline to talk about, and primes you for the basis-functions story we
# unfold in class.
#
# **Time:** ~75 minutes.
#
# ## What this homework is
#
# Three short workouts, all anchored on the same idea:
#
# > **A "linear" model is linear in its parameters, not in its features.** Polynomial,
# > spline, Fourier, and wavelet models all live inside the same loss-minimisation
# > framework — they only differ in which *basis functions* you choose.
#
# | Part | Min | Topic | Lecture anchor |
# |---|---:|---|---|
# | A | 30 | PyTorch baseline on a real materials dataset | MFML §"supervised learning framework" |
# | B | 25 | FFT — frequency content as a basis | ML-PC §18 |
# | C | 15 | Wavelets — when the basis must be local | ML-PC §19 |
# | D | 5  | Reflection paragraph linking A→B→C | bridge to Thursday's class |
#
# ## What you must hand in (or be able to show on Thursday)
#
# 1. Train/val MSE curves from Part A.
# 2. Spectrum + reconstruction figure from Part B.
# 3. Scalogram + FFT comparison figure from Part C.
# 4. Your written answer to the reflection question (Part D).

# %%
# Standard imports for the whole homework. Same seeds idiom as week 2.
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from ai4mat.datasets import TensileTestDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# # Part A — PyTorch baseline on TensileTestDataset
#
# We fit a 1-feature linear model `stress = w * strain + b` to a tensile-test
# stress–strain curve at 600 °C. The dataset is small (350 samples), the model has
# 2 parameters, and the loss is MSE. Everything is intentionally minimal — this is
# the **strawman** that the in-class notebook will then break apart and improve.

# %%
# Load the data and look at it.
dataset = TensileTestDataset(temperature=600)
print(f"Dataset size: {len(dataset)}")
x0, y0 = dataset[0]
print(f"Sample 0: strain={float(x0):.4f}  stress={float(y0):.2f} MPa")

X_all = torch.stack([dataset[i][0] for i in range(len(dataset))]).squeeze(1)  # (N,)
y_all = torch.stack([dataset[i][1] for i in range(len(dataset))])             # (N,)

plt.figure(figsize=(6, 3.5))
plt.scatter(X_all.numpy(), y_all.numpy(), s=8, alpha=0.5)
plt.xlabel("strain"); plt.ylabel("stress (MPa)")
plt.title("Tensile test, 600 °C — what we are about to fit a line to")
plt.tight_layout(); plt.show()


# %% [markdown]
# Notice the curve is *not* a line — it has an elastic regime, a yield knee, and a
# work-hardening tail. A linear model is *wrong* for this data, on purpose. Part A
# establishes the wrong baseline; the in-class notebook will fix it with bases that
# can express curvature.

# %%
# Train/val split with a fixed generator so the split is reproducible.
gen = torch.Generator().manual_seed(0)
n_train = int(0.8 * len(dataset))
n_val   = len(dataset) - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  generator=torch.Generator().manual_seed(0))
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
print(f"train={n_train}  val={n_val}")


# %%
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


model = LinearModel()
print(model)
print("params:", sum(p.numel() for p in model.parameters()))


# %%
# Manual training loop — no Lightning, no Trainer, on purpose.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
n_epochs = 100

train_losses, val_losses = [], []
for epoch in range(n_epochs):
    model.train()
    epoch_train = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train += loss.item() * len(x_batch)
    train_losses.append(epoch_train / n_train)

    model.eval()
    with torch.no_grad():
        epoch_val = 0.0
        for x_batch, y_batch in val_loader:
            epoch_val += criterion(model(x_batch), y_batch).item() * len(x_batch)
        val_losses.append(epoch_val / n_val)


# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(train_losses, label="train"); ax[0].plot(val_losses, label="val")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE"); ax[0].legend(); ax[0].set_title("training curves")
ax[0].grid(alpha=0.3)

with torch.no_grad():
    xs_grid = torch.linspace(X_all.min(), X_all.max(), 200).unsqueeze(1)
    y_pred = model(xs_grid)
ax[1].scatter(X_all.numpy(), y_all.numpy(), s=8, alpha=0.4, label="data")
ax[1].plot(xs_grid.numpy().squeeze(), y_pred.numpy(), 'r-', lw=2, label="linear fit")
ax[1].set_xlabel("strain"); ax[1].set_ylabel("stress"); ax[1].legend()
ax[1].set_title("the wrong-on-purpose baseline")
plt.tight_layout(); plt.show()

print(f"final train MSE = {train_losses[-1]:.2f}")
print(f"final val   MSE = {val_losses[-1]:.2f}")


# %% [markdown]
# Hold on to the train/val MSE numbers above — Block 1 of the in-class notebook
# will check that closed-form OLS lands on the same answer to within numerical
# precision.
