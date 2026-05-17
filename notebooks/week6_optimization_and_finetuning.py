# %% [markdown]
# # Week 6 — Optimization and fine-tuning
#
# This week we braid three lectures around a single decision: **what does
# the optimizer do?**
#
# 1. **MFML Unit 6**: Optimization for deep learning — SGD, momentum, Adam,
#    ill-conditioning, learning-rate schedules, batch effects.
# 2. **ML-PC Unit 6**: Transfer learning *as continued optimization* on a
#    related loss landscape — catastrophic forgetting, discriminative LRs,
#    cosine schedules + warm-up, why Adam often hurts during fine-tuning.
# 3. **MG Unit 5**: Graph-based crystal representations — when the input is
#    a *graph* of atoms instead of an image, the same optimizer choices
#    matter, but the failure modes change.
#
# **Red thread:** *The optimizer is one toolbox. Today we apply it to two
# inputs — a 64×64 phase-field image being fine-tuned across distributions
# (Ising → Cahn–Hilliard) and a small crystal graph being trained from
# scratch. Each block takes one MFML W6 concept and shows it materialising
# in one of those two settings.*
#
# > **Pre-flight check.** This notebook **assumes** you have run
# > `notebooks/week6_homework.py`. Block 1 picks up directly from your Part A
# > trajectories and Part C optimizer table; we will not re-derive them.
#
# ## Agenda (90 min)
#
# | Block | Min | Topic |
# |------:|:---:|:------|
# | 1 | ~6  | Recap from homework — optimizer trajectories + element baseline |
# | 2 | ~14 | Catastrophic forgetting: Ising → Cahn–Hilliard with too-high LR |
# | 3 | ~12 | Discriminative learning rates: backbone vs head |
# | 4 | ~14 | Cosine schedule + warm-up rescues fine-tuning |
# | 5 | ~12 | Modern optimizer bake-off — AdamW vs Adam+L2 vs Lion vs Schedule-Free AdamW |
# | 6 | ~12 | Crystal graphs: a tiny hand-rolled message-passing GNN |
# | 7 | ~22 | Student exercises (3 core + 1 stretch) |

# %%
# Standard imports. Same idiom as weeks 2-5: explicit seeds, no hidden state.
import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt

from ai4mat.datasets import (
    IsingDataset, CahnHilliardDataset, CrystalGraphsDataset,
)

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# ## Helpers used by every block
#
# We use the *same* CNN backbone as in homework Part B but at 64×64 (Ising
# *full*, not light) so it has enough capacity to be interesting under
# fine-tuning. The architecture has two halves — a convolutional **backbone**
# and a small **head**. The split lets us freeze, unfreeze, and re-LR each
# half independently.

# %%
class CNNBackbone(nn.Module):
    """Two conv blocks ending in 32 channels at 16x16."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))     # (B, 32, 16, 16)
        return x


class ClassifierHead(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, feats):
        h = F.relu(self.fc1(feats.flatten(1)))
        return self.fc2(h)


class CNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = ClassifierHead(n_classes=n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def evaluate(model, loader):
    model.eval()
    n_correct = n_total = 0
    with torch.no_grad():
        for xb, yb in loader:
            n_correct += (model(xb).argmax(1) == yb).sum().item()
            n_total += yb.numel()
    return n_correct / n_total


# %% [markdown]
# # Block 1 — Recap from homework
#
# Two takeaways from the homework that frame the rest of the lecture:
#
# 1. **From Part A (Rosenbrock):** vanilla SGD crawls; momentum builds
#    velocity along the valley axis; Adam normalizes the step per axis.
# 2. **From Part C (ChemicalElementsDataset):** on a 4-feature regressor
#    with ~27 training examples, all three optimizers reach roughly the
#    same accuracy. The *descriptor* (composition-only, MG U6 ladder bottom
#    rung) carries the signal. The optimizer's job was easy.
#
# Today the optimizer's job will not be easy. Two ways the input gets
# harder: (a) **the loss landscape changes** when we fine-tune (Blocks 2–5);
# (b) **the input is a graph of atoms**, not a 4-vector or an image (Block 6).

# %%
# Build a *source-task* CNN: Ising-full classification, trained from scratch
# in ~30 seconds on CPU.  We will reuse this exact set of weights as the
# starting point for every fine-tuning experiment in Blocks 2-5.
ising = IsingDataset(size="full")
n_train = int(0.8 * len(ising))
ising_train, ising_test = random_split(
    ising, [n_train, len(ising) - n_train],
    generator=torch.Generator().manual_seed(0),
)
ising_train_loader = DataLoader(ising_train, batch_size=64, shuffle=True)
ising_test_loader = DataLoader(ising_test, batch_size=256)

torch.manual_seed(0)
source_model = CNN(n_classes=2)
optim_ = torch.optim.SGD(source_model.parameters(), lr=0.03, momentum=0.9)

print("Pretraining source model on Ising-full (3 epochs, ~30 s on CPU)...")
for epoch in range(3):
    source_model.train()
    for xb, yb in ising_train_loader:
        optim_.zero_grad()
        F.cross_entropy(source_model(xb), yb).backward()
        optim_.step()
print(f"  source-task test acc on Ising-full: {evaluate(source_model, ising_test_loader):.3f}")

# Freeze a deep copy of the trained weights so every fine-tuning experiment
# starts from the *exact same* initialization.  This is the moral equivalent
# of loading a checkpoint.
SOURCE_STATE = copy.deepcopy(source_model.state_dict())


# %%
# Build a *target-task* dataset: Cahn-Hilliard with the free energy
# binarised at its median.  64x64 grayscale, two classes — same shape and
# task type as Ising, but a *different* generative process.  This is the
# "related but not identical" distribution shift that makes fine-tuning
# well-defined.
print("Loading Cahn-Hilliard (3 simulations, ~3000 samples)...")
ch_full = CahnHilliardDataset(simulation_number=[0, 1, 2])
print(f"  loaded {len(ch_full)} samples; energy range = [{ch_full.y.min():.3g}, {ch_full.y.max():.3g}]")

ch_y_bin = (ch_full.y > ch_full.y.median()).long()


class CahnHilliardBinary(torch.utils.data.Dataset):
    def __init__(self, base, y_bin, idx):
        self.base = base
        self.y_bin = y_bin
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return self.base.X[j], self.y_bin[j]


# Use a small subset (1500 train, 500 test) for fast iteration during the lecture.
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(ch_full), generator=g)[:2000]
ch_train_idx, ch_test_idx = perm[:1500], perm[1500:2000]
ch_train = CahnHilliardBinary(ch_full, ch_y_bin, ch_train_idx)
ch_test = CahnHilliardBinary(ch_full, ch_y_bin, ch_test_idx)
ch_train_loader = DataLoader(ch_train, batch_size=64, shuffle=True)
ch_test_loader = DataLoader(ch_test, batch_size=256)
print(f"  fine-tuning subset: {len(ch_train)} train, {len(ch_test)} test")


# %% [markdown]
# # Block 2 — Catastrophic forgetting at high LR
#
# Naive transfer learning recipe: load source weights, swap the head, train
# on the target task. We do exactly that, with a *deliberately too-high*
# learning rate, and watch *both* test accuracies — source (Ising) and
# target (Cahn-Hilliard) — across the fine-tune.
#
# What you should see: the source-task accuracy collapses within the first
# 1-2 epochs while the target accuracy is still climbing. The model is
# being yanked out of the basin that solves Ising and dragged into a basin
# that solves Cahn-Hilliard. From the optimizer's point of view this is
# **SGD on a non-stationary loss** — yesterday's loss is forgotten.
#
# *(see ML-PC §"Catastrophic forgetting = SGD on a non-stationary loss",
# §"Why fine-tuning needs careful optimization")*

# %%
def fine_tune(state_dict, lr_backbone, lr_head, n_epochs=4,
              optimizer_cls=torch.optim.SGD, scheduler=None,
              freeze_backbone=False, momentum=0.9):
    """Fine-tune source weights on Cahn-Hilliard.  Track BOTH accuracies."""
    torch.manual_seed(0)
    model = CNN(n_classes=2)
    model.load_state_dict(state_dict)

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        param_groups = [{"params": model.head.parameters(), "lr": lr_head}]
    else:
        param_groups = [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.head.parameters(),     "lr": lr_head},
        ]

    if optimizer_cls is torch.optim.SGD:
        optim_ = optimizer_cls(param_groups, momentum=momentum)
    else:
        optim_ = optimizer_cls(param_groups)

    sched = scheduler(optim_) if scheduler is not None else None
    src_acc, tgt_acc = [], []
    for ep in range(n_epochs):
        model.train()
        for xb, yb in ch_train_loader:
            optim_.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            optim_.step()
            if sched is not None:
                sched.step()
        src_acc.append(evaluate(model, ising_test_loader))
        tgt_acc.append(evaluate(model, ch_test_loader))
    return np.array(src_acc), np.array(tgt_acc)


# %%
# Naive too-high-LR fine-tune.  Watch source acc collapse.
src_naive, tgt_naive = fine_tune(SOURCE_STATE, lr_backbone=0.1, lr_head=0.1)
print(f"Naive (lr=0.1, full unfreeze):  source={src_naive[-1]:.3f}  target={tgt_naive[-1]:.3f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
ep = np.arange(1, len(src_naive) + 1)
ax.plot(ep, src_naive, "o-", label="source (Ising)", c="C3")
ax.plot(ep, tgt_naive, "o-", label="target (Cahn-H)", c="C0")
ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
ax.set_ylim(0.4, 1.05); ax.set_title("Block 2 — naive fine-tune at too-high LR")
ax.legend(); plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the plot.** Source accuracy starts near 1.0 (the source-task
# checkpoint), then collapses to chance (~0.5) within 1-2 epochs as
# backbone weights drift. Target accuracy climbs but slowly. This is the
# textbook *catastrophic-forgetting* curve.
#
# Two ways out — both standard ML-PC W6 fixes — explored next.

# %% [markdown]
# # Block 3 — Discriminative learning rates
#
# **Classical recipe (ULMFiT, Howard & Ruder 2018):** lower LR on the
# backbone than on the head, because the backbone *already knows* something
# useful (Ising textures = Cahn-Hilliard textures, roughly) and only needs
# fine refinement; the head is fresh and can take big steps. We compare
# three protocols at the same total budget:
#
# 1. **Frozen backbone** — head only, LR 1e-2.
# 2. **Discriminative LRs** — backbone 1e-4, head 1e-2 (factor 100).
# 3. **Full unfreezing at uniform LR** — same LR everywhere, 1e-3.
#
# *(see ML-PC §"Layer-wise / discriminative learning rates")*

# %%
src_frozen, tgt_frozen = fine_tune(
    SOURCE_STATE, lr_backbone=0.0, lr_head=1e-2,
    n_epochs=4, freeze_backbone=True,
)
src_disc, tgt_disc = fine_tune(
    SOURCE_STATE, lr_backbone=1e-4, lr_head=1e-2, n_epochs=4,
)
src_uni, tgt_uni = fine_tune(
    SOURCE_STATE, lr_backbone=1e-3, lr_head=1e-3, n_epochs=4,
)

print("Block 3 — three protocols at fixed budget:")
print(f"  frozen backbone:        source={src_frozen[-1]:.3f}  target={tgt_frozen[-1]:.3f}")
print(f"  discriminative LRs:     source={src_disc[-1]:.3f}  target={tgt_disc[-1]:.3f}")
print(f"  uniform LR (small):     source={src_uni[-1]:.3f}  target={tgt_uni[-1]:.3f}")


# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
ep = np.arange(1, 5)
for src, label, c in [(src_frozen, "frozen", "C0"),
                      (src_disc,   "discr.", "C1"),
                      (src_uni,    "uniform 1e-3", "C2"),
                      (src_naive,  "naive 1e-1 (B2)", "C3")]:
    a1.plot(ep, src, "o-", label=label, c=c)
a1.set_title("Source (Ising) accuracy retention")
a1.set_xlabel("epoch"); a1.set_ylabel("source-task acc")
a1.set_ylim(0.4, 1.05); a1.legend(fontsize=9)

for tgt, label, c in [(tgt_frozen, "frozen", "C0"),
                      (tgt_disc,   "discr.", "C1"),
                      (tgt_uni,    "uniform 1e-3", "C2"),
                      (tgt_naive,  "naive 1e-1 (B2)", "C3")]:
    a2.plot(ep, tgt, "o-", label=label, c=c)
a2.set_title("Target (Cahn-Hilliard) accuracy")
a2.set_xlabel("epoch"); a2.set_ylabel("target-task acc")
a2.set_ylim(0.4, 1.05); a2.legend(fontsize=9)
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the comparison.** Frozen backbone preserves source accuracy
# perfectly (we never touched it) but caps the target accuracy where the
# *unchanged* features happen to land. Discriminative LRs preserve most
# of the source-task knowledge while letting the target task improve
# beyond the frozen ceiling. Uniform-small-LR is a workable compromise
# that needs no per-group bookkeeping.
#
# **Take-home.** Discriminative LRs are an *implementation* of the prior
# we want: "the backbone is approximately right; nudge it; the head is
# fresh; train it normally."

# %% [markdown]
# # Block 4 — Cosine schedule + warm-up rescues a high-LR fine-tune
#
# The naive Block-2 collapse came from one decision: **constant LR at the
# source-task value**. The fix is mechanical, not magical: anneal the LR
# along a cosine, with a short warm-up at the start so the optimizer does
# not take a huge first step into a fresh head.
#
# *(see MFML §"Cosine annealing", §"Warm-up", ML-PC §"Cosine schedules /
# warm-up for fine-tuning")*

# %%
def cosine_with_warmup(optim_, n_warmup_steps, n_total_steps, lr_max_per_group):
    """Custom step-level scheduler: linear warm-up then cosine to ~0."""
    step = {"i": 0}

    class Sched:
        def step(self_):
            i = step["i"]
            for g, lr_max in zip(optim_.param_groups, lr_max_per_group):
                if i < n_warmup_steps:
                    factor = (i + 1) / n_warmup_steps
                else:
                    frac = (i - n_warmup_steps) / max(1, n_total_steps - n_warmup_steps)
                    factor = 0.5 * (1 + math.cos(math.pi * frac))
                g["lr"] = lr_max * factor
            step["i"] = i + 1

    return Sched()


N_EPOCHS_FT = 4
N_STEPS_FT = N_EPOCHS_FT * len(ch_train_loader)
LR_MAX_BB, LR_MAX_HEAD = 1e-2, 1e-2

src_warm, tgt_warm = fine_tune(
    SOURCE_STATE,
    lr_backbone=LR_MAX_BB, lr_head=LR_MAX_HEAD,
    n_epochs=N_EPOCHS_FT,
    scheduler=lambda opt: cosine_with_warmup(
        opt, n_warmup_steps=N_STEPS_FT // 10,
        n_total_steps=N_STEPS_FT,
        lr_max_per_group=[LR_MAX_BB, LR_MAX_HEAD],
    ),
)
print(f"Cosine + warm-up at lr_max=1e-2:  source={src_warm[-1]:.3f}  target={tgt_warm[-1]:.3f}")

# Same model, same lr_max, *no* schedule — for the comparison.
src_const, tgt_const = fine_tune(
    SOURCE_STATE, lr_backbone=LR_MAX_BB, lr_head=LR_MAX_HEAD,
    n_epochs=N_EPOCHS_FT,
)
print(f"Constant lr=1e-2 (no schedule):   source={src_const[-1]:.3f}  target={tgt_const[-1]:.3f}")


# %%
fig, ax = plt.subplots(figsize=(6, 3.5))
ep = np.arange(1, N_EPOCHS_FT + 1)
ax.plot(ep, src_const, "o-", label="constant — source", c="C0", ls=":")
ax.plot(ep, tgt_const, "o-", label="constant — target", c="C0")
ax.plot(ep, src_warm,  "o-", label="cosine+warm-up — source", c="C2", ls=":")
ax.plot(ep, tgt_warm,  "o-", label="cosine+warm-up — target", c="C2")
ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.set_ylim(0.4, 1.05)
ax.set_title("Block 4 — schedule rescues source-task retention at lr_max=1e-2")
ax.legend(fontsize=8, loc="lower right"); plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the rescue plot.** At the *same* lr_max, cosine + warm-up keeps
# more source-task accuracy and reaches a similar target-task accuracy as
# the constant-LR baseline. The schedule is doing the work of the
# discriminative LR from Block 3 — except it does it *over time* rather
# than *across parameter groups*. In practice you usually combine both.

# %% [markdown]
# # Block 5 — Modern optimizer bake-off: AdamW vs Adam+L2 vs Lion vs Schedule-Free AdamW
#
# Three years ago this block would have been "Adam vs SGD-with-momentum"
# and the take-home would have been *"Adam is the obvious default."* In
# 2026 the landscape has shifted. The production default is now **AdamW**
# (decoupled weight decay), with **Lion** [@chen_2023_lion] gaining
# adoption when optimizer-state memory is the bottleneck, and
# **Schedule-Free AdamW** removing the LR schedule entirely. We retire the
# SGD-vs-Adam comparison and run a 4-way bake-off on the small
# `ChemicalElementsDataset` regressor — the same model the homework Part C
# already touched, but with the *modern* optimizer roster:
#
# 1. **AdamW** — Adam with *decoupled* weight decay applied directly to
#    the parameters (`p ← p − lr · wd · p` outside the gradient step).
#    State per parameter: `(m, v)` → 2 tensors.
# 2. **Adam + L2** — vanilla Adam with the L2 penalty added to the *loss*
#    (`loss + wd · Σ p²`). Same nominal regularisation, but the penalty
#    gets divided by Adam's variance estimate, which is *not* the same.
#    State per parameter: `(m, v)` → 2 tensors.
# 3. **Lion** [@chen_2023_lion] — sign-of-momentum update,
#    `p ← p − lr · sign(β₁·m + (1−β₁)·g)`. Halves the optimizer state
#    (1 tensor instead of 2) and tends to need ~3-10× smaller LR.
# 4. **Schedule-Free AdamW** — eliminates the LR schedule. Couples a
#    running-average iterate `z` with the gradient point `y`, removing
#    the bias-variance trade-off of choosing a cosine length up front.
#
# **Predicted headline result.** Lion roughly matches AdamW's validation
# loss with about **half** the optimizer-state memory; AdamW beats Adam+L2
# by a hair — the *decoupling effect* is small on a small model but real.
# Schedule-Free AdamW lands close to AdamW + cosine without any schedule
# tuning, which is its main appeal in practice.
#
# We also mention **Sophia** (Liu et al. 2023) for completeness: a
# second-order optimizer using a diagonal Hessian estimate. It is a useful
# data point in the lecture slide but we do not benchmark it here — the
# Hessian estimator adds enough code to outshadow the pedagogical point.
#
# *(see MFML §"AdamW: decoupled weight decay", §"Modern alternatives to
# AdamW (2023–2024)"; ML-PC §"Optimizer state memory in fine-tuning")*

# %%
# We sweep a small LR grid per optimizer (3 LRs) and pick the best by
# final validation loss.  Everything else is held fixed: same model, same
# data, same seed, same epoch count.

from ai4mat.datasets import ChemicalElementsDataset
import time

elements_b5 = ChemicalElementsDataset()
X_b5 = elements_b5.X
X_b5 = (X_b5 - X_b5.mean(0)) / X_b5.std(0)
y_b5 = elements_b5.y

g = torch.Generator().manual_seed(0)
perm_b5 = torch.randperm(len(X_b5), generator=g)
n_tr_b5 = int(0.7 * len(X_b5))
tr_b5, te_b5 = perm_b5[:n_tr_b5], perm_b5[n_tr_b5:]
X_tr_b5, y_tr_b5 = X_b5[tr_b5], y_b5[tr_b5]
X_te_b5, y_te_b5 = X_b5[te_b5], y_b5[te_b5]


class TinyMLP_B5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# %%
# Hand-rolled Lion — ~20 lines, no extra dependency.
# Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms",
# NeurIPS 2023 [@chen_2023_lion].
class Lion(torch.optim.Optimizer):
    """Sign-of-momentum optimizer.

    Update rule (per parameter):
        u = sign(beta1 * m + (1 - beta1) * g)
        p <- p - lr * (u + wd * p)              # decoupled weight decay
        m <- beta2 * m + (1 - beta2) * g
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                update = torch.sign(b1 * m + (1.0 - b1) * g)
                # Decoupled weight decay (same trick as AdamW)
                p.add_(update + wd * p, alpha=-lr)
                # Update momentum AFTER the parameter step (Lion convention)
                m.mul_(b2).add_(g, alpha=1.0 - b2)


# %%
# Schedule-Free AdamW: prefer the pip package if installed.  Otherwise
# fall back to a minimal inline implementation so the lecture cell runs.
# Reference: Defazio et al., "The Road Less Scheduled" (2024).
try:
    import schedulefree
    SF_AVAILABLE = True
    print("schedulefree package detected -- using the reference implementation.")
except ImportError:
    SF_AVAILABLE = False
    print("schedulefree not installed -- using inline minimal implementation.")
    print("  (install with: pip install schedulefree)")


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """Minimal inline Schedule-Free AdamW (Defazio et al. 2024).

    The idea: maintain two iterates, z (the gradient-step iterate) and
    x (the running average that is reported as the parameter). The trick
    is that no learning-rate schedule is required; an internal
    momentum-like averaging plays the same role.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, warmup_steps=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup_steps=warmup_steps)
        super().__init__(params, defaults)
        self.train()

    def train(self):
        # Swap params to the gradient point y for forward/backward.
        for group in self.param_groups:
            b1 = group["betas"][0]
            for p in group["params"]:
                st = self.state[p]
                if "z" in st:
                    p.data.mul_(b1).add_(st["z"], alpha=1.0 - b1)

    def eval(self):
        # Swap params to the running average x for evaluation.
        for group in self.param_groups:
            b1 = group["betas"][0]
            for p in group["params"]:
                st = self.state[p]
                if "z" in st:
                    p.data.add_(st["z"] - p.data, alpha=1.0 - b1)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]; b1, b2 = group["betas"]
            eps = group["eps"]; wd = group["weight_decay"]
            warmup = group["warmup_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "z" not in st:
                    st["z"] = p.data.clone()
                    st["v"] = torch.zeros_like(p)
                    st["k"] = 0
                st["k"] += 1
                k = st["k"]
                lr_t = lr * min(1.0, k / max(1, warmup)) if warmup > 0 else lr
                # AdamW-style second-moment running average on g (no first moment).
                st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                v_hat = st["v"] / (1.0 - b2 ** k)
                # Update z (gradient iterate) with decoupled weight decay.
                z = st["z"]
                z.addcdiv_(g, v_hat.sqrt().add_(eps), value=-lr_t)
                z.add_(p.data, alpha=-lr_t * wd)
                # Set p to a y = (1 - b1) z + b1 x convex combination.
                # We approximate x ≈ running mean by ck = 1/k weighting.
                ck = 1.0 / k
                # x <- (1 - ck) x + ck z, but x is implicit; we keep p as y.
                p.data.mul_(b1).add_(z, alpha=1.0 - b1)


# %% [markdown]
# **Bake-off setup.** Same architecture (`TinyMLP_B5`), same train/test
# split, same 200 full-batch epochs, same `torch.manual_seed(0)`. The only
# things that vary across runs are the optimizer class and its LR.

# %%
N_EPOCHS_B5 = 200
WD_B5 = 1e-2          # the same nominal weight decay applied everywhere


def train_b5(optimizer_factory, label, lr, n_states_per_param):
    """Train one TinyMLP_B5 full-batch for N_EPOCHS_B5.

    Returns: (final_val_loss, wall_time_per_epoch_s, optim_state_tensors).
    """
    torch.manual_seed(0)
    model = TinyMLP_B5()
    optim_ = optimizer_factory(model.parameters(), lr)
    n_params_tensors = sum(1 for _ in model.parameters())
    optim_state_tensors = n_params_tensors * n_states_per_param

    t0 = time.perf_counter()
    for ep in range(N_EPOCHS_B5):
        model.train()
        optim_.zero_grad()
        pred = model(X_tr_b5)
        # For Adam+L2 we add the explicit L2 penalty here.
        loss = F.binary_cross_entropy_with_logits(pred, y_tr_b5)
        if label.startswith("Adam+L2"):
            loss = loss + WD_B5 * sum(p.pow(2).sum() for p in model.parameters())
        loss.backward()
        optim_.step()
    wall = (time.perf_counter() - t0) / N_EPOCHS_B5

    model.eval()
    with torch.no_grad():
        val_loss = F.binary_cross_entropy_with_logits(model(X_te_b5), y_te_b5).item()
    return val_loss, wall, optim_state_tensors


# %%
# Per-optimizer LR grids (a small budget of 3 candidates each).
adamw_grid     = [3e-3, 1e-2, 3e-2]
adam_l2_grid   = [3e-3, 1e-2, 3e-2]
lion_grid      = [3e-4, 1e-3, 3e-3]      # Lion likes 3-10x smaller LR
sf_adamw_grid  = [3e-3, 1e-2, 3e-2]


def sweep(name, factory, grid, n_states):
    best = (float("inf"), None, None, None)
    for lr in grid:
        val, wall, st = train_b5(factory, name, lr, n_states)
        if val < best[0]:
            best = (val, lr, wall, st)
    return best   # (val_loss, lr_best, wall_per_ep, n_state_tensors)


print("Sweeping 4 optimizers x 3 LRs each on ChemicalElementsDataset...")

adamw_best = sweep(
    "AdamW",
    lambda p, lr: torch.optim.AdamW(p, lr=lr, weight_decay=WD_B5),
    adamw_grid, n_states=2,
)
adam_l2_best = sweep(
    "Adam+L2",
    lambda p, lr: torch.optim.Adam(p, lr=lr),            # L2 added in loss
    adam_l2_grid, n_states=2,
)
lion_best = sweep(
    "Lion",
    lambda p, lr: Lion(p, lr=lr, weight_decay=WD_B5),
    lion_grid, n_states=1,
)
if SF_AVAILABLE:
    sf_factory = lambda p, lr: schedulefree.AdamWScheduleFree(
        p, lr=lr, weight_decay=WD_B5,
    )
else:
    sf_factory = lambda p, lr: ScheduleFreeAdamW(p, lr=lr, weight_decay=WD_B5)
sf_best = sweep("SF-AdamW", sf_factory, sf_adamw_grid, n_states=2)


# %%
# Pretty-print the bake-off table.
print(f"\n{'optimizer':<14} {'best LR':>10} {'val loss':>10} "
      f"{'sec/epoch':>12} {'state tensors':>16}")
print("-" * 64)
for name, b in [("AdamW",     adamw_best),
                ("Adam+L2",   adam_l2_best),
                ("Lion",      lion_best),
                ("SF-AdamW",  sf_best)]:
    val, lr_best, wall, st = b
    print(f"{name:<14} {lr_best:>10.1e} {val:>10.4f} "
          f"{wall * 1000:>10.2f} ms {st:>16d}")


# %%
# Bar plot: validation loss + optimizer state.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))

names = ["AdamW", "Adam+L2", "Lion", "SF-AdamW"]
vals = [b[0] for b in (adamw_best, adam_l2_best, lion_best, sf_best)]
sts = [b[3] for b in (adamw_best, adam_l2_best, lion_best, sf_best)]

ax1.bar(names, vals, color=["C0", "C3", "C2", "C1"])
ax1.set_ylabel("final val loss (BCE)")
ax1.set_title("Block 5 — val loss across modern optimizers")

ax2.bar(names, sts, color=["C0", "C3", "C2", "C1"])
ax2.set_ylabel("optimizer state tensors  (n_params x n_states)")
ax2.set_title("Optimizer-state memory — Lion is half")
plt.tight_layout(); plt.show()


# %% [markdown]
# **Take-home from Block 5.**
#
# 1. **AdamW is the modern default.** Decoupling weight decay from the
#    gradient step removes a subtle bias that Adam+L2 carries — visible
#    here as a slightly higher val loss for Adam+L2 at the same nominal
#    `wd`. The gap is small on a small model, but it grows with model
#    size; it is the reason every modern foundation-model recipe uses
#    AdamW, not Adam+L2.
# 2. **Lion buys you memory for free.** Sign-of-momentum needs *one*
#    momentum tensor per parameter instead of two; the optimizer state is
#    halved. The val loss is competitive with AdamW once you re-tune the
#    LR (Lion likes ~3-10× smaller LR). For a 7B-parameter model that
#    halving is measured in tens of GB.
# 3. **Schedule-Free AdamW removes one knob.** It reaches AdamW-class val
#    loss without a cosine schedule. Useful in practice because you no
#    longer need to know `n_total_steps` up front — a constant pain in
#    early-stopping / continual-training regimes.
# 4. **Sophia (mention only).** Second-order, diagonal-Hessian — promising
#    in language-model pretraining but adds enough code to obscure the
#    pedagogical point. See the MFML W6 slide deck for the cartoon.
#
# This is the optimizer roster you should reach for in 2026; "Adam" by
# itself is no longer a complete answer.

# %% [markdown]
# # Block 6 — Crystal graphs: a tiny hand-rolled message-passing GNN
#
# We now switch input modality entirely. The dataset is
# `CrystalGraphsDataset` — 200 toy crystals across 5 prototype templates
# (rocksalt, zincblende, wurtzite, fluorite, perovskite), each populated
# with a randomly chosen cation/anion pair. Targets are toy formation
# energies built from electronegativity differences and radius mismatches
# (see the dataset docstring for the exact recipe).
#
# The point of this block is **not** to teach CGCNN. The point is to show
# that the optimizer toolkit you used on the CNN above carries over to a
# GNN with one twist: each crystal is a different graph, so per-step work
# is variable and the loss landscape is rougher.
#
# We hand-roll a 25-line message-passing GNN. No PyTorch Geometric, no
# pymatgen, no DFT — just enough to demonstrate the optimizer choices.
#
# *(see MG §"Crystals as periodic graphs", §"Message passing on crystal
# graphs"; MFML §"Per-parameter LR vs uniform LR")*

# %%
class TinyCGNN(nn.Module):
    """Atom-embedding -> n_layers of edge-conditioned message passing -> mean-pool -> MLP head.

    Each crystal is a single small graph (~8-12 atoms, ~12-32 edges) so
    we run one crystal at a time inside the inner loop.  The architecture
    is intentionally simple to fit in one screen; the point is the
    optimization story, not the GNN sophistication.
    """

    def __init__(self, n_elements=120, embed_dim=16, n_layers=3):
        super().__init__()
        self.embed = nn.Embedding(n_elements, embed_dim)
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * embed_dim + 1, embed_dim), nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, species, edge_index, edge_distance):
        h = self.embed(species)                                  # (n_nodes, d)
        for layer in self.msg_mlps:
            src, dst = edge_index[0], edge_index[1]
            msg_in = torch.cat(
                [h[src], h[dst], edge_distance.unsqueeze(-1)], dim=-1
            )
            msg = layer(msg_in)                                  # (n_edges, d)
            agg = torch.zeros_like(h).index_add_(0, dst, msg)
            h = h + agg
        return self.head(h.mean(0)).squeeze(-1)                  # scalar y


# %%
crystals = CrystalGraphsDataset()
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(crystals), generator=g)
n_tr = int(0.8 * len(crystals))
tr_idx, te_idx = perm[:n_tr].tolist(), perm[n_tr:].tolist()

# Standardise targets to ~unit scale so the optimizer LRs we used on Ising
# transfer over without retuning.
y_all = crystals.y
y_mean, y_std = y_all.mean().item(), y_all.std().item()
print(f"target stats: mean={y_mean:+.3f} eV/atom  std={y_std:.3f}")


def gnn_train(optim_factory, label, n_epochs=8, grad_clip=None):
    """Train a fresh TinyCGNN with the given optimizer.  One crystal per step."""
    torch.manual_seed(0)
    model = TinyCGNN()
    optim_ = optim_factory(model.parameters())
    train_mae, test_mae = [], []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        order = torch.randperm(len(tr_idx)).tolist()
        for i in order:
            sample = crystals[tr_idx[i]]
            y_norm = (sample["y"] - y_mean) / y_std
            optim_.zero_grad()
            pred = model(sample["species"], sample["edge_index"],
                         sample["edge_distance"])
            loss = (pred - y_norm) ** 2
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim_.step()
            epoch_loss += loss.item()
        train_mae.append(epoch_loss / len(tr_idx))

        model.eval()
        with torch.no_grad():
            errs = []
            for j in te_idx:
                s = crystals[j]
                y_norm = (s["y"] - y_mean) / y_std
                p = model(s["species"], s["edge_index"], s["edge_distance"])
                errs.append((p - y_norm).abs().item())
            test_mae.append(float(np.mean(errs)) * y_std)         # de-normalise
    print(f"  {label:25s}  final test MAE = {test_mae[-1]:.3f} eV/atom")
    return np.array(train_mae), np.array(test_mae)


# %%
print("Block 6 — three optimizer presets on CrystalGraphsDataset:")
tr_sgd_g, te_sgd_g = gnn_train(
    lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    "SGD-mom (lr=0.05)",
)
tr_adam_g, te_adam_g = gnn_train(
    lambda p: torch.optim.Adam(p, lr=0.005),
    "Adam (lr=0.005)",
)
tr_adam_clip_g, te_adam_clip_g = gnn_train(
    lambda p: torch.optim.Adam(p, lr=0.005),
    "Adam + clip(1.0)",
    grad_clip=1.0,
)


# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
for tr, label, c in [(tr_sgd_g, "SGD-mom", "C0"),
                     (tr_adam_g, "Adam", "C3"),
                     (tr_adam_clip_g, "Adam + clip(1)", "C2")]:
    a1.plot(tr, "o-", label=label, c=c, lw=1.4)
a1.set_xlabel("epoch"); a1.set_ylabel("train MSE (normalised)")
a1.set_yscale("log"); a1.set_title("Crystal GNN — train MSE"); a1.legend()

for te, label, c in [(te_sgd_g, "SGD-mom", "C0"),
                     (te_adam_g, "Adam", "C3"),
                     (te_adam_clip_g, "Adam + clip(1)", "C2")]:
    a2.plot(te, "o-", label=label, c=c, lw=1.4)
a2.set_xlabel("epoch"); a2.set_ylabel("test MAE (eV/atom)")
a2.set_title("Crystal GNN — test MAE"); a2.legend()
plt.tight_layout(); plt.show()


# %% [markdown]
# **Three things are happening on this plot.**
#
# 1. **SGD with momentum is a perfectly reasonable choice** — the GNN is
#    small, the loss is well-conditioned once we standardise targets, and
#    the per-step noise (one crystal at a time) acts as light regularisation.
# 2. **Adam without gradient clipping can spike** — message passing on a
#    fresh embedding occasionally produces large gradients that Adam's
#    running variance cannot dampen quickly. You may see one or two epochs
#    where the loss jumps before settling.
# 3. **Gradient clipping fixes (2) cheaply.** A single line of code
#    (`torch.nn.utils.clip_grad_norm_`) bounds the worst case and makes Adam
#    train as smoothly as SGD on this problem.
#
# **Forward link to MG U7 / U8.** Real CGCNN / MEGNet / M3GNet training
# uses *exactly* this template — Adam + cosine schedule + gradient
# clipping. Knowing why each ingredient is there is the point of Week 6.

# %% [markdown]
# # Block 6b — From toy graphs to *real* crystal graphs (MG Unit 5 core)
#
# Block 6 trained on the dataset's **pre-baked** fixed graphs and fed the
# message MLP a raw scalar distance with an implicit hard cutoff. That was
# fine for the *optimizer* story but it skips the three pieces of machinery
# MG Unit 5 spends its whole lecture on:
#
# 1. **Periodic boundary conditions.** A crystal is infinite. The "graph"
#    is whatever you get by searching for neighbours within a cutoff
#    radius *across periodic images* of the unit cell — not a fixed
#    template. Change the cutoff and you change the graph.
# 2. **Distance featurization.** Real crystal GNNs (CGCNN, SchNet, MEGNet)
#    never feed the raw scalar bond length. They expand it in a basis of
#    **Gaussian radial basis functions (RBF)** and multiply by a **smooth
#    cutoff envelope** so the edge feature → 0 *continuously* as an atom
#    leaves the cutoff sphere.
# 3. **Why the envelope matters.** A hard cutoff makes the predicted
#    energy *discontinuous* in atomic position — fatal for forces and a
#    reproducibility landmine. We show that artifact directly.
#
# Frame the Block-6 `TinyCGNN` honestly: it is **CGCNN/SchNet minus
# RBF + PBC**. This block adds the two missing pieces on small toy
# lattices and re-runs the same optimizer presets so the comparison is
# apples-to-apples.
#
# *(see MG §"Crystals as periodic graphs", §"Minimum-image convention",
# §"RBF edge features + smooth cutoff", §"Ranking metrics for screening";
# MFML §"What did the optimizer actually fit?")*

# %% [markdown]
# ## 6b.1 — Toy periodic lattices
#
# We do not have `pymatgen` or `ase` as exercise dependencies, so we
# hand-roll the minimal thing: each crystal becomes a small cubic lattice
# (a `(N, 3)` array of fractional coordinates in `[0, 1)` plus a cubic
# box length `L` in Å). We reuse the dataset's species and toy formation
# energies unchanged — only the **graph construction** becomes physical.
#
# The atom count per prototype is kept identical to the dataset templates
# so the targets `crystals.y` stay meaningful; we just place those atoms
# on a real periodic lattice instead of using the abstract template edges.

# %%
# A deterministic toy lattice per prototype: a simple cubic arrangement of
# the prototype's atoms inside a cubic box. The box length is chosen so the
# nearest-neighbour spacing is ~ the sum of covalent radii, i.e. the same
# physical scale the dataset used for its toy bond lengths.
_PROTO_GRID = {
    # prototype_index : (n_atoms, grid_shape)  with n_atoms == prod(grid)
    0: (8, (2, 2, 2)),    # rocksalt   — 8-atom cube
    1: (8, (2, 2, 2)),    # zincblende
    2: (8, (2, 2, 2)),    # wurtzite
    3: (12, (3, 2, 2)),   # fluorite   — 12 atoms
    4: (10, (5, 2, 1)),   # perovskite — 10 atoms
}


def build_periodic_lattice(species, prototype, rng):
    """Place `species` on a small cubic lattice; return (frac, L).

    frac : (N, 3) float64 fractional coordinates in [0, 1)
    L    : float, cubic box edge length in Angstrom

    The lattice is the integer grid for the prototype, rescaled into the
    unit cube, plus a small random displacement so different crystals see
    different geometry (mirrors the dataset's distance distortion).
    """
    n = len(species)
    _, grid = _PROTO_GRID[prototype]
    gx, gy, gz = grid
    coords = np.array(
        [(i, j, k) for i in range(gx) for j in range(gy) for k in range(gz)],
        dtype=np.float64,
    )[:n]
    # Nearest-neighbour spacing target ~ mean covalent-radius sum of the cell.
    from ai4mat.datasets.crystal_graphs import _RADIUS
    r_mean = float(np.mean([_RADIUS[int(z)] for z in species]))
    spacing = 1.8 * r_mean                       # Å between adjacent sites
    L = spacing * max(gx, gy, gz)
    frac = coords / np.array([gx, gy, gz], dtype=np.float64)
    # Small random rattle (±3% of the box) so geometry varies per crystal.
    frac = frac + rng.uniform(-0.03, 0.03, size=frac.shape)
    frac = np.mod(frac, 1.0)                      # wrap back into the cell
    return frac, float(L)


# %% [markdown]
# ## 6b.2 — Minimum-image neighbour search
#
# The core PBC primitive. For a cubic box of edge `L`, the **minimum-image
# convention** says: the distance between atoms `i` and `j` is the
# distance to the *closest periodic image* of `j`. For a cubic cell that
# is one line of code on the fractional displacement:
#
# $$\Delta f \;\leftarrow\; \Delta f - \operatorname{round}(\Delta f),
#   \qquad d \;=\; L\,\lVert \Delta f \rVert .$$
#
# We build a directed edge `i → j` (both directions) for every pair within
# `r_cut`. This is the `O(N^2)` brute-force version — correct and fine for
# our ≤12-atom toy cells; real codes use a cell list for `O(N · k̄)`.

# %%
def pbc_neighbor_graph(frac, L, r_cut):
    """Minimum-image neighbour search inside a cubic box.

    Parameters
    ----------
    frac : (N, 3) fractional coordinates in [0, 1)
    L    : cubic box edge length (Å)
    r_cut: cutoff radius (Å)

    Returns
    -------
    edge_index : int64 tensor (2, M)  directed edges (i->j and j->i)
    edge_dist  : float32 tensor (M,)  minimum-image distances (Å)
    """
    frac = np.asarray(frac)
    n = len(frac)
    src, dst, dists = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            df = frac[j] - frac[i]
            df = df - np.round(df)               # minimum-image convention
            d = L * float(np.linalg.norm(df))
            if d <= r_cut:
                src.append(i)
                dst.append(j)
                dists.append(d)
    if not src:                                  # degenerate: r_cut too small
        return (torch.zeros((2, 0), dtype=torch.int64),
                torch.zeros((0,), dtype=torch.float32))
    edge_index = torch.tensor([src, dst], dtype=torch.int64)
    edge_dist = torch.tensor(dists, dtype=torch.float32)
    return edge_index, edge_dist


# %%
# Build a PBC graph view of the whole dataset once, at a fixed cutoff.
# We keep the dataset's species/y; only edges + distances are recomputed.
R_CUT = 3.2                                      # Å — captures 1st (and some 2nd) shell

_lat_rng = np.random.default_rng(0)
pbc_graphs = []
for idx in range(len(crystals)):
    s = crystals[idx]
    sp = s["species"]
    proto = s["prototype"]
    frac, L = build_periodic_lattice(sp.numpy(), proto, _lat_rng)
    ei, ed = pbc_neighbor_graph(frac, L, R_CUT)
    pbc_graphs.append({"species": sp, "edge_index": ei,
                       "edge_distance": ed, "y": s["y"],
                       "frac": frac, "L": L})

_deg = np.array([g["edge_index"].shape[1] / len(g["species"])
                 for g in pbc_graphs])
print(f"PBC graphs built: r_cut={R_CUT} Å  "
      f"mean degree = {_deg.mean():.2f}  "
      f"(min {_deg.min():.1f}, max {_deg.max():.1f})")
print("Note: degree now DEPENDS on the cutoff — it is no longer a fixed "
      "template.")


# %% [markdown]
# ## 6b.3 — Gaussian RBF expansion + smooth cutoff envelope
#
# Instead of feeding the scalar distance `d`, we expand it on a grid of
# Gaussians centred at `mu_k` and multiply by a **cosine cutoff envelope**
# `0.5 (1 + cos(pi d / r_cut))` that decays smoothly to exactly 0 at
# `r_cut` (Behler 2011 / SchNet). The hard-cutoff baseline instead uses a
# step indicator `1[d <= r_cut]` — and *that* step is the artifact.

# %%
class RBFExpansion(nn.Module):
    """Gaussian RBF edge featurizer with a switchable cutoff envelope.

    edge_feat_k = exp(-gamma (d - mu_k)^2) * envelope(d)

    envelope = smooth cosine  (default, physical)
             | hard step      (pedagogical 'wrong' baseline)
    """

    def __init__(self, r_cut, n_rbf=16, smooth=True):
        super().__init__()
        centers = torch.linspace(0.0, r_cut, n_rbf)
        self.register_buffer("centers", centers)
        # Width = one grid spacing (standard SchNet-style choice).
        spacing = float(centers[1] - centers[0])
        self.gamma = 1.0 / (spacing ** 2)
        self.r_cut = float(r_cut)
        self.smooth = smooth

    def envelope(self, d):
        if self.smooth:
            env = 0.5 * (1.0 + torch.cos(math.pi * d / self.r_cut))
        else:
            env = torch.ones_like(d)             # hard cutoff = no taper
        return env * (d <= self.r_cut).float()   # zero strictly outside

    def forward(self, d):
        rbf = torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)
        return rbf * self.envelope(d).unsqueeze(-1)            # (M, n_rbf)


# %% [markdown]
# ### The hard-cutoff discontinuity artifact
#
# Take one atom and slide a neighbour radially outward through `r_cut`.
# With the **hard** cutoff the summed edge feature (a proxy for the
# energy contribution of that bond) jumps to zero discontinuously the
# instant the neighbour crosses `r_cut`. With the **smooth** envelope it
# decays to zero continuously. The discontinuity is what breaks forces
# (= −∂E/∂x) and makes a hard-cutoff model irreproducible near the shell.

# %%
d_scan = torch.linspace(2.0, 4.0, 400)           # sweep a bond length through r_cut
rbf_hard = RBFExpansion(R_CUT, n_rbf=16, smooth=False)
rbf_soft = RBFExpansion(R_CUT, n_rbf=16, smooth=True)

# "Bond energy proxy" = total RBF activation on that single edge.
e_hard = rbf_hard(d_scan).sum(-1)
e_soft = rbf_soft(d_scan).sum(-1)

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(d_scan, e_hard, c="C3", lw=1.8, label="hard cutoff (step)")
ax.plot(d_scan, e_soft, c="C2", lw=1.8, label="smooth cosine envelope")
ax.axvline(R_CUT, ls=":", c="0.4", label=f"r_cut = {R_CUT} Å")
ax.set_xlabel("bond length d (Å)")
ax.set_ylabel("Σ RBF activation  (bond-energy proxy)")
ax.set_title("Block 6b — hard cutoff is discontinuous at r_cut")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

print(f"jump at r_cut (hard)  = {abs(e_hard[d_scan <= R_CUT][-1]):.3f}  "
      f"-> 0 across one step  (discontinuous)")
print(f"value at r_cut (soft) = {e_soft[d_scan <= R_CUT][-1]:.3e}  "
      f"(continuous, ->0)")


# %% [markdown]
# ## 6b.4 — `TinyCGNN_RBF`: the Block-6 GNN with RBF edge features
#
# Same message-passing skeleton as `TinyCGNN`; the *only* change is the
# edge channel. Where `TinyCGNN` concatenated a single raw distance,
# `TinyCGNN_RBF` concatenates the `n_rbf`-dim smooth RBF vector. We also
# expose the **readout** (mean vs sum) because it is a one-line change
# with a real physical meaning (intensive vs extensive energy).

# %%
class TinyCGNN_RBF(nn.Module):
    """CGCNN/SchNet-flavoured: atom embedding -> RBF-conditioned message
    passing -> pooled readout -> MLP head.

    readout = "mean"  -> intensive target (energy per atom)
            = "sum"   -> extensive target (total energy)
    """

    def __init__(self, rbf, n_elements=120, embed_dim=16, n_layers=3,
                 readout="mean"):
        super().__init__()
        self.rbf = rbf
        n_rbf = len(rbf.centers)
        self.embed = nn.Embedding(n_elements, embed_dim)
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * embed_dim + n_rbf, embed_dim), nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.readout = readout

    def forward(self, species, edge_index, edge_distance):
        h = self.embed(species)
        if edge_index.shape[1] > 0:
            edge_feat = self.rbf(edge_distance)               # (M, n_rbf)
            for layer in self.msg_mlps:
                src, dst = edge_index[0], edge_index[1]
                msg_in = torch.cat([h[src], h[dst], edge_feat], dim=-1)
                msg = layer(msg_in)
                agg = torch.zeros_like(h).index_add_(0, dst, msg)
                h = h + agg
        pooled = h.sum(0) if self.readout == "sum" else h.mean(0)
        return self.head(pooled).squeeze(-1)


# %% [markdown]
# ## 6b.5 — Ranking / discovery metrics
#
# For screening you do not care about absolute MAE; you care whether the
# model **ranks** candidates correctly and whether the true best few are
# in the model's top-k shortlist. We add three rank-aware metrics
# alongside MSE/MAE. We hand-roll the rank correlations (no scipy import
# in this notebook) so the definitions are visible:
#
# - **Spearman ρ** — Pearson correlation of the *ranks*.
# - **Kendall τ** — fraction of concordant minus discordant pairs.
# - **Top-k recall** — of the `k` truly lowest-energy crystals, how many
#   appear in the model's predicted lowest-`k`.

# %%
def _ranks(x):
    """Average ranks (ties shared) of a 1-D numpy array."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # average tied ranks
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman_rho(y_true, y_pred):
    rt, rp = _ranks(np.asarray(y_true)), _ranks(np.asarray(y_pred))
    rt = rt - rt.mean()
    rp = rp - rp.mean()
    denom = np.sqrt((rt ** 2).sum() * (rp ** 2).sum())
    return float((rt * rp).sum() / denom) if denom > 0 else 0.0


def kendall_tau(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    n = len(yt)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(yt[i] - yt[j]) * np.sign(yp[i] - yp[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    tot = c + d
    return float((c - d) / tot) if tot > 0 else 0.0


def topk_recall(y_true, y_pred, k):
    """Fraction of the true lowest-k that land in the predicted lowest-k.

    Lower energy = better candidate, so we take the *smallest* values.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    true_best = set(np.argsort(yt, kind="mergesort")[:k])
    pred_best = set(np.argsort(yp, kind="mergesort")[:k])
    return len(true_best & pred_best) / k


# %% [markdown]
# ## 6b.6 — Re-run the optimizer presets on the *physical* graphs
#
# Same training loop shape as Block 6, but now (a) the graphs come from
# the PBC neighbour search, (b) edges carry smooth RBF features, and
# (c) we report the ranking metrics next to MSE/MAE. Same three optimizer
# presets so the optimizer story still lines up with Block 6.

# %%
def gnn_train_rbf(optim_factory, label, n_epochs=8, grad_clip=None,
                  readout="mean", graphs=None, smooth=True):
    """Train a fresh TinyCGNN_RBF on the PBC graphs. One crystal per step.

    Returns (train_mse_curve, test_mae_curve, metrics_dict) where metrics
    are computed on the held-out split at the final epoch.
    """
    graphs = pbc_graphs if graphs is None else graphs
    torch.manual_seed(0)
    rbf = RBFExpansion(R_CUT, n_rbf=16, smooth=smooth)
    model = TinyCGNN_RBF(rbf, readout=readout)
    optim_ = optim_factory(model.parameters())
    train_mse, test_mae = [], []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        order = torch.randperm(len(tr_idx)).tolist()
        for i in order:
            g_ = graphs[tr_idx[i]]
            y_norm = (g_["y"] - y_mean) / y_std
            optim_.zero_grad()
            pred = model(g_["species"], g_["edge_index"],
                         g_["edge_distance"])
            loss = (pred - y_norm) ** 2
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim_.step()
            epoch_loss += loss.item()
        train_mse.append(epoch_loss / len(tr_idx))

        model.eval()
        with torch.no_grad():
            errs = []
            for j in te_idx:
                g_ = graphs[j]
                y_norm = (g_["y"] - y_mean) / y_std
                p = model(g_["species"], g_["edge_index"],
                          g_["edge_distance"])
                errs.append((p - y_norm).abs().item())
            test_mae.append(float(np.mean(errs)) * y_std)

    # Final-epoch predictions on the held-out split, de-normalised.
    model.eval()
    with torch.no_grad():
        y_true, y_hat = [], []
        for j in te_idx:
            g_ = graphs[j]
            p = model(g_["species"], g_["edge_index"], g_["edge_distance"])
            y_hat.append(p.item() * y_std + y_mean)
            y_true.append(g_["y"].item())
    y_true, y_hat = np.array(y_true), np.array(y_hat)
    k = max(1, len(te_idx) // 5)                 # top-20% shortlist
    metrics = {
        "MSE": float(np.mean((y_true - y_hat) ** 2)),
        "MAE": float(np.mean(np.abs(y_true - y_hat))),
        "Spearman": spearman_rho(y_true, y_hat),
        "Kendall": kendall_tau(y_true, y_hat),
        f"top{k}_recall": topk_recall(y_true, y_hat, k),
    }
    print(f"  {label:24s}  MAE={metrics['MAE']:.3f} eV/atom  "
          f"rho={metrics['Spearman']:.3f}  tau={metrics['Kendall']:.3f}  "
          f"top{k}-recall={metrics[f'top{k}_recall']:.2f}")
    return np.array(train_mse), np.array(test_mae), metrics


# %%
print("Block 6b — same optimizer presets, now on PBC + RBF graphs:")
tr_sgd_r, te_sgd_r, m_sgd_r = gnn_train_rbf(
    lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    "SGD-mom (lr=0.05)",
)
tr_adam_r, te_adam_r, m_adam_r = gnn_train_rbf(
    lambda p: torch.optim.Adam(p, lr=0.005),
    "Adam (lr=0.005)",
)
tr_adam_clip_r, te_adam_clip_r, m_adam_clip_r = gnn_train_rbf(
    lambda p: torch.optim.Adam(p, lr=0.005),
    "Adam + clip(1.0)",
    grad_clip=1.0,
)


# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
for tr, label, c in [(tr_sgd_r, "SGD-mom", "C0"),
                     (tr_adam_r, "Adam", "C3"),
                     (tr_adam_clip_r, "Adam + clip(1)", "C2")]:
    a1.plot(tr, "o-", label=label, c=c, lw=1.4)
a1.set_xlabel("epoch"); a1.set_ylabel("train MSE (normalised)")
a1.set_yscale("log"); a1.set_title("PBC+RBF crystal GNN — train MSE")
a1.legend()

for te, label, c in [(te_sgd_r, "SGD-mom", "C0"),
                     (te_adam_r, "Adam", "C3"),
                     (te_adam_clip_r, "Adam + clip(1)", "C2")]:
    a2.plot(te, "o-", label=label, c=c, lw=1.4)
a2.set_xlabel("epoch"); a2.set_ylabel("test MAE (eV/atom)")
a2.set_title("PBC+RBF crystal GNN — test MAE"); a2.legend()
plt.tight_layout(); plt.show()


# %%
# Side-by-side metric table: ranking metrics are what screening cares about.
print(f"\n{'optimizer':<18}{'MSE':>8}{'MAE':>8}"
      f"{'Spearman':>10}{'Kendall':>9}{'topk-rec':>10}")
print("-" * 63)
for name, m in [("SGD-mom", m_sgd_r),
                ("Adam", m_adam_r),
                ("Adam+clip", m_adam_clip_r)]:
    kkey = [x for x in m if x.startswith("top")][0]
    print(f"{name:<18}{m['MSE']:>8.3f}{m['MAE']:>8.3f}"
          f"{m['Spearman']:>10.3f}{m['Kendall']:>9.3f}{m[kkey]:>10.2f}")


# %% [markdown]
# **Reading the metric table.** MAE alone hides what matters for
# *discovery*: two models with similar MAE can rank candidates very
# differently. Spearman ρ / Kendall τ measure whether the model orders
# crystals by stability correctly; top-k recall measures whether the
# truly most-stable crystals make the shortlist you would actually send
# to DFT. A screening model with mediocre MAE but high ρ is often more
# useful than the reverse — this is the MG "metrics for screening" point.

# %% [markdown]
# ## 6b.7 — Readout: sum (extensive) vs mean (intensive)
#
# One-line change, real physics. The dataset target is energy **per atom**
# (intensive) → `mean` pooling is the physically-consistent readout.
# `sum` pooling predicts an *extensive* quantity and must learn to undo
# the variable atom count itself, which on a fixed per-atom target just
# injects an N-dependent nuisance. We show the gap with everything else
# held fixed.

# %%
print("Block 6b — readout contrast (Adam, identical everything else):")
_, te_mean, m_mean = gnn_train_rbf(
    lambda p: torch.optim.Adam(p, lr=0.005), "mean-pool (intensive)",
    readout="mean",
)
_, te_sum, m_sum = gnn_train_rbf(
    lambda p: torch.optim.Adam(p, lr=0.005), "sum-pool  (extensive)",
    readout="sum",
)
print(f"  mean-pool final test MAE = {te_mean[-1]:.3f} eV/atom  "
      f"(rho={m_mean['Spearman']:.3f})")
print(f"  sum-pool  final test MAE = {te_sum[-1]:.3f} eV/atom  "
      f"(rho={m_sum['Spearman']:.3f})")
print("Take-home: match the readout to the target's extensivity. Per-atom "
      "target -> mean; total-energy target -> sum.")


# %% [markdown]
# **Block 6b take-home.**
#
# 1. **The graph is a modelling choice, not a given.** PBC + a cutoff
#    *define* the neighbour list; change `R_CUT` and you change the model
#    input. (Try `R_CUT = 2.6` vs `4.0` and re-run — mean degree and
#    accuracy both move.)
# 2. **Smooth cutoff is not optional.** The hard-cutoff discontinuity
#    plot in 6b.3 is exactly why production crystal GNNs use an envelope:
#    discontinuous energy ⇒ undefined/garbage forces ⇒ irreproducible MD.
# 3. **Report ranking metrics for screening.** MSE/MAE answer "how close",
#    Spearman/Kendall/top-k answer "did we find the good ones" — the
#    question discovery actually asks.
# 4. **Readout encodes a physical assumption** (intensive vs extensive);
#    it is one line and it matters.
#
# `TinyCGNN_RBF` is still "CGCNN/SchNet minus the learned filter-generating
# network and equivariance". Closing *that* gap (and the rotation-
# invariance check) is the MG Unit 5 / Unit 7 reading task.

# %% [markdown]
# # Block 7 — Student exercises (~22 min)

# %% [markdown]
# ## Exercise 1 (core) — LP-FT (Kumar et al. 2022)
#
# **Setup.** "Linear-probe-then-fine-tune" claims that *first* training only
# the head (linear probe), then unfreezing the backbone for a few more
# epochs, *beats* either protocol alone on out-of-distribution targets.
#
# **Task.** Implement LP-FT for the Ising → Cahn-Hilliard transfer:
#
# 1. Phase 1 (probe): freeze the backbone, train the head for 2 epochs at
#    `lr_head=1e-2` using `SOURCE_STATE` as init.
# 2. Phase 2 (FT): unfreeze and continue for 2 more epochs at
#    `lr_backbone=1e-4`, `lr_head=1e-3` (cosine + warm-up).
#
# Compare the final source/target accuracies against (a) frozen-only and
# (b) full-FT-from-scratch from Block 3.  Plot all three on one set of axes.
#
# **Expected:** LP-FT typically retains source acc better than full FT
# *and* reaches a slightly higher target acc than frozen-only.
#
# *(reference: Kumar, Raghunathan, Jones, Ma, Liang, "Fine-Tuning can
# Distort Pretrained Features and Underperform Out-of-Distribution",
# ICLR 2022)*

# %% [markdown]
# ## Exercise 2 (core) — Gradient clipping in catastrophic-forgetting territory
#
# **Setup.** Block 2's naive lr=0.1 fine-tune collapsed source accuracy
# inside one epoch. You verified in Block 6 that gradient clipping tames
# Adam on the GNN.
#
# **Task.** Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
# to the fine-tune loop and rerun Block 2 at lr=0.1. Two questions:
#
# 1. Does clipping prevent the source-acc collapse, or does it merely
#    *delay* it? Plot 4-epoch source/target curves with and without clip.
# 2. What is the smallest clip threshold (try 0.5, 0.25, 0.1) that recovers
#    "useful" fine-tuning (target acc > 0.7 *and* source acc > 0.7)?
#
# **Expected:** clipping at 1.0 helps but is not enough at lr=0.1; tighter
# clip values recover both metrics but slow target-task progress.

# %% [markdown]
# ## Exercise 3 (core) — Edge of stability
#
# **Setup.** MFML W6 mentions that for a quadratic loss with Hessian
# top-eigenvalue λ_max, vanilla GD is stable iff `lr < 2 / λ_max`. Above
# that threshold the loss diverges.
#
# **Task.** Take the Block 6 TinyCGNN and find the *empirical* edge of
# stability for SGD without momentum:
#
# 1. Train at lr ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0} for 3 epochs each;
#    record final train loss.
# 2. Plot final loss vs lr (log x). Identify the lr where it explodes.
# 3. Use that to back out a Hessian-eigenvalue estimate `λ_max ≈ 2 / lr*`.
#
# **Question to answer in writing:** is the `λ_max` you find compatible
# with the LR you used in Block 6? If not, why does Block 6's optimizer
# still converge? (Hint: MFML §"Implicit regularisation of SGD".)

# %% [markdown]
# ## Exercise 4 (stretch) — Optimizer-as-prior across modalities
#
# **Setup.** You have now run the same three optimizers on three
# qualitatively different inputs:
#
# - Homework Part A: 2-D analytic landscapes
# - Homework Part C: 4-feature element vectors
# - Block 6: variable-size atom graphs
#
# **Task.** Pick one optimizer (Adam) and one diagnostic (e.g., gradient
# norm distribution per epoch). Plot it for *all three* settings on the
# same axes. Comment in 3 sentences on what the optimizer's *implicit
# prior* looks like in each setting and where it stops being a good prior.
#
# **No expected answer.** This is the synthesis exercise — your reading
# of Week 6.

# %% [markdown]
# ---
# **Bridge to Week 7.** Next week MFML moves to the *probabilistic view of
# learning* (MLE, MAP, conformal prediction) and ML-PC pairs that with
# *generalization, robustness, and process windows*.  Week 6's optimizer
# toolkit + fine-tuning discipline is the prerequisite for both — without
# honest fine-tuning, you cannot honestly measure either uncertainty or
# generalization.
