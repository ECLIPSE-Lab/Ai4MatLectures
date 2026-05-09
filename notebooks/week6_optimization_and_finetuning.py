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
# | 5 | ~12 | Adam vs SGD-with-momentum for fine-tuning — why Adam often hurts |
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
# # Block 5 — Adam vs SGD-with-momentum for fine-tuning
#
# The classic ML-PC W6 anti-pattern: "I'll just swap SGD for Adam, it's
# always better." On *fine-tuning* this is often wrong, because Adam's
# per-parameter step size is tuned by *recent* gradients — and on the
# fresh head those are noisy and large. Adam ends up overshooting the
# good neighborhood of the source weights and lands in a basin that solves
# the target task at the cost of the source-task structure.
#
# Same setup as Block 4 (cosine + warm-up, lr_max = 1e-3 to give both a
# fair chance) — only the optimizer differs.
#
# *(see MFML §"Adam vs SGD-with-momentum — when each is preferable",
# ML-PC §"Adam often hurts fine-tuning")*

# %%
LR_MAX_FAIR = 1e-3
N_STEPS_FAIR = N_EPOCHS_FT * len(ch_train_loader)

src_sgd, tgt_sgd = fine_tune(
    SOURCE_STATE, lr_backbone=LR_MAX_FAIR, lr_head=LR_MAX_FAIR,
    n_epochs=N_EPOCHS_FT, optimizer_cls=torch.optim.SGD,
    scheduler=lambda opt: cosine_with_warmup(
        opt, n_warmup_steps=N_STEPS_FAIR // 10,
        n_total_steps=N_STEPS_FAIR,
        lr_max_per_group=[LR_MAX_FAIR, LR_MAX_FAIR],
    ),
)
src_adam, tgt_adam = fine_tune(
    SOURCE_STATE, lr_backbone=LR_MAX_FAIR, lr_head=LR_MAX_FAIR,
    n_epochs=N_EPOCHS_FT, optimizer_cls=torch.optim.Adam,
    scheduler=lambda opt: cosine_with_warmup(
        opt, n_warmup_steps=N_STEPS_FAIR // 10,
        n_total_steps=N_STEPS_FAIR,
        lr_max_per_group=[LR_MAX_FAIR, LR_MAX_FAIR],
    ),
)

print(f"SGD+momentum, cosine+warm-up:  source={src_sgd[-1]:.3f}  target={tgt_sgd[-1]:.3f}")
print(f"Adam,          cosine+warm-up:  source={src_adam[-1]:.3f}  target={tgt_adam[-1]:.3f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
ep = np.arange(1, N_EPOCHS_FT + 1)
ax.plot(ep, src_sgd,  "o-", label="SGD-mom — source", c="C0", ls=":")
ax.plot(ep, tgt_sgd,  "o-", label="SGD-mom — target", c="C0")
ax.plot(ep, src_adam, "o-", label="Adam — source",    c="C3", ls=":")
ax.plot(ep, tgt_adam, "o-", label="Adam — target",    c="C3")
ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.set_ylim(0.4, 1.05)
ax.set_title("Block 5 — Adam often forgets faster than SGD-mom")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()


# %% [markdown]
# **Take-home from Block 5.** When the goal is *fine-tuning* (= staying
# near a good initialization while picking up a small new signal), the
# update rule's *implicit prior* matters. SGD with small momentum is a
# conservative prior — small steps in the direction of the current
# gradient. Adam is an aggressive prior — per-axis normalized steps that
# *don't shrink* when the gradient is small, because the running variance
# is also small. On fine-tuning, "don't shrink the step when the gradient
# is small" is exactly the wrong inductive bias.

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
# **Bridge to Week 7.** Next week MFML moves to *generalization* (bias-
# variance, regularisation, model selection) and ML-PC pairs that with
# *process windows + robustness*.  Week 6's optimizer toolkit + fine-tuning
# discipline is the prerequisite for that — without honest fine-tuning,
# you cannot honestly measure generalization.
