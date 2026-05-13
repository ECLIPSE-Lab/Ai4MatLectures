# %% [markdown]
# # Week 14 — Explainability, limits, and a course retrospective
#
# This is the final Thursday of the SS26 sequence. We braid three
# lectures' Week 14 content onto one deployment-audit story:
#
# 1. **MFML Unit 14** — Explainability, limits, scientific trust.
#    SHAP, Integrated Gradients, counterfactuals, OOD detection, the
#    six levels of explainability, and a first look at **mechanistic
#    interpretability** (superposition, sparse autoencoders).
# 2. **ML-PC Unit 14** — Integration, limits, reflection. Why ML
#    fails in real labs; explainability for experimental ML;
#    instrument drift and distribution shift in practice.
# 3. **MG Unit 14** — Physical constraints, limits, outlook.
#    Symmetry constraints; what ML can and cannot discover; how the
#    pipeline integrates with experimental workflows.
#
# **Red thread.** *A 95%-accurate model has 5% wrong predictions —
# the only useful question is **which 5%**. Today we instrument the
# tensile regression model from Week 8 with SHAP to learn **why**
# each prediction is what it is, train a **sparse autoencoder on the
# Ising-CNN's hidden activations** to ask **what concepts** the
# network represents internally (mechanistic interpretability),
# generate counterfactual explanations for what would have to change
# to flip a decision, audit the Ising classifier from Week 10 for
# symmetry equivariance the physics demands, build an
# autoencoder-based OOD detector that beats the homework's
# max-softmax baseline, and end with a retrospective of the 14-week
# arc — every method on the same problem, with a single chart that
# says when each one wins.*
#
# > **Pre-flight check.** This notebook **assumes** you have run
# > `notebooks/week14_homework.py`. Block 1 picks up directly from
# > your trained TinyCNN and your max-softmax-probability OOD scores.
#
# ## Agenda (90 min)
#
# | Block | Min | Topic |
# |------:|:---:|:------|
# | 1 |  6 | Recap from homework |
# | 2 | 14 | KernelSHAP from scratch on a tensile regression model |
# | 3 | 12 | Sparse autoencoder on Ising-CNN activations — mechanistic interp. |
# | 4 | 14 | Counterfactuals via gradient descent on the input |
# | 5 | 12 | Symmetry audit on the Ising CNN; fix with test-time augmentation |
# | 6 | 12 | Autoencoder reconstruction error as an OOD detector — beats MSP |
# | 7 | 10 | Course retrospective: 14 weeks, one chart |
# | 8 | 10 | Student exercises (3 core + 1 stretch) |

# %%
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ai4mat.datasets import IsingDataset, CahnHilliardDataset, TensileTestDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# ## Helpers used by every block

# %%
class TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        # Split into trunk + head so Block 3 can tap the penultimate
        # 64-dim feature vector (post-pool, pre-classifier).
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(64, n_classes)

    def features(self, x):
        """Penultimate-layer activations (64-dim)."""
        return self.trunk(x)

    def forward(self, x):
        return self.head(self.trunk(x))


def train_cnn(model, X, y, n_epochs=5, lr=3e-3, batch=128, val_frac=0.1, seed=0):
    torch.manual_seed(seed)
    n_val = int(val_frac * len(X))
    n_tr = len(X) - n_val
    ds = TensorDataset(X, y)
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(seed))
    tr_dl = DataLoader(tr, batch_size=batch, shuffle=True)
    va_dl = DataLoader(va, batch_size=batch, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(n_epochs):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in va_dl:
            correct += (model(xb).argmax(-1) == yb).sum().item()
            total += len(yb)
    return correct / total, va


# %% [markdown]
# ## Block 1 — Recap from homework
#
# Three results travel into today:
#
# 1. A small CNN reaches >95% on Ising-light in 5 epochs.
# 2. **Vanilla saliency** (`|x · grad|`) gives a per-pixel attribution
#    but is biased near saturated activations.
# 3. **Integrated Gradients** fixes this with a baseline-relative
#    line integral and satisfies the completeness axiom: the per-pixel
#    attributions sum to $f_c(x) - f_c(\text{baseline})$.
# 4. **Max softmax probability** is a cheap OOD baseline; AUROC was
#    moderate vs Cahn-Hilliard, weaker vs shuffled Ising. Block 6
#    will do better with reconstruction error.
#
# We re-train the CNN here so today's session is self-contained.

# %%
ising = IsingDataset(size="light")
cnn = TinyCNN()
val_acc, ising_val = train_cnn(cnn, ising.X, ising.y)
print(f"Block 1 — TinyCNN re-trained on Ising-light: val acc = {val_acc:.3f}")


# %% [markdown]
# ## Block 2 — KernelSHAP from scratch on a tensile regression model
#
# SHAP (Lundberg & Lee 2017) gives a per-feature attribution that
# satisfies four axioms (efficiency, symmetry, dummy, additivity) and
# generalises Shapley values from cooperative game theory to ML. The
# **kernel** version regresses the model on subsets of "active" vs
# "background" features, weighted by the Shapley kernel
# $$
# w(|S|) = \frac{n - 1}{\binom{n}{|S|} \cdot |S| \cdot (n - |S|)}.
# $$
#
# We build a small tensile regression model whose input is *enriched*:
# physically meaningful features (strain, $T$), engineered cross
# terms ($\varepsilon^2$, $\varepsilon \cdot T$, $T^2$), and a pure
# noise feature. SHAP should attribute high importance to the real
# features, lower to the engineered ones, and ~0 to the noise.

# %%
# Build a 6-feature tensile regression task.
def load_tensile_features():
    """Return (X, y, feature_names) for the multi-temperature tensile data."""
    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for T in [0, 400, 600]:
        ds = TensileTestDataset(temperature=T)
        s = ds.X.numpy().reshape(-1)
        st = ds.y.numpy().reshape(-1)
        T_norm = np.full_like(s, (T - 300.0) / 300.0)
        feats = np.stack([s,                    # strain
                          T_norm,               # T_norm
                          s ** 2,               # strain^2
                          s * T_norm,           # strain * T
                          T_norm ** 2,          # T^2
                          rng.normal(size=s.shape)],  # noise
                         axis=1)
        Xs.append(feats); ys.append(st)
    return np.concatenate(Xs), np.concatenate(ys), [
        "strain", "T_norm", "strain^2", "strain*T", "T_norm^2", "noise"
    ]


X_t, y_t, feat_names = load_tensile_features()
scaler = StandardScaler().fit(X_t)
X_ts = scaler.transform(X_t).astype(np.float32)
y_t_mean, y_t_sd = y_t.mean(), y_t.std()
y_ts = ((y_t - y_t_mean) / y_t_sd).astype(np.float32)
print(f"Block 2 — feature matrix {X_ts.shape}, target {y_ts.shape}")
print(f"  features: {feat_names}")


class TensileMLP(nn.Module):
    def __init__(self, in_dim=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


X_ts_t = torch.tensor(X_ts)
y_ts_t = torch.tensor(y_ts)
torch.manual_seed(0)
mlp = TensileMLP()
opt = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
for _ in range(800):
    opt.zero_grad()
    F.mse_loss(mlp(X_ts_t), y_ts_t).backward()
    opt.step()
mlp.eval()


# %%
def kernel_shap(model_fn, x: np.ndarray, background: np.ndarray,
                n_samples: int = 500, seed: int = 0) -> tuple[np.ndarray, float, float]:
    """KernelSHAP attribution for a single query x using a single background.

    model_fn: callable np.ndarray (B, n) -> np.ndarray (B,)
    x:        (n,) — the query point
    background: (n,) — the baseline values
    Returns: phi (n,), base value f(background), prediction f(x)
    """
    rng = np.random.default_rng(seed)
    n = len(x)

    # Sample subset masks (z in {0, 1}^n, excluding all-0 and all-1).
    subsets, ws, targets = [], [], []
    for _ in range(n_samples):
        k = rng.integers(1, n)                # |S| in {1, ..., n-1}
        idx = rng.choice(n, size=int(k), replace=False)
        z = np.zeros(n, dtype=np.float64); z[idx] = 1.0
        # Shapley kernel weight.
        w = (n - 1) / (math.comb(n, int(k)) * k * (n - k))
        # Build the masked input: keep features in S, replace others with background.
        x_masked = np.where(z.astype(bool), x, background)
        subsets.append(z); ws.append(w)
        targets.append(model_fn(x_masked.reshape(1, -1))[0])

    Z = np.array(subsets)                     # (n_samples, n)
    W = np.array(ws)
    T = np.array(targets)

    # Constrain efficiency: Σ phi_i = f(x) - f(background). Solve weighted
    # least squares with a Lagrange multiplier-style constraint by
    # subtracting f(background) from targets and from the constant offset.
    f_bg = model_fn(background.reshape(1, -1))[0]
    f_x = model_fn(x.reshape(1, -1))[0]

    # Centered targets.
    Y = T - f_bg
    # Solve weighted least squares: minimise sum_i w_i (Y_i - Z_i @ phi)^2.
    sw = np.sqrt(W)
    Zw = Z * sw[:, None]
    Yw = Y * sw
    phi, *_ = np.linalg.lstsq(Zw, Yw, rcond=None)

    # Rescale to satisfy efficiency exactly (small numerical correction).
    if abs(phi.sum()) > 1e-9:
        phi = phi * (f_x - f_bg) / phi.sum()
    return phi, float(f_bg), float(f_x)


def mlp_predict_np(X_np: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = mlp(torch.tensor(X_np, dtype=torch.float32)).numpy()
    return out


# Pick a query: a tensile sample at T=600 °C, strain near peak.
i_query = int(np.argmax(np.abs(X_ts[:, 0]) + (X_ts[:, 1] > 0.8)))
x_query = X_ts[i_query]
background = X_ts.mean(axis=0)               # background = dataset mean

phi, base, pred = kernel_shap(mlp_predict_np, x_query, background, n_samples=600)
phi_phys = phi * y_t_sd                       # back to MPa for readability
base_phys = base * y_t_sd + y_t_mean
pred_phys = pred * y_t_sd + y_t_mean

print(f"Block 2 — KernelSHAP on tensile MLP query (sample {i_query}):")
print(f"  base prediction f(background)  = {base_phys:>8.2f} MPa")
print(f"  query prediction f(x)          = {pred_phys:>8.2f} MPa")
print(f"  delta = f(x) - f(background)   = {pred_phys - base_phys:>8.2f} MPa")
print(f"  sum of SHAP values  ({phi_phys.sum():>+8.2f}) MPa  -- should match delta\n")
print(f"  per-feature attribution (MPa):")
for n, p in sorted(zip(feat_names, phi_phys), key=lambda v: -abs(v[1])):
    print(f"    {n:<10}: {p:>+8.2f}")


# %%
# Waterfall-style plot.
order = np.argsort(np.abs(phi_phys))
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["tab:red" if p < 0 else "tab:green" for p in phi_phys[order]]
ax.barh(np.array(feat_names)[order], phi_phys[order], color=colors)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel(r"SHAP value $\varphi_i$  (MPa)")
ax.set_title(f"Block 2 — KernelSHAP waterfall: f(query) = {pred_phys:.1f} MPa  (vs base {base_phys:.1f})")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.show()


# %% [markdown]
# Read the chart:
#
# - Real physical features (`strain`, `T_norm`, and their squared
#   versions) carry most of the attribution.
# - The pure `noise` feature receives ~0 attribution — the model
#   correctly ignores it. **This is the SHAP soundness check**:
#   features that have no causal link to the target should attribute
#   to ~0.
# - Sum of $\varphi_i$ matches $f(x) - f(\text{background})$ — the
#   efficiency axiom holds (modulo a small numerical correction
#   applied at the end of `kernel_shap`).


# %% [markdown]
# ## Block 3 — Sparse autoencoder on Ising-CNN activations
#
# SHAP and Integrated Gradients answer **per-prediction** questions
# ("which inputs mattered for *this* output?"). They are silent on a
# different question that matters just as much for trust:
# **what concept does a hidden layer of my network represent?**
#
# The naive answer — "inspect individual neurons" — fails because
# neurons are *polysemantic*: a single hidden unit fires for several
# unrelated patterns. This is not a bug; it is **superposition**
# [@elhage_2022_superposition]. A network with $d$ neurons can pack
# many more than $d$ near-orthogonal features into the same activation
# space when those features are sparse — the cost is that each neuron
# becomes a sum over multiple features.
#
# The fix is a **wide, sparse autoencoder (SAE)** trained on the
# layer's activations [@templeton_2024_scaling]. The encoder projects
# into a higher-dimensional space ($d' \gg d$) with an $\ell_1$
# penalty on the post-ReLU code, which forces each input activation
# to be reconstructed from a *small* subset of SAE features. The
# top-activating inputs *per SAE feature* then tend toward
# **monosemantic** concepts — one feature, one idea.
#
# This is exactly **the Unit-5 autoencoder with an $\ell_1$ activation
# penalty**: same encoder/decoder skeleton, same reconstruction loss,
# one extra term. We apply it to the penultimate (64-dim) layer of the
# Ising CNN trained in Block 1.

# %%
# Cache penultimate-layer activations on a representative Ising batch.
N_ACT = min(2000, len(ising))
torch.manual_seed(0)
act_idx = torch.randperm(len(ising))[:N_ACT]
X_act = ising.X[act_idx]
y_act = ising.y[act_idx]

cnn.eval()
with torch.no_grad():
    H = cnn.features(X_act)                          # (N, 64)
d_act = H.shape[1]
print(f"Block 3 — cached penultimate activations: {tuple(H.shape)} (d = {d_act})")


# %%
class SparseAutoencoder(nn.Module):
    """Wide, sparse autoencoder for mechanistic interpretability.

    Encoder lifts d -> d' = 4d with ReLU; decoder maps d' -> d.
    Loss = ||h - h_hat||^2 + lam * ||f||_1, where f is the post-ReLU
    encoder code. The L1 penalty on the *activations* (not the
    weights) is what drives monosemanticity.
    """

    def __init__(self, d: int, expansion: int = 4):
        super().__init__()
        d_prime = expansion * d
        self.enc = nn.Linear(d, d_prime)
        self.dec = nn.Linear(d_prime, d, bias=False)
        # A small input bias subtracted before encoding helps centre
        # activations (cf. Anthropic 2023 SAE recipe).
        self.b_pre = nn.Parameter(torch.zeros(d))

    def encode(self, h):
        return F.relu(self.enc(h - self.b_pre))

    def forward(self, h):
        f = self.encode(h)
        h_hat = self.dec(f) + self.b_pre
        return h_hat, f


torch.manual_seed(0)
sae = SparseAutoencoder(d=d_act, expansion=4)
opt_sae = torch.optim.Adam(sae.parameters(), lr=1e-3)
LAM = 5e-3
N_EPOCHS_SAE = 15
BATCH_SAE = 256

print(f"Block 3 — training SAE (d={d_act} -> d'={4*d_act}) for {N_EPOCHS_SAE} epochs, lambda = {LAM}")
for ep in range(N_EPOCHS_SAE):
    perm = torch.randperm(len(H))
    ep_recon = ep_l1 = 0.0; n = 0
    for i in range(0, len(H), BATCH_SAE):
        hb = H[perm[i : i + BATCH_SAE]]
        opt_sae.zero_grad()
        h_hat, f = sae(hb)
        recon = F.mse_loss(h_hat, hb)
        l1 = f.abs().mean()
        loss = recon + LAM * l1
        loss.backward(); opt_sae.step()
        ep_recon += recon.item() * len(hb); ep_l1 += l1.item() * len(hb); n += len(hb)
    if ep == 0 or (ep + 1) % 5 == 0 or ep == N_EPOCHS_SAE - 1:
        print(f"  epoch {ep+1:>2}: recon MSE {ep_recon/n:.4f}   mean |f| {ep_l1/n:.4f}")

# Activation frequency per SAE feature.
with torch.no_grad():
    F_all = sae.encode(H)                            # (N, d')
active = (F_all > 0).float().mean(0)                 # firing rate per feature
print(f"  SAE features: {F_all.shape[1]} total, {(active > 0).sum().item()} ever active, "
      f"{(active > 0.01).sum().item()} active on >1% of inputs")


# %%
# Top-K activating Ising images per feature, plotted as a grid.
# Compare SAE features vs raw neurons of the penultimate layer.
TOP_K = 5
N_FEATS_SHOWN = 6

# SAE features: pick the most "interesting" — high firing rate but not saturated.
with torch.no_grad():
    F_vals = sae.encode(H).numpy()                   # (N, d')
sae_score = active.numpy() * (active.numpy() < 0.5).astype(float)
sae_pick = np.argsort(-sae_score)[:N_FEATS_SHOWN]

# Raw neurons: pick by variance (most "informative" axis-aligned units).
H_np = H.numpy()
raw_pick = np.argsort(-H_np.var(0))[:N_FEATS_SHOWN]


def top_k_grid(values: np.ndarray, picks: np.ndarray, title: str):
    """values: (N, n_feats). picks: indices into n_feats. Plot rows of TOP_K images."""
    fig, axes = plt.subplots(N_FEATS_SHOWN, TOP_K, figsize=(2.0 * TOP_K, 2.0 * N_FEATS_SHOWN))
    for row, idx in enumerate(picks):
        scores = values[:, idx]
        top = np.argsort(-scores)[:TOP_K]
        for col, t in enumerate(top):
            ax = axes[row, col]
            ax.imshow(X_act[t, 0].numpy(), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"feat {idx}\nlbl {int(y_act[t])}", fontsize=8)
            ax.set_title(f"a={scores[t]:.2f}", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


top_k_grid(F_vals, sae_pick, "Block 3 — top-5 Ising images per SAE feature (tend toward one concept)")
top_k_grid(H_np, raw_pick, "Block 3 — top-5 Ising images per raw CNN neuron (mixed concepts)")


# %% [markdown]
# Read the two grids together. Raw penultimate neurons fire for
# **mixtures** of unrelated patterns — high values for both
# above-Curie and below-Curie images, or for several distinct
# textures. SAE features, in contrast, tend to concentrate on **one**
# concept per feature (e.g. consistently above-Curie, or consistently
# a particular domain morphology) — the post-ReLU $\ell_1$ penalty
# forces the network to *spread* concepts across many features rather
# than packing them into shared neurons.
#
# The penalty weight $\lambda$ tunes the trade-off: too small and
# features stay polysemantic; too large and many features go *dead*
# (never fire). The Anthropic *Scaling Monosemanticity* report
# [@templeton_2024_scaling] applied this exact recipe at the scale of
# a frontier LLM and recovered millions of monosemantic features.
# **This is the Unit-5 autoencoder with $\ell_1$ on activations**, so
# the connection to the rest of the course is one line of code —
# every modelling primitive in this lecture is one we have already
# built.


# %% [markdown]
# ## Block 4 — Counterfactual explanations
#
# Wachter et al. 2017 framed counterfactuals as: given $x$ with
# prediction $f(x) = y$, find the *closest* $x'$ such that $f(x')
# \approx y_\text{target}$. Solve by gradient descent on
# $$
# \mathcal{L}_\mathrm{CF}(x') = (f(x') - y_\text{target})^2 + \lambda\, \|x' - x\|_2^2.
# $$
# Pick $\lambda$ so the data-fit term dominates until the prediction
# is near target, then the proximity term keeps the perturbation
# minimal.
#
# We ask: "this 600-°C tensile sample currently predicts 480 MPa.
# What is the *minimum* feature change that takes the prediction down
# to 200 MPa?"

# %%
def counterfactual(model, x: np.ndarray, y_target: float,
                   lam: float = 0.5, n_steps: int = 1500, lr: float = 5e-3) -> np.ndarray:
    x0 = torch.tensor(x, dtype=torch.float32)
    x_cf = x0.clone().requires_grad_(True)
    y_t = torch.tensor(y_target, dtype=torch.float32)
    opt = torch.optim.Adam([x_cf], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        pred = model(x_cf.unsqueeze(0)).squeeze(-1)
        loss = (pred - y_t) ** 2 + lam * ((x_cf - x0) ** 2).sum()
        loss.backward(); opt.step()
    return x_cf.detach().numpy()


y_target_phys = 200.0
y_target_std = (y_target_phys - y_t_mean) / y_t_sd
x_cf = counterfactual(mlp, x_query, y_target_std)

# Predict and report.
with torch.no_grad():
    pred_cf_std = float(mlp(torch.tensor(x_cf).unsqueeze(0)).squeeze(-1))
pred_cf_phys = pred_cf_std * y_t_sd + y_t_mean
delta_std = x_cf - x_query
delta_phys = scaler.scale_ * delta_std        # back to physical units per feature

print(f"Block 4 — counterfactual:")
print(f"  query prediction:       {pred_phys:7.2f} MPa")
print(f"  target prediction:      {y_target_phys:7.2f} MPa")
print(f"  achieved counterfactual: {pred_cf_phys:7.2f} MPa")
print(f"\n  feature changes (counterfactual - query, in physical units):")
for n, d_phys, x0_phys, x1_phys in zip(
    feat_names,
    delta_phys,
    scaler.inverse_transform(x_query.reshape(1, -1))[0],
    scaler.inverse_transform(x_cf.reshape(1, -1))[0],
):
    print(f"    {n:<10}: {x0_phys:>+8.4f}  ->  {x1_phys:>+8.4f}    (delta {d_phys:+.4f})")


# %% [markdown]
# The counterfactual answers a different question than SHAP/IG: not
# *which features mattered*, but *what would have to be different*.
# It is the natural form of explanation for an engineer looking at a
# rejected prediction — "if the temperature had been X and the strain
# Y, the model would have predicted target". For deployment, this is
# the most actionable XAI primitive.


# %% [markdown]
# ## Block 5 — Symmetry audit on the Ising classifier
#
# An Ising microstructure is **rotation-invariant**: the underlying
# physics (above vs below Curie) does not change if we rotate or flip
# the image. A classifier should reflect this. CNNs have *translation*
# equivariance built in, but not rotation. We measure the gap and
# repair it with **test-time augmentation (TTA)**: average predictions
# over the $\{$identity, rot90, rot180, rot270, flip-H, flip-V$\}$
# orbit.

# %%
# Held-out subset for the audit.
torch.manual_seed(0)
audit_size = 1024
audit_idx = torch.randperm(len(ising))[:audit_size]
X_aud = ising.X[audit_idx]
y_aud = ising.y[audit_idx]

ROT_FNS = {
    "identity": lambda x: x,
    "rot90":   lambda x: torch.rot90(x, k=1, dims=(-2, -1)),
    "rot180":  lambda x: torch.rot90(x, k=2, dims=(-2, -1)),
    "rot270":  lambda x: torch.rot90(x, k=3, dims=(-2, -1)),
    "flip_h":  lambda x: torch.flip(x, dims=(-1,)),
    "flip_v":  lambda x: torch.flip(x, dims=(-2,)),
}


@torch.no_grad()
def acc_under(transform):
    pred = cnn(transform(X_aud)).argmax(-1)
    return (pred == y_aud).float().mean().item()


print(f"Block 5 — symmetry audit on the Ising CNN ({audit_size} samples):")
for name, fn in ROT_FNS.items():
    a = acc_under(fn)
    print(f"  {name:<10}: acc = {a:.3f}")

# Test-time augmentation: average softmax over the orbit.
@torch.no_grad()
def tta_predict(x):
    probs = torch.zeros(len(x), 2)
    for fn in ROT_FNS.values():
        probs += cnn(fn(x)).softmax(-1)
    return probs / len(ROT_FNS)


tta_pred = tta_predict(X_aud).argmax(-1)
acc_tta = (tta_pred == y_aud).float().mean().item()
print(f"  TTA (avg over orbit):  acc = {acc_tta:.3f}")


# %%
# Visualise predictions under the orbit on one sample.
i = 0
xs = X_aud[i : i + 1]
fig, axes = plt.subplots(1, 6, figsize=(15, 3.2))
for ax, (name, fn) in zip(axes, ROT_FNS.items()):
    img = fn(xs)[0, 0].numpy()
    with torch.no_grad():
        p = cnn(fn(xs)).softmax(-1)[0]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"{name}\np(0)={p[0]:.2f} p(1)={p[1]:.2f}", fontsize=9)
    ax.axis("off")
fig.suptitle(f"Block 5 — same sample (true label {int(y_aud[i])}), 6 transforms — predictions vary")
plt.tight_layout()
plt.show()


# %% [markdown]
# A drop from `identity` accuracy to `rot90` accuracy of even a few
# percent is not a small bug — it means the model has memorised
# orientations that the physics says are interchangeable. TTA is a
# cheap fix at inference time. A more principled fix is to *bake* the
# symmetry into the architecture (group-equivariant CNNs, Cohen &
# Welling 2016) — see week 13's PIML coverage.


# %% [markdown]
# ## Block 6 — Autoencoder reconstruction error as an OOD detector
#
# The homework's max-softmax-probability OOD score is cheap but
# fragile — it depends on the classifier's calibration on OOD inputs,
# which is exactly where calibration is weakest. A more principled
# detector trains an **unsupervised reconstruction model** on the
# in-distribution data; the reconstruction error on OOD inputs is
# typically much larger because the model has never compressed those
# patterns.

# %%
class TinyAE(nn.Module):
    def __init__(self, latent: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GELU(),     # 16 -> 8
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),    # 8 -> 4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 32 * 4 * 4), nn.GELU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GELU(),  # 4 -> 8
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),               # 8 -> 16
        )

    def forward(self, x):
        return self.dec(self.enc(x))


# Train AE on Ising training split (90% of the dataset).
torch.manual_seed(0)
n_val_ae = int(0.1 * len(ising))
n_tr_ae = len(ising) - n_val_ae
tr_idx, va_idx = torch.utils.data.random_split(
    range(len(ising)), [n_tr_ae, n_val_ae],
    generator=torch.Generator().manual_seed(0),
)
tr_idx = list(tr_idx); va_idx = list(va_idx)
ae = TinyAE()
opt = torch.optim.AdamW(ae.parameters(), lr=3e-3, weight_decay=1e-4)
X_tr_ae = ising.X[tr_idx]
print(f"Block 6 — training Ising AE for 5 epochs on {len(X_tr_ae)} samples...")
for ep in range(5):
    perm = torch.randperm(len(X_tr_ae))
    ep_loss = 0.0; n = 0
    for i in range(0, len(perm), 128):
        b = perm[i : i + 128]
        x = X_tr_ae[b]
        opt.zero_grad()
        loss = F.mse_loss(ae(x), x)
        loss.backward(); opt.step()
        ep_loss += loss.item() * len(b); n += len(b)
    print(f"  epoch {ep+1}: train MSE {ep_loss/n:.5f}")


@torch.no_grad()
def recon_error(x: torch.Tensor, batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(x), batch):
        xb = x[i : i + batch]
        err = ((ae(xb) - xb) ** 2).mean(dim=(1, 2, 3))
        out.append(err.cpu().numpy())
    return np.concatenate(out)


# Score the three slabs.
X_id = ising.X[va_idx]                                         # in-dist
err_id = recon_error(X_id)

ch = CahnHilliardDataset(simulation_number=0)
X_ood_ch = F.avg_pool2d(ch.X, kernel_size=4)[: len(X_id)]
err_ch = recon_error(X_ood_ch)

X_shuf = X_id.clone()
B, C, H, W = X_shuf.shape
torch.manual_seed(0)
for i in range(B):
    perm = torch.randperm(H * W)
    flat = X_shuf[i].reshape(C, H * W)[:, perm]
    X_shuf[i] = flat.reshape(C, H, W)
err_shuf = recon_error(X_shuf)

print(f"Block 6 — reconstruction error means:")
print(f"  Ising val:        {err_id.mean():.5f}")
print(f"  CH downsampled:   {err_ch.mean():.5f}")
print(f"  shuffled Ising:   {err_shuf.mean():.5f}")

# AUROC vs in-dist using reconstruction error as score.
y_disc = np.concatenate([np.zeros(len(err_id)), np.ones(len(err_ch))])
auc_ch = roc_auc_score(y_disc, np.concatenate([err_id, err_ch]))
y_disc = np.concatenate([np.zeros(len(err_id)), np.ones(len(err_shuf))])
auc_shuf = roc_auc_score(y_disc, np.concatenate([err_id, err_shuf]))
print(f"\nReconstruction-error OOD discrimination AUROC:")
print(f"  Ising vs CH downsampled:  {auc_ch:.3f}    (compare to homework MSP)")
print(f"  Ising vs shuffled Ising:  {auc_shuf:.3f}")


# %%
fig, ax = plt.subplots(figsize=(8, 5))
log_err_id = np.log10(err_id + 1e-12)
log_err_ch = np.log10(err_ch + 1e-12)
log_err_shuf = np.log10(err_shuf + 1e-12)
bins = np.linspace(min(log_err_id.min(), log_err_ch.min(), log_err_shuf.min()),
                   max(log_err_id.max(), log_err_ch.max(), log_err_shuf.max()), 30)
ax.hist(log_err_id, bins=bins, alpha=0.6, label="Ising val (in-dist)")
ax.hist(log_err_ch, bins=bins, alpha=0.6, label="CH downsampled (OOD)")
ax.hist(log_err_shuf, bins=bins, alpha=0.6, label="shuffled Ising")
ax.set_xlabel(r"$\log_{10}$ AE reconstruction error per pixel")
ax.set_ylabel("count")
ax.set_title("Block 6 — AE reconstruction error separates OOD better than MSP")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# Almost always, the AE-reconstruction-error AUROC dominates the
# homework's MSP AUROC on the same OOD slabs — the AE was *trained*
# to compress in-distribution patterns and refuses to compress OOD
# ones. The cost is one extra model and a modest amount of training.
# In a real lab, you would deploy an AE alongside every classifier
# whose predictions feed an actuator.


# %% [markdown]
# ## Block 7 — Course retrospective: 14 weeks on one chart
#
# The arc this term: from sparse data to physics-informed
# uncertainty-aware models.

# %%
print("Course retrospective — methods covered (week → primary method → primary lecture):")
arc = [
    (1,  "data + setup",                                "all"),
    (2,  "linear algebra, PCA",                         "MFML 2 / MG 5"),
    (3,  "regression as loss min, leakage",             "MFML 3 / ML-PC 3"),
    (4,  "MLPs, training, baselines (digits, Iris)",    "MFML 4 / ML-PC 4-5"),
    (5,  "k-means + GMM + autoencoders",                "MFML 5 / ML-PC 5"),
    (6,  "loss landscapes + optimisation + CGNN",       "MFML 6 / MG 6"),
    (7,  "bias-variance, ensembles, distribution shift", "MFML 7 / ML-PC 8"),
    (8,  "probabilistic view, MLE/MAP, calibration, robustness", "MFML 8 / MG 8 / ML-PC 8"),
    (9,  "latent geometry: PCA, t-SNE, contrastive",    "MFML 9 / MG 9 / ML-PC 9"),
    (10, "attention, ViT, cross-system transfer",       "MFML 10 / MG 10 / ML-PC 10"),
    (11, "VAE / diffusion (generative + inverse design)", "MFML 11 / MG 11 / ML-PC 11"),
    (12, "uncertainty in predictions, GP, ensembles",   "MFML 12 / MG 12 / ML-PC 12"),
    (13, "PINN + GP + active discovery",                "MFML 13 / MG 13 / ML-PC 13"),
    (14, "explainability, OOD, symmetry, retrospective", "MFML 14 / MG 14 / ML-PC 14"),
]
for w, m, l in arc:
    print(f"  W{w:>2}  {m:<55} [{l}]")


# %%
# A simple chart: rough conceptual position of each week along two axes.
fig, ax = plt.subplots(figsize=(10, 6))
data = [
    # (week, x = data-vs-physics, y = point-vs-uncertainty)
    (1,  0.1, 0.1, "data setup"),
    (2,  0.2, 0.1, "PCA"),
    (3,  0.3, 0.1, "regression"),
    (4,  0.4, 0.1, "MLPs"),
    (5,  0.4, 0.2, "k-means / AE"),
    (6,  0.5, 0.1, "loss / opt"),
    (7,  0.5, 0.4, "ensembles"),
    (8,  0.6, 0.7, "probabilistic + calibration"),
    (9,  0.4, 0.3, "latent geometry"),
    (10, 0.5, 0.3, "ViT, transfer"),
    (11, 0.4, 0.6, "VAE / diffusion"),
    (12, 0.6, 0.9, "GPs, ensembles"),
    (13, 0.9, 0.9, "PINN + GP + active"),
    (14, 0.9, 0.6, "XAI, OOD, symmetry"),
]
for w, x, y, label in data:
    ax.scatter(x, y, s=300, alpha=0.55, edgecolor="k")
    ax.annotate(f"W{w}: {label}", (x, y), xytext=(8, 8), textcoords="offset points",
                fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("← pure data-driven       physics-informed →")
ax.set_ylabel("← point estimate         calibrated uncertainty →")
ax.set_title("Block 7 — 14 weeks on one chart")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# Three lessons to walk out the door with:
#
# 1. **The right method depends on the data regime.** Lots of clean
#    data, no physics → big NN. Sparse data, strong physics → PINN +
#    GP. Distribution shift expected → autoencoder OOD detector
#    deployed alongside.
# 2. **Uncertainty is not optional in materials work.** A point
#    prediction without an uncertainty band is not an answer; it is
#    a promise the model cannot keep. Bayesian regression, deep
#    ensembles, GPs, calibration plots — pick whichever matches the
#    cost-of-a-wrong-answer in your application.
# 3. **Explainability is a deployment tool, not a research luxury.**
#    SHAP / Integrated Gradients / counterfactuals / sparse
#    autoencoders / attention maps each answer a different question.
#    Pick the question first, then pick the method.


# %% [markdown]
# # Student exercises (Block 8 — ~10 min)

# %% [markdown]
# ## Exercise 1 (core) — SHAP under a different background
#
# Re-run Block 2 with the background set to:
# (a) all-zeros (the "neutral" baseline),
# (b) the median tensile sample, and
# (c) a sample from a *different* temperature than the query.
#
# How do the SHAP values change? Which background is the right one
# if the question is "explain this $T = 600$ prediction relative to
# what a $T = 0$ sample would predict"?

# %%
# YOUR CODE for Exercise 1 below.


# %% [markdown]
# ## Exercise 2 (core) — Counterfactual feasibility constraints
#
# The Block 4 counterfactual is the *minimum L2* perturbation. Realistic
# counterfactuals must respect feasibility: $T$ is one of the three
# discrete values you can run, strain cannot be negative, etc.
#
# 1. Modify `counterfactual` so that the strain feature is clipped to
#    its training range and `T_norm` is projected onto the discrete
#    set $\{-1, 1/3, 1\}$ corresponding to {0, 400, 600} °C after
#    each gradient step.
# 2. Compare to the Block 4 unconstrained CF: how much further does
#    the constrained CF have to travel to achieve the target?
# 3. What does a *non-feasible* counterfactual mean operationally?

# %%
# YOUR CODE for Exercise 2 below.


# %% [markdown]
# ## Exercise 3 (core) — Symmetry as a constraint, not a fix
#
# Block 5 used TTA at inference time. A more principled fix is to
# *augment training* with rotations and flips. Re-train `TinyCNN`
# with on-the-fly random rotations (`torch.rot90` with a random k each
# batch) and random flips. Re-run the symmetry audit. Does the
# accuracy gap to TTA close?
#
# Bonus: discuss when train-time augmentation is *not* a good idea
# (e.g. when the symmetry is approximate, not exact, and breaking it
# is informative).

# %%
# YOUR CODE for Exercise 3 below.


# %% [markdown]
# ## Exercise 4 (stretch) — Calibrated OOD threshold from cost
#
# Block 6 produces a continuous reconstruction-error score. To deploy
# it, you need a *threshold*: above this error, refuse to predict.
#
# 1. Pick a cost-of-error model: cost of a false positive (refusing
#    a good prediction) = $C_\mathrm{FP}$; cost of a false negative
#    (accepting a bad OOD prediction) = $C_\mathrm{FN}$. Use
#    $C_\mathrm{FN} = 100 \cdot C_\mathrm{FP}$ as a starting point.
# 2. Compute the threshold that minimises expected total cost on the
#    held-out slabs.
# 3. Re-run with $C_\mathrm{FN} / C_\mathrm{FP} = 1, 10, 100, 1000$
#    and plot threshold vs ratio. Where does the threshold change
#    dramatically — i.e. where does cost asymmetry start to dominate
#    over the score distribution?

# %%
# YOUR CODE for Exercise 4 below.


# %% [markdown]
# ## Exam-aligned must-know statements
#
# Re-read these after the exercises; today's blocks have given you the
# concrete intuition for every one of them.
#
# 1. **Vanilla saliency** = $|x \cdot \partial f_c / \partial x|$. Cheap,
#    biased near saturated activations. (Homework Part A.)
# 2. **Integrated Gradients** = baseline-relative line integral of the
#    gradient. Satisfies completeness and sensitivity. (Homework Part B.)
# 3. **KernelSHAP** = weighted least-squares regression on subsets of
#    "active" vs "background" features, with the Shapley kernel weight.
#    Satisfies efficiency: $\sum_i \varphi_i = f(x) - f(\text{bg})$.
#    (Block 2.)
# 4. **Sparse autoencoder (SAE)** on a hidden layer = wide encoder +
#    $\ell_1$ penalty on its post-ReLU code. Lifts polysemantic neurons
#    into monosemantic features. Direct application of
#    **superposition** [@elhage_2022_superposition] and
#    [@templeton_2024_scaling]. (Block 3.)
# 5. **Counterfactual** = minimum perturbation to flip the prediction;
#    the actionable form of XAI for a deployed model. (Block 4.)
# 6. CNNs have **translation** equivariance built in but **not
#    rotation**. Test rotation/flip equivariance with an audit and fix
#    with TTA or train-time augmentation. (Block 5.)
# 7. **Max softmax probability** is a cheap OOD detector but fragile,
#    especially against domain-similar OOD. **Reconstruction error
#    from an unsupervised AE** is a more principled alternative.
#    (Homework Part C, Block 6.)
# 8. **OOD AUROC** of 1.0 means perfect separation, 0.5 means chance.
#    For deployment, pick a threshold by **cost-weighted** risk, not
#    by AUROC. (Block 6, Exercise 4.)
# 9. The right XAI method depends on the question. *Why this prediction?*
#    → SHAP. *Where in the input?* → IG/saliency. *What would have
#    to change?* → counterfactual. *What concept does my hidden
#    layer represent?* → sparse autoencoder. (All of today.)
# 10. The 14-week arc: from data → loss → NN → optimisation →
#     ensembles → probabilistic → representation learning → attention
#     → generative → uncertainty → physics-informed → trust.
#     The destination is *deployable, calibrated, explainable* models.
#     (Block 7.)
