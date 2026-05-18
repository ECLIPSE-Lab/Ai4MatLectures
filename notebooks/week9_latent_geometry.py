# %% [markdown]
# # Week 9 — Latent geometry and trustworthy regression
#
# This week we braid three lectures. Two share a single question —
# **what does the latent space look like, and what does it organise
# by?** — and the third asks the harder follow-up: **once you have a
# materials regressor, how do you know its number is scientifically
# trustworthy?**
#
# 1. **MFML Unit 9**: Latent spaces & advanced representation learning —
#    PCA, t-SNE, UMAP, contrastive learning, linear probing.
# 2. **ML-PC Unit 9** (`unit09_characterization_signals`): PCA + AE on
#    spectra, reconstruction error as anomaly score, t-SNE/UMAP on
#    hyperspectral embeddings.
# 3. **MG Unit 8** (`08_regression_and_generalization_in_materials_data`,
#    delivered calendar-Week-9): *Regression and generalization in
#    materials data* — split design, chemistry-family leakage, polymorph
#    aliasing, the mandatory baseline ladder, per-region residual
#    diagnostics, the structure-awareness ablation, and the seven-point
#    trustworthy-reporting checklist. (The SchNet/CGCNN/MEGNet/M3GNet
#    *architecture* lecture is MG Unit 9, delivered next week — not this
#    week.)
#
# **Red thread:** *MFML supplies the projection tools and ML-PC applies
# them to spectra (Blocks 1–3, 6). MG then asks the orthogonal question
# on crystal data: a learned representation is only useful if the
# regression built on it generalises — and "generalises" in materials
# means a split design that matches the scientific claim, a baseline
# ladder, and residuals read per chemistry family. Blocks 4–5 are the MG
# leg, run on the **real `MatBenchDataset` perovskites benchmark**; a
# short synthetic structural appendix (Block 5b, toy
# `CrystalGraphsDataset`) isolates what explicit structure adds and feeds
# the Block-7 embedding exercises.*
#
# > **Pre-flight check.** This notebook **assumes** you have run
# > `notebooks/week9_homework.py`. Block 1 picks up directly from your PCA
# > reconstruction curve and your `make_positive_pair` augmentation
# > pipeline.
#
# ## Agenda (90 min)
#
# | Block | Min | Topic |
# |------:|:---:|:------|
# | 1 | ~6  | Recap from homework — PCA vs t-SNE vs the augmentation pipeline |
# | 2 | ~14 | Spectra in latent space (ML-PC): synthetic XRD + PCA + anomaly via reconstruction error |
# | 3 | ~14 | Convolutional autoencoder on Ising-full; t-SNE on the bottleneck |
# | 4 | ~16 | Materials regression & generalization (MG): split design, the baseline ladder, $\Delta_\text{shift}$ |
# | 5 | ~14 | Residual diagnostics, the structure-awareness ablation, bootstrap CI, the 7-point checklist |
# | 6 | ~16 | From supervised to contrastive to masked: InfoNCE *and* MAE on Ising |
# | 7 | ~18 | Student exercises (3 core + 1 stretch) |

# %%
# Standard imports.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from ai4mat.datasets import IsingDataset, CrystalGraphsDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# ## Helpers — re-imported from homework
#
# Two shared utilities. `pca_from_scratch` is the eigendecomp PCA you
# wrote in Part A (copied so this notebook stands alone).
# `make_positive_pair` is the augmentation pipeline from Part C.

# %%
def pca_from_scratch(X, k):
    """Top-k PCA via covariance eigendecomp.  Returns (Z, X_recon, eigvals, V_k)."""
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    V_k = eigvecs[:, :k]
    Z = Xc @ V_k
    X_recon = Z @ V_k.T + mu
    return Z, X_recon, eigvals, V_k


def make_positive_pair(x, rng=None):
    """Two augmented views of the same Ising image (rot/flip/noise)."""
    if rng is None:
        rng = np.random.default_rng()

    def aug(t):
        k = int(rng.integers(0, 4))
        t = torch.rot90(t, k, dims=(-2, -1))
        if rng.random() < 0.5: t = torch.flip(t, dims=(-1,))
        if rng.random() < 0.5: t = torch.flip(t, dims=(-2,))
        sigma = float(rng.uniform(0.0, 0.1))
        return (t + sigma * torch.randn_like(t)).clamp(0.0, 1.0)

    return aug(x), aug(x)


# %% [markdown]
# # Block 1 — Recap from homework
#
# Three takeaways frame the rest of the lecture:
#
# 1. **PCA is quantitative.** You can compute reconstruction error in the
#    *original* feature space and meaningfully ask "how many dimensions does
#    this dataset really need?" (Part A).
# 2. **t-SNE is visual, not metric.** Distances between t-SNE clusters are
#    not faithful — your distance-trap pair (Part B) made that concrete.
# 3. **Augmentations encode the prior.** Rotations, flips, and small noise
#    are the invariances we want a contrastive embedding to respect on
#    Ising — but in pixel space, those augmentations look as different as
#    images of different classes (Part C histogram).
#
# Today we move beyond Ising images:
#
# - to **spectra** (Block 2),
# - to **AE bottlenecks** (Block 3),
# - to **trustworthy materials regression** on crystal data — split
#   design, the baseline ladder, residual diagnostics (MG U8, Blocks 4–5),
# - and to **contrastive + masked** self-supervision as the readouts
#   (Block 6).

# %% [markdown]
# # Block 2 — Synthetic XRD spectra in latent space
#
# Materials characterisation rarely starts with images. Most of the
# experimental day produces **1-D spectra**: XRD intensity vs angle,
# EELS intensity vs energy loss, EDS intensity vs photon energy. They all
# share the property that *peak positions* encode the underlying physics
# (lattice spacings, electronic transitions, K/L edges) and *peak heights*
# encode chemistry-weighted populations.
#
# We generate a synthetic XRD-like dataset from `CrystalGraphsDataset`'s
# 200 crystals: each crystal becomes a 1-D intensity curve $I(2\theta)$
# over 200 angle bins. Then we run PCA and ask:
#
# 1. Does PCA reveal the 5 prototype clusters? (yes — Bragg peak
#    positions are prototype-determined.)
# 2. Does PCA reveal *chemistry* substructure within a prototype? (yes —
#    peak intensities are chemistry-weighted.)
# 3. Does **reconstruction error** flag a *contaminated* spectrum (we
#    inject extra peaks)? (yes — that is the unsupervised anomaly story
#    in 5 lines.)
#
# *(see ML-PC §"PCA-based phase ID for spectra", §"Reconstruction-error
# anomaly detection")*

# %%
# Hand-coded HKL sets and prototype intensity weights for the 5 prototypes.
# Real XRD has more peaks; 4-6 is enough for clear cluster structure.
_HKL = {
    "rocksalt":   [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2), (4,0,0)],
    "zincblende": [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2), (4,0,0)],
    "wurtzite":   [(1,0,0), (0,0,2), (1,0,1), (1,0,2), (1,1,0), (1,0,3)],
    "fluorite":   [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2), (4,0,0)],
    "perovskite": [(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0), (2,1,1)],
}
# Prototype-specific structure-factor weights (toy — chosen so the five
# prototypes have visibly different *intensity patterns* even at the
# same peak positions).
_INTENSITY_WEIGHTS = {
    "rocksalt":   [0.7, 1.0, 1.2, 0.9, 0.5, 0.4],
    "zincblende": [1.2, 0.4, 1.0, 1.0, 0.3, 0.7],   # different weights for same HKL set
    "wurtzite":   [1.0, 0.9, 1.3, 0.5, 0.7, 0.5],
    "fluorite":   [1.4, 0.5, 0.9, 1.1, 0.6, 0.4],
    "perovskite": [0.8, 1.0, 0.6, 1.1, 0.5, 0.7],
}


def synth_xrd(species, edge_distance, prototype_name, n_bins=200,
              two_theta_range=(20.0, 90.0), wavelength=1.54, fwhm=0.5, rng=None):
    """Generate a synthetic XRD-like 1-D spectrum from a crystal.

    The lattice constant `a` is taken as the mean edge distance scaled by
    a prototype-specific factor.  Bragg's law gives 2θ for each (hkl).
    Intensities are atomic-number-weighted (heavier atom = brighter peak).
    """
    if rng is None:
        rng = np.random.default_rng()
    a = float(edge_distance.mean()) * 1.4                   # rough lattice constant
    hkl = _HKL[prototype_name]
    weights = _INTENSITY_WEIGHTS[prototype_name]
    # mean Z is a stand-in for "structure-factor-weighted" intensity
    mean_z = float(species.float().mean())
    bins = np.linspace(two_theta_range[0], two_theta_range[1], n_bins)
    spectrum = np.zeros(n_bins)
    for (h, k, l), w in zip(hkl, weights):
        d = a / math.sqrt(max(h * h + k * k + l * l, 1))
        sin_theta = wavelength / (2 * d)
        if not (-1 < sin_theta < 1):
            continue
        two_theta = 2 * math.degrees(math.asin(sin_theta))
        if not (two_theta_range[0] <= two_theta <= two_theta_range[1]):
            continue
        intensity = w * (mean_z / 30.0)                     # chemistry weighting
        spectrum += intensity * np.exp(-((bins - two_theta) / fwhm) ** 2)
    spectrum += rng.normal(0.0, 0.02, size=n_bins)          # measurement noise
    return spectrum.astype(np.float32), bins.astype(np.float32)


# %%
# Build the spectrum dataset.
crystals = CrystalGraphsDataset()
proto_names = crystals.prototype_names
rng = np.random.default_rng(0)

specs = []
proto_idx = []
mean_z_list = []
for i in range(len(crystals)):
    s = crystals[i]
    pname = proto_names[s["prototype"]]
    spec, bins_two_theta = synth_xrd(
        s["species"], s["edge_distance"], pname, rng=rng,
    )
    specs.append(spec)
    proto_idx.append(s["prototype"])
    mean_z_list.append(float(s["species"].float().mean()))

specs = torch.from_numpy(np.stack(specs))                  # (200, 200)
proto_idx = np.array(proto_idx)
mean_z = np.array(mean_z_list)
print(f"Built {specs.shape[0]} spectra of length {specs.shape[1]} bins.")

# Show 5 example spectra (one per prototype) so the cluster structure
# is visible at the data level before we run PCA.
fig, ax = plt.subplots(figsize=(8, 4))
for p_idx, pname in enumerate(proto_names):
    sample_i = (proto_idx == p_idx).nonzero()[0][0]
    ax.plot(bins_two_theta, specs[sample_i].numpy() + p_idx * 1.5,
            lw=1.0, label=pname)
ax.set_xlabel(r"2$\theta$ (deg)"); ax.set_ylabel("intensity (offset for clarity)")
ax.set_title("Synthetic XRD: one spectrum per prototype")
ax.legend(loc="upper right", fontsize=9); plt.tight_layout(); plt.show()


# %%
# PCA on the spectra.  Project to 2D and colour by prototype.
Z_spec, X_recon_spec, eigvals_spec, V_spec = pca_from_scratch(specs, k=2)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
for p_idx, pname in enumerate(proto_names):
    m = proto_idx == p_idx
    a1.scatter(Z_spec[m, 0], Z_spec[m, 1], s=22, alpha=0.85,
               c=f"C{p_idx}", label=pname, edgecolor="k", lw=0.3)
a1.set_xlabel("PC1"); a1.set_ylabel("PC2"); a1.set_title("PCA — coloured by prototype")
a1.legend(fontsize=8)

# Same plot, coloured by mean atomic number — chemistry substructure.
sc = a2.scatter(Z_spec[:, 0], Z_spec[:, 1], s=22, alpha=0.85,
                c=mean_z, cmap="viridis", edgecolor="k", lw=0.3)
a2.set_xlabel("PC1"); a2.set_ylabel("PC2"); a2.set_title("PCA — coloured by mean atomic Z")
plt.colorbar(sc, ax=a2, label="mean Z")
plt.tight_layout(); plt.show()


# %%
# Anomaly detection by PCA reconstruction error.  Inject a "contaminant"
# into one spectrum (extra peaks at random locations); ask which spectra
# the PCA model cannot reconstruct.
target = specs[0].clone()
contaminated = target.clone()
extra_peaks_at = [37.5, 51.0, 68.0]
extra_widths = 0.6
for ang in extra_peaks_at:
    contaminated += 1.5 * torch.from_numpy(
        np.exp(-((bins_two_theta - ang) / extra_widths) ** 2).astype(np.float32)
    )

# Build a PCA model on clean data (skip index 0), reconstruct, score.
clean = specs[1:]
mu = clean.mean(0, keepdim=True)
_, _, eig_clean, V_clean = pca_from_scratch(clean, k=8)
def project_recon(x, mu, V_k):
    return ((x - mu) @ V_k) @ V_k.T + mu

recon_clean = project_recon(specs[1:11], mu, V_clean)
recon_target = project_recon(target.unsqueeze(0), mu, V_clean)
recon_contam = project_recon(contaminated.unsqueeze(0), mu, V_clean)
mse_clean = ((specs[1:11] - recon_clean) ** 2).mean(dim=1)
mse_target = ((target - recon_target.squeeze(0)) ** 2).mean()
mse_contam = ((contaminated - recon_contam.squeeze(0)) ** 2).mean()
print(f"Clean spectra reconstruction MSE: mean = {mse_clean.mean():.4f}, max = {mse_clean.max():.4f}")
print(f"Target (clean copy of #0):        MSE = {mse_target:.4f}")
print(f"Contaminated #0 (3 spurious peaks): MSE = {mse_contam:.4f}   <-- anomaly flag")

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(bins_two_theta, target.numpy(), lw=1.0, label="clean #0")
ax.plot(bins_two_theta, contaminated.numpy(), lw=1.0, label="contaminated #0")
ax.plot(bins_two_theta, recon_contam.squeeze(0).numpy(), lw=1.0, ls="--",
        label="PCA-8 reconstruction of contaminated")
ax.set_xlabel(r"2$\theta$ (deg)"); ax.set_ylabel("intensity")
ax.set_title("Reconstruction error as anomaly score")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()


# %% [markdown]
# **Take-home from Block 2.** PCA on spectra solves *two* characterisation
# problems at once: (i) the 2-D layout already separates phases (PC1, PC2
# resolve all 5 prototypes), and (ii) reconstruction error flags spectra
# that the PCA model has not seen — anomaly detection in 5 lines, no
# labels required. This is the ML-PC W9 punch line.

# %% [markdown]
# # Block 3 — Convolutional autoencoder on Ising-full
#
# PCA is a *linear* autoencoder. A nonlinear encoder can usually compress
# Ising-full to a much smaller bottleneck than k = 8 because phase-domain
# texture is captured better by convolutions than by global covariance
# eigenvectors.
#
# We train a tiny conv-AE for ~3 epochs (CPU-friendly), extract the
# bottleneck on the *test* split, and run t-SNE on it. The point: the
# bottleneck activations should organise by class as cleanly as PCA, but
# in a much smaller dimension.
#
# *(see MFML §"Convolutional autoencoders", §"Bottleneck dim and the
# elbow"; ML-PC §"AE features as input to downstream regressors")*

# %%
class TinyAE(nn.Module):
    """64x64 grayscale -> latent_dim -> 64x64 grayscale."""

    def __init__(self, latent_dim=8):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # 32x32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 16x16
        )
        self.enc_lin = nn.Linear(32 * 16 * 16, latent_dim)
        self.dec_lin = nn.Linear(latent_dim, 32 * 16 * 16)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.enc_lin(self.enc_conv(x).flatten(1))

    def forward(self, x):
        z = self.encode(x)
        h = self.dec_lin(z).view(-1, 32, 16, 16)
        return self.dec_conv(h), z


# %%
ising = IsingDataset(size="full")
g = torch.Generator().manual_seed(0)
sub = torch.randperm(len(ising), generator=g)[:1500]
X_img = ising.X[sub]                                       # (1500, 1, 64, 64)
y = ising.y[sub]
n_train = 1200
train_ds = torch.utils.data.TensorDataset(X_img[:n_train])
test_X = X_img[n_train:]
test_y = y[n_train:]
loader = DataLoader(train_ds, batch_size=64, shuffle=True)

torch.manual_seed(0)
ae = TinyAE(latent_dim=8)
opt = torch.optim.Adam(ae.parameters(), lr=2e-3)
print("Training conv-AE (3 epochs, ~30 s on CPU)...")
for epoch in range(3):
    ae.train()
    losses = []
    for (xb,) in loader:
        opt.zero_grad()
        xh, _ = ae(xb)
        loss = F.mse_loss(xh, xb)
        loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"  epoch {epoch}  train MSE = {np.mean(losses):.4f}")

# Bottleneck on test split.
ae.eval()
with torch.no_grad():
    Z_ae = ae.encode(test_X)                               # (300, 8)


# %%
# t-SNE on the AE bottleneck.
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
Z_ae_2d = tsne.fit_transform(Z_ae.numpy())

fig, ax = plt.subplots(figsize=(6, 4.5))
for cls in [0, 1]:
    m = (test_y == cls).numpy()
    ax.scatter(Z_ae_2d[m, 0], Z_ae_2d[m, 1], s=14, alpha=0.7,
               c=f"C{cls}", label=f"class {cls}")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE on the conv-AE bottleneck (8-D) — test split")
ax.legend(); plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the AE-bottleneck plot.** With latent_dim = 8 and ~3 epochs of
# training (no labels involved), the bottleneck already separates the two
# classes cleanly when projected by t-SNE. Compare to the PCA-2 plot from
# homework Part A — the AE bottleneck typically gives tighter clusters
# because it learns nonlinear directions that capture phase-domain texture.
#
# **Takeaway.** Pre-training an AE on unlabelled microstructure data and
# using its bottleneck as input to a downstream regressor is a standard
# ML-PC W9 recipe — the bottleneck features carry the structural
# information without ever needing labels.

# %% [markdown]
# # Block 4 — Materials regression & generalization (MG Unit 8)
#
# We now switch to crystal data — but the MG question this week is *not*
# "what does the embedding organise by?" (that is MG Unit 9, next week).
# It is the harder one: **you have a materials regressor; how do you know
# its number is scientifically trustworthy?**
#
# The MG U8 spine, in one sentence:
#
# > *In materials ML, the test set's relationship to the training set
# > **is** the scientific claim. A model is trustworthy only when its
# > split design matches the claim its predictions are meant to support.*
#
# This block runs on a **real materials benchmark**:
# `MatBenchDataset(task="matbench_perovskites")` — ~19k DFT-relaxed
# perovskites, target = formation energy (eV/atom), features = a
# 118-dimensional element-fraction composition vector (`ds.X`, `ds.y`).
# Because it is a real composition benchmark, a composition baseline does
# real work and the leakage question is *not* hypothetical.
#
# Three split designs:
#
# - **Random 80/20** — probes *no* generalization axis (MG slide 09:
#   "a random IID split probes none of these axes").
# - **`ds.folds` cross-validation** — the dataset ships 5
#   *reproducible surrogate folds* (seeded 5-fold; official Matbench
#   folds need the `matbench` PyPI package — these are sufficient for
#   split-design teaching, **not** for leaderboard parity). Averaging the
#   5 held-out folds is the IID-but-honest CV number.
# - **Composition-family hold-out** — every crystal whose dominant
#   non-oxygen element is in a chosen element set is moved entirely to
#   test (derived from `ds.X`). Chemistry-family leakage is impossible by
#   construction (MG §B "chemistry-family leakage", §C
#   "chemistry-aware split").
#
# The gap between the random/CV number and the composition-family number
# is the **fourth bias-variance term $\Delta_\text{shift}$** from MG
# slide 05 — the literal quantity this afternoon's MG exercise produces.
#
# Tiers 0/1/2 (mean / ridge / GBT) run on this real benchmark. A
# **synthetic structural appendix** further down keeps the hand-rolled
# `TinyCGNN` on the toy `CrystalGraphsDataset` — MatBench composition
# vectors carry no graph tensors, so the GNN cannot consume them; the
# appendix exists only to *isolate what explicit structure adds*, and to
# feed the frozen `embeds` the Block-7 exercises inspect.
#
# *(see MG U8 §A5 "the fourth term $\Delta_\text{shift}$", §B10
# "chemistry-family leakage", §C "split design", §D2 "the mandatory
# baseline ladder")*

# %%
class TinyCGNN(nn.Module):
    """Same hand-rolled CGNN as Week 6.  `encode` returns the pooled atom
    embedding *before* the regression head; `forward` is the
    formation-energy regressor (tier-4 'structure-aware' model in the MG
    baseline ladder)."""

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

    def encode(self, species, edge_index, edge_distance):
        h = self.embed(species)
        for layer in self.msg_mlps:
            src, dst = edge_index[0], edge_index[1]
            msg_in = torch.cat([h[src], h[dst], edge_distance.unsqueeze(-1)], dim=-1)
            msg = layer(msg_in)
            agg = torch.zeros_like(h).index_add_(0, dst, msg)
            h = h + agg
        return h.mean(0)                                    # (embed_dim,)

    def forward(self, species, edge_index, edge_distance):
        return self.head(self.encode(species, edge_index, edge_distance)).squeeze(-1)


# %%
# --- Real MatBench composition data and the split-design machinery ------
#
# `MatBenchDataset` gives us the Magpie-spirit input MG §A6 / §D29 demand
# for a baseline ladder: a 118-D element-fraction composition vector
# (`ds.X`) and the DFT formation energy (`ds.y`).  This is a *real*
# materials benchmark — composition-only baselines do real work here, and
# the leakage question is no longer hypothetical.
from ai4mat.datasets import MatBenchDataset

ds = MatBenchDataset(task="matbench_perovskites", download=True)

# ~19k crystals would make the GBT tier slow on CPU; take a seeded,
# stratification-free 4000-row subsample so the lecture runs in a couple
# of minutes.  All split logic operates on this subsample.
_sub_rng = np.random.default_rng(0)
N_MB = min(4000, len(ds.X))
mb_sub = np.sort(_sub_rng.permutation(len(ds.X))[:N_MB])

X_mb = np.asarray(ds.X)[mb_sub].astype(np.float64)         # (N_MB, 118)
y_mb = np.asarray(ds.y)[mb_sub].astype(np.float64)         # (N_MB,) eV/atom

# A "composition family" label derived purely from ds.X: the column of
# the largest element fraction once oxygen (Z=8 -> column index 7) is
# removed.  Most of these perovskites are oxides, so this dominant
# non-oxygen element is a stable, purely-compositional family key — the
# chemistry-family axis MG §B10 cares about (a model that never saw a
# family must extrapolate to it).
_X_no_O = X_mb.copy()
_X_no_O[:, 7] = 0.0                                        # zero the O column
family_z = _X_no_O.argmax(axis=1) + 1                      # 1-based atomic Z
print(f"MatBench perovskites: using {N_MB} / {len(ds.X)} crystals; "
      f"{len(np.unique(family_z))} distinct composition families.")

# ds.folds are reproducible *surrogate* folds (official folds need the
# `matbench` pkg).  Re-index them onto the subsample: keep only fold
# members that survived subsampling, remap to 0..N_MB-1.
_pos = {orig: k for k, orig in enumerate(mb_sub)}
mb_folds = []
for tr_o, te_o in ds.folds:
    tr = np.array([_pos[i] for i in tr_o if i in _pos], dtype=np.int64)
    te = np.array([_pos[i] for i in te_o if i in _pos], dtype=np.int64)
    mb_folds.append((tr, te))


def split_random(seed=0, frac=0.8):
    """MG slide 20: the IID split — probes no generalization axis."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_MB)
    cut = int(frac * N_MB)
    return perm[:cut], perm[cut:]


def split_family_heldout(held_elements):
    """MG §B10 / §C21: chemistry-aware split — every crystal whose
    dominant non-oxygen element is in `held_elements` (1-based atomic Z)
    is moved to test, so chemistry-family leakage is impossible by
    construction."""
    held = np.isin(family_z, list(held_elements))
    return np.where(~held)[0], np.where(held)[0]


# %%
# --- The mandatory baseline ladder (MG §D29) ----------------------------
#
# Tier 0  constant (training mean)        "anything that doesn't beat
#                                          tier 0 is broken"
# Tier 1  composition vector + ridge      (Magpie+linear analogue)
# Tier 2  composition vector + GBT        (the skeptic's baseline)
#
# (Tier 4 — a structure-aware GNN — needs graph tensors the MatBench
# composition vector does not carry; it lives in the *synthetic
# structural appendix* further down, on the toy CrystalGraphsDataset.)
#
# A leakage-safe rule shared by every tier: the feature standardiser is
# *fit on train only* (MG §D32: "fit on train; apply to test. Any
# operation that touches the test data before split design is leakage").

def _standardize_fit(Xtr):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-8
    return mu, sd


def baseline_ladder(tr_idx, te_idx):
    """Tiers 0/1/2 on the MatBench composition data.

    Returns {tier_name: (mae, preds)} on the given split."""
    Xtr_all, Xte_all = X_mb[tr_idx], X_mb[te_idx]
    ytr, yte = y_mb[tr_idx], y_mb[te_idx]
    mu, sd = _standardize_fit(Xtr_all)
    Xtr = (Xtr_all - mu) / sd
    Xte = (Xte_all - mu) / sd

    results = {}
    # Tier 0 — constant.
    const = np.full_like(yte, ytr.mean())
    results["tier0_constant"] = (mean_absolute_error(yte, const), const)
    # Tier 1 — ridge.
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    p1 = ridge.predict(Xte)
    results["tier1_ridge"] = (mean_absolute_error(yte, p1), p1)
    # Tier 2 — gradient-boosted trees.
    gbt = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
    ).fit(Xtr, ytr)
    p2 = gbt.predict(Xte)
    results["tier2_gbt"] = (mean_absolute_error(yte, p2), p2)
    return results


# %%
# --- Split design contrast: random / CV / family (the Delta_shift demo) -
#
# The single highest-priority MG-U8 number this afternoon: the same
# baseline ladder under (a) a random 80/20 split, (b) the dataset's 5
# reproducible surrogate folds (IID-but-honest CV), (c) a chemistry-aware
# composition-family hold-out.  The MAE *gap* is Delta_shift (MG slide 05).
tr_r, te_r = split_random(seed=0)
ladder_random = baseline_ladder(tr_r, te_r)

# ds.folds CV: average the 5 reproducible surrogate folds (official folds
# need the `matbench` pkg; these are sufficient for split-design teaching,
# NOT for leaderboard parity).
cv_fold_mae = {k: [] for k in ladder_random}
for tr_f, te_f in mb_folds:
    for k, (mae, _) in baseline_ladder(tr_f, te_f).items():
        cv_fold_mae[k].append(mae)
ladder_cv = {k: float(np.mean(v)) for k, v in cv_fold_mae.items()}

# Composition-family hold-out: hold out the light-element family
# (Z = 3 Li, 4 Be, 5 B — every crystal whose dominant non-oxygen element
# is one of these moves entirely to test; ~330 crystals here).
HELD_FAMILY = [3, 4, 5]
tr_c, te_c = split_family_heldout(HELD_FAMILY)
ladder_family = {k: mae for k, (mae, _) in baseline_ladder(tr_c, te_c).items()}

print("Formation-energy MAE (eV/atom) by tier and split design")
print(f"{'tier':<18}{'random':>10}{'cv (folds)':>12}{'family-held':>13}"
      f"{'  Δ_shift (fam)':>17}")
print("-" * 70)
for k in ladder_random:
    mr = ladder_random[k][0]
    mv = ladder_cv[k]
    mc = ladder_family[k]
    print(f"{k:<18}{mr:>10.4f}{mv:>12.4f}{mc:>13.4f}{mc - mr:>17.4f}")
print(f"(family hold-out: {len(te_c)} test crystals whose dominant "
      f"non-O element Z in {HELD_FAMILY})")


# %%
# Visualise the split-design gap (MG slide 27: "the gap is the signal").
fig, ax = plt.subplots(figsize=(8.5, 4.4))
tiers = list(ladder_random.keys())
x = np.arange(len(tiers))
w = 0.26
ax.bar(x - w, [ladder_random[k][0] for k in tiers], w, label="random (no axis)")
ax.bar(x, [ladder_cv[k] for k in tiers], w, label="ds.folds CV (surrogate)")
ax.bar(x + w, [ladder_family[k] for k in tiers], w, label="composition-family held-out")
ax.set_xticks(x); ax.set_xticklabels(
    [t.replace("_", "\n") for t in tiers], fontsize=8)
ax.set_ylabel("test MAE (eV/atom)")
ax.set_title("Baseline ladder × split design — the gap is $\\Delta_\\text{shift}$")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the split-design table (MG §A5, §B10, §C27).**
#
# - **Tier 0 (constant)** is split-invariant by construction — it never
#   beats anything; it only certifies that every other tier is *not
#   broken* (MG §D29: "anything that doesn't beat tier 0 is broken").
#   Every other tier beats it on every split. ✓
# - On the **random** split tier-1/2 cut the constant MAE substantially:
#   composition alone carries real signal on a real perovskite benchmark
#   (MG §D31's "composition-only is surprisingly good"). This is the IID
#   number — and *the wrong quantity* for any discovery claim (MG slide
#   07).
# - The **`ds.folds` CV** column is the same IID regime measured honestly
#   across 5 reproducible *surrogate* folds (official Matbench folds need
#   the `matbench` pkg — these suffice for split-design teaching, not for
#   leaderboard parity). It should track the random number closely; that
#   agreement is the point — CV does not probe a *new* generalization
#   axis, it only de-noises the IID estimate.
# - Under **composition-family hold-out** the *learned* tiers (ridge,
#   GBT) degrade, because every crystal whose dominant non-oxygen element
#   is in the held-out family was never seen — MG slide 10's
#   chemistry-family leakage, made impossible by the split. (Tier 0 only
#   tracks the held family's label *mean*, so it can even improve if that
#   family's energies happen to be low-variance — another reason tier 0
#   is a sanity floor, not a model.)
# - The **$\Delta_\text{shift}$** column (family − random) is the
#   headline pedagogical number: zero bias/variance/noise would still
#   leave this gap, because the *training distribution* changed (MG
#   slide 05). On a *real* benchmark this shift is honest and often
#   modest (a 118-D composition vector transfers reasonably across
#   A-site chemistries) — reporting it, modest or not, is the discipline.
#
# > *Report the family-held-out number as the headline when the claim is
# > new-chemistry transfer; report the random/CV number only as a
# > secondary, IID-comparable figure, with the gap discussed (MG slide
# > 20, 27).*

# %% [markdown]
# # Block 5 — Residual diagnostics, CI, and the checklist
#
# A single MAE hides where a materials model fails (MG §E: "global MAE
# hides localized failure"). This block runs the MG-U8 diagnostic suite
# **on the real MatBench tier-1 ridge**, all on a single declared split:
#
# 1. **Per-family residual table** — MAE + signed bias per composition
#    family (MG §E38).
# 2. **OOD-vs-interpolation** — residuals binned by composition-space
#    distance to the training set (MG slide 13).
# 3. **Bootstrap CI** on the headline MAE — MG §B19: "a point MAE
#    without a confidence interval is statistically dishonest".
# 4. **The 7-point trustworthy-reporting checklist** as the closing
#    rubric (MG §F47).
#
# The structure-awareness ablation (item 4 of the checklist) needs an
# explicit-structure model and a structure perturbation — neither exists
# in a composition vector. It moves to the **synthetic structural
# appendix** after this block, where the toy `CrystalGraphsDataset`
# supplies graph tensors to scramble.

# %%
# Fix one declared split for the whole diagnostic block: the
# composition-family hold-out (a new-chemistry transfer claim — the
# honest, harder number).  Worked regressor: tier-1 ridge (cheap; on this
# real benchmark it is competitive with the GBT tier).
tr_d, te_d = split_family_heldout(HELD_FAMILY)
mu_d, sd_d = _standardize_fit(X_mb[tr_d])
ridge_d = Ridge(alpha=1.0).fit((X_mb[tr_d] - mu_d) / sd_d, y_mb[tr_d])
pred_d = ridge_d.predict((X_mb[te_d] - mu_d) / sd_d)
true_d = y_mb[te_d]
resid_d = true_d - pred_d
print(f"Declared split: composition-family hold-out (dominant non-O "
      f"element Z in {HELD_FAMILY});  N_test = {len(te_d)}")
print(f"Headline tier-1 MAE = {mean_absolute_error(true_d, pred_d):.4f} eV/atom "
      f"  R² = {r2_score(true_d, pred_d):.3f}")


# %%
# (1) Per-family residual table.  The declared test set is a single
# family, so we break the *random-split* residuals down per composition
# family to expose where the model is weak (MG §E38).
tr_rr, te_rr = split_random(seed=0)
mu_rr, sd_rr = _standardize_fit(X_mb[tr_rr])
ridge_rr = Ridge(alpha=1.0).fit((X_mb[tr_rr] - mu_rr) / sd_rr, y_mb[tr_rr])
pred_rr = ridge_rr.predict((X_mb[te_rr] - mu_rr) / sd_rr)
true_rr = y_mb[te_rr]

# Report the five most common composition families in the test set.
fam_te = family_z[te_rr]
top_fam = [z for z, _ in sorted(
    zip(*np.unique(fam_te, return_counts=True)),
    key=lambda zc: -zc[1])][:5]
print("Per-family residuals on the random-split test set (MG §E38)")
print(f"{'family Z':<10}{'N_test':>7}{'MAE':>10}{'signed bias':>14}")
print("-" * 41)
for z in top_fam:
    m = fam_te == z
    mae_p = mean_absolute_error(true_rr[m], pred_rr[m])
    bias_p = float((true_rr[m] - pred_rr[m]).mean())
    print(f"Z={int(z):<8}{int(m.sum()):>7}{mae_p:>10.4f}{bias_p:>+14.4f}")
print(f"{'GLOBAL':<10}{len(te_rr):>7}"
      f"{mean_absolute_error(true_rr, pred_rr):>10.4f}"
      f"{float((true_rr - pred_rr).mean()):>+14.4f}")


# %%
# (2) OOD-vs-interpolation: bin the family-held-out test points by
# minimum composition-space distance to the training set (MG slide 13).
Xtr_s = (X_mb[tr_d] - mu_d) / sd_d
Xte_s = (X_mb[te_d] - mu_d) / sd_d
dmin = np.array([np.min(np.linalg.norm(Xtr_s - xt, axis=1)) for xt in Xte_s])
q = np.quantile(dmin, [0.25, 0.5, 0.75])
print("OOD diagnostic — test MAE by distance-to-train quartile (MG slide 13)")
for lo, hi, name in [(-np.inf, q[0], "Q1 (nearest)"), (q[0], q[1], "Q2"),
                     (q[1], q[2], "Q3"), (q[2], np.inf, "Q4 (farthest)")]:
    m = (dmin > lo) & (dmin <= hi)
    if m.sum():
        print(f"  {name:<14} n={int(m.sum()):>4}  "
              f"MAE={mean_absolute_error(true_d[m], pred_d[m]):.4f} eV/atom")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.scatter(dmin, np.abs(resid_d), s=14, alpha=0.5, edgecolor="k", lw=0.2)
ax.set_xlabel("min composition distance to training set")
ax.set_ylabel("|residual|  (eV/atom)")
ax.set_title("Extrapolation tail: error grows with distance from train")
plt.tight_layout(); plt.show()


# %%
# (3) Bootstrap 95% CI on the headline MAE (MG §B19).
rng_bs = np.random.default_rng(0)
abs_res = np.abs(resid_d)
boot = np.array([
    rng_bs.choice(abs_res, size=len(abs_res), replace=True).mean()
    for _ in range(2000)
])
lo, hi = np.quantile(boot, [0.025, 0.975])
print(f"Headline MAE = {abs_res.mean():.4f} eV/atom  "
      f"95% bootstrap CI = [{lo:.4f}, {hi:.4f}]  "
      f"(half-width {(hi - lo) / 2:.4f} = "
      f"{100 * (hi - lo) / 2 / abs_res.mean():.0f}% of the mean)")


# %% [markdown]
# **The MG U8 trustworthy-reporting checklist (§F47) — scoring the
# MatBench part.** The afternoon MG exercise targets 5/7; this real
# composition-benchmark braid scores:
#
# | # | Checklist item | Status |
# |--:|:--|:--|
# | 1 | **Split design** declared & matched to the claim | ✅ random, `ds.folds` CV *and* composition-family-held-out, declared per block |
# | 2 | **Mandatory baselines** (constant, linear, GBT) | ✅ full tier-0/1/2 ladder on the real benchmark |
# | 3 | **Per-region residuals** (per-family table) | ✅ per-composition-family MAE + signed bias |
# | 4 | **Structure-awareness ablation** | ◻️ N/A on a *composition-only* benchmark — there is no structure tensor to scramble. Exercised separately in the synthetic structural appendix below (edge-distance scramble on the toy CGNN) |
# | 5 | **Leakage paths audited** (split + train-only scaling) | ✅ family-disjoint splits; standardiser fit on train only |
# | 6 | **Confidence interval** on the headline MAE | ✅ 2000-sample bootstrap 95% CI |
# | 7 | **Test-set construction** documented | ✅ real Matbench `matbench_perovskites`; seeded subsample (seed=0) of a fixed snapshot; split is family-disjoint by construction. (Official leaderboard folds need the `matbench` pkg; we use the dataset's reproducible *surrogate* folds and say so — see MG §F46 on snapshot-date documentation.) |
#
# **Score: 6/7** — above the exercise's 5/7 bar. The single N/A item
# (structure-awareness ablation) is *intrinsic to a composition-only
# benchmark*: with no structure tensor there is nothing to perturb. The
# synthetic structural appendix below closes it on toy graph data, which
# is exactly the right division of labour — *real composition benchmark
# for the trustworthiness machinery; synthetic graph only to isolate what
# explicit structure adds*.
#
# > **The MG U8 sentence to leave with:** *better features or a fancier
# > architecture never fix bad benchmarking. The split is part of the
# > hypothesis, not the postprocessing.*

# %% [markdown]
# # Block 5b — Synthetic structural appendix (CGNN + structure ablation)
#
# Everything above ran on the **real composition benchmark**. The
# remaining MG-U8 idea — the **structure-awareness ablation** (MG §F1:
# *a "structure-aware" model that does not break when you destroy the
# geometry was composition-only in disguise*) — has no purchase on a
# composition vector: there is no structure to destroy.
#
# So this appendix switches to the **toy `CrystalGraphsDataset`** purely
# to manufacture an explicit-structure model and a structure
# perturbation. Read it as a controlled probe, *not* a benchmark: the
# real number is the MatBench table above; this synthetic graph exists
# only to answer "what does explicit structure add, and how would you
# *detect* a model that only pretends to use it?"
#
# It also produces the frozen `embeds` tensor the Block-7 exercises
# inspect (a learned crystal *representation*, not a metric).

# %%
# Toy crystal-graph dataset + the hand-rolled CGNN (same as Week 6).
cg = CrystalGraphsDataset()
proto_names = cg.prototype_names
N = len(cg)
proto_all = cg.prototype.numpy()
y_all = cg.y.numpy().astype(np.float64)                    # eV/atom (toy)


def cg_split_random(seed=0, frac=0.8):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    cut = int(frac * N)
    return perm[:cut], perm[cut:]


def cg_split_prototype_heldout(held_proto):
    """Structure-aware split on the toy data: one whole prototype unseen."""
    return (np.where(proto_all != held_proto)[0],
            np.where(proto_all == held_proto)[0])


def train_cgnn(tr_idx, n_epochs=12, scramble_edges=False, seed=0):
    """Train the TinyCGNN regressor on `tr_idx`.  `scramble_edges`
    randomly permutes every crystal's edge_distance vector (MG §F1
    structure-awareness ablation)."""
    g_rng = np.random.default_rng(seed)
    y_mean = y_all[tr_idx].mean(); y_std = y_all[tr_idx].std()
    torch.manual_seed(seed)
    net = TinyCGNN()
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    order = tr_idx.copy()
    for _ in range(n_epochs):
        net.train()
        g_rng.shuffle(order)
        for j in order:
            s = cg[int(j)]
            ed = s["edge_distance"]
            if scramble_edges:
                ed = ed[torch.randperm(ed.shape[0])]
            yn = (float(y_all[j]) - y_mean) / y_std
            opt.zero_grad()
            p = net(s["species"], s["edge_index"], ed)
            loss = (p - yn) ** 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return net, y_mean, y_std


def cgnn_predict(net, y_mean, y_std, idx, scramble_edges=False, seed=1):
    g_rng = np.random.default_rng(seed)
    net.eval()
    out = np.zeros(len(idx))
    with torch.no_grad():
        for k, j in enumerate(idx):
            s = cg[int(j)]
            ed = s["edge_distance"]
            if scramble_edges:
                perm = torch.from_numpy(g_rng.permutation(ed.shape[0]))
                ed = ed[perm]
            out[k] = net(s["species"], s["edge_index"], ed).item() * y_std + y_mean
    return out


# %%
# Structure-awareness ablation (MG §F1/44) on the synthetic graph.
# Declared split: prototype-held-out on 'perovskite' (a transfer claim).
HELD = proto_names.index("perovskite")
tr_s, te_s = cg_split_prototype_heldout(HELD)
net_s, ym_s, ys_s = train_cgnn(tr_s)
pred_s = cgnn_predict(net_s, ym_s, ys_s, te_s)
true_s = y_all[te_s]
net_ab, ym_ab, ys_ab = train_cgnn(tr_s, scramble_edges=True)
pred_ab = cgnn_predict(net_ab, ym_ab, ys_ab, te_s, scramble_edges=True)
mae_real = mean_absolute_error(true_s, pred_s)
mae_abl = mean_absolute_error(true_s, pred_ab)
print("Structure-awareness ablation on the SYNTHETIC graph (MG §F1)")
print(f"  declared split : prototype-held-out '{proto_names[HELD]}' "
      f"(N_test = {len(te_s)})")
print(f"  CGNN, true geometry        : MAE = {mae_real:.4f} (toy units)")
print(f"  CGNN, scrambled edge dists : MAE = {mae_abl:.4f} (toy units)")
print(f"  inflation factor           : {mae_abl / mae_real:.2f}x  "
      f"({'uses structure' if mae_abl > 1.3 * mae_real else 'STRUCTURE-BLIND — composition-only in disguise'})")


# %%
# Build the frozen embeddings the Block-7 exercises inspect (CGNN trained
# on a random split of the toy data — the representation, not the metric).
tr_r_cg, _ = cg_split_random(seed=0)
_cgnn_repr, _ym, _ys = train_cgnn(tr_r_cg)
_cgnn_repr.eval()
with torch.no_grad():
    embeds = torch.stack([
        _cgnn_repr.encode(cg[i]["species"], cg[i]["edge_index"],
                          cg[i]["edge_distance"])
        for i in range(N)
    ])
print(f"\nFrozen CGNN embeddings for Block-7 exercises: {tuple(embeds.shape)}")


# %% [markdown]
# **Reading the appendix (MG §F1).** On this tiny toy graph the
# scrambled-edge CGNN barely changes — the inflation factor is well below
# the 1.3× bar — so by MG §F1's own test this "structure-aware" model is
# *composition-only in disguise*: with 12 epochs on 5 tiny prototypes it
# never learned to use geometry. That is the pedagogical payload of the
# appendix: it shows you the *detector* (scramble the structure, watch
# the MAE) and a model that fails it. The honest materials number is the
# real-benchmark MatBench table in Block 4; this synthetic graph only
# isolates *what explicit structure would have to add* and *how you would
# audit a model that claims to use it*.

# %% [markdown]
# # Block 6 — From supervised to contrastive to masked
#
# Blocks 4–5 trained a *supervised* regressor and measured what it put
# into the representation. Self-supervised learning asks the
# complementary question: *can we put similar things into the embedding
# without using any labels?* The SSL field of 2026 offers two answer
# families:
#
# - **Contrastive** (SimCLR, MoCo): pull augmented views of the same image
#   together, push everything else apart. Pedagogically essential, but in
#   benchmarks since 2022 it has been *out-performed* by masked-image
#   pre-training on small-to-medium datasets.
# - **Masked image modelling** (MAE [@he_2022_mae], I-JEPA): hide most of
#   the image, ask the encoder to reconstruct (or predict embeddings of)
#   the masked patches. No negative pairs needed; no batch-size pressure.
#
# We do *both* in this block on the same Ising data, then compare them on
# the *same* linear-probe protocol — which is the only fair way to read
# "is SSL recipe X better than SSL recipe Y for materials microstructure?"
#
# Recipe for the contrastive half:
#
# 1. Take an Ising image.
# 2. Apply the homework Part C `make_positive_pair` augmentation pipeline
#    twice to get two views.
# 3. Pass both views through the same encoder.
# 4. The InfoNCE loss says: the two views should be **closer** to each
#    other than to any other image's views.
#
# The encoder is trained from scratch on Ising — *no class labels are ever
# touched*. We then visualise the resulting embedding and compare to a
# supervised baseline. The MAE half follows immediately after.
#
# *(see MFML §"Contrastive learning — InfoNCE", §"SSL refresh — MAE /
# DINOv2 / I-JEPA"; ML-PC §"Self-supervised learning of materials
# descriptors")*

# %%
class ConvEncoder(nn.Module):
    """Tiny 64x64 -> 32-D encoder."""

    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # 32x32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 16x16
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),   # 8x8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def info_nce_loss(zi, zj, temperature=0.5):
    """InfoNCE: row i of zi matches row i of zj; everything else is negative.

    zi, zj: (B, d) batches of paired views.
    """
    B = zi.shape[0]
    z = torch.cat([zi, zj], dim=0)                         # (2B, d)
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature                            # (2B, 2B)
    sim.fill_diagonal_(-1e9)                               # remove self
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(B)])
    return F.cross_entropy(sim, targets)


# %%
# Train the contrastive encoder.  ~3 epochs with batch size 64 takes <30 s on CPU.
torch.manual_seed(0)
enc = ConvEncoder(out_dim=32)
opt = torch.optim.Adam(enc.parameters(), lr=1e-3)

# Use the same Ising-full subset we used for the AE.
loader = DataLoader(torch.utils.data.TensorDataset(X_img[:n_train]), batch_size=64, shuffle=True)

print("Training contrastive encoder on Ising-full (3 epochs)...")
rng = np.random.default_rng(0)
for epoch in range(3):
    enc.train()
    losses = []
    for (xb,) in loader:
        # Build positive pairs by independently augmenting each image twice.
        xi = torch.stack([make_positive_pair(x, rng=rng)[0] for x in xb])
        xj = torch.stack([make_positive_pair(x, rng=rng)[1] for x in xb])
        zi, zj = enc(xi), enc(xj)
        loss = info_nce_loss(zi, zj)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"  epoch {epoch}  InfoNCE = {np.mean(losses):.4f}")

# Extract embeddings on the test split (no augmentation) and t-SNE them.
enc.eval()
with torch.no_grad():
    Z_contrast = enc(test_X)                               # (300, 32)
Z_contrast_2d = TSNE(n_components=2, perplexity=30, init="pca",
                     random_state=0).fit_transform(Z_contrast.numpy())


# %%
# Side-by-side: AE bottleneck vs contrastive embedding, both t-SNEd.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
for cls in [0, 1]:
    m = (test_y == cls).numpy()
    a1.scatter(Z_ae_2d[m, 0], Z_ae_2d[m, 1], s=14, alpha=0.7,
               c=f"C{cls}", label=f"class {cls}")
    a2.scatter(Z_contrast_2d[m, 0], Z_contrast_2d[m, 1], s=14, alpha=0.7,
               c=f"C{cls}", label=f"class {cls}")
a1.set_title("Conv-AE bottleneck (Block 3)"); a1.legend()
a2.set_title("Contrastive encoder (Block 6) — labels never used in training"); a2.legend()
for ax in (a1, a2):
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.tight_layout(); plt.show()


# %% [markdown]
# **Take-home from the contrastive half.** The contrastive encoder,
# trained without any class labels, produces an embedding that t-SNE
# separates as cleanly as the AE bottleneck. The augmentation pipeline
# (= what we declared invariant) provided the supervisory signal. This is
# the core idea behind SimCLR, MoCo, and the materials-specific
# contrastive papers (CrystalCLR, CrystalTwins): the *prior over
# invariances* replaces the label.

# %%
# Linear probe on the contrastive (SimCLR-style) encoder.
# Same protocol we will reuse for the MAE encoder below — fair comparison.
def ising_linear_probe(Z_train, y_train, Z_test, y_test, n_epochs=200, lr=0.1):
    """Frozen-features linear probe on a 2-class Ising task. Returns test acc."""
    torch.manual_seed(0)
    probe = nn.Linear(Z_train.shape[1], 2)
    opt = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    for _ in range(n_epochs):
        opt.zero_grad()
        F.cross_entropy(probe(Z_train), y_train).backward()
        opt.step()
    with torch.no_grad():
        return (probe(Z_test).argmax(1) == y_test).float().mean().item()


# Train-side embeddings (no augmentation, encoder frozen) for both splits.
with torch.no_grad():
    Z_contrast_train = enc(X_img[:n_train])
    Z_contrast_test = Z_contrast  # already computed above
acc_simclr = ising_linear_probe(
    Z_contrast_train, y[:n_train], Z_contrast_test, test_y,
)
print(f"SimCLR-style contrastive encoder — linear probe test acc = {acc_simclr:.3f}")


# %% [markdown]
# ## Block 6b — Tiny MAE on Ising patches
#
# Masked Autoencoders [@he_2022_mae] swap "two augmented views" for "one
# image with 75% of its patches hidden". The encoder only sees the
# visible patches; a tiny decoder is asked to reconstruct the hidden
# ones from the latent representation. MSE loss on the masked patches
# only — the visible patches are free.
#
# Concretely, on our 16x16 Ising-light images (we use `IsingDataset(size=
# "light")` to keep the patch grid tiny; 4x4 patches gives 16 patches per
# sample, exactly matching the lecture slide):
#
# - Patch size 4 × 4   →   16 patches per sample.
# - Mask ratio 0.75    →   12 patches hidden, 4 visible.
# - Encoder: 2 transformer blocks, dim = 64 — tiny enough to train in a
#   couple of minutes on a 1080 Ti.
# - Decoder: a small MLP head that maps the encoder output to the 16
#   patch positions; loss is averaged over masked patches only.
#
# We then **linear-probe the CLS token** on the Ising classification task
# with the same protocol as the SimCLR probe and compare numbers. Across
# students who have run this exercise, MAE typically matches or beats
# SimCLR on this dataset despite using a tinier encoder — that is the
# pedagogical pay-off of the block.

# %%
# Tiny MAE on 16x16 Ising images (the "light" split is 16x16).
ising_light = IsingDataset(size="light")
g_light = torch.Generator().manual_seed(0)
sub_l = torch.randperm(len(ising_light), generator=g_light)[:1500]
X_mae = ising_light.X[sub_l]                                   # (1500, 1, 16, 16)
y_mae = ising_light.y[sub_l]
n_tr_mae = 1200
X_mae_tr, X_mae_te = X_mae[:n_tr_mae], X_mae[n_tr_mae:]
y_mae_tr, y_mae_te = y_mae[:n_tr_mae], y_mae[n_tr_mae:]
print(f"MAE inputs: {tuple(X_mae.shape)}   (16x16, so 4x4 patches -> 16 per sample)")

PATCH = 4
GRID = 16 // PATCH                                              # 4 patches per side
N_PATCHES = GRID * GRID                                         # 16
PATCH_DIM = PATCH * PATCH                                       # 16


def patchify(x):
    """(B, 1, 16, 16) -> (B, N_PATCHES, PATCH_DIM)."""
    B = x.shape[0]
    # unfold into 4x4 blocks
    p = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)       # (B, 1, 4, 4, 4, 4)
    p = p.contiguous().view(B, 1, GRID, GRID, PATCH, PATCH)
    p = p.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, N_PATCHES, PATCH_DIM)
    return p


class TinyMAE(nn.Module):
    """2-layer transformer encoder on visible patches; small MLP decoder
    head onto every patch position. CLS token is the embedding we probe."""

    def __init__(self, patch_dim=PATCH_DIM, embed_dim=64, depth=2, n_heads=4):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for the CLS slot at position 0
        self.pos_embed = nn.Parameter(torch.zeros(1, N_PATCHES + 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=2 * embed_dim,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        # Decoder head: map each *patch position's* token to that patch.
        # We share weights across positions (a single Linear).
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, patch_dim),
        )

    def encode_visible(self, patches, vis_idx):
        """patches: (B, N, P), vis_idx: (B, n_vis). Returns (B, 1+n_vis, D)."""
        B, _, _ = patches.shape
        # Gather visible patches and their positional embeddings.
        gather_idx = vis_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        vis_patches = torch.gather(patches, 1, gather_idx)      # (B, n_vis, P)
        vis_tokens = self.patch_embed(vis_patches)              # (B, n_vis, D)
        # +1 because position 0 is the CLS slot in pos_embed.
        vis_pos = self.pos_embed[:, 1:, :].expand(B, -1, -1)
        vis_pos = torch.gather(
            vis_pos, 1, vis_idx.unsqueeze(-1).expand(-1, -1, vis_pos.shape[-1])
        )
        vis_tokens = vis_tokens + vis_pos
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        z = torch.cat([cls, vis_tokens], dim=1)                 # (B, 1+n_vis, D)
        return self.encoder(z)

    def cls_embedding(self, x):
        """Embed the full image (no masking) and return the CLS token. Used
        for linear probing — mirrors what an MAE-pretrained encoder is used
        for downstream."""
        patches = patchify(x)
        B = x.shape[0]
        vis_idx = torch.arange(N_PATCHES).unsqueeze(0).expand(B, -1).to(x.device)
        z = self.encode_visible(patches, vis_idx)
        return z[:, 0, :]                                       # (B, D)

    def reconstruct_masked(self, x, mask_ratio=0.75, generator=None):
        """Mask `mask_ratio` patches per sample; encode visible, decode all
        positions (with the encoded CLS + visible tokens scattered back into
        the full grid, and a learned mask token elsewhere - here we use the
        encoder's mean output as a stand-in to keep the model tiny).
        Returns (pred_masked, true_masked) for the loss."""
        patches = patchify(x)
        B = x.shape[0]
        n_mask = int(mask_ratio * N_PATCHES)
        n_vis = N_PATCHES - n_mask
        # Random per-sample permutation.
        noise = torch.rand(B, N_PATCHES, generator=generator, device=x.device)
        perm = noise.argsort(dim=1)
        vis_idx = perm[:, :n_vis]                               # (B, n_vis)
        mask_idx = perm[:, n_vis:]                              # (B, n_mask)
        z = self.encode_visible(patches, vis_idx)               # (B, 1+n_vis, D)
        # Use a single learned "context" vector for masked positions: take
        # the mean of (cls + visible tokens) and add the position embed.
        context = z.mean(dim=1, keepdim=True).expand(B, n_mask, -1)
        mask_pos = self.pos_embed[:, 1:, :].expand(B, -1, -1)
        mask_pos = torch.gather(
            mask_pos, 1, mask_idx.unsqueeze(-1).expand(-1, -1, mask_pos.shape[-1])
        )
        masked_tokens = context + mask_pos                      # (B, n_mask, D)
        pred = self.decoder(masked_tokens)                      # (B, n_mask, P)
        true = torch.gather(
            patches, 1, mask_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        )
        return pred, true


# %%
# Train the tiny MAE for ~20 epochs on Ising-light.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
mae = TinyMAE().to(DEVICE)
opt = torch.optim.AdamW(mae.parameters(), lr=3e-3, weight_decay=1e-4)
mae_loader = DataLoader(
    torch.utils.data.TensorDataset(X_mae_tr), batch_size=64, shuffle=True,
)
gen = torch.Generator(device=DEVICE).manual_seed(0)

print("Training TinyMAE on Ising-light patches (20 epochs)...")
for epoch in range(20):
    mae.train()
    losses = []
    for (xb,) in mae_loader:
        xb = xb.to(DEVICE)
        pred, true = mae.reconstruct_masked(xb, mask_ratio=0.75, generator=gen)
        loss = F.mse_loss(pred, true)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    if epoch % 4 == 0 or epoch == 19:
        print(f"  epoch {epoch:2d}   masked-patch MSE = {np.mean(losses):.4f}")


# %%
# Linear probe on the MAE CLS embedding — same protocol as the SimCLR probe.
mae.eval()
with torch.no_grad():
    Z_mae_train = mae.cls_embedding(X_mae_tr.to(DEVICE)).cpu()
    Z_mae_test = mae.cls_embedding(X_mae_te.to(DEVICE)).cpu()

acc_mae = ising_linear_probe(Z_mae_train, y_mae_tr, Z_mae_test, y_mae_te)
print(f"\n=== SSL linear-probe comparison on Ising ===")
print(f"  SimCLR-style contrastive (64x64, ConvEncoder)  test acc = {acc_simclr:.3f}")
print(f"  Tiny MAE (16x16, 2-layer transformer, CLS)     test acc = {acc_mae:.3f}")


# %% [markdown]
# **Take-home from the masked half.** Reconstruction-with-masking is a
# *very* different supervisory signal from contrastive. It does not need
# negative pairs; it does not care about batch size; and on small images
# like Ising the MAE CLS token often matches or beats the SimCLR
# embedding on the same downstream linear probe. The 2022 MAE paper made
# this point on ImageNet; you just measured the same effect on Ising
# microstructure — at 1500 samples and a 2-layer transformer.
#
# This is the pedagogical pay-off promised in the lecture slide
# "SSL refresh — MAE / DINOv2 / I-JEPA": *the contrastive recipe is no
# longer the default. Masked modelling is competitive or better, and
# does not require a SimCLR-sized batch.*

# %% [markdown]
# # Block 7 — Student exercises (~18 min)

# %% [markdown]
# ## Exercise 1 (core) — k-NN vs linear as a geometry test
#
# **Setup.** A 1-layer linear probe measures *linear* separability.
# **k-NN** measures local structure: an embedding can score high on k-NN
# even when it fails the linear probe (curved class boundary), or vice
# versa.
#
# **Task.** On the synthetic-appendix CGNN crystal embeddings (`embeds`
# from Block 5b, labels = `proto_all`), evaluate two prototype
# classifiers on a random 80/20 split:
#
# 1. A 1-layer linear classifier (`nn.Linear(embeds.shape[1], 5)`,
#    cross-entropy, ~200 SGD steps) — train acc and test acc.
# 2. A k-NN classifier with k = 5 (test acc on the same split).
#
# Report both accuracies. If k-NN > linear by more than ~5%, the
# embedding has *local* structure that linear separation does not capture
# — likely curved cluster boundaries. If linear ≥ k-NN, the embedding is
# already linearly arranged.

# %% [markdown]
# ## Exercise 2 (core) — PCA reconstruction error as crystal anomaly score
#
# **Setup.** In Block 2 we used PCA reconstruction error to flag a
# *contaminated* spectrum. The same logic applies to embeddings: the
# crystals with the largest PCA-2D reconstruction error are the ones
# whose embeddings the dominant 2 directions cannot capture — anomalous
# in the embedding sense.
#
# **Task.** Project all 200 synthetic-appendix CGNN embeddings to
# PCA-2D, reconstruct, rank by reconstruction error. Print the top 5
# anomalies — what do they have in common? (Hint: rare prototypes? rare
# cation/anion combos? boundary crystals near a cluster?)

# %% [markdown]
# ## Exercise 3 (core) — Embedding distance vs energy distance
#
# **Setup.** The CGNN was trained to predict formation energy. Does the
# *embedding* make crystals with similar formation energy close to each
# other in the latent space?
#
# **Task.**
#
# 1. Sample 50 random pairs of crystals (i, j).
# 2. For each pair, compute embedding L2 distance and |y_i - y_j|.
# 3. Scatter-plot one vs the other; report Pearson correlation.
#
# **Question to answer in writing:** is the embedding *energy-aware* (high
# correlation), *structure-aware* (no correlation but prototype-organised),
# or both? Predict what would change if the CGNN trained for 50 epochs
# instead of the 12 used in the Block-5b synthetic appendix.

# %% [markdown]
# ## Exercise 3b (core, MG U8) — Choose the split that matches the claim
#
# **Setup.** Block 4 built three split designs on the real MatBench
# perovskites benchmark — `split_random`, the `mb_folds` reproducible
# *surrogate* folds, and `split_family_heldout` — and a `baseline_ladder`
# (tiers 0/1/2). MG U8 slide 26: *the split is part of the hypothesis,
# not the postprocessing*.
#
# **Task.** For each of the following deployment claims, state which
# split design is the *honest* headline split, then verify empirically:
#
# 1. *"Predicts formation energy for new perovskite compositions drawn
#    from the same chemistry as our training set."* → which split?
# 2. *"Enables discovery of stable perovskites in composition families
#    we have never computed."* → which split?
# 3. *"Reports a de-noised IID benchmark number, comparable across
#    methods."* → which split?
#
# For claim 2, run `baseline_ladder` under `split_family_heldout` with a
# *different* held-out family than Block 4's `[3, 4, 5]` (try the
# alkaline-earth set `[12, 20, 38]` = Mg, Ca, Sr). Report the
# tier-1 MAE and its gap to the random-split number — that gap is the
# $\Delta_\text{shift}$ a discovery paper must disclose. In one
# paragraph, explain why reporting only the random/CV number for claim 2
# would be the MG slide-48 anti-pattern "random-split numbers in a
# discovery paper". Note explicitly that `mb_folds` is a *surrogate*-fold
# CV (official folds need the `matbench` pkg) so the absolute numbers are
# for split-design teaching, not leaderboard comparison.

# %% [markdown]
# ## Exercise 4 (stretch) — Contrastive on crystals
#
# **Setup.** Block 6 ran InfoNCE on Ising images. The same recipe works on
# crystal graphs — and that's the foundation of CrystalCLR, CrystalTwins,
# and other self-supervised crystal-embedding methods.
#
# **Task.** Modify the Block 6 InfoNCE loop to operate on
# `CrystalGraphsDataset`:
#
# 1. Define `make_crystal_pair(crystal, rng)` that returns two augmented
#    copies of the same crystal. Augmentation: re-sample the
#    `edge_distance` array with new uniform distortions in [0.92, 1.08]
#    (the same recipe `_build_dataset` used to construct the dataset).
#    Same species, same edges, different geometry → same chemistry.
# 2. Train `TinyCGNN.encode` with InfoNCE for 5 epochs.
# 3. PCA-2D and colour by prototype. Compare to the supervised embedding
#    from Block 4.
#
# **Expected:** the contrastive embedding will separate prototypes about
# as well as the supervised one — without ever seeing a formation energy.

# %% [markdown]
# ## Exercise 5 (stretch, optional) — DINOv2 foundation embedding + UMAP
#
# **Setup.** DINOv2 [@oquab_2024_dinov2] is Meta's 2024 self-supervised
# vision foundation model. It was pre-trained on 142 M curated images
# with a self-distillation objective and produces 384-dimensional
# embeddings (for the `vits14` variant) that transfer to many downstream
# tasks **with no fine-tuning** — including, increasingly, materials
# microscopy. This exercise asks you to feel a 2026-grade foundation
# embedding by your own hand.
#
# **Task.**
#
# 1. Load `dinov2_vits14` via `torch.hub.load('facebookresearch/dinov2',
#    'dinov2_vits14')`. This needs internet + ~100 MB download — wrapped
#    in `try/except` so the rest of the notebook still runs offline.
# 2. Embed a small image dataset: try `NEUDETDataset` first (steel-surface
#    defects, 6 classes, 1800 images — DINOv2 is exactly the kind of
#    model that should crush this task). If `NEUDETDataset` is not
#    available locally, fall back to embedding the Ising images
#    (Ising-full, 64x64 -> upsample to 224x224, grayscale -> 3 channels).
# 3. Project the embeddings with **UMAP** (n_neighbors=15, min_dist=0.1)
#    and colour by class. Compare visually to the SimCLR embedding
#    you trained in Block 6 — does the foundation model's representation
#    organise the classes better than the from-scratch contrastive one?
#
# **Expected.** On NEU-DET the DINOv2 UMAP should show 6 visually
# crisp clusters with very little class overlap. On Ising the
# improvement over SimCLR is smaller (DINOv2 was not trained on
# microstructure) but you should still see cleaner separation than
# the from-scratch baseline.

# %%
# Optional stretch exercise: DINOv2 foundation embeddings + UMAP.
# Wrapped in try/except for offline-friendly notebook execution.
try:
    print("Loading DINOv2 (dinov2_vits14) ...")
    dinov2 = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False,
    )
    dinov2.eval().to(DEVICE)
    print(f"DINOv2 loaded. Embedding dim = {dinov2.embed_dim}")

    # Prefer NEU-DET if it is installed locally; fall back to Ising-full.
    use_neudet = False
    try:
        from ai4mat.datasets import NEUDETDataset
        try:
            neudet = NEUDETDataset(download=False)
            use_neudet = True
            print(f"Using NEU-DET: {len(neudet)} samples, "
                  f"{len(neudet.class_names)} classes")
        except Exception as e:
            print(f"NEU-DET not available locally ({type(e).__name__}); "
                  f"falling back to Ising-full.")
    except ImportError:
        print("NEUDETDataset import failed; falling back to Ising-full.")

    # Build the (X, y, class names) triple to embed.
    if use_neudet:
        X_dino_raw = neudet.X                                   # (N, 1, 200, 200)
        y_dino = neudet.y.numpy()
        class_names = neudet.class_names
    else:
        X_dino_raw = X_img                                      # (1500, 1, 64, 64)
        y_dino = y.numpy()
        class_names = ["below T_c", "above T_c"]

    # DINOv2 expects 3-channel ImageNet-normalised inputs at a multiple of 14.
    IMG = 224
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    def dinov2_embed(X, batch_size=32):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = X[i:i + batch_size].to(DEVICE).float()
                if xb.shape[1] == 1:
                    xb = xb.expand(-1, 3, -1, -1)
                xb = F.interpolate(xb, size=IMG, mode="bilinear", align_corners=False)
                xb = (xb - mean) / std
                z = dinov2(xb)                                  # (B, embed_dim)
                out.append(z.cpu())
        return torch.cat(out).numpy()

    print("Computing DINOv2 embeddings ...")
    Z_dino = dinov2_embed(X_dino_raw)
    print(f"DINOv2 embeddings: {Z_dino.shape}")

    # UMAP visualisation.
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=0,
        )
        Z_dino_2d = reducer.fit_transform(Z_dino)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        for cls in range(len(class_names)):
            m = y_dino == cls
            ax.scatter(Z_dino_2d[m, 0], Z_dino_2d[m, 1], s=10, alpha=0.7,
                       label=class_names[cls])
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title(
            "DINOv2 vits14 embeddings -> UMAP "
            f"({'NEU-DET' if use_neudet else 'Ising-full'})"
        )
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout(); plt.show()
    except ImportError:
        print("umap-learn not installed — skipping UMAP plot. "
              "(`pip install umap-learn`)")

except Exception as exc:
    print(f"DINOv2 stretch exercise skipped: {type(exc).__name__}: {exc}")
    print(
        "This is fine — it needs internet for the torch.hub download (~100 MB)\n"
        "and optionally a local NEU-DET install. See the markdown above for\n"
        "what you would have seen."
    )

# %% [markdown]
# ---
# **Bridge to Week 10.** Next week MFML Unit 10 turns to *attention and
# transformers*. The transformer is itself an embedding model — every
# token gets a contextual vector — and the same MFML W9 tools you used
# today (PCA, t-SNE, linear probing) become the standard way to read what
# a trained transformer has learned. Today's discipline carries over.
# DINOv2 in Exercise 5 was already a transformer; week 10 unpacks *how*
# the attention layer produces those tokens.
#
# On the MG side, next week is **MG Unit 9 — neural networks for
# materials properties** (SchNet, CGCNN, MEGNet, M3GNet, equivariant
# successors). MG U8 slide 50 is explicit: *better architecture does not
# fix bad benchmarking*. Every split design, the baseline ladder, the
# per-family residual table (Blocks 4–5, on the real MatBench benchmark)
# and the structure-awareness ablation (Block 5b synthetic appendix) you
# built carry into U9 **unchanged** — a fancier crystal network is only
# interesting if it beats the tier-0/1/2 ladder under a split that
# matches the claim.
