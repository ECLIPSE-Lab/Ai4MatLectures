# %% [markdown]
# # Week 9 — Latent geometry across three lenses
#
# This week we braid three lectures around a single question: **what does
# the latent space look like, and what does it organise by?**
#
# 1. **MFML Unit 9**: Latent spaces & advanced representation learning —
#    PCA, t-SNE, UMAP, contrastive learning, linear probing.
# 2. **ML-PC Unit 9** (delivered title; folder still
#    `unit10_characterization_signals`): PCA + AE on spectra,
#    reconstruction error as anomaly score, t-SNE/UMAP on hyperspectral
#    embeddings.
# 3. **MG Unit 8** (delivered as W9): NN architectures for materials
#    (SchNet/CGCNN/MEGNet/M3GNet) — *the trained network is itself an
#    embedding model*, and its penultimate layer carries chemistry,
#    prototype identity, and (sometimes) property structure.
#
# **Red thread:** *MFML supplies the projection tools, ML-PC applies them
# to spectra, and MG applies them to a trained crystal-graph network. The
# same PCA / t-SNE / linear-probe machinery reads three different feature
# types — and each one tells a different story about what the model has
# learned.*
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
# | 4 | ~14 | Materials NN as embedding model (MG): train TinyCGNN, freeze, embed, project |
# | 5 | ~12 | Linear probing the CGNN embedding; held-out-prototype generalisation |
# | 6 | ~18 | From supervised to contrastive to masked: InfoNCE *and* MAE on Ising |
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
# - to **trained materials-NN embeddings** (Block 4),
# - and to **linear probing + contrastive** as the readouts (Blocks 5–6).

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
# # Block 4 — Materials NN as embedding model
#
# We now switch to crystal data. The plan is straightforward: train a
# small CGCNN-style network on `CrystalGraphsDataset` (5 epochs); freeze
# it; extract the **penultimate-layer activations** for every crystal;
# project them to 2-D with PCA and t-SNE; ask what the geometry organises
# by.
#
# Spoiler: the embedding organises by **prototype** *and* by **chemistry**,
# in two separable directions. The supervised loss (formation-energy
# regression) does *not* tell the network to separate prototypes — but the
# inductive bias of the message-passing architecture does.
#
# *(see MG §"What a trained CGNN learns", §"Crystal embeddings as
# foundation features"; MFML §"Probing what a network has learned")*

# %%
class TinyCGNN(nn.Module):
    """Same as Week 6.  We add an `encode` method that returns the
    pooled atom embedding *before* the head."""

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
# Train the CGNN for 5 epochs on CrystalGraphsDataset.  Same protocol as
# Week 6 Block 6: Adam + grad clipping; one crystal per step.
cg = CrystalGraphsDataset()
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(cg), generator=g)
n_tr_cg = int(0.8 * len(cg))
tr_idx_cg, te_idx_cg = perm[:n_tr_cg].tolist(), perm[n_tr_cg:].tolist()
y_mean = cg.y.mean().item(); y_std = cg.y.std().item()

torch.manual_seed(0)
cgnn = TinyCGNN()
opt = torch.optim.Adam(cgnn.parameters(), lr=5e-3)
print("Training TinyCGNN on CrystalGraphsDataset (5 epochs)...")
for epoch in range(5):
    cgnn.train()
    losses = []
    for i in torch.randperm(len(tr_idx_cg)).tolist():
        s = cg[tr_idx_cg[i]]
        yn = (s["y"] - y_mean) / y_std
        opt.zero_grad()
        p = cgnn(s["species"], s["edge_index"], s["edge_distance"])
        loss = (p - yn) ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cgnn.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    print(f"  epoch {epoch}  train MSE = {np.mean(losses):.4f}")

# Extract the embedding for every crystal.
cgnn.eval()
with torch.no_grad():
    embeds = torch.stack([
        cgnn.encode(cg[i]["species"], cg[i]["edge_index"], cg[i]["edge_distance"])
        for i in range(len(cg))
    ])
print(f"\nCrystal embeddings: {tuple(embeds.shape)}")


# %%
# PCA on the crystal embeddings, coloured first by prototype, then by mean Z.
proto_all = cg.prototype.numpy()
mean_z_all = np.array([float(cg[i]["species"].float().mean()) for i in range(len(cg))])

Z_embed_2d, _, _, _ = pca_from_scratch(embeds, k=2)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
for p_idx, pname in enumerate(cg.prototype_names):
    m = proto_all == p_idx
    a1.scatter(Z_embed_2d[m, 0], Z_embed_2d[m, 1], s=24, alpha=0.85,
               c=f"C{p_idx}", label=pname, edgecolor="k", lw=0.3)
a1.set_xlabel("embed PC1"); a1.set_ylabel("embed PC2")
a1.set_title("CGNN embedding — coloured by prototype")
a1.legend(fontsize=8)

sc = a2.scatter(Z_embed_2d[:, 0], Z_embed_2d[:, 1], s=24, alpha=0.85,
                c=mean_z_all, cmap="viridis", edgecolor="k", lw=0.3)
a2.set_xlabel("embed PC1"); a2.set_ylabel("embed PC2")
a2.set_title("CGNN embedding — coloured by mean atomic Z")
plt.colorbar(sc, ax=a2, label="mean Z")
plt.tight_layout(); plt.show()


# %% [markdown]
# **What the colouring reveals.** Two directions in the latent space:
#
# - **Prototype** is captured by one axis (or one cluster pattern). The
#   message-passing topology encodes the *graph structure*, and crystals
#   that share a prototype share that topology — naturally grouped.
# - **Chemistry** (mean Z) is captured by a different direction. Lower-Z
#   crystals (Li, Na, Mg compounds) cluster together; heavier Z (Ag, Sn,
#   Ba compounds) cluster at the other end. The model learns this without
#   ever being told to.
#
# This is the **two-axis disentanglement** that makes a learned embedding
# useful for downstream tasks beyond the one it was trained on (formation
# energy). Block 5 quantifies this with linear probing.

# %% [markdown]
# # Block 5 — Linear probing the embedding
#
# Eyeballing a t-SNE plot is not measurement. Linear probing *is*: freeze
# the encoder, train a 1-layer linear classifier on top, and report
# accuracy. The probe accuracy is a **quantitative** answer to "how
# linearly separable are the classes in the embedding?"
#
# We probe two things:
#
# 1. **In-distribution probe**: train and test on all 5 prototypes
#    (random 80/20 split). High accuracy = the embedding "knows" prototype.
# 2. **Held-out-prototype probe**: train on 4 prototypes, test on the
#    5th. Low accuracy = the embedding does not extrapolate to unseen
#    prototypes — a *latent-space distribution shift* readout.
#
# *(see MFML §"Linear probing as a representation diagnostic"; MG
# §"Diagnosing learned representations")*

# %%
def linear_probe(embeds, labels, train_idx, test_idx, n_classes, n_epochs=200, lr=0.1):
    """Train a 1-layer linear classifier on the embedding."""
    torch.manual_seed(0)
    probe = nn.Linear(embeds.shape[1], n_classes)
    opt = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    Xtr, ytr = embeds[train_idx], labels[train_idx]
    Xte, yte = embeds[test_idx], labels[test_idx]
    for _ in range(n_epochs):
        opt.zero_grad()
        F.cross_entropy(probe(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        tr_acc = (probe(Xtr).argmax(1) == ytr).float().mean().item()
        te_acc = (probe(Xte).argmax(1) == yte).float().mean().item()
    return tr_acc, te_acc


# In-distribution: random 80/20 across all 200 crystals.
proto_torch = cg.prototype
g = torch.Generator().manual_seed(0)
perm_p = torch.randperm(len(cg), generator=g)
n_tr_p = int(0.8 * len(cg))
tr_p, te_p = perm_p[:n_tr_p].tolist(), perm_p[n_tr_p:].tolist()
tr_id, te_id = linear_probe(embeds, proto_torch, tr_p, te_p, n_classes=5)
print(f"In-distribution probe (random 80/20):  train acc = {tr_id:.3f}  test acc = {te_id:.3f}")

# Held-out prototype: train on 4 prototypes, test on the 5th.
print("\nHeld-out-prototype probes:")
for held in range(5):
    tr_h = [i for i in range(len(cg)) if proto_torch[i].item() != held]
    te_h = [i for i in range(len(cg)) if proto_torch[i].item() == held]
    # Re-label train classes to {0..3} and skip the test (chance is 1/4 there).
    proto_remap = proto_torch.clone()
    map_ = {p: idx for idx, p in enumerate([p for p in range(5) if p != held])}
    for p_old, p_new in map_.items():
        proto_remap[proto_torch == p_old] = p_new
    proto_remap[proto_torch == held] = 0                   # placeholder; ignored in test
    tr_acc, _ = linear_probe(embeds, proto_remap, tr_h, tr_h[:5], n_classes=4)
    # For the held-out probe, "test accuracy" = how often the probe puts the
    # held-out crystal into *some* class.  We report the entropy of the
    # prediction distribution as a soft "the model doesn't know" score.
    torch.manual_seed(0)
    probe = nn.Linear(embeds.shape[1], 4)
    opt2 = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9)
    Xtr, ytr = embeds[tr_h], proto_remap[tr_h]
    for _ in range(200):
        opt2.zero_grad()
        F.cross_entropy(probe(Xtr), ytr).backward(); opt2.step()
    with torch.no_grad():
        logits = probe(embeds[te_h])
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * probs.log()).sum(dim=1).mean().item()
    print(f"  held-out = {cg.prototype_names[held]:11s}   "
          f"in-dist train acc = {tr_acc:.3f}   held-out mean entropy = {entropy:.3f}  "
          f"(log4 = {math.log(4):.3f} = max uncertainty)")


# %% [markdown]
# **Reading the held-out probe.** When a prototype is *not* seen during
# probe training, the probe's prediction on those crystals is forced to
# belong to one of the 4 training prototypes. The entropy column tells us
# how confused the probe is: high entropy = "the embedding doesn't fit
# any of the 4 known prototypes well", low entropy = "the embedding looks
# like one of the seen prototypes". Both outcomes are diagnostic — neither
# is a model failure, just a *measurement of embedding generalisation*.

# %% [markdown]
# # Block 6 — From supervised to contrastive to masked
#
# Linear probing measured *what the supervised loss put into the
# embedding*. Self-supervised learning asks: *can we put similar things
# into the embedding without using any labels?* The SSL field of 2026
# offers two answer families:
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
# ## Exercise 1 (core) — k-NN as a geometry test
#
# **Setup.** Linear probing measures *linear* separability. **k-NN**
# measures local structure: an embedding can score high on k-NN even when
# it fails the linear probe (curved class boundary), or vice versa.
#
# **Task.** On the Block 4 CGNN crystal embeddings, evaluate two probes:
#
# 1. The 1-layer linear probe from Block 5 (train acc, test acc).
# 2. A k-NN classifier with k = 5 (test acc on the same 80/20 split).
#
# Report both accuracies. If k-NN > linear probe by more than ~5%, the
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
# **Task.** Project all 200 CGNN embeddings to PCA-2D, reconstruct, rank
# by reconstruction error. Print the top 5 anomalies — what do they have
# in common? (Hint: rare prototypes? rare cation/anion combos? boundary
# crystals near a cluster?)

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
# instead of 5.

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
