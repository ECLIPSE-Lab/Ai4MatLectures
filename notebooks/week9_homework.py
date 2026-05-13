# %% [markdown]
# # Week 9 — Homework (do BEFORE the Thursday exercise)
#
# This notebook is the **mandatory warm-up** for the Week 9 in-class exercise.
# It puts the three latent-space lenses from MFML Unit 9 in your hands —
# linear (PCA), nonlinear visual (t-SNE), and learned-without-labels
# (contrastive) — so Thursday can spend its 90 minutes on the integrated
# story: latent spaces of *characterisation signals* (ML-PC) and of *trained
# materials NNs* (MG).
#
# **Time:** ~75 minutes.
#
# ## Red thread
#
# > *Once a model is trained, its penultimate layer is a coordinate system
# > of its own. The same questions apply whether that coordinate system was
# > set by PCA, by t-SNE, by an autoencoder, or by a contrastive loss:
# > **what does the geometry organise by, and how do we measure that
# > quantitatively?** Today you build the three projection tools on a clean
# > image dataset (Ising). Thursday we apply them to spectra and to a
# > trained crystal-graph network.*
#
# ## What this homework is
#
# | Part | Min | Topic | Lecture anchor |
# |---|---:|---|---|
# | A | 25 | Hand-rolled PCA on Ising-full; reconstruction error vs latent dim | MFML §"PCA as a linear AE", §"Reconstruction error vs latent dim" |
# | B | 30 | t-SNE **and UMAP** on Ising-full latents; distance trap; n_neighbors sweep | MFML §"t-SNE", §"UMAP — the 2026 default" [@mcinnes_2018_umap] |
# | C | 15 | Augmentation pipeline for contrastive learning — *no* training; data-prep only | MFML §"Positive pair construction", §"Augmentations as the prior" |
# | D | 10 | Reflection: when does each method help? | bridge to Thursday Block 5 (probing) |
#
# ## What you must hand in (or be able to show on Thursday)
#
# 1. Part A: reconstruction-error-vs-k curve for k ∈ {1, 2, 4, 8, 16, 32}
#    on Ising-full; printed table of MSE values.
# 2. Part B: 2-D t-SNE plot of the Ising bottleneck features, with the
#    "distance trap" pair annotated; pixel-vs-tSNE distance comparison;
#    2×2 grid of UMAP layouts at `n_neighbors ∈ {5, 15, 50, 200}` with
#    runtimes; one-sentence verdict on which method preserves global
#    structure better.
# 3. Part C: pairwise pixel-distance histograms showing your `make_positive_pair`
#    breaks pixel similarity while preserving class.
# 4. Part D: paragraph (~5 sentences) on PCA vs t-SNE vs contrastive.

# %%
# Standard imports for the whole homework. Same idiom as weeks 2-6.
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE

from ai4mat.datasets import IsingDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# # Part A — Hand-rolled PCA on Ising-full
#
# The whole of PCA is two lines once you stop fearing the covariance
# matrix. We compute it explicitly so you can read off where the variance
# lives, then project the 64×64 = 4096-D Ising images down to k principal
# components and reconstruct.
#
# Recipe:
#
# 1. Centre the data: $\tilde X = X - \bar x$.
# 2. Covariance matrix: $C = \tilde X^\top \tilde X / (N - 1)$.
# 3. Eigendecomposition: $C = V \Lambda V^\top$.
# 4. Top-k components are the columns of $V$ paired with the largest
#    eigenvalues. Project: $Z = \tilde X V_{:,1:k}$.
# 5. Reconstruct: $\hat X = Z V_{:,1:k}^\top + \bar x$.
#
# We compute the eigendecomposition with `torch.linalg.eigh` (uses the
# symmetry of $C$ — faster and more accurate than `eig`).
#
# *(see MFML §"PCA as a linear autoencoder", §"From SVD to PCA")*

# %%
ising = IsingDataset(size="full")
print(f"Ising-full: {len(ising)} samples, image shape {tuple(ising.X.shape[1:])}")

# Subsample for speed on CPU. 1500 images is plenty for PCA.
g = torch.Generator().manual_seed(0)
sub = torch.randperm(len(ising), generator=g)[:1500]
X_img = ising.X[sub]                                       # (1500, 1, 64, 64)
y = ising.y[sub]                                           # (1500,)
X_flat = X_img.flatten(1)                                  # (1500, 4096)
print(f"flattened: {tuple(X_flat.shape)}, classes: {y.unique().tolist()}")


# %%
def pca_from_scratch(X, k):
    """Return (Z, X_recon, eigvals_descending, components_DxK)."""
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)                    # (D, D)
    eigvals, eigvecs = torch.linalg.eigh(cov)              # ascending
    # eigh returns ascending; reverse for descending.
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    V_k = eigvecs[:, :k]                                   # (D, k)
    Z = Xc @ V_k                                           # (N, k)
    X_recon = Z @ V_k.T + mu                               # (N, D)
    return Z, X_recon, eigvals, V_k


# Sanity-check at k=2: hand-rolled vs sklearn.
Z2, X2_recon, eigvals_full, V_full = pca_from_scratch(X_flat, k=2)
sk = SklearnPCA(n_components=2).fit(X_flat.numpy())
ours_evr = (eigvals_full[:2] / eigvals_full.sum()).tolist()
print(f"Top-2 explained variance ratio:")
print(f"  hand-rolled: {ours_evr[0]:.4f}, {ours_evr[1]:.4f}")
print(f"  sklearn:     {sk.explained_variance_ratio_[0]:.4f}, {sk.explained_variance_ratio_[1]:.4f}")

# Plot the 2-D projection coloured by class.
Z2_np = Z2.numpy()
fig, ax = plt.subplots(figsize=(6, 4.5))
for cls in [0, 1]:
    m = (y == cls).numpy()
    ax.scatter(Z2_np[m, 0], Z2_np[m, 1], s=8, alpha=0.6,
               label=f"class {cls}", c=f"C{cls}")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Hand-rolled PCA on Ising-full (top 2 components)")
ax.legend(); plt.tight_layout(); plt.show()


# %%
# Reconstruction error vs latent dim k.  We sweep k and report MSE
# averaged over the training set.
ks = [1, 2, 4, 8, 16, 32, 64]
mses = []
for k in ks:
    _, X_rec, _, _ = pca_from_scratch(X_flat, k=k)
    mse = ((X_flat - X_rec) ** 2).mean().item()
    mses.append(mse)
    print(f"  k = {k:3d}   MSE = {mse:.5f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, mses, "o-", lw=1.6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("latent dim k"); ax.set_ylabel("reconstruction MSE")
ax.set_title("PCA reconstruction error vs k (Ising-full)")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the curve.** Reconstruction MSE decays monotonically — that's
# the spectrum of the covariance matrix in disguise. The "elbow" you see
# (around k = 8 or so) is where additional dimensions are paying for
# diminishing returns. This is the *linear* answer to "how many degrees
# of freedom does the data have?"; the autoencoder we build on Thursday
# (Block 3) gives the *nonlinear* answer and usually finds a much smaller k.

# %% [markdown]
# # Part B — t-SNE *and UMAP* on Ising-full latents
#
# t-SNE is the most popular nonlinear visualisation of high-D data. It
# preserves *local* neighbourhood structure: two points that were near each
# other in the original space *should* end up near each other in the
# 2-D layout. But the converse is *not* true: two points that end up near
# each other in 2-D may have been far apart in the original space, because
# t-SNE exaggerates distances between separated clusters.
#
# This is the **distance trap**: people read t-SNE plots as if pixel-space
# distance equalled t-SNE distance. It does not.
#
# **UMAP** [@mcinnes_2018_umap] is the 2026 default for 2-D embeddings of
# materials-science feature spaces. It is built on the same fuzzy-simplicial
# foundation as t-SNE but trades the heavy KL-divergence optimisation for a
# cross-entropy on a sparse k-NN graph: faster, scalable to millions of
# points, and — most importantly for this exercise — it preserves more of
# the **global** structure. After running both, you should be able to *see*
# the difference.
#
# *(see MFML §"t-SNE — perplexity, t-distribution, and what it is for",
# §"UMAP — the 2026 default for 2-D visualisation",
# §"What latent visualisations don't tell you")*

# %%
# Project Ising-full to 2-D with t-SNE.  We project the *flattened pixels*
# directly; on a real problem you would first reduce to k=50 with PCA for
# speed, but on Ising-full at N=1500 the direct path is fast enough.
#
# Using the PCA-50 path is recommended in production - it's also what the
# MFML W9 lecture shows.
print("Pre-reducing to PCA-50 for t-SNE...")
Z50, _, _, _ = pca_from_scratch(X_flat, k=50)
print("Running t-SNE on the PCA-50 features...")
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
Z_tsne = tsne.fit_transform(Z50.numpy())
print(f"t-SNE output shape: {Z_tsne.shape}")


# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for cls in [0, 1]:
    m = (y == cls).numpy()
    ax1.scatter(Z2_np[m, 0], Z2_np[m, 1], s=8, alpha=0.6, c=f"C{cls}", label=f"class {cls}")
    ax2.scatter(Z_tsne[m, 0], Z_tsne[m, 1], s=8, alpha=0.6, c=f"C{cls}", label=f"class {cls}")
ax1.set_title("PCA-2D"); ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.legend()
ax2.set_title("t-SNE-2D (from PCA-50)"); ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2"); ax2.legend()
plt.tight_layout(); plt.show()


# %%
# The distance trap: pick two points close in t-SNE but far in pixel space.
tsne_pts = torch.tensor(Z_tsne, dtype=torch.float32)
N = tsne_pts.shape[0]

# Pairwise t-SNE distances and pairwise pixel distances on a 200-point sub.
sub_idx = torch.randperm(N, generator=torch.Generator().manual_seed(1))[:200]
d_tsne = torch.cdist(tsne_pts[sub_idx], tsne_pts[sub_idx])
d_pix = torch.cdist(X_flat[sub_idx], X_flat[sub_idx])

# Find a pair with t-SNE rank in the top 5% but pixel rank in the bottom 50%.
flat = torch.triu_indices(d_tsne.shape[0], d_tsne.shape[1], offset=1)
i_, j_ = flat[0], flat[1]
d_t = d_tsne[i_, j_]; d_p = d_pix[i_, j_]
t_rank = d_t.argsort().argsort().float() / d_t.numel()
p_rank = d_p.argsort().argsort().float() / d_p.numel()
candidates = ((t_rank < 0.05) & (p_rank > 0.5))            # close in t-SNE, far in pixels
if candidates.any():
    pick = candidates.nonzero()[0, 0].item()
    a, b = int(i_[pick]), int(j_[pick])
    print(f"Distance-trap pair: t-SNE dist = {d_t[pick]:.2f}  (rank {t_rank[pick].item():.0%}), "
          f"pixel dist = {d_p[pick]:.2f}  (rank {p_rank[pick].item():.0%})")
else:
    print("No clear distance-trap pair found — try perplexity or seed.")


# %% [markdown]
# ## UMAP — n_neighbors sweep, runtime, and side-by-side with t-SNE
#
# `umap-learn` is not in the default course env. Install it with
# `pip install umap-learn` (wrapped in try/except like the TabPFN pattern
# you saw in Week 8) and re-run this cell. If the import fails the cell
# prints an install hint and skips — your hand-in is then just the t-SNE
# half plus a note that you did not get UMAP running.
#
# **What we sweep.** `n_neighbors` is UMAP's single most consequential
# hyperparameter. Small values (5) over-fragment the manifold and behave
# like t-SNE; large values (200) push UMAP toward a global, almost
# linear-looking layout. We measure runtime alongside the layout so you
# can feel the cost-quality knob.

# %%
import time

try:
    import umap  # noqa: F401  (the actual API is umap.UMAP)
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False
    print(
        "umap-learn not installed — skipping the UMAP sweep.\n"
        "  pip install umap-learn\n"
        "and re-run this cell to complete Part B."
    )

if HAVE_UMAP:
    import umap as umap_lib

    nn_values = [5, 15, 50, 200]
    umap_embeddings = {}
    umap_runtimes = {}
    for nn in nn_values:
        reducer = umap_lib.UMAP(
            n_components=2, n_neighbors=nn, min_dist=0.1, random_state=0,
        )
        t0 = time.time()
        Z_u = reducer.fit_transform(Z50.numpy())
        umap_runtimes[nn] = time.time() - t0
        umap_embeddings[nn] = Z_u
        print(f"  UMAP n_neighbors = {nn:3d}   runtime = {umap_runtimes[nn]:.2f} s")

    # 2x2 grid of UMAPs + the t-SNE on a 5th panel (use a 2x3 layout, leave one blank).
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    for ax, nn in zip(axes.flat[:4], nn_values):
        Z_u = umap_embeddings[nn]
        for cls in [0, 1]:
            m = (y == cls).numpy()
            ax.scatter(Z_u[m, 0], Z_u[m, 1], s=8, alpha=0.6,
                       c=f"C{cls}", label=f"class {cls}")
        ax.set_title(f"UMAP   n_neighbors = {nn}   ({umap_runtimes[nn]:.2f} s)")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=8)

    # Panel 5: the t-SNE we already ran, for direct comparison.
    ax = axes.flat[4]
    for cls in [0, 1]:
        m = (y == cls).numpy()
        ax.scatter(Z_tsne[m, 0], Z_tsne[m, 1], s=8, alpha=0.6,
                   c=f"C{cls}", label=f"class {cls}")
    ax.set_title("t-SNE (perplexity = 30)   — comparison")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(fontsize=8)

    # Panel 6: blank.
    axes.flat[5].axis("off")
    plt.suptitle("Ising-full (PCA-50 features) — UMAP n_neighbors sweep vs t-SNE", y=1.02)
    plt.tight_layout(); plt.show()


# %% [markdown]
# **Reflection question (write 1-2 sentences in your hand-in).** Across
# the four UMAP panels and the t-SNE panel, **which method preserves the
# *global* structure better** — i.e. which one keeps the two Ising
# classes on a sensible relative scale rather than blowing them apart
# into isolated islands? Note that at low `n_neighbors` UMAP behaves
# closer to t-SNE; at high `n_neighbors` it pushes toward a more
# PCA-like global layout.
#
# *(Answer for the marker: UMAP, especially at `n_neighbors ≥ 50`,
# keeps inter-cluster distances meaningful. t-SNE's KL objective
# explicitly exaggerates them. See [@mcinnes_2018_umap].)*


# %% [markdown]
# # Part C — Augmentation pipeline for contrastive learning
#
# Contrastive learning needs **positive pairs** — two views of the same
# underlying object — and **negatives** — views from different objects.
# A "view" is just an augmented version of an image.  The augmentations
# encode the *invariances* you want the latent space to respect: a model
# trained with rotations as positives will learn rotation-invariant
# features; a model trained with colour-jitter positives will learn
# colour-invariant features.
#
# For Ising microstructures, the natural invariances are:
#
# - **Rotation** by k × 90° (the lattice has no preferred direction).
# - **Flips** (the spin-up and spin-down domains are exchangeable below T_c
#   and the order parameter is symmetric across mirrors).
# - **Small additive noise** (we'd like the embedding to be robust to
#   imaging noise / measurement noise).
#
# Crucially, these augmentations preserve the **class label** (above vs
# below T_c) — that's how we know they're "valid invariances" for this
# task. *No model training* in this part. Just the data-prep that
# Thursday Block 6's InfoNCE loop will consume.
#
# *(see MFML §"Positive-pair construction", §"Augmentations as an
# implicit prior on what to make invariant")*

# %%
def make_positive_pair(x, rng=None):
    """Two augmented views of the same Ising image.

    x: (1, H, W) float tensor in [0, 1].
    Returns (xi, xj), both same shape and dtype.

    Augmentations applied independently to each view:
      - random rotation by 0/90/180/270 degrees
      - random horizontal flip with p=0.5
      - random vertical flip with p=0.5
      - additive Gaussian noise, sigma in [0.0, 0.1]
    """
    if rng is None:
        rng = np.random.default_rng()

    def aug(t):
        # rotation
        k = int(rng.integers(0, 4))
        t = torch.rot90(t, k, dims=(-2, -1))
        # h-flip
        if rng.random() < 0.5:
            t = torch.flip(t, dims=(-1,))
        # v-flip
        if rng.random() < 0.5:
            t = torch.flip(t, dims=(-2,))
        # noise
        sigma = float(rng.uniform(0.0, 0.1))
        t = t + sigma * torch.randn_like(t)
        return t.clamp(0.0, 1.0)

    return aug(x), aug(x)


# %%
# Verify the augmentations *preserve class* but *break pixel similarity*.
rng = np.random.default_rng(0)
n_pairs = 300
intra_pos_dists = []                                       # pos-pair pixel L2
intra_class_dists = []                                     # different image, same class
inter_class_dists = []                                     # different image, different class

idx0 = (y == 0).nonzero().squeeze()[:200]
idx1 = (y == 1).nonzero().squeeze()[:200]

# Same-image positive pairs
for i in range(n_pairs):
    j = int(idx0[i % len(idx0)])
    xi, xj = make_positive_pair(X_img[j], rng=rng)
    intra_pos_dists.append((xi - xj).pow(2).mean().sqrt().item())

# Different-image, same-class
for i in range(n_pairs):
    a = int(idx0[i % len(idx0)])
    b = int(idx0[(i + 7) % len(idx0)])
    intra_class_dists.append((X_img[a] - X_img[b]).pow(2).mean().sqrt().item())

# Different-image, different-class
for i in range(n_pairs):
    a = int(idx0[i % len(idx0)])
    b = int(idx1[i % len(idx1)])
    inter_class_dists.append((X_img[a] - X_img[b]).pow(2).mean().sqrt().item())

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(intra_pos_dists, bins=30, alpha=0.6, label="positive pair (same image, augmented)", color="C2")
ax.hist(intra_class_dists, bins=30, alpha=0.6, label="same class, different image", color="C0")
ax.hist(inter_class_dists, bins=30, alpha=0.6, label="different class", color="C3")
ax.set_xlabel("pixel-space RMS distance")
ax.set_ylabel("count")
ax.set_title("Pair-distance histograms — pixel L2 cannot tell positive pairs from negatives")
ax.legend(); plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the histogram.** The three distributions overlap heavily.
# In pixel space, "two augmentations of the same Ising image" are *not*
# closer than "two different images of the same class". This is precisely
# why we need a learned encoder — the model must discover an embedding
# in which positive pairs become close and unrelated images become far.
# That is the contrastive job description.

# %% [markdown]
# # Part D — Reflection (1 paragraph, ~5 sentences)
#
# Write a paragraph answering: **PCA, t-SNE, and contrastive learning all
# produce a low-dimensional embedding of high-D data — but they answer
# different questions.** Specifically address:
#
# 1. Which method is **quantitative** (you can compute reconstruction error
#    in the original space)?
# 2. Which method is purely **visual** (the embedding axes are not
#    interpretable, distances are not metric)?
# 3. Which method **uses no labels at all** for training, even implicitly?
# 4. Which method is **most computationally expensive**?
#
# Reference one observation from your Part A reconstruction-error curve
# and one from your Part C distance histograms.
#
# **Bridge to Thursday.** On Thursday you'll see a fourth source of
# embeddings — the **penultimate layer of a trained CGNN on
# `CrystalGraphsDataset`** — and the same MFML W9 tools (PCA, t-SNE,
# linear probing) used to read what that embedding has learned about
# materials chemistry. Your Part D paragraph should already have an
# opinion about which tool to pick first, and what to follow up with.

# %% [markdown]
# ---
# You're done with Week 9 homework. Bring your four deliverables on Thursday:
#
# 1. PCA reconstruction error vs latent dim curve (Part A).
# 2. t-SNE 2-D plot with the distance-trap pair annotated (Part B).
# 3. Pixel-distance histogram for positive vs negative pairs (Part C).
# 4. Method-comparison paragraph (Part D).
