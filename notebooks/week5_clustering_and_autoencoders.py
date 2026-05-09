# %% [markdown]
# # Week 5 — Clustering and autoencoders
#
# This week we braid two lectures:
#
# 1. **MFML Unit 5**: Clustering and autoencoders — K-means and Lloyd's
#    algorithm, the GMM/EM relaxation, PCA as a linear autoencoder, the
#    non-linear autoencoder bottleneck.
# 2. **ML-PC Unit 5**: Unsupervised learning in materials — applying these
#    methods to *unlabelled* lab data, validating without ground truth, and
#    using reconstruction error as an anomaly score for distribution shift.
#
# **Red thread:** *K-means quantises data into K discrete codes; an
# autoencoder quantises into a continuous code. Both ask the same question —
# "what is the smallest description of $\mathbf{x}$ that still lets us
# recognise it?" — answered with hard labels (K-means), soft probabilities
# (GMM/EM), and learned latent vectors (autoencoder). Today we run all
# three on the same Ising microstructure data and compare what each one
# discovers without ever seeing a phase label.*
#
# > **Pre-flight check.** This notebook **assumes** you have run
# > `notebooks/week5_homework.py`. Block 1 picks up directly from your Lloyd
# > loop and Part C nanoindentation result; we will not re-derive them.
#
# ## Agenda (90 min)
#
# | Block | Min | Topic |
# |------:|:---:|:------|
# | 1 | ~6  | Recap from homework — Lloyd on blobs, K-means on nanoindentation |
# | 2 | ~12 | Hard → soft assignments: GMM/EM, responsibilities, BIC for K |
# | 3 | ~10 | K-means as the σ²→0 limit of an isotropic GMM |
# | 4 | ~10 | PCA on Ising-full — the linear baseline (and why it is not enough) |
# | 5 | ~18 | A tiny convolutional autoencoder on Ising-full; latent scatter |
# | 6 | ~10 | Reconstruction error as anomaly score — Ising vs Cahn–Hilliard |
# | 7 | ~24 | Student exercises (3 core + 1 stretch) |

# %%
# Standard imports. Same idiom as weeks 2-4: explicit seeds, no hidden state.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, roc_auc_score, roc_curve

from ai4mat.datasets import IsingDataset, NanoindentationDataset, CahnHilliardDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# ## Helpers used by every block

# %%
def standardise(z, dim=0):
    """Return (z - mean) / std along `dim`. Used for clustering preprocessing."""
    mu = z.mean(dim=dim, keepdim=True)
    sd = z.std(dim=dim, keepdim=True)
    return (z - mu) / (sd + 1e-12)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %% [markdown]
# # Block 1 — Recap from homework
#
# In Part A you implemented Lloyd's algorithm in two NumPy lines and watched
# the centroids migrate. In Part C you ran K-means with K=4 on
# `NanoindentationDataset` *after standardising*; the standardised version
# beat the raw-feature one by a wide ARI margin.
#
# We restate both results in 8 lines so the rest of the lecture has a
# baseline to compare against.
#
# *(see MFML §"Lloyd's algorithm"; ML-PC §"What unsupervised buys you")*

# %%
ds_nano = NanoindentationDataset()
Xn = ds_nano.X.numpy()
yn = ds_nano.y.numpy()

Xn_std = (Xn - Xn.mean(axis=0)) / Xn.std(axis=0)
km4 = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Xn_std)
ari_km = adjusted_rand_score(yn, km4.labels_)
print(f"recap (homework Part C): K-means K=4 on standardised (E, H)   ARI = {ari_km:.3f}")


# %% [markdown]
# # Block 2 — Hard → soft assignments: GMM and EM
#
# K-means hands every point exactly one label. A point sitting between two
# clusters is forced to commit even when both choices are nearly equally
# good. A **Gaussian mixture model** relaxes that commitment: every point
# gets a probability vector $\boldsymbol\gamma_i = (\gamma_{i1}, \dots,
# \gamma_{iK})$ with $\sum_k \gamma_{ik} = 1$. The EM algorithm fits the
# mixture by alternating an E-step (compute the $\gamma_{ik}$) and an M-step
# (re-estimate $\pi_k, \boldsymbol\mu_k, \boldsymbol\Sigma_k$).
#
# We use `sklearn.mixture.GaussianMixture` here — the algorithm is exactly
# the EM you saw in MFML §E "EM — E-step and M-step". We focus on the two
# things that are new compared to K-means: (i) **soft** membership, and
# (ii) the **BIC** as a principled, likelihood-based way to choose K.
#
# *(see MFML §"The Gaussian Mixture Model", §"EM — E-step and M-step")*

# %%
# Two overlapping, anisotropic 2-D clusters — the kind of data K-means
# struggles with and a full-covariance GMM handles cleanly.
rng = np.random.default_rng(7)
A = rng.multivariate_normal([-1.0, 0.0], [[0.6, 0.3], [0.3, 0.4]], 200)
B = rng.multivariate_normal([+1.5, 0.5], [[0.5, -0.2], [-0.2, 0.8]], 200)
X_overlap = np.vstack([A, B])
y_overlap = np.array([0] * 200 + [1] * 200)

gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(X_overlap)
gamma = gmm.predict_proba(X_overlap)         # (N, 2) responsibilities
print(f"GMM converged in {gmm.n_iter_} EM iterations   log-lik = {gmm.score(X_overlap)*len(X_overlap):.1f}")
print(f"mixing weights pi = {gmm.weights_.round(3)}")


# %%
# Plot soft-assignment heatmap on the overlap region.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].scatter(X_overlap[:, 0], X_overlap[:, 1], c=gamma[:, 0],
                cmap="bwr", s=14, alpha=0.85, vmin=0, vmax=1)
for k, color in zip(range(2), ["#1f77b4", "#d62728"]):
    axes[0].scatter(*gmm.means_[k], c=color, s=200, marker="X", edgecolors="k", linewidths=1.5)
axes[0].set_title("GMM responsibilities $\\gamma_{i,1}$ (blue=cluster 0, red=cluster 1)")
axes[0].set_aspect("equal")

# BIC vs K on the same data — the model-selection diagnostic.
K_range = list(range(1, 7))
bics = []
for K in K_range:
    g = GaussianMixture(n_components=K, covariance_type="full", random_state=0).fit(X_overlap)
    bics.append(g.bic(X_overlap))
axes[1].plot(K_range, bics, "o-", color="#2ca02c")
axes[1].set_xlabel("K (number of components)"); axes[1].set_ylabel("BIC (lower is better)")
axes[1].axvline(2, color="gray", ls="--", alpha=0.5, label="truth K=2")
axes[1].legend(); axes[1].set_title("Bayesian information criterion vs K")

plt.tight_layout()
plt.show()


# %% [markdown]
# **Read the responsibility heatmap.** Points deep in either cluster get
# $\gamma \approx 0$ or $\gamma \approx 1$; points in the overlap zone get
# $\gamma \approx 0.5$. The model honestly says "I'm not sure" exactly where
# we would expect — that uncertainty is the upgrade soft assignments buy
# over hard ones, and it is *exactly* the property exam Q5 tests in MFML.


# %% [markdown]
# # Block 3 — K-means as a special case of GMM
#
# MFML §"K-Means is a special case of GMM" claimed that K-means is what you
# get from a GMM if you (i) fix every covariance to $\boldsymbol\Sigma_k =
# \sigma^2 \mathbf{I}$ (isotropic, equal width across clusters), and (ii)
# let $\sigma^2 \to 0$. We verify the claim numerically: a tiny-$\sigma$
# `spherical`-covariance GMM and a K-means run on the same data should
# produce the same hard assignment.
#
# *(see MFML §"K-Means is a special case of GMM")*

# %%
X_blob, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=0)

km_blob = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X_blob)

# `spherical` => one scalar variance per component; small `reg_covar` keeps
# the variance pinned near zero so we are effectively in the σ²→0 regime.
gmm_tight = GaussianMixture(
    n_components=4, covariance_type="spherical", reg_covar=1e-6, random_state=0
).fit(X_blob)

# Compare hard assignments (label IDs may permute, so use ARI).
labels_km  = km_blob.labels_
labels_gmm = gmm_tight.predict(X_blob)
print(f"agreement (ARI) between K-means and tight-σ spherical GMM: {adjusted_rand_score(labels_km, labels_gmm):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, lbl, title in zip(axes, [labels_km, labels_gmm],
                          ["K-means (hard, mean update)",
                           "GMM, spherical Σ, σ²→0 (hard limit)"]):
    ax.scatter(X_blob[:, 0], X_blob[:, 1], c=lbl, cmap="tab10", s=14, alpha=0.7)
    ax.set_title(title); ax.set_aspect("equal")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Take-away.** ARI ≈ 1: the two assignments coincide up to a permutation
# of cluster labels. K-means is not "a different algorithm from GMM" — it is
# GMM with maximally constrained covariances and zero soft-assignment
# temperature. This is the first clean example in the course of "simple
# algorithm = limit of richer probabilistic model"; the same pattern returns
# in MFML Unit 8 (probabilistic view of learning) and Unit 11 (unsupervised
# learning, contrastive losses).


# %% [markdown]
# # Block 4 — PCA on Ising — the linear baseline
#
# Now we leave 2-D toy data behind and do unsupervised learning on real
# microstructure images. `IsingDataset(size='full')` gives 5,000 grayscale
# 64×64 spin configurations, half above the Curie temperature ("disordered"),
# half below ("ordered"). We will pretend we do not know the labels.
#
# **The classical first move** is PCA: flatten each 64×64 image into a
# 4,096-D vector, take the top-2 principal components, and look at the
# scatter. PCA is the **linear** autoencoder (encoder = $\mathbf{U}_k^\top$,
# decoder = $\mathbf{U}_k$, MSE) — it is the right baseline to argue against
# before reaching for a neural network.
#
# *(see MFML §"Recap — PCA from Unit 2", §"Why PCA fails on a manifold")*

# %%
ds_ising = IsingDataset(size="full")
X_img = ds_ising.X                     # (5000, 1, 64, 64) in [0, 1]
y_img = ds_ising.y                     # (5000,)
print(f"Ising-full: N={len(ds_ising)}   image shape={tuple(X_img[0].shape)}   classes={y_img.unique().tolist()}")

# Flatten to (N, 4096) and run PCA via torch.pca_lowrank (no sklearn, no
# n^2 covariance matrix in memory).
Xf = X_img.reshape(len(X_img), -1)     # (5000, 4096)
Xf = Xf - Xf.mean(dim=0, keepdim=True) # centre
U, S, V = torch.pca_lowrank(Xf, q=8, center=False)
Z_pca = Xf @ V[:, :2]                  # (5000, 2)
print(f"top-2 singular values (proxy for explained variance): {S[:2].tolist()}")


# %%
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c=y_img.numpy(), cmap="coolwarm",
                s=8, alpha=0.6)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Ising-full in PCA(2)  (colour = phase label, *not* used for fitting)")
plt.colorbar(sc, ax=ax, label="phase (0=disordered, 1=ordered)")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Read this plot.** PCA picks the 2-D *plane* of maximum variance. On the
# Ising microstructure, that plane already separates the two phases by mean
# magnetisation (PC1) — but the two clouds still overlap and PC2 carries
# very little extra information. A linear projection cannot capture the
# *texture* of an ordered configuration vs the texture of a disordered one.
# That is the gap a non-linear autoencoder is built to close.


# %% [markdown]
# # Block 5 — A tiny convolutional autoencoder on Ising-full
#
# Replace PCA's linear $\mathbf{U}_k^\top$ by a 2-layer convolutional encoder,
# and PCA's linear $\mathbf{U}_k$ by a 2-layer transposed-convolution decoder.
# Bottleneck $k=2$ so we can plot the latent space; train on raw images
# without ever showing the labels; minimise MSE reconstruction loss.
#
# *(see MFML §"The autoencoder concept", §"Forward pass through an autoencoder";
# ML-PC §"From classical to learned representations")*

# %%
class ConvAE(nn.Module):
    """Tiny conv-AE for 64x64 binary Ising images. ~80k params."""
    def __init__(self, k=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # (16, 32, 32)
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # (32, 16, 16)
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, k),                             # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, 32 * 16 * 16), nn.ReLU(),
            nn.Unflatten(1, (32, 16, 16)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # (16, 32, 32)
            nn.ConvTranspose2d(16,  1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),# (1, 64, 64)
        )

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x): return self.decode(self.encode(x))


# %%
# Train/val split, loaders. We hold back 1000 images for held-out
# reconstruction-error scoring in Block 6.
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
train_ds, val_ds = random_split(ds_ising, [4000, 1000], generator=gen)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                          generator=torch.Generator().manual_seed(0))
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)


# %%
torch.manual_seed(0)
model = ConvAE(k=2)
print(f"ConvAE params: {n_params(model):,}")

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 3 epochs is enough on CPU to get a sensible latent space for the demo.
n_epochs = 3
for epoch in range(n_epochs):
    model.train()
    for x, _ in train_loader:
        opt.zero_grad()
        x_hat = model(x)
        loss = loss_fn(x_hat, x)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(model(x), x).item() * x.size(0) for x, _ in val_loader) / len(val_ds)
    print(f"epoch {epoch + 1}   val MSE = {val_loss:.4f}")


# %%
# Encode the held-out validation set into the 2-D latent space and plot,
# coloured by the never-seen phase label.
model.eval()
Z_list, y_list = [], []
with torch.no_grad():
    for x, y in val_loader:
        Z_list.append(model.encode(x))
        y_list.append(y)
Z_ae = torch.cat(Z_list).numpy()
Y_ae = torch.cat(y_list).numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc1 = axes[0].scatter(Z_ae[:, 0], Z_ae[:, 1], c=Y_ae, cmap="coolwarm", s=10, alpha=0.7)
axes[0].set_xlabel("$z_1$"); axes[0].set_ylabel("$z_2$")
axes[0].set_title("ConvAE latent on Ising-full val (colour = phase, *not* used for training)")
plt.colorbar(sc1, ax=axes[0], label="phase")

# Side-by-side: a few reconstructions to confirm the AE actually learned to draw spins.
with torch.no_grad():
    x_some, _ = next(iter(val_loader))
    x_hat = model(x_some[:5])
for i in range(5):
    axes[1].imshow(np.hstack([x_some[i, 0].numpy(), x_hat[i, 0].numpy()]),
                   cmap="gray", extent=(i, i + 1, 0, 1))
axes[1].set_xlim(0, 5); axes[1].set_ylim(0, 1)
axes[1].set_xticks([]); axes[1].set_yticks([])
axes[1].set_title("5 originals (left of each pair) vs reconstructions (right)")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Read this latent space.** Without ever seeing a label, the AE has placed
# ordered configurations and disordered configurations in roughly different
# regions of the 2-D latent — *not* perfectly, but with much sharper
# separation than PCA's plane in Block 4. This is the empirical observation
# that motivates representation learning: useful structure emerges *for
# free* from a reconstruction objective.


# %% [markdown]
# # Block 6 — Reconstruction error as anomaly score
#
# A trained autoencoder reconstructs *what it was trained on* well, and
# *anything else* badly. That is the entire idea behind reconstruction-based
# anomaly detection: train on "normal" data, score test images by the MSE
# between input and reconstruction, threshold to flag anomalies.
#
# We score (i) held-out **Ising** images (in-distribution — the AE was
# trained on Ising) and (ii) **Cahn–Hilliard** phase-field snapshots
# (out-of-distribution — same 64×64 grayscale shape, completely different
# physics). The reconstruction MSE should separate the two clearly.
#
# *(see ML-PC §"Specimen leakage", §"Reconstruction shortcuts"; MFML §"Denoising autoencoders")*

# %%
# In-distribution: held-out Ising val set.
def per_image_mse(model, loader, max_n=1000):
    model.eval()
    errs = []
    with torch.no_grad():
        for x, _ in loader:
            x_hat = model(x)
            err = ((x - x_hat) ** 2).mean(dim=(1, 2, 3))   # one MSE per image
            errs.append(err)
            if sum(e.numel() for e in errs) >= max_n:
                break
    return torch.cat(errs)[:max_n].numpy()


id_err = per_image_mse(model, val_loader, max_n=1000)


# %%
# Out-of-distribution: a single Cahn-Hilliard simulation as the "anomaly" set.
ds_ch = CahnHilliardDataset(simulation_number=0)
print(f"Cahn-Hilliard sim 0: N={len(ds_ch)}   image shape={tuple(ds_ch.X[0].shape)}")

ch_loader = DataLoader(ds_ch, batch_size=256, shuffle=False)
ood_err = per_image_mse(model, ch_loader, max_n=1000)

# ROC: label 0 = in-distribution Ising, label 1 = out-of-distribution Cahn-Hilliard.
y_score = np.concatenate([id_err, ood_err])
y_lab   = np.concatenate([np.zeros_like(id_err), np.ones_like(ood_err)])
auc = roc_auc_score(y_lab, y_score)
fpr, tpr, _ = roc_curve(y_lab, y_score)
print(f"per-image MSE   in-dist median = {np.median(id_err):.4f}   ood median = {np.median(ood_err):.4f}")
print(f"ROC AUC (Ising vs Cahn-Hilliard, AE trained on Ising) = {auc:.3f}")


# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(id_err,  bins=40, alpha=0.6, label="Ising (in-dist)",       color="#1f77b4")
axes[0].hist(ood_err, bins=40, alpha=0.6, label="Cahn-Hilliard (ood)",   color="#d62728")
axes[0].set_xlabel("per-image reconstruction MSE"); axes[0].set_ylabel("count")
axes[0].set_title("Reconstruction error as anomaly score"); axes[0].legend()

axes[1].plot(fpr, tpr, lw=2)
axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
axes[1].set_xlabel("false-positive rate"); axes[1].set_ylabel("true-positive rate")
axes[1].set_title(f"ROC, AUC = {auc:.3f}")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Take-away.** The same conv-AE that gave us a meaningful latent space
# (Block 5) also gives us a defensible anomaly score (Block 6) for *free* —
# we trained one model and got two unsupervised use cases out of it. That
# is the practical reason these two topics share a single MFML unit and a
# single ML-PC unit: in materials practice, the same encoder/decoder is
# usually doing both jobs.
#
# **Honest caveat.** This demo conflates "anomaly" with "out-of-distribution
# from a different physics generator", which is the easy regime. Detecting a
# *subtle* defect inside Ising microstructures themselves is much harder and
# is the subject of MFML Unit 11 and ML-PC Unit 11 later in the semester.


# %% [markdown]
# # Block 7 — Student exercises
#
# **Three core (do all three) + one stretch (optional).** Write your code
# in the empty cells below; bring printed plots / numbers to the next class
# for the 5-minute walk-through.

# %% [markdown]
# ## Exercise 1 (core) — K-medoids on contaminated nanoindentation
#
# K-means uses the *mean* as the cluster prototype, which has breakdown
# point 0: a single outlier can drag the centroid arbitrarily far. K-medoids
# replaces the mean with an actual data point — the medoid — which gives it
# robustness to outliers.
#
# **Your task:**
#
# 1. Take the standardised nanoindentation features from Block 1
#    (`Xn_std`, 938 × 2). Inject 5 % synthetic outliers at random by
#    replacing those rows with samples from `np.random.normal(loc=10, scale=2)`
#    in both features.
# 2. Implement K-medoids with the simple **PAM-style swap loop** below
#    (skeleton given). Run it for K=4 on (a) the clean standardised data and
#    (b) the contaminated data.
# 3. Run K-means on the same two datasets. Report adjusted-Rand index
#    against the original Cr-content labels for all four runs.
# 4. Conclusion in two sentences: how much does K-medoids buy you here, and
#    what is the cost in seconds per run?
#
# *Hint: the simplest correct PAM implementation alternates an "assign each
# point to its nearest current medoid" step with a "for each cluster, pick
# the data point with smallest sum-of-distances to the rest of the cluster
# as the new medoid" step. ~25 lines.*

# %%
# YOUR CODE for Exercise 1 below. Skeleton:
#
# def kmedoids(X, K, n_iter=20, rng=None):
#     rng = rng or np.random.default_rng(0)
#     # 1. random initial medoids: pick K data indices uniformly
#     # 2. for each iteration:
#     #      - assignment step: c[i] = argmin_k ||X[i] - X[medoid_k]||
#     #      - update step:    new_medoid_k = argmin_{j in C_k} sum_{i in C_k} ||X[i] - X[j]||
#     # 3. return medoids, assignments
#     ...


# %% [markdown]
# ## Exercise 2 (core) — Tied-weight autoencoder
#
# A common practical trick: force the decoder weights to be the transpose of
# the encoder weights. This roughly halves the parameter count and is a soft
# regulariser ("decoder is the encoder run backwards").
#
# **Your task:**
#
# 1. Build a *fully-connected* tied-weight AE for flattened Ising images with
#    bottleneck $k=8$:
#    - encoder: `z = ReLU(W @ x + b_e)` with `W` of shape `(k, d)` and `d = 4096`.
#    - decoder: `x_hat = sigmoid(W.T @ z + b_d)`  (note: same `W`, transposed).
# 2. Train for 3 epochs on flattened Ising-full images using MSE.
# 3. Report: parameter count vs the `ConvAE` from Block 5, and validation
#    MSE.
#
# *Hint: subclass `nn.Module`, store `self.W = nn.Parameter(torch.randn(k, d) * 0.01)`,
# and call `F.linear(x, self.W)` for the encoder and `F.linear(z, self.W.t())`
# for the decoder. The two bias parameters are still independent.*

# %%
# YOUR CODE for Exercise 2 below.


# %% [markdown]
# ## Exercise 3 (core) — Latent-dim sweep ("elbow for autoencoders")
#
# Just like K-means has an elbow plot for K, autoencoders have one for the
# latent dimension $k$. Below a critical $k$, the bottleneck throws away
# information the data really needs; above it, you are wasting parameters.
#
# **Your task:**
#
# 1. Re-train the `ConvAE` from Block 5 for $k \in \{1, 2, 4, 8, 16, 32\}$,
#    each for 2 epochs (CPU-friendly).
# 2. Record the *validation* MSE at convergence for each $k$.
# 3. Plot validation MSE vs $k$. Where is the elbow? Annotate it.
# 4. Sanity-check: at $k = 32$ the AE has more than enough capacity to
#    reconstruct Ising-full. Why is the MSE not (close to) zero?
#
# *Hint: write a small `train(model, n_epochs)` helper to avoid copy-pasting
# the loop from Block 5 six times.*

# %%
# YOUR CODE for Exercise 3 below.


# %% [markdown]
# ## Exercise 4 (stretch) — Denoising autoencoder
#
# A **denoising** AE is trained to reconstruct a *clean* input from a
# *corrupted* version of it. This forces the network to model the manifold
# the data actually lives on, rather than learning the identity through the
# bottleneck. Empirically, denoising AEs were the breakthrough that revived
# autoencoders before VAEs took over (Vincent et al., 2008), and they are
# the conceptual ancestor of modern diffusion models.
#
# **Your task:**
#
# 1. Wrap the `ConvAE(k=2)` architecture in a training loop that, for each
#    batch `x`, computes `x_noisy = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)`
#    and trains with `loss = MSE(model(x_noisy), x)` (target is the *clean*
#    image).
# 2. Train for 5 epochs on Ising-full.
# 3. On the held-out validation set, take 5 images, corrupt them with the
#    same noise level, run them through the trained denoising AE, and plot
#    `[clean | noisy | reconstruction]` triplets.
# 4. Compare the latent-space scatter of the *denoising* AE against the
#    plain AE from Block 5. Is it cleaner / messier / different? Write
#    one sentence interpreting the difference.

# %%
# YOUR CODE for Exercise 4 below.


# %% [markdown]
# ## Exam-aligned must-know statements (from MFML Unit 5 §"Exam-aligned")
#
# Re-read these after the exercises; today's blocks have given you the
# concrete intuition for every one of them.
#
# 1. K-means minimises within-cluster variance via Lloyd's alternating
#    update.
# 2. K-means converges to a *local* minimum; initialisation matters; use
#    k-means++.
# 3. K-medoids replaces the mean with an actual data point — robust to
#    outliers, works with any dissimilarity (Exercise 1).
# 4. GMMs give **soft** assignments via responsibilities $\gamma_{ik}$
#    (Block 2).
# 5. EM alternates an E-step and an M-step; it monotonically increases the
#    log-likelihood (Block 2).
# 6. K-means is the limit of GMM with isotropic, equal-variance covariances
#    and zero temperature (Block 3).
# 7. PCA = linear encoder + linear decoder + MSE; cannot capture non-linear
#    manifolds (Block 4).
# 8. An autoencoder is non-linear encoder/decoder + reconstruction loss; it
#    is **self-supervised** (Block 5).
# 9. A linear autoencoder is equivalent to PCA; non-linear AEs gain power
#    *only* from the activations (theory bridge between Blocks 4 and 5).
# 10. Reconstruction error doubles as an anomaly score for distribution
#     shift (Block 6).
