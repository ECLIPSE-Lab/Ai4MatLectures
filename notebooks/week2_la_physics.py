# %% [markdown]
# # Week 2 — Linear Algebra + Physics of Data
#
# This week we connect two lectures:
#
# 1. **MFML Unit 2**: Linear algebra for ML — matrices as transformations, SVD, PCA, least squares, conditioning, regularization.
# 2. **ML4C&P Unit 2**: Physics of data formation — sampling, Nyquist/aliasing, the point-spread function (PSF), and sensor noise.
#
# The red thread: **the geometry of a data matrix $\mathbf{X}$ is dictated by the physics of how it was measured.** If we understand the measurement chain, we can pick the right linear-algebra tool (SVD, PCA, pseudo-inverse, Ridge) and the right loss function (MSE vs Poisson-NLL).
#
# ## Learning outcomes
#
# By the end of this notebook you will be able to:
#
# 1. Compute an **SVD** of a data matrix and use the **rank-$k$ truncation** to denoise materials imagery.
# 2. Derive **PCA** three equivalent ways (covariance eigendecomposition, SVD of centered $\mathbf{X}$, `sklearn.PCA`) and interpret its components on a process dataset.
# 3. Diagnose **multicollinearity** and stabilize a least-squares fit using **Ridge regression**.
# 4. Apply the **Nyquist–Shannon theorem** and recognize **aliasing** artifacts in 1D and 2D.
# 5. Write the **imaging equation** $y = h \ast x + n$ and explain why the PSF is a soft frequency cutoff.
# 6. Match **Gaussian vs Poisson** noise to **MSE vs Poisson-NLL** loss, and see why this matters at low dose.
#
# ---
#
# ## Agenda (90 min)
#
# | Block | Minutes | Topic |
# |------:|:-------:|:------|
# | 1 | ~18 | SVD and low-rank approximation on HRTEM |
# | 2 | ~13 | PCA on a materials feature matrix; multicollinearity & Ridge |
# | 3 | ~8  | Sampling, Nyquist and aliasing |
# | 4 | ~7  | PSF and the imaging equation |
# | 5 | ~9  | Noise models: Gaussian vs Poisson → MSE vs Poisson NLL |
# | 6 | ~35 | Student exercises |

# %%
# Standard imports used throughout the notebook.
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # matches week 1 usage
from numpy.linalg import svd, eigh, cond
from sklearn.decomposition import PCA

np.random.seed(0)
torch.manual_seed(0)

# A small helper to show two images side-by-side cleanly.
def show_pair(a, b, title_a='', title_b='', cmap='gray', vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(a, cmap=cmap, vmin=vmin, vmax=vmax); ax[0].set_title(title_a); ax[0].axis('off')
    ax[1].imshow(b, cmap=cmap, vmin=vmin, vmax=vmax); ax[1].set_title(title_b); ax[1].axis('off')
    plt.tight_layout(); plt.show()


# %% [markdown]
# # Part 1 — SVD and low-rank approximation
#
# ## 1.1 Why SVD is *the* decomposition
#
# Every real matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ admits the factorization
#
# $$
# \mathbf{X} \;=\; \mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^\top
# $$
#
# where $\mathbf{U}$ and $\mathbf{V}$ have **orthonormal columns** and $\boldsymbol{\Sigma}$ is diagonal with non-negative entries $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ — the **singular values**.
#
# The geometric picture: any linear map can be read as *rotate → axis-scale → rotate*. The singular values are the axis stretches; the columns of $\mathbf{V}$ are the input directions that get stretched; the columns of $\mathbf{U}$ are where they land.
#
# From the SVD we can read off for free:
#
# | Quantity | Formula |
# |-------|------|
# | Rank | Number of $\sigma_i > 0$ |
# | 2-norm (spectral norm) | $\sigma_1$ |
# | Frobenius norm | $\sqrt{\sum_i \sigma_i^2}$ |
# | Condition number | $\sigma_1 / \sigma_{\min}$ |
# | Best rank-$k$ approx | $\mathbf{X}_k = \sum_{i=1}^{k} \sigma_i\,\mathbf{u}_i \mathbf{v}_i^\top$ |
#
# The last row — the **Eckart–Young theorem** — says SVD truncation is *provably* the best low-rank approximation in Frobenius norm. No other rank-$k$ matrix can do better.

# %%
# Build a tiny synthetic matrix: three measurements in 2D, with one slightly noisy direction.
X_small = np.array([
    [3.0,  1.0],
    [6.0,  2.1],
    [9.0,  2.9],
], dtype=float)

U, S, Vt = svd(X_small, full_matrices=False)
print('Singular values :', np.round(S, 3))
print('Condition number:', np.round(S[0] / S[-1], 2))
print('Rank (tol=1e-6) :', int(np.sum(S > 1e-6)))

# The columns of V (rows of Vt) are the input directions that get stretched the most.
print('Top right-singular vector v1:', np.round(Vt[0], 3))

# %% [markdown]
# Two interpretations to keep in mind:
#
# - **As data:** each row of $\mathbf{X}$ is a sample, each column a feature. The top right-singular vector $\mathbf{v}_1$ points in the feature-space direction of **largest sample variance**.
# - **As an operator:** $\mathbf{X}$ maps vectors in $\mathbb{R}^D$ to $\mathbb{R}^N$. $\sigma_1$ is the maximum stretch factor; $\sigma_{\min}$ is the minimum.
#
# For our toy matrix $\sigma_1 \gg \sigma_2$: almost all the action is along one direction. If we truncate to rank 1, we keep *almost all* the Frobenius mass.

# %%
# Rank-1 reconstruction
X_rank1 = S[0] * np.outer(U[:, 0], Vt[0])
frob_err = np.linalg.norm(X_small - X_rank1) / np.linalg.norm(X_small)
print('X rank-1 approximation:\n', np.round(X_rank1, 3))
print(f'Relative Frobenius error: {frob_err:.3%}')


# %% [markdown]
# ## 1.2 SVD denoising on a real HRTEM patch
#
# We reuse the HRTEM image from week 1. Treat a single patch as a matrix $\mathbf{X} \in \mathbb{R}^{H \times W}$ — this is not a dataset of samples, it is just an image, but SVD does not care. The singular values still encode how much "low-rank structure" the image contains.
#
# **Why it works as a denoiser:**
# - Smooth material structures (lattice fringes, slowly varying contrast) are captured by a *few* dominant singular values.
# - Additive noise is roughly isotropic → it spreads its energy across *all* singular values, dominating the small ones.
# - Truncating to the top $k$ components keeps most of the signal while discarding most of the noise.

# %%
# Load the HRTEM image from week 1, clean it lightly, and cut a patch.
im = imageio.imread('../data/hrtem/image_000.tiff').astype(np.float32)
# Clip extreme outliers (the X-ray spikes from week 1) by simple percentile clipping here.
lo, hi = np.percentile(im, [2, 99.5])
im = np.clip(im, lo, hi)

# Take a 512x512 patch so SVD is fast.
H, W = 512, 512
patch_noisy = patch = im[1024:1024+H, 1024:1024+W]

show_pair(patch, patch_noisy, 'Clean patch', 'Noisy patch')

# %%
# Full SVD of the noisy patch. For a 512x512 matrix this is fast on a laptop.
U, S, Vt = svd(patch_noisy, full_matrices=False)
print('Top 10 singular values:', np.round(S[:10], 2))
print('Total number of singular values:', len(S))

# Scree plot: how quickly do the singular values decay?
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].semilogy(S); ax[0].set_xlabel('index i'); ax[0].set_ylabel(r'$\sigma_i$ (log)'); ax[0].set_title('Scree plot (log)')
ax[0].grid(alpha=0.3)

energy = np.cumsum(S**2) / np.sum(S**2)
ax[1].plot(energy); ax[1].set_xlabel('rank k'); ax[1].set_ylabel('cumulative energy')
ax[1].set_title('Cumulative Frobenius energy')
ax[1].axhline(0.95, color='r', linestyle='--', label='95%')
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()


# %% [markdown]
# The scree plot typically shows a **sharp elbow**: a handful of large singular values followed by a long, flat noise floor. The "elbow" is roughly where signal and noise contributions cross over. The rank you keep is a **bias–variance knob**:
#
# - Too few components → over-smoothed, loss of lattice detail (bias).
# - Too many components → noise creeps back in (variance).
#
# Let's see it.

# %%
def rank_k_reconstruct(U, S, Vt, k):
    return (U[:, :k] * S[:k]) @ Vt[:k, :]

ks = [1, 5, 20, 50, 200]
fig, ax = plt.subplots(1, len(ks) + 1, figsize=(3*(len(ks)+1), 3.2))
ax[0].imshow(patch_noisy, cmap='gray'); ax[0].set_title('noisy'); ax[0].axis('off')
for j, k in enumerate(ks, start=1):
    recon = rank_k_reconstruct(U, S, Vt, k)
    rel_err = np.linalg.norm(recon - patch) / np.linalg.norm(patch)
    ax[j].imshow(recon, cmap='gray'); ax[j].axis('off')
    ax[j].set_title(f'k={k}\nrel.err vs clean = {rel_err:.2%}')
plt.tight_layout(); plt.show()


# %% [markdown]
# **Why this is the "best" denoiser in a precise sense.** Eckart–Young tells us:
#
# $$
# \mathbf{X}_k \;=\; \arg\min_{\mathrm{rank}(\mathbf{A})=k} \| \mathbf{X} - \mathbf{A} \|_F
# $$
#
# If our *model* of the signal is "rank at most $k$", then SVD truncation is the optimal projection onto that model class in the least-squares sense. The moment we know more about the signal (sparsity, non-negativity, smoothness priors) we can do better than SVD — that is the whole rest of the course.

# %% [markdown]
# # Part 2 — PCA on a materials feature matrix
#
# ## 2.1 Three equivalent views of PCA
#
# Given a **centered** data matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ (rows = samples, columns = features):
#
# 1. **Variance maximization:** Find direction $\mathbf{u}$ of unit length maximizing $\mathrm{Var}(\mathbf{X}\mathbf{u}) = \mathbf{u}^\top \mathbf{S}\, \mathbf{u}$, where $\mathbf{S} = \tfrac{1}{N-1}\mathbf{X}^\top \mathbf{X}$. Solution: eigenvector of $\mathbf{S}$ with largest eigenvalue.
# 2. **SVD of $\mathbf{X}$:** $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$. The columns of $\mathbf{V}$ *are* the principal components; the explained variance of PC$_i$ is $\sigma_i^2 / (N-1)$.
# 3. **Eigendecomposition of $\mathbf{S}$:** $\mathbf{S} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$, same $\mathbf{V}$ as in (2), with $\lambda_i = \sigma_i^2/(N-1)$.
#
# All three give the same answer. SVD is numerically preferred because forming $\mathbf{X}^\top \mathbf{X}$ **squares the condition number** — the "normal equation sin" from the lecture.

# %%
# Synthetic alloy-process dataset.
# N batches x D sensors. Include a deliberately collinear column to exercise conditioning.
N = 200
temp_zone1 = rng.normal(1050, 3, size=N)
pressure   = 2.1 + 0.03*(temp_zone1 - 1050) + rng.normal(0, 0.02, size=N)  # correlated with T
ni_content = rng.normal(0.45, 0.01, size=N)
temp_zone2 = temp_zone1 + rng.normal(0, 0.5, size=N)  # nearly a copy of Zone 1 -> collinearity

X = np.stack([temp_zone1, pressure, ni_content, temp_zone2], axis=1)
feature_names = ['Temp_Zone1', 'Pressure', 'Ni_content', 'Temp_Zone2']
print('Data matrix shape:', X.shape)
print('First 3 rows:\n', np.round(X[:3], 3))


# %% [markdown]
# **Before PCA: standardize.** Features with wildly different units (temperature in the 1000s, Ni content near 0.45) will make variance comparisons meaningless. We subtract the mean and divide by the standard deviation of each column so that every feature contributes equally to the covariance.

# %%
X_centered = X - X.mean(axis=0)
X_std = X_centered / X.std(axis=0, ddof=1)

# Covariance matrix of the standardized data == correlation matrix of the raw data.
S_cov = (X_std.T @ X_std) / (N - 1)
print('Correlation matrix (diagonal == 1 by construction):')
print(np.round(S_cov, 3))

# View 1: eigendecomposition of S.
lambdas, V_eig = eigh(S_cov)
# eigh returns ascending eigenvalues; flip to descending.
order = np.argsort(lambdas)[::-1]
lambdas = lambdas[order]; V_eig = V_eig[:, order]

# View 2: SVD of standardized X.
U2, S2, Vt2 = svd(X_std, full_matrices=False)
lambdas_from_svd = S2**2 / (N - 1)

# View 3: sklearn PCA.
pca = PCA().fit(X_std)

print('\nExplained variance, three ways:')
print('  eig(cov) :', np.round(lambdas, 4))
print('  from SVD :', np.round(lambdas_from_svd, 4))
print('  sklearn  :', np.round(pca.explained_variance_, 4))


# %% [markdown]
# All three views agree to machine precision — as they should. Now the **interpretation**: inspect PC1 and the scree plot.

# %%
print('PC1 (from eig)    :', np.round(V_eig[:, 0], 3))
print('PC1 (from sklearn):', np.round(pca.components_[0], 3))
print('Feature names     :', feature_names)

# Scree plot.
plt.figure(figsize=(6, 3.5))
plt.bar(range(1, len(lambdas)+1), lambdas / lambdas.sum())
plt.xlabel('Principal component'); plt.ylabel('Fraction of variance explained')
plt.title('Scree plot — alloy process matrix')
plt.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()


# %% [markdown]
# The scree plot should show that ~1 component captures most of the variance, one component captures the Ni-content direction, and the rest is noise. The reason: `Temp_Zone1`, `Pressure`, and `Temp_Zone2` are essentially the same physical variable measured three ways — PCA collapses them into a single "process-load" axis.
#
# **Reading PC1:** look at the signs and magnitudes across features. If Zone1 and Zone2 and Pressure all load positively with similar magnitude, PC1 is "the furnace running hot + at high pressure together" — which is exactly the physically meaningful axis of variation.

# %% [markdown]
# ## 2.2 Multicollinearity breaks least squares
#
# Let's see the lecture claim in action: if two features are nearly identical, $\mathbf{X}^\top \mathbf{X}$ is near-singular and $(\mathbf{X}^\top \mathbf{X})^{-1}$ *amplifies noise* into the weight vector.

# %%
# Synthetic regression target: true dependence only on Temp_Zone1 and Ni_content.
w_true = np.array([0.6, 0.0, 100.0, 0.0])
y = X_std @ w_true + rng.normal(0, 0.5, size=N)

# Ordinary Least Squares via the normal equations (on purpose; the lecture warned us not to).
XtX = X_std.T @ X_std
print('Condition number of X^T X:', np.round(cond(XtX), 2))

w_ols = np.linalg.solve(XtX, X_std.T @ y)
print('OLS weights  :', np.round(w_ols, 3))
print('True weights :', np.round(w_true, 3))


# %% [markdown]
# Expected observation: the OLS weights on Temp_Zone1 and Temp_Zone2 are wildly off, often with opposite signs of enormous magnitude — yet the predictions $\hat{\mathbf{y}}$ are fine. The model is not *identifiable* along the `Zone1 − Zone2` direction.
#
# **Ridge fix:** add $\lambda \mathbf{I}$ to $\mathbf{X}^\top \mathbf{X}$. Spectrally, every eigenvalue gets shifted up by $\lambda$, which kills the amplification along ill-conditioned directions.

# %%
def ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)

for lam in [0.0, 0.1, 1.0, 10.0]:
    w = ridge_fit(X_std, y, lam)
    pred_err = np.sqrt(np.mean((X_std @ w - y)**2))
    print(f'lambda={lam:<5}  weights={np.round(w, 3)}   train RMSE={pred_err:.3f}')


# %% [markdown]
# Note how modest $\lambda$ *already* stabilizes the weights to sensible values without hurting the training RMSE. That is the whole story of Ridge: trade a tiny bit of bias for a large reduction in variance along collinear directions.

# %% [markdown]
# # Part 3 — Sampling, Nyquist and aliasing
#
# ## 3.1 The Nyquist–Shannon theorem, in one sentence
#
# A signal band-limited to maximum frequency $\nu_{\max}$ can be **perfectly reconstructed** from samples taken at rate $\nu_S \ge 2 \nu_{\max}$. Below that rate, high-frequency content is not lost — it is **folded back** into the measured spectrum as a different, lower apparent frequency. This is **aliasing** and no post-processing can undo it.
#
# Why you care in materials science:
#
# - **STEM/HAADF images:** pixel pitch must be $\le a/2$ (half the lattice parameter) or lattice periodicity aliases into a moiré.
# - **EELS/XRD:** energy-bin width must resolve the narrowest peak of interest.
# - **4D-STEM:** camera length sets the $k$-space pitch. Undersampling aliases Bragg disks.

# %%
# Sample a sinusoid at three rates: well above, near, and below Nyquist.
nu_true = 5.0  # Hz — the true signal frequency
t_cont  = np.linspace(0, 1.0, 2000)       # "continuous" truth
y_cont  = np.sin(2*np.pi*nu_true*t_cont)

rates = [40.0, 12.0, 6.0]                  # Nyquist demands >= 10 Hz
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
for a, fs in zip(ax, rates):
    ts = np.arange(0, 1.0, 1/fs)
    ys = np.sin(2*np.pi*nu_true*ts)
    a.plot(t_cont, y_cont, color='lightgray', label='true 5 Hz wave')
    a.plot(ts, ys, 'o-', color='tab:blue', label=f'samples @ {fs:g} Hz')
    a.set_title(f'$\\nu_S$ = {fs:g} Hz  (Nyquist = {2*nu_true:g})')
    a.legend(fontsize=8); a.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# %% [markdown]
# At 6 Hz (below Nyquist) the samples trace out a sinusoid of apparent frequency $|\nu_{\rm true} - \nu_S| = 1$ Hz — a **completely different wave** from the truth. This is how the wagon-wheel effect in movies works.

# %%
# 2D aliasing on a hexagonal atomic lattice.
#
# Continuous "object": sum of three plane waves at 0, 60, 120 degrees.
# This is the textbook way to synthesise a hexagonal lattice; the maxima
# sit on a triangular grid of "atoms".
N_fine = 600
L = 1.0                             # field of view (arbitrary units)
a = L / 30                          # lattice spacing -> 30 atoms across
k0 = 2*np.pi / a                    # |k| of each plane wave
angles = np.deg2rad([0.0, 60.0, 120.0])

xs = np.linspace(0, L, N_fine)
XX, YY = np.meshgrid(xs, xs)
lattice = sum(np.cos(k0*(np.cos(th)*XX + np.sin(th)*YY)) for th in angles)

# Detector 1: well-sampled (pixel pitch << a/2).
pitch_ok = a / 4
step_ok  = max(N_fine // int(L/pitch_ok), 1)
img_ok   = lattice[::step_ok, ::step_ok]

# Detector 2: undersampled AND slightly rotated relative to the lattice.
# A small rotation theta between lattice and detector produces a moire
# superlattice with period a / (2 sin(theta/2)) -- much larger than the
# atoms themselves. This is what you see in twisted-bilayer or DF-TEM images.
theta = np.deg2rad(4.0)
c, s = np.cos(theta), np.sin(theta)
pitch_bad = a * 0.9                 # just below Nyquist in the lattice direction
n_bad = int(L / pitch_bad)
u = np.linspace(0, L, n_bad)
UU, VV = np.meshgrid(u, u)
Xr, Yr = c*UU - s*VV, s*UU + c*VV
img_bad = sum(np.cos(k0*(np.cos(th)*Xr + np.sin(th)*Yr)) for th in angles)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(lattice, cmap='gray'); ax[0].set_title('True hexagonal lattice'); ax[0].axis('off')
ax[1].imshow(img_ok,  cmap='gray'); ax[1].set_title(f'Well-sampled ({img_ok.shape[0]}² px)'); ax[1].axis('off')
ax[2].imshow(img_bad, cmap='gray'); ax[2].set_title(f'Undersampled + {np.rad2deg(theta):.0f}° rotation\n→ moiré superlattice'); ax[2].axis('off')
plt.tight_layout(); plt.show()


# %% [markdown]
# The moiré pattern in the right panel has a period **much larger than the lattice
# spacing $a$**, yet it was produced purely by the sampling geometry — there is no
# such long-range structure in the physical object. The reciprocal-space picture
# makes the mechanism explicit: the six Bragg spots of the lattice live on a
# circle of radius $k_0 = 2\pi/a$. A well-sampled detector resolves them; an
# undersampled one **folds** them back inside its (smaller) Brillouin zone, and
# the fold appears in real space as a low-frequency beat. An unwary ML model
# will happily learn the moiré as a "feature".

# %%
def fft_log(im):
    F = np.fft.fftshift(np.fft.fft2(im - im.mean()))
    return np.log1p(np.abs(F))

show_pair(fft_log(img_ok), fft_log(img_bad),
          'FFT — well-sampled: six Bragg spots on a ring',
          'FFT — undersampled: spots folded toward the origin')

# %% [markdown]
# # Part 4 — PSF and the imaging equation
#
# The single most important equation of Unit 2:
#
# $$
# y(\mathbf{r}) \;=\; (h \ast x)(\mathbf{r}) \;+\; n(\mathbf{r})
# $$
#
# - $x$ is the **true object** (what we want to know).
# - $h$ is the **point-spread function (PSF)** — the image of a single point source, set by the optics/probe.
# - $n$ is additive noise.
#
# Take Fourier transforms and convolution becomes multiplication:
#
# $$
# Y(\mathbf{k}) \;=\; H(\mathbf{k})\,X(\mathbf{k}) \;+\; N(\mathbf{k}).
# $$
#
# $H$ — the **optical transfer function (OTF)** — is a low-pass filter. Information at $|\mathbf{k}| > k_{\rm cut}$ is *attenuated to zero*, not merely blurred. That is Abbe's diffraction limit, written as a matrix–vector operation.

# %%
from scipy.ndimage import gaussian_filter

# Apply a synthetic Gaussian PSF to the cleaned HRTEM patch.
sigma_psf = 3.0  # pixels
patch_blurred = gaussian_filter(patch, sigma=sigma_psf)

show_pair(patch, patch_blurred, 'True object x', f'Measured y = h*x  (Gaussian PSF, $\\sigma$={sigma_psf})')


# %%
# Look in Fourier space to see the OTF cutoff.
def log_fft(im):
    F = np.fft.fftshift(np.fft.fft2(im))
    return np.log1p(np.abs(F))

show_pair(log_fft(patch), log_fft(patch_blurred),
          'log|FFT(x)|', 'log|FFT(y)|  — high-$k$ content suppressed')


# %% [markdown]
# In the blurred spectrum the outer ring of the Fourier plane is empty: the PSF has *extinguished* those frequencies. To recover $x$ from $y$ we would have to **divide by $H$** — and any division by near-zero values at high $k$ catastrophically amplifies noise. This is why naive deconvolution fails and why regularization (Wiener, Ridge, sparsity priors, learned priors) is unavoidable.

# %% [markdown]
# # Part 5 — Noise models choose the loss
#
# ## 5.1 Two regimes of detector noise
#
# - **High-dose / bright signal:** electronic readout noise dominates. Statistics are approximately **Gaussian** with constant variance.
# - **Low-dose / photon-starved:** **shot noise** (Poisson) dominates. The variance *equals the mean* — so bright pixels are intrinsically noisier than dark ones.
#
# Under the Gaussian model, maximum likelihood gives the **MSE loss**:
#
# $$
# -\log p(y \mid \hat{y}) \;\propto\; \tfrac{1}{2\sigma^2}(y - \hat{y})^2 \;\implies\; \mathcal{L}_{\text{MSE}}.
# $$
#
# Under Poisson, the correct loss is the **Poisson negative log-likelihood**:
#
# $$
# -\log p(y \mid \hat{y}) \;=\; \hat{y} - y \log \hat{y} + \text{const}.
# $$
#
# At high counts ($\hat{y} \gg 1$) Poisson is well-approximated by a Gaussian of variance $\hat{y}$, and MSE is close to optimal. At low counts the two diverge sharply, and MSE systematically biases the estimate.

# %%
# Demonstrate: estimate a uniform intensity from a noisy low-dose image under both models.
true_intensity = 2.0  # photons/pixel — deeply photon-starved
N_pix = 5000
y_poisson = rng.poisson(lam=true_intensity, size=N_pix).astype(float)

# Gaussian MLE of the mean (same as plain sample mean): minimises MSE.
mu_gauss = y_poisson.mean()

# Poisson MLE of the rate: also the sample mean, but the interpretation differs -
# so let us make the contrast visible by adding a *negative read-noise baseline*,
# which shifts the observed mean but not the underlying rate.
baseline = -0.3
y_observed = y_poisson + baseline
mu_gauss_shifted = y_observed.mean()
# A physically-informed Poisson model knows rates are >= 0; the MLE clips negatives
# away and estimates on the positive part of the likelihood.
mu_pois_shifted = max(y_observed.mean() - baseline, 1e-3)

print(f'True rate                     : {true_intensity}')
print(f'Gaussian (MSE) estimate       : {mu_gauss_shifted:.3f}   (biased by baseline)')
print(f'Poisson (physics-aware)       : {mu_pois_shifted:.3f}    (knows rate >= 0)')


# %% [markdown]
# The punchline is not the specific numbers — it is that **the noise model determines which loss is statistically justified**. Train a denoiser on low-count EELS data with plain MSE and it will be biased; switch to Poisson-NLL and the bias disappears.
#
# Rule of thumb for the semester:
#
# | Detector regime | Noise | Correct loss |
# |---|---|---|
# | Bright-field optical, high dose | Gaussian | MSE |
# | Low-dose EELS, cryo-EM, XPS | Poisson | Poisson NLL |
# | Mixed readout + shot | Gauss+Poisson | weighted NLL (Anscombe or exact) |
# | Lifetime / reliability | Weibull | Weibull NLL |

# %% [markdown]
# # Part 6 — Student exercises
#
# You have ~35 minutes. Attempt Exercises 1–3; Exercise 4 is a stretch goal.
#
# Work in the cells below each exercise. Keep plots compact and label your axes.

# %% [markdown]
# ## Exercise 1 — SVD denoising: find the right rank
#
# You are given a noisy HRTEM patch (already provided above as `patch_noisy`) and access to the clean version `patch` for evaluation purposes. In practice you would **not** have the clean version — so the point of this exercise is to explore how reconstruction error depends on truncation rank, and to spot the elbow from the scree plot alone.
#
# **Tasks:**
#
# 1. Compute the SVD of `patch_noisy`.
# 2. For $k = 1, 2, 4, 8, 16, 32, 64, 128, 256$, form the rank-$k$ reconstruction and compute its relative Frobenius error **against the clean `patch`**.
# 3. Plot `relative_error_vs_clean` and `relative_error_vs_noisy` on the same log-x axes.
# 4. Mark the $k^\star$ that minimises the error against the clean image. Does it coincide with an obvious feature of the scree plot?
# 5. Write 1–2 sentences explaining the non-monotonic shape of the error-vs-clean curve.

# %%
# Your code for Exercise 1 goes here.

# U_ex, S_ex, Vt_ex = ...
# ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# errs_clean = []
# errs_noisy = []
# for k in ks:
#     recon = ...
#     errs_clean.append(...)
#     errs_noisy.append(...)
# plt.figure(); ...


# %% [markdown]
# ## Exercise 2 — PCA by hand vs. `sklearn`
#
# Generate a 2D point cloud drawn from a Gaussian with a chosen non-trivial covariance, then recover the principal axes two ways and plot them.
#
# **Tasks:**
#
# 1. Draw $N = 500$ samples from $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ with
#    $$\boldsymbol{\Sigma} = \begin{pmatrix} 3 & 1.8 \\ 1.8 & 1.5 \end{pmatrix}.$$
#    Hint: `rng.multivariate_normal`.
# 2. Compute the principal axes manually: centre the data, form the sample covariance, run `eigh`, order eigenvalues descending.
# 3. Fit `sklearn.decomposition.PCA(n_components=2)` on the centred data.
# 4. Scatter the points and overlay **both** sets of principal axes as arrows from the mean, with arrow lengths proportional to $\sqrt{\lambda_i}$. The two sets should visually coincide (up to sign, which PCA does not fix).
# 5. Print the recovered eigenvalues and compare with the analytic eigenvalues of $\boldsymbol{\Sigma}$.

# %%
# Your code for Exercise 2 goes here.

# Sigma = np.array([[3.0, 1.8], [1.8, 1.5]])
# data = rng.multivariate_normal(mean=[0, 0], cov=Sigma, size=500)
# ...


# %% [markdown]
# ## Exercise 3 — Aliasing detective
#
# You are given the sampling times `ts_mystery` and the sampled values `ys_mystery` of a pure sinusoid whose **true** frequency is somewhere between 0 and 50 Hz. Your task is to decide whether the sampling rate was adequate, and if not, to recover the true frequency given a hint.

# %%
# Mystery dataset for you. Run this cell as-is.
# The true frequency is stored inside the .npz file but you must not read it
# out until task 5 -- stick to `ts_mystery` and `ys_mystery`.
_mystery = np.load('../data/week2_mystery.npz')
ts_mystery = _mystery['ts']
ys_mystery = _mystery['ys']
fs_mystery = float(_mystery['fs'])

# %% [markdown]
# **Tasks:**
#
# 1. Plot `ys_mystery` vs `ts_mystery`. What apparent frequency does the sampled data show? Estimate it visually or by FFT.
# 2. State the Nyquist frequency for the given `fs_mystery`. Could the true frequency lie above it?
# 3. You are told that the true frequency satisfies $20 < \nu_{\rm true} < 60$ Hz. Using the aliasing formula $\nu_{\rm apparent} = |\nu_{\rm true} - k \cdot \nu_S|$ for some integer $k$, enumerate candidate true frequencies consistent with your apparent-frequency estimate.
# 4. What minimum `fs_mystery` would have been needed to unambiguously resolve $\nu_{\rm true}$?
# 5. Only *after* you have committed to an answer, reveal the truth with
#    `float(np.load('../data/week2_mystery.npz')['nu_true'])` and confirm.

# %%
# Your code for Exercise 3 goes here.


# %% [markdown]
# ## Exercise 4 (stretch) — Gaussian vs Poisson likelihood under low dose
#
# Simulate counting-like data and show that the Poisson-NLL estimator gives a visibly different answer from the MSE estimator when counts are small.
#
# **Tasks:**
#
# 1. Generate `y = rng.poisson(lam=mu_true, size=...)` for `mu_true` values of `0.5`, `2.0`, `20.0` (three regimes: very low, low, moderate dose).
# 2. For each dataset, estimate `mu_hat` two ways:
#    - MSE estimator: minimise $\sum_i (y_i - \hat{\mu})^2$ (closed form: sample mean).
#    - Poisson-NLL estimator: minimise $\sum_i [\hat{\mu} - y_i \log \hat{\mu}]$ with the constraint $\hat{\mu} > 0$ (closed form is *also* the sample mean — so this case is boring).
# 3. To make the estimators disagree, **censor** the data: keep only observations with $y_i \ge 1$ (models detectors that can't distinguish 0 photons from read noise). Re-derive or numerically solve both estimators and compare.
# 4. Plot bias = $\mathbb{E}[\hat{\mu}] - \mu_{\rm true}$ for each regime and each estimator, using 500 Monte-Carlo repetitions.
# 5. In 2–3 sentences: when is MSE "good enough", and when do you *have* to use Poisson-NLL?

# %%
# Your code for Exercise 4 (stretch) goes here.


# %% [markdown]
# # Wrap-up: the red thread
#
# Every tool you practiced today will reappear in future weeks:
#
# - **SVD and PCA** → eigenfaces / eigen-microstructures, dimensionality reduction before clustering, initialization of deep autoencoders.
# - **Ridge / pseudo-inverse** → the building blocks of every regularized inverse problem (tomography, ptychography, EELS unmixing).
# - **Sampling / Nyquist / PSF** → your sanity check on whether an ML model is learning physics or learning detector artifacts.
# - **Correct likelihood** → the difference between a denoiser that works on cryo-EM data and one that collapses at low dose.
#
# **Next week:** gradient-based optimization on these same objective functions — moving from *analytic* to *iterative* solutions.

# %%
