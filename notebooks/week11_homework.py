# %% [markdown]
# # Week 11 — Homework (do BEFORE the Thursday exercise)
#
# This notebook is the **mandatory warm-up** for the Week 11 in-class
# exercise.  It puts the VAE machinery from MFML Unit 11 in your hands —
# encoder, reparameterisation, decoder, and the ELBO as
# *reconstruction + KL* — so Thursday can spend its 90 minutes on the
# integrated story: **conditional generation for property targeting**
# (ML-PC W11 inverse design) and **latent-space arithmetic on materials
# embeddings** (MG W11).
#
# **Time:** ~75 minutes.
#
# ## Red thread
#
# > *Generation is interpretation in reverse.* Week 9 read the latent
# > space; Week 11 writes it. The VAE you build today turns a latent
# > vector $z$ into a microstructure image — a continuous knob over the
# > Cahn–Hilliard space. Once that knob exists, Thursday will use it for
# > inverse design (sample $z$ that gives a target free energy) and
# > compare to a diffusion model (sample $z$ in noise, denoise back).
#
# ## What this homework is
#
# | Part | Min | Topic | Lecture anchor |
# |---|---:|---|---|
# | A | 25 | Hand-rolled VAE on Cahn–Hilliard; ELBO = reconstruction + KL | MFML §"VAE — encoder, decoder, ELBO", §"Reparameterisation trick" |
# | B | 20 | ELBO decomposition; β-VAE sweep at β ∈ {0.1, 1.0, 4.0} | MFML §"ELBO decomposition", §"β-VAE" |
# | C | 20 | Latent-space interpolation between two CH images | MFML §"Latent traversal"; MG §"Latent-space arithmetic" |
# | D | 10 | Reflection — what does β do? | bridge to Thursday Block 3 |
#
# ## What you must hand in (or be able to show on Thursday)
#
# 1. Part A: 8-image grid with 4 reconstructions and 4 prior samples;
#    final-epoch ELBO with reconstruction term and KL term reported separately.
# 2. Part B: training-curve overlay of reconstruction term and KL term
#    for the three β values.
# 3. Part C: 5-image interpolation grid annotated with linearly-interpolated
#    free energy estimates.
# 4. Part D: paragraph (~5 sentences) on what β does.

# %%
# Standard imports.  Same idiom as weeks 2-9.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from ai4mat.datasets import CahnHilliardDataset

np.random.seed(0)
torch.manual_seed(0)

# Use GPU if available; otherwise CPU.  Everything below is device-agnostic.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# %% [markdown]
# ## Helpers — load data and define the VAE

# %%
print("Loading Cahn-Hilliard (3 simulations, ~3000 samples)...")
ch = CahnHilliardDataset(simulation_number=[0, 1, 2])
print(f"  {len(ch)} samples, image shape {tuple(ch.X.shape[1:])}, "
      f"energy range = [{ch.y.min():.3g}, {ch.y.max():.3g}]")

# Standardise the energy targets for downstream use.  We will not use them
# during VAE training in Part A (the VAE is unconditional), but Part C and
# Thursday will.
y_mean = ch.y.mean().item(); y_std = ch.y.std().item()
y_norm = (ch.y - y_mean) / y_std

# Subsample for speed; 1500 train, 300 test is plenty for the VAE to learn
# basic CH structure.
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(ch), generator=g)
n_tr = 1500; n_te = 300
tr_idx, te_idx = perm[:n_tr], perm[n_tr:n_tr + n_te]
X_tr, X_te = ch.X[tr_idx].to(DEVICE), ch.X[te_idx].to(DEVICE)
y_tr_n, y_te_n = y_norm[tr_idx].to(DEVICE), y_norm[te_idx].to(DEVICE)
print(f"  train {X_tr.shape}, test {X_te.shape}")


# %% [markdown]
# # Part A — Hand-rolled VAE on Cahn-Hilliard
#
# A VAE is three pieces and one trick:
#
# 1. **Encoder** $q_\phi(z|x)$: a network that maps an image $x$ to two
#    vectors $\mu(x)$ and $\log \sigma^2(x)$ — the parameters of a
#    diagonal Gaussian over the latent.
# 2. **Reparameterisation:** sample $z = \mu + \sigma \cdot \varepsilon$
#    with $\varepsilon \sim \mathcal{N}(0, I)$. The randomness is moved
#    *outside* the gradient path so backprop flows through $\mu, \sigma$.
# 3. **Decoder** $p_\theta(x|z)$: a network that maps $z$ back to a
#    reconstructed image $\hat x$.
# 4. **ELBO loss**:
#    $$
#    \mathcal{L} = \underbrace{\| x - \hat x \|^2}_{\text{reconstruction}}
#                  + \underbrace{D_{\text{KL}}(q_\phi(z|x) \,\|\, p(z))}_{\text{KL to }\mathcal{N}(0,I)}
#    $$
#    The KL has a closed form for diagonal Gaussians:
#    $D_{\text{KL}} = -\tfrac{1}{2} \sum (1 + \log \sigma^2 - \mu^2 - \sigma^2)$.
#
# *(see MFML §"VAE — encoder, decoder, ELBO", §"Reparameterisation trick")*

# %%
class TinyVAE(nn.Module):
    """64x64 grayscale -> latent_dim -> 64x64 grayscale.

    Same backbone as Week 9's conv-AE; the only changes are:
      - encoder produces 2 * latent_dim numbers (mu and log_var),
      - forward reparameterises z and returns (x_hat, mu, log_var, z).
    """

    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # 32x32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # 16x16
        )
        self.enc_lin = nn.Linear(32 * 16 * 16, 2 * latent_dim)
        self.dec_lin = nn.Linear(latent_dim, 32 * 16 * 16)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x).flatten(1)
        mu_logvar = self.enc_lin(h)
        return mu_logvar.chunk(2, dim=-1)                  # (mu, log_var)

    def reparameterise(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z):
        return self.dec_conv(self.dec_lin(z).view(-1, 32, 16, 16))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        return self.decode(z), mu, log_var, z


def vae_loss(x, x_hat, mu, log_var, beta=1.0):
    """Sum-reduced reconstruction + beta * KL.  Returns (total, recon, kl)."""
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
    # KL(N(mu, sigma^2) || N(0, I)) per-element, summed over latent dim.
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()
    return recon + beta * kl, recon, kl


# %%
# Train the VAE for 3 epochs at beta=1.  ~30 s on CPU, ~10 s on GPU.
def train_vae(beta, n_epochs=3, latent_dim=8, lr=1e-3, log=True):
    torch.manual_seed(0)
    model = TinyVAE(latent_dim=latent_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_tr), batch_size=64, shuffle=True)
    rec_curve, kl_curve = [], []
    for ep in range(n_epochs):
        model.train()
        recs, kls = [], []
        for (xb,) in loader:
            x_hat, mu, log_var, _ = model(xb)
            loss, rec, kl = vae_loss(xb, x_hat, mu, log_var, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
            recs.append(rec.item()); kls.append(kl.item())
        rec_curve.append(float(np.mean(recs)))
        kl_curve.append(float(np.mean(kls)))
        if log:
            print(f"  beta={beta:>3.1f}  epoch {ep}  recon = {rec_curve[-1]:.4f}  KL = {kl_curve[-1]:.4f}")
    return model, rec_curve, kl_curve


print("Training VAE at beta=1.0:")
vae, rec1, kl1 = train_vae(beta=1.0)


# %%
# Show 4 reconstructions and 4 samples from the prior.
vae.eval()
with torch.no_grad():
    x_4 = X_te[:4]
    x_hat_4, _, _, _ = vae(x_4)
    z_prior = torch.randn(4, vae.latent_dim, device=DEVICE)
    x_prior = vae.decode(z_prior)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i in range(4):
    axes[0, i].imshow(x_4[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"orig {i}", fontsize=9); axes[0, i].axis("off")
    axes[1, i].imshow(x_hat_4[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, i].set_title(f"recon {i}", fontsize=9); axes[1, i].axis("off")
plt.suptitle("VAE reconstructions (β=1)")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
for i in range(4):
    axes[i].imshow(x_prior[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[i].set_title(f"sample {i}", fontsize=9); axes[i].axis("off")
plt.suptitle("VAE samples from prior z ~ N(0, I)  (β=1)")
plt.tight_layout(); plt.show()


# %% [markdown]
# **What you should see.** Reconstructions of test images are recognisably
# CH-like (phase-domain texture, no stripes or salt-and-pepper noise);
# they are noticeably blurrier than the originals because the decoder is
# trying to satisfy *both* reconstruction and the KL pull toward
# $\mathcal{N}(0, I)$. Samples from the prior are valid CH-like
# microstructures — fuzzier than reconstructions because the prior has no
# information about which mode of the CH manifold to pick.

# %% [markdown]
# # Part B — ELBO decomposition + β-VAE sweep
#
# β-VAE adds a single knob: the KL term gets multiplied by $\beta$.
# Smaller $\beta$ → reconstruction wins, the latent is busy and not well
# regularised; larger $\beta$ → KL wins, the latent collapses toward the
# prior and reconstructions become bland.
#
# We sweep $\beta \in \{0.1, 1.0, 4.0\}$, plot the **reconstruction term**
# and **KL term** *separately* across training, and observe the trade-off.
#
# *(see MFML §"ELBO decomposition into reconstruction and KL", §"β-VAE")*

# %%
print("\nTraining β-VAE sweep (3 betas × 3 epochs each)...")
results = {}
for beta in [0.1, 1.0, 4.0]:
    _, rec, kl = train_vae(beta=beta, log=False)
    results[beta] = (rec, kl)
    print(f"  beta = {beta:>3.1f}   final recon = {rec[-1]:.4f}   final KL = {kl[-1]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
for beta, (rec, kl) in results.items():
    ax1.plot(rec, "o-", label=f"β={beta}")
    ax2.plot(kl,  "o-", label=f"β={beta}")
ax1.set_xlabel("epoch"); ax1.set_ylabel("reconstruction term"); ax1.set_title("Reconstruction (lower is better)"); ax1.legend()
ax2.set_xlabel("epoch"); ax2.set_ylabel("KL term"); ax2.set_title("KL to prior (lower = closer to N(0,I))"); ax2.legend()
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the curves.**
#
# - **β = 0.1**: reconstruction term plummets (the decoder fits the data),
#   but the KL term is large — the latent uses much wider distributions
#   than the prior allows. Sampling from $\mathcal{N}(0, I)$ at this β
#   typically misses the data manifold.
# - **β = 1.0**: the textbook setting — both terms reach an equilibrium.
#   Reconstructions are decent and prior samples are CH-like.
# - **β = 4.0**: the KL term is small (latent is close to the prior) but
#   reconstructions are blurry — the decoder has lost individual-sample
#   information. This is *posterior collapse*.

# %% [markdown]
# # Part C — Latent-space interpolation
#
# A useful VAE has a smooth latent: small steps in $z$ produce small
# changes in $\hat x$. We test this by picking two test images with very
# different free energies, encoding both, and walking linearly between
# their latents.
#
# *(see MFML §"Latent traversal as a smoothness diagnostic"; MG
# §"Latent-space arithmetic for property targeting")*

# %%
# Find a low-energy and a high-energy CH image in the test split.
y_te = ch.y[te_idx]
i_low = int(y_te.argmin().item())
i_high = int(y_te.argmax().item())
print(f"low-energy test idx = {i_low}, y = {y_te[i_low]:.3g}")
print(f"high-energy test idx = {i_high}, y = {y_te[i_high]:.3g}")

vae.eval()
with torch.no_grad():
    x_low  = X_te[i_low: i_low + 1]
    x_high = X_te[i_high: i_high + 1]
    mu_low,  _ = vae.encode(x_low)
    mu_high, _ = vae.encode(x_high)
    alphas = torch.linspace(0, 1, 5, device=DEVICE)
    interp = []
    for a in alphas:
        z = (1 - a) * mu_low + a * mu_high
        interp.append(vae.decode(z).cpu().numpy())
    grid = np.concatenate(interp, axis=0)

# Linearly interpolate the *energy* values too — to annotate the figure.
y_low  = float(y_te[i_low]); y_high = float(y_te[i_high])
y_alpha = [(1 - float(a)) * y_low + float(a) * y_high for a in alphas]

fig, axes = plt.subplots(1, 5, figsize=(11, 2.7))
for i, (a, y_a) in enumerate(zip(alphas, y_alpha)):
    axes[i].imshow(grid[i, 0], cmap="gray", vmin=0, vmax=1)
    axes[i].set_title(f"α = {float(a):.2f}\nE ≈ {y_a:.0f}", fontsize=9)
    axes[i].axis("off")
plt.suptitle("Latent-space interpolation: low-energy → high-energy")
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the interpolation grid.** Adjacent panels look like *gradual*
# transitions from one phase pattern to the other — that's the smoothness
# we wanted from the VAE prior. Notice: the linear interpolation in
# *energy* ($E$ on each panel) is just an annotation; the energies of the
# generated images are not actually the linearly-interpolated values.
# Verifying that "walking the latent at constant rate produces images
# with linearly-spaced energies" is the inverse-design question for
# Thursday Block 3 (latent-space gradient descent).

# %% [markdown]
# # Part D — Reflection (1 paragraph, ~5 sentences)
#
# Write a paragraph answering: **what does β actually control?**
#
# Specifically address:
#
# 1. What β = 0.1 does to the latent space (use your Part B curves).
# 2. What β = 4.0 does to reconstructions (use your Part B curves and
#    one observation from your Part C grid if you re-trained at β=4.0).
# 3. What "useful" β means for an inverse-design downstream task —
#    do you want a sharp decoder or a well-regularised latent?
#
# **Bridge to Thursday.** On Thursday you'll add a *condition* (free
# energy) to the encoder and decoder — that's a CVAE, the simplest form of
# property-targeted generation. Then you'll build a second method —
# *latent-space gradient descent* on a frozen VAE — and compare which
# performs better when the target energy is in vs out of the training
# distribution. Your Part D paragraph should already have an opinion
# about which method is more honest about its OOD behaviour.

# %% [markdown]
# ---
# You're done with Week 11 homework. Bring your four deliverables on Thursday:
#
# 1. Reconstruction grid + prior samples (Part A).
# 2. Reconstruction-vs-KL training curves at three β values (Part B).
# 3. Latent-interpolation grid with energy annotations (Part C).
# 4. β-VAE paragraph (Part D).
