# %% [markdown]
# # Week 11 — Generative models, embedding diagnostics & lab automation
#
# This week braids the three lectures that materials students actually
# sit on 23.06, around one spine: **a learned representation is only as
# good as what it lets you do downstream — generate, diagnose, automate.**
#
# 1. **MFML Unit 11**: Generative models — VAE, β-VAE, conditional VAE,
#    DDPM as historical anchor, **flow matching** as the 2026 default
#    [@lipman_2023_flow_matching], **consistency models** for one-step
#    sampling [@song_2023_consistency]. *(Blocks 1–4, 6; the spine.)*
# 2. **MG Unit 10** (Representation Learning & Feature Discovery, the true
#    calendar-W11 MG lecture): **embedding diagnostics** — linear probe vs
#    random-init baseline vs engineered (Magpie-style) features,
#    nearest-neighbour retrieval, and the deck's signature **"pretty
#    t-SNE / dead downstream"** anti-pattern. The W11 deck explicitly
#    *defers* generative latent arithmetic to MG U12, so Block 5 is now
#    diagnostics, not property-targeted generation. *(Block 5.)*
# 3. **ML-PC Unit 10** (Automation in microscopy & characterization, the
#    true calendar-W11 ML-PC lecture): an autonomous
#    **acquire → model → decide → acquire** loop — active-learning /
#    self-driving-lab style, with the deck's conformal automate/escalate
#    decision rule. *(Block 5b.)*
#
# **Red thread:** *Week 9 read the latent space; today we use it three
# ways.* MFML supplies the VAE/diffusion machinery to **generate**
# candidates. MG supplies the discipline to **diagnose** whether a learned
# embedding is actually doing work (probe before you trust). ML-PC closes
# the loop: an embedding good enough to retrieve in is good enough to
# **steer an autonomous experiment** — the self-driving-lab loop that
# decides what to measure next.
#
# > **Pre-flight check.** This notebook **assumes** you have run
# > `notebooks/week11_homework.py`. Block 1 picks up directly from your
# > Part B β-VAE curves and your Part C interpolation grid.
#
# ## Agenda (90 min)
#
# | Block | Min | Topic |
# |------:|:---:|:------|
# | 1  | ~5  | Recap from homework — VAE, ELBO, β trade-off |
# | 2  | ~12 | Conditional VAE on Cahn–Hilliard — generation under target free energy |
# | 3  | ~12 | Latent-space gradient descent — inverse design as differentiable optimization |
# | 4  | ~12 | Flow matching — train tiny ODE-velocity U-Net; 10-step Heun sampling |
# | 5  | ~14 | Embedding diagnostics — linear probe vs random-init vs Magpie, retrieval, the t-SNE trap (MG W11) |
# | 5b | ~12 | Self-driving-lab loop — autonomous acquire→model→decide, conformal escalate (ML-PC W11) |
# | 6  | ~8  | Honest limitations — mode collapse, posterior collapse, OOD generation |
# | 7  | ~15 | Student exercises (3 core) |

# %%
# Standard imports.
import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from ai4mat.datasets import CahnHilliardDataset, CrystalGraphsDataset

np.random.seed(0)
torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_GPU = (DEVICE.type == "cuda")
print(f"Using device: {DEVICE}  (GPU available: {HAS_GPU})")


# %% [markdown]
# ## Helpers — load data, copy VAE definition from homework

# %%
print("Loading Cahn-Hilliard (3 simulations, ~3000 samples)...")
ch = CahnHilliardDataset(simulation_number=[0, 1, 2])
y_mean = ch.y.mean().item(); y_std = ch.y.std().item()
print(f"  loaded {len(ch)} samples; energy mean={y_mean:.1f}  std={y_std:.1f}")

g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(ch), generator=g)
n_tr = 1500; n_te = 300
tr_idx, te_idx = perm[:n_tr], perm[n_tr:n_tr + n_te]
X_tr = ch.X[tr_idx].to(DEVICE)
X_te = ch.X[te_idx].to(DEVICE)
y_tr = ch.y[tr_idx].to(DEVICE)
y_te = ch.y[te_idx].to(DEVICE)
y_tr_n = (y_tr - y_mean) / y_std
y_te_n = (y_te - y_mean) / y_std


# %%
class TinyVAE(nn.Module):
    """Same backbone as homework Part A.  Unconditional."""

    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.enc_lin = nn.Linear(32 * 16 * 16, 2 * latent_dim)
        self.dec_lin = nn.Linear(latent_dim, 32 * 16 * 16)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x).flatten(1)
        return self.enc_lin(h).chunk(2, dim=-1)

    def reparameterise(self, mu, log_var):
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def decode(self, z):
        return self.dec_conv(self.dec_lin(z).view(-1, 32, 16, 16))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        return self.decode(z), mu, log_var, z


def vae_loss(x, x_hat, mu, log_var, beta=1.0):
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()
    return recon + beta * kl, recon, kl


# %% [markdown]
# # Block 1 — Recap from homework
#
# Three takeaways frame the rest of the lecture:
#
# 1. **The VAE compresses CH microstructure into 8 latent dimensions**
#    from which the decoder regenerates plausible images.
# 2. **β controls the trade-off**: small β → busy, expressive latent;
#    large β → near-prior latent at the cost of reconstruction quality.
# 3. **The latent space is smooth**: walking from $z_{\text{low E}}$ to
#    $z_{\text{high E}}$ produces a continuous sequence of plausible
#    microstructures, not a sequence of nearest-neighbour copies.
#
# Today we *use* that smoothness for inverse design: given a target free
# energy, can we find a latent that decodes to a microstructure with that
# energy? Two methods:
#
# - **Conditional VAE** (Block 2): bake the target into the model.
# - **Latent-space gradient descent** (Block 3): freeze the model, solve
#   for $z$ at inference time.

# %%
# Quick warm-up: train an unconditional VAE that we will reuse in Blocks
# 3 and 6 (for the latent-GD baseline and the failure-mode demos).
print("Pretraining unconditional VAE for downstream blocks...")
torch.manual_seed(0)
vae = TinyVAE(latent_dim=8).to(DEVICE)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(X_tr), batch_size=64, shuffle=True)
for ep in range(4):
    vae.train()
    losses = []
    for (xb,) in loader:
        x_hat, mu, log_var, _ = vae(xb)
        loss, _, _ = vae_loss(xb, x_hat, mu, log_var, beta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"  epoch {ep}  total ELBO loss = {np.mean(losses):.4f}")


# %% [markdown]
# # Block 2 — Conditional VAE on Cahn-Hilliard
#
# A **conditional VAE** is the cleanest form of property-targeted
# generation: append the target $y$ to both the encoder input and the
# decoder input. Concretely:
#
# - encoder: $q_\phi(z \mid x, y)$
# - decoder: $p_\theta(x \mid z, y)$
#
# At inference, sample $z \sim \mathcal{N}(0, I)$, pick any $y^*$, decode
# $\hat x = \text{dec}(z, y^*)$. This is the simplest "give me a sample
# with energy = X" pipeline.
#
# *(see ML-PC §"VAE-based inverse design", §"Conditional generation
# under target property"; MFML §"Conditional VAE")*

# %%
class TinyCVAE(nn.Module):
    """Conditional VAE. The condition y is broadcast as a constant feature
    map and concatenated with the input image (encoder side); on the
    decoder side it is concatenated with z."""

    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.ReLU(),    # 2 channels: image + y-broadcast
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.enc_lin = nn.Linear(32 * 16 * 16, 2 * latent_dim)
        self.dec_lin = nn.Linear(latent_dim + 1, 32 * 16 * 16)      # +1 for y
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def encode(self, x, y):
        # broadcast y as a 64x64 feature plane
        y_plane = y.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        h = self.enc_conv(torch.cat([x, y_plane], dim=1)).flatten(1)
        return self.enc_lin(h).chunk(2, dim=-1)

    def decode(self, z, y):
        zy = torch.cat([z, y.view(-1, 1)], dim=1)
        return self.dec_conv(self.dec_lin(zy).view(-1, 32, 16, 16))

    def forward(self, x, y):
        mu, log_var = self.encode(x, y)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return self.decode(z, y), mu, log_var, z


# %%
print("Training CVAE on (image, energy) pairs...")
torch.manual_seed(0)
cvae = TinyCVAE(latent_dim=8).to(DEVICE)
opt = torch.optim.Adam(cvae.parameters(), lr=1e-3)
ds_cvae = TensorDataset(X_tr, y_tr_n)
loader_cvae = DataLoader(ds_cvae, batch_size=64, shuffle=True)
for ep in range(4):
    cvae.train()
    losses = []
    for xb, yb in loader_cvae:
        x_hat, mu, log_var, _ = cvae(xb, yb)
        loss, _, _ = vae_loss(xb, x_hat, mu, log_var, beta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"  epoch {ep}  total ELBO loss = {np.mean(losses):.4f}")


# %%
# Sample from the CVAE prior at three target energies: low, median, high.
y_low_n = float(y_te_n.min().item())
y_med_n = 0.0
y_high_n = float(y_te_n.max().item())
targets_n = [y_low_n, y_med_n, y_high_n]
target_labels = ["low E", "median E", "high E"]
print(f"Target energies (de-normalised): "
      f"low = {y_low_n*y_std + y_mean:.0f}, "
      f"median = {y_mean:.0f}, "
      f"high = {y_high_n*y_std + y_mean:.0f}")

cvae.eval()
n_samples_per_y = 4
with torch.no_grad():
    fig, axes = plt.subplots(3, n_samples_per_y, figsize=(10, 7))
    for r, (yt, lbl) in enumerate(zip(targets_n, target_labels)):
        z = torch.randn(n_samples_per_y, cvae.latent_dim, device=DEVICE)
        y_t = torch.full((n_samples_per_y,), yt, device=DEVICE)
        samples = cvae.decode(z, y_t)
        for c in range(n_samples_per_y):
            axes[r, c].imshow(samples[c, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(lbl, rotation=0, ha="right",
                                      fontsize=11, va="center")
plt.suptitle("CVAE samples conditioned on target free energy")
plt.tight_layout(); plt.show()


# %% [markdown]
# **What you should see.** Across the three rows, the microstructures
# differ visibly: at low target energy, the patterns are well-separated
# phase domains; at high target energy, the patterns are noisier with
# more interface area (CH thermodynamics: higher energy = more interface).
# Within a row, the four samples are *different but similar* — the VAE
# generates *diverse* candidates that all share the target property.
#
# This is the cleanest form of inverse design: one model, two minutes of
# training, and a control knob that produces property-targeted samples
# on demand.

# %% [markdown]
# # Block 3 — Latent-space gradient descent (inverse design as optimization)
#
# An alternative recipe: **don't** retrain the VAE — instead, use the
# *frozen* unconditional VAE plus a *frozen* property regressor, and
# solve for the latent that gives the target property by gradient
# descent.
#
# $$
# z^* = \arg\min_z \; \big( f(\text{dec}(z)) - y^* \big)^2
# $$
#
# The chain `z → decode → regressor → property` is fully differentiable,
# so this is a 5-line training loop. The advantage: works on *any*
# pretrained generator + regressor, no joint retraining required. The
# disadvantage: each target $y^*$ requires its own optimisation (the
# CVAE amortises this).
#
# *(see ML-PC §"Latent optimization for inverse design"; MFML
# §"Differentiable generation")*

# %%
# Train a small CNN regressor on (image, energy) for use in Block 3.
class EnergyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(32 * 16, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


print("Training energy regressor on CH images...")
torch.manual_seed(0)
reg = EnergyRegressor().to(DEVICE)
opt = torch.optim.Adam(reg.parameters(), lr=1e-3)
loader_reg = DataLoader(TensorDataset(X_tr, y_tr_n), batch_size=64, shuffle=True)
for ep in range(4):
    reg.train()
    losses = []
    for xb, yb in loader_reg:
        opt.zero_grad()
        loss = F.mse_loss(reg(xb), yb)
        loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"  epoch {ep}  train MSE = {np.mean(losses):.4f}")
with torch.no_grad():
    test_mae = (reg(X_te) - y_te_n).abs().mean().item() * y_std
print(f"  test MAE (de-normalised): {test_mae:.1f} energy units")


# %%
def latent_gd(target_y_n, n_steps=300, lr=0.05):
    """Gradient-descend in z for a target normalised energy.

    Frozen unconditional VAE decoder + frozen regressor; only z requires grad.
    """
    z = torch.randn(1, vae.latent_dim, device=DEVICE, requires_grad=True)
    opt_z = torch.optim.Adam([z], lr=lr)
    target = torch.tensor([target_y_n], device=DEVICE)
    losses = []
    for _ in range(n_steps):
        opt_z.zero_grad()
        x_hat = vae.decode(z)
        y_pred = reg(x_hat)
        loss = (y_pred - target).pow(2).mean()
        loss.backward(); opt_z.step()
        losses.append(loss.item())
    return z.detach(), losses


# Run latent-GD for the three target energies and visualise.
fig, axes = plt.subplots(2, 3, figsize=(9, 5.5))
for r, (yt, lbl) in enumerate(zip(targets_n, target_labels)):
    z_star, loss_curve = latent_gd(yt)
    with torch.no_grad():
        x_star = vae.decode(z_star)
        y_pred = reg(x_star).item()
    axes[0, r].plot(loss_curve, lw=1.0)
    axes[0, r].set_yscale("log")
    axes[0, r].set_title(f"{lbl}: target = {yt*y_std+y_mean:.0f}, achieved = {y_pred*y_std+y_mean:.0f}")
    axes[0, r].set_xlabel("step"); axes[0, r].set_ylabel("(y_pred - y_target)^2")
    axes[1, r].imshow(x_star[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, r].axis("off"); axes[1, r].set_title(f"latent-GD candidate")
plt.tight_layout(); plt.show()


# %% [markdown]
# **CVAE vs latent-GD — which wins?** Both produce candidates with the
# target energy. The CVAE produces *diverse* candidates per call (different
# z, same y); latent-GD produces a *deterministic* candidate per random
# init. Latent-GD requires no retraining — useful when a new target is
# requested at inference time. CVAE amortises the optimisation — useful
# when many targets need many samples.
#
# Real-world materials inverse design uses both: a CVAE for proposal
# generation, latent-GD for refinement.

# %% [markdown]
# # Block 4 — Flow matching for inverse design
#
# **Pedagogical anchor — DDPM then flow matching.** The other major
# generative family used to be **DDPM** (denoising diffusion probabilistic
# models): an *SDE-based* generator with a noisy forward process
#
# $$
# q(x_t \mid x_0) = \mathcal{N}\!\big(\sqrt{\bar\alpha_t}\, x_0,\, (1 - \bar\alpha_t) I\big)
# $$
#
# and a learned, *stochastic*, 1000-step reverse process that predicts the
# noise $\varepsilon_\theta(x_t, t)$. DDPM is what got the field excited in
# 2020; we keep the equation above as a historical anchor and move on.
#
# In 2026 the **default new image generator is flow matching**
# [@lipman_2023_flow_matching]: an *ODE-based* generator with a *simpler
# loss* and *fewer sampling steps*. Same U-Net backbone, different
# training target. The recipe is:
#
# - **Forward path** (no learning): sample $x_0 \sim \mathcal{N}(0, I)$
#   and $x_1$ from data. Sample $t \sim \mathrm{Uniform}(0, 1)$ and form
#   the linear interpolant $x_t = (1 - t) x_0 + t x_1$.
# - **Target velocity:** $u^*(x_t, t) = x_1 - x_0$.
# - **Loss:** $\mathcal{L} = \| u_\theta(x_t, t) - u^* \|^2$.
# - **Sample:** start from $x \leftarrow x_0 \sim \mathcal{N}(0, I)$ and
#   integrate the learned ODE $\dot x = u_\theta(x, t)$ from $t = 0$ to
#   $t = 1$ with a small number of solver steps.
#
# DDPM and flow matching share the U-Net; flow matching replaces "predict
# noise on a noisy schedule" with "predict the straight-line velocity",
# which trains faster and samples in **10 ODE steps** instead of hundreds.
#
# *(see MFML §"Flow matching"; ML-PC §"Flow-matching microstructure
# inverse design")*

# %%
# Visualise the linear interpolant x_t = (1-t) x_0 + t x_1 used by flow
# matching.  No training; just forward path inspection.
x1_demo = X_tr[0:1]                                        # one data sample
x0_demo = torch.randn_like(x1_demo)                        # one Gaussian sample
t_show = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
fig, axes = plt.subplots(1, 6, figsize=(13, 2.5))
for i, t_i in enumerate(t_show):
    x_t_demo = (1 - t_i) * x0_demo + t_i * x1_demo
    axes[i].imshow(x_t_demo[0, 0].cpu().numpy(), cmap="gray")
    axes[i].set_title(f"t = {t_i:.1f}", fontsize=9); axes[i].axis("off")
plt.suptitle("Flow-matching interpolant x_t = (1-t) x_0 + t x_1  (no training)")
plt.tight_layout(); plt.show()


# %%
class TinyUNet(nn.Module):
    """Minimal U-Net for the flow-matching velocity field.  64x64 in/out;
    sinusoidal time embedding; channel mults [16, 32, 64].

    Same architecture we used for DDPM in the 2025 edition of this
    notebook — flow matching only changes the *loss* and *sampler*, not
    the backbone."""

    def __init__(self, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 128), nn.SiLU(), nn.Linear(128, 128),
        )
        self.in_conv = nn.Conv2d(1, 16, 3, padding=1)
        self.down1 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU())
        self.down2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU())
        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
        )
        self.up2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(16, 1, 3, padding=1)
        # Project time embedding into each scale.
        self.time_proj_mid = nn.Linear(128, 64)
        self.time_proj_up2 = nn.Linear(128, 32)
        self.time_proj_up1 = nn.Linear(128, 16)

    def time_embedding(self, t):
        """t is a continuous tensor in [0, 1] here (flow matching), not
        an integer step index as it was in DDPM.  We rescale to keep the
        sinusoidal frequencies in a useful range."""
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = (t.float() * 1000.0)[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embedding(t))      # (B, 128)
        h0 = self.in_conv(x)                               # (B, 16, 64, 64)
        h1 = self.down1(h0)                                # (B, 32, 32, 32)
        h2 = self.down2(h1)                                # (B, 64, 16, 16)
        h_mid = self.mid(h2 + self.time_proj_mid(t_emb)[:, :, None, None])
        u2 = self.up2(h_mid)
        u2 = u2 + h1                                       # skip
        u1 = self.up1(u2 + self.time_proj_up2(t_emb)[:, :, None, None])
        u1 = u1 + h0
        return self.out_conv(u1 + self.time_proj_up1(t_emb)[:, :, None, None])


# %%
if HAS_GPU:
    print("GPU detected — training tiny flow-matching U-Net (~1-2 min on 1080Ti)")
    torch.manual_seed(0)
    unet = TinyUNet().to(DEVICE)
    opt = torch.optim.Adam(unet.parameters(), lr=2e-4)
    loader_fm = DataLoader(TensorDataset(X_tr), batch_size=32, shuffle=True)
    n_epochs_fm = 20
    for ep in range(n_epochs_fm):
        unet.train()
        losses = []
        for (xb,) in loader_fm:
            # Flow-matching loss: predict the straight-line velocity.
            x1 = xb
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=DEVICE)
            x_t = (1 - t).view(-1, 1, 1, 1) * x0 + t.view(-1, 1, 1, 1) * x1
            u_star = x1 - x0                               # target velocity
            u_pred = unet(x_t, t)
            loss = F.mse_loss(u_pred, u_star)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 4 == 0 or ep == n_epochs_fm - 1:
            print(f"  epoch {ep:2d}  velocity-matching MSE = {np.mean(losses):.4f}")

    # Sample with a 10-step Heun ODE solver (predictor + corrector).
    @torch.no_grad()
    def fm_sample_heun(unet, n=6, n_steps=10):
        """Heun's method for x' = u_theta(x, t), t: 0 -> 1.

        Each step does a predictor (Euler) and a corrector (average of
        velocities at both ends).  10 steps is plenty for this small
        model — that's the flow-matching efficiency story."""
        x = torch.randn(n, 1, 64, 64, device=DEVICE)
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=DEVICE)
        for k in range(n_steps):
            t_k = ts[k]; t_k1 = ts[k + 1]
            dt = t_k1 - t_k
            t_vec = torch.full((n,), float(t_k), device=DEVICE)
            u_k = unet(x, t_vec)                           # predictor velocity
            x_pred = x + dt * u_k                          # Euler step
            t_vec1 = torch.full((n,), float(t_k1), device=DEVICE)
            u_k1 = unet(x_pred, t_vec1)                    # corrector velocity
            x = x + 0.5 * dt * (u_k + u_k1)                # Heun average
        return x.clamp(0, 1)

    samples = fm_sample_heun(unet, n=6, n_steps=10)
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.5))
    for i in range(6):
        axes[i].imshow(samples[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off"); axes[i].set_title(f"FM sample {i}", fontsize=9)
    plt.suptitle("Samples from the trained flow-matching ODE (10 Heun steps)")
    plt.tight_layout(); plt.show()
else:
    print("No GPU detected — skipping flow-matching training.")
    print("The interpolant visualisation above is the qualitative point.")
    print("On a GPU machine, the next cell would train a tiny U-Net flow-matching")
    print("model in ~1-2 min and sample in 10 ODE steps.")
    unet = None                                            # used by Exercise 5


# %% [markdown]
# **Take-home from Block 4.** Flow matching reaches similar visual quality
# to a tiny DDPM with **~10x fewer sampling steps** and a simpler loss
# (MSE on a velocity, no noise schedule, no $\bar\alpha_t$ bookkeeping).
# That's the 2026 default for new image generators.
#
# **Flow matching vs VAE — when to pick which?**
#
# - Flow matching produces sharper samples *with enough training* — no
#   posterior collapse, no blurry-mean-of-modes artefact.
# - Flow matching is slower at sampling than a VAE (10 ODE steps vs 1
#   decode), but much faster than DDPM (which used hundreds of steps).
# - VAE sampling is one decode call — fastest — but blurrier.
# - **Conditional generation in a CVAE is one extra input** (Block 2);
#   conditional generation in flow matching uses classifier-free guidance
#   on the velocity field — the same idea as guided diffusion, applied
#   to the ODE.
#
# **Where DDPM lives now.** As the SDE-based ancestor: the 2 lines of math
# at the top of this block are the only DDPM you need today. Everything
# downstream — score matching, classifier-free guidance, consistency
# distillation — generalises naturally to the flow-matching ODE.

# %% [markdown]
# # Block 5 — Embedding diagnostics (MG W11)
#
# We now switch from CH images to crystal embeddings — the MG home turf.
# **But the true MG-W11 lecture is *not* latent-space arithmetic for
# property targeting** (that is explicitly deferred to MG U12, "Generative
# Models & Inverse Design"). MG-W11 is *Representation Learning & Feature
# Discovery*, and its conceptual centre is a discipline, not a generator:
#
# > *Probe before you project.* A learned embedding is only useful to the
# > extent it contains information about the property you care about — and
# > a pretty 2-D projection does **not** establish that.
#
# We exercise the deck's diagnostic stack on a small TinyCGNN trained on
# toy formation energies:
#
# 1. **Linear probe vs random-init baseline vs engineered features**
#    (MG §F slide 42): freeze the encoder, fit a linear head on a
#    *held-out prototype*, and compare against (i) a *random-init*
#    encoder of the same architecture and (ii) a hand-engineered
#    composition baseline (a Magpie-style stand-in). Without the
#    random-init row you cannot tell whether *training* helped or whether
#    the *architecture* did the work — "the most-omitted comparison in
#    published work".
# 2. **Nearest-neighbour retrieval** (MG §F slide 43): the diagnostic the
#    deck trusts most — full-dimension, per-query, manually inspectable.
# 3. **The "pretty t-SNE / dead downstream" trap** (MG §F slide 45):
#    construct it on purpose — a PCA scatter that *looks* structured while
#    a probe of a *metadata artefact* (prototype id / atom count) scores
#    high and the *property* probe is comparatively weak.
#
# *(see MG §F "Diagnosing Learned Representations", slides 41–46;
# §"Linear Probe Protocol"; §"Nearest-Neighbour Retrieval Check")*

# %%
# Train a fresh TinyCGNN (same backbone as Week 6/9): supervised on toy
# formation energy.  We keep its `encode()` (frozen embedding) and `head`
# for the probes below.
class TinyCGNN(nn.Module):
    def __init__(self, n_elements=120, embed_dim=16, n_layers=3):
        super().__init__()
        self.embed = nn.Embedding(n_elements, embed_dim)
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * embed_dim + 1, embed_dim), nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            ) for _ in range(n_layers)
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
        return h.mean(0)

    def forward(self, species, edge_index, edge_distance):
        return self.head(self.encode(species, edge_index, edge_distance)).squeeze(-1)


cg = CrystalGraphsDataset()
y_cg = cg.y; y_cg_mean = y_cg.mean().item(); y_cg_std = y_cg.std().item()
y_true = cg.y                                              # 200 floats (property)
prototype = cg.prototype                                   # 200 int (metadata)


def cgnn_embeddings(model):
    """Mean-pooled 16-D embedding for every crystal in `cg`."""
    model.eval()
    with torch.no_grad():
        return torch.stack([
            model.encode(cg[i]["species"], cg[i]["edge_index"],
                         cg[i]["edge_distance"])
            for i in range(len(cg))
        ])                                                 # (200, 16)


# Engineered "Magpie-style" composition baseline: per-crystal summary
# statistics of atomic number (mean, std, min, max, n_atoms) — the kind
# of cheap hand-crafted descriptor MG U6 builds and §F slide 42 makes the
# probe compare against.
def magpie_style_features():
    feats = []
    for i in range(len(cg)):
        z = cg[i]["species"].float()
        feats.append(torch.tensor([
            z.mean(), z.std(unbiased=False), z.min(), z.max(),
            float(z.numel()),
        ]))
    return torch.stack(feats)                              # (200, 5)


# Train the supervised CGNN (this is our "pretrained" encoder).
torch.manual_seed(0)
cgnn = TinyCGNN()
opt_g = torch.optim.Adam(cgnn.parameters(), lr=5e-3)
print("Training TinyCGNN encoder (5 epochs, supervised on formation energy)...")
for ep in range(5):
    cgnn.train()
    losses = []
    for i in torch.randperm(len(cg)).tolist():
        s = cg[i]
        yn = (s["y"] - y_cg_mean) / y_cg_std
        opt_g.zero_grad()
        p = cgnn(s["species"], s["edge_index"], s["edge_distance"])
        loss = (p - yn) ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cgnn.parameters(), 1.0)
        opt_g.step()
        losses.append(loss.item())
    if ep % 2 == 0 or ep == 4:
        print(f"  epoch {ep}  train MSE = {np.mean(losses):.4f}")

# A *random-init* encoder of the exact same architecture: never trained.
# This is the MG §F slide-42 baseline that isolates "did pretraining help?"
torch.manual_seed(1)
cgnn_rand = TinyCGNN()

emb_trained = cgnn_embeddings(cgnn)                        # (200, 16)
emb_random = cgnn_embeddings(cgnn_rand)                    # (200, 16)
feat_magpie = magpie_style_features()                      # (200, 5)


# %% [markdown]
# ## 5.1 — Linear probe on a held-out prototype
#
# **Protocol (MG §F slide 42).** Freeze each representation. Hold out one
# *entire prototype* (here `perovskite`) — this is the deck's
# *held-out-chemistry* requirement: a probe evaluated on the same
# distribution it was fit on is a memorisation test, not a transfer test.
# Fit a closed-form ridge-regression linear head on the other 4
# prototypes, evaluate $R^2$ / MAE on the held-out one. Compare four
# rows: trained encoder, random-init encoder, Magpie-style features, and
# a raw-mean-atomic-number scalar (a deliberately weak floor).

# %%
def linear_probe(Z, y, train_mask, test_mask, ridge=1.0):
    """Closed-form ridge linear probe.  Returns (R2, MAE) on the test set.

    Standardise features on the train split, append a bias, solve the
    normal equations with an L2 penalty (no torch grad needed)."""
    Z = Z.float()
    mu = Z[train_mask].mean(0, keepdim=True)
    sd = Z[train_mask].std(0, keepdim=True).clamp_min(1e-6)
    Zs = (Z - mu) / sd
    Ztr = torch.cat([Zs[train_mask], torch.ones(train_mask.sum(), 1)], dim=1)
    Zte = torch.cat([Zs[test_mask], torch.ones(test_mask.sum(), 1)], dim=1)
    ytr = y[train_mask].unsqueeze(1)
    d = Ztr.shape[1]
    reg = ridge * torch.eye(d); reg[-1, -1] = 0.0          # don't penalise bias
    w = torch.linalg.solve(Ztr.T @ Ztr + reg, Ztr.T @ ytr)
    yhat = (Zte @ w).squeeze(1)
    yte = y[test_mask]
    ss_res = ((yte - yhat) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum().clamp_min(1e-12)
    r2 = (1 - ss_res / ss_tot).item()
    mae = (yte - yhat).abs().mean().item()
    return r2, mae


held_out_proto = cg.prototype_names.index("perovskite")
test_mask = (prototype == held_out_proto)
train_mask = ~test_mask
print(f"Held-out prototype: 'perovskite'  "
      f"({int(test_mask.sum())} test / {int(train_mask.sum())} train crystals)")

# A deliberately weak floor: a single per-crystal scalar (mean atomic
# number).  Anything that cannot beat this is not a representation.
floor_feat = torch.stack([cg[i]["species"].float().mean()
                          for i in range(len(cg))]).unsqueeze(1)  # (200,1)

probe_rows = [
    ("trained CGNN encoder", emb_trained),
    ("random-init CGNN encoder", emb_random),
    ("Magpie-style features", feat_magpie),
    ("mean-Z scalar (weak floor)", floor_feat),
]
print(f"\n{'representation':<30}  {'R2':>7}  {'MAE (eV/atom)':>14}")
print("-" * 56)
probe_results = {}
for name, Z in probe_rows:
    r2, mae = linear_probe(Z, y_true, train_mask, test_mask)
    probe_results[name] = (r2, mae)
    print(f"{name:<30}  {r2:7.3f}  {mae:14.3f}")


# %% [markdown]
# **Reading the probe table.** The row that matters most is **random-init
# CGNN encoder**. If it scores close to the trained encoder, the
# *architecture* (graph message passing + mean pooling) is doing the work
# and the supervised training added little — exactly the "most-omitted
# comparison" the MG deck (slide 42) insists on. The Magpie-style row is
# the engineered baseline any learned embedding must *beat to justify its
# cost* (MG §G slide 47). On this toy dataset the formation energy is
# largely an electronegativity/radius-mismatch function of composition, so
# expect the cheap composition features to be a *strong* baseline — the
# deck's "always use the foundation model is wrong" point, measured.

# %% [markdown]
# ## 5.2 — Nearest-neighbour retrieval (the honest diagnostic)
#
# **Protocol (MG §F slide 43).** For each query crystal, retrieve its
# $k$ nearest neighbours **in full embedding dimension** (no 2-D
# projection) and ask: do the neighbours share the query's prototype, and
# are their formation energies clustered near the query's? We report
# `precision@k` for prototype and the mean absolute energy gap to the
# query — the per-query, manually-inspectable diagnostic the deck trusts
# more than any t-SNE.

# %%
def retrieval_metrics(Z, k=5):
    """Mean prototype precision@k and mean |Δenergy| to query, full-dim."""
    Z = F.normalize(Z.float(), dim=1)
    sim = Z @ Z.T
    sim.fill_diagonal_(-2.0)                                # exclude self
    nn_idx = sim.topk(k, dim=1).indices                     # (N, k)
    proto_hit = (prototype[nn_idx] == prototype[:, None]).float().mean().item()
    e_gap = (y_true[nn_idx] - y_true[:, None]).abs().mean().item()
    return proto_hit, e_gap


print(f"{'representation':<30}  {'proto P@5':>10}  {'mean |ΔE| (eV/atom)':>20}")
print("-" * 64)
for name, Z in [("trained CGNN encoder", emb_trained),
                ("random-init CGNN encoder", emb_random),
                ("Magpie-style features", feat_magpie)]:
    p_at_k, e_gap = retrieval_metrics(Z, k=5)
    print(f"{name:<30}  {p_at_k:10.3f}  {e_gap:20.3f}")

# A worked example: 1 query, its 5 nearest neighbours in the trained space.
Zn = F.normalize(emb_trained, dim=1)
q = 0
sims = Zn @ Zn[q]; sims[q] = -2.0
nn5 = sims.topk(5).indices
print(f"\nQuery crystal {q}: prototype="
      f"{cg.prototype_names[int(prototype[q])]}, E={y_true[q]:+.2f} eV/atom")
for j in nn5.tolist():
    print(f"  neighbour {j:3d}: "
          f"prototype={cg.prototype_names[int(prototype[j])]:<10}  "
          f"E={y_true[j]:+.2f}  (ΔE={abs(y_true[j]-y_true[q]):.2f})")


# %% [markdown]
# ## 5.3 — The "pretty t-SNE / dead downstream" trap
#
# **The anti-pattern (MG §F slide 45).** A 2-D projection can look
# beautifully clustered while the embedding is *useless for the property*
# — because the projection latches onto a high-variance **metadata
# artefact**. The deck's canonical concrete example: "the embedding had
# learned to *count atoms* — the cluster picture was by number of atoms in
# the cell, and the property probe was at chance." We reproduce exactly
# that: the artefact is **atom count per cell** (`n_atoms`), a real but
# physically-irrelevant-to-stability quantity.
#
# We PCA-scatter the trained embedding (it *looks* structured), then put
# two probes **on the same random split** side by side — a probe of
# `n_atoms` (the metadata artefact) vs a probe of **formation energy**
# (the property we actually want). The lesson lands when the artefact
# probe scores high while the property probe is comparatively weak: the
# projection organised the embedding by the artefact, not the property.

# %%
def pca_2d(X):
    X = X.float()
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    V = eigvecs[:, order][:, :2]
    return Xc @ V


Z2 = pca_2d(emb_trained)                                    # (200, 2)

# The metadata artefact: number of atoms in the cell — real, but
# physically irrelevant to formation energy.  Probe it vs the property on
# the SAME random 75/25 split (both targets are in-distribution here; the
# point is "same embedding, same split, artefact wins").
n_atoms = torch.tensor([float(cg[i]["species"].numel())
                        for i in range(len(cg))])
g_split = torch.Generator().manual_seed(0)
perm_e = torch.randperm(len(cg), generator=g_split)
n_tr_e = int(0.75 * len(cg))
rand_train = torch.zeros(len(cg), dtype=torch.bool)
rand_train[perm_e[:n_tr_e]] = True
rand_test = ~rand_train

r2_artefact, _ = linear_probe(emb_trained, n_atoms, rand_train, rand_test)
r2_property, _ = linear_probe(emb_trained, y_true, rand_train, rand_test)
print(f"trained-embedding probe (random split):  "
      f"n_atoms artefact R2 = {r2_artefact:.3f}   "
      f"formation-energy R2 = {r2_property:.3f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
sc = a1.scatter(Z2[:, 0], Z2[:, 1], s=20, alpha=0.7,
                c=n_atoms.numpy(), cmap="viridis")
a1.set_xlabel("embed PC1"); a1.set_ylabel("embed PC2")
a1.set_title("PCA of the embedding — looks structured\n(coloured by atom count)")
fig.colorbar(sc, ax=a1, label="n_atoms")

a2.bar(["metadata\n(n_atoms)", "property\n(formation E)"],
       [r2_artefact, r2_property], color=["C3", "C0"])
a2.axhline(0.0, c="grey", lw=0.8)
a2.set_ylabel("held-out probe $R^2$")
a2.set_title("Probe the projection, don't trust it")
for i, v in enumerate([r2_artefact, r2_property]):
    a2.text(i, v + 0.02 * (1 if v >= 0 else -1), f"{v:.2f}",
            ha="center", fontsize=10)
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the trap.** The left scatter *looks* like the embedding "knows
# something" — and the colour reveals what: the structure tracks
# **atom count**, not stability. The right panel is the honest verdict:
# the metadata-artefact probe (`n_atoms`) scores high while the property
# probe (formation energy) is comparatively weak — the projection
# organised the embedding by *how big the cell is*, not by *how stable*
# the crystal is. A downstream inverse-design or discovery pipeline built
# on a "pretty t-SNE" alone would inherit exactly this blind spot.
# **Probe before you project** is the single transferable discipline of
# MG-W11. (If on this toy dataset the property probe is *also* strong,
# that is the §F46 "good downstream, bad t-SNE" mirror image — still the
# same lesson: trust the probe, not the picture.)

# %% [markdown]
# # Block 5b — The self-driving-lab loop (ML-PC W11)
#
# The true calendar-W11 ML-PC lecture is **Unit 10 — Automation in
# microscopy & characterization**, not inverse problems. Its spine is the
# *self-driving lab*: an agent that **defines an objective** ("find the
# most stable composition") instead of issuing commands, and runs an
# autonomous
#
# > **acquire → model → decide → acquire** loop
#
# until the objective is met or the budget is spent. The deck frames this
# as RL / active experimentation with a reward signal, plus a discipline
# for *when to hand back to a human* (conformal "emit a set, not a
# label").
#
# We make this concrete and cheap by reusing the **embedding from Block
# 5** as the lab's state representation — closing the deck's own forward
# link ("retrieval ... generalises directly to the discovery loop", MG
# §F slide 43; ML-PC §"Self-Driving Lab Framework"). The "instrument" is
# the toy formation-energy oracle `cg.y`; "measuring" a crystal is
# expensive, so the agent may only query a small budget. Active-learning
# loop:
#
# 1. **Model.** Fit a cheap linear surrogate on all crystals measured so
#    far (state = frozen CGNN embedding).
# 2. **Decide.** Score every *unmeasured* crystal by an acquisition
#    function (expected improvement-style: predicted stability minus an
#    uncertainty-aware exploration bonus from k-NN embedding distance).
# 3. **Acquire.** "Measure" the top candidate (reveal its true energy),
#    add it to the labelled pool, loop.
# 4. **Escalate.** A conformal-style calibrated band decides
#    automate-vs-escalate: a *wide* prediction band → the surrogate is
#    unsure → flag for the (simulated) human operator instead of
#    auto-accepting.
#
# *(see ML-PC §"The Self-Driving Lab Framework", §"Reinforcement Learning
# Foundations" (state/action/reward), §"Conformal Classification — emit
# prediction sets, not single labels")*

# %%
# State = frozen Block-5 trained embedding.  Goal: find the most stable
# (lowest formation-energy) crystal under a tight measurement budget,
# without measuring all 200.
emb_state = F.normalize(emb_trained, dim=1)                 # (200, 16) frozen
energy_oracle = y_true                                       # "instrument": expensive
N = emb_state.shape[0]

rng = np.random.default_rng(0)
budget = 30
n_seed = 5

measured = list(rng.choice(N, size=n_seed, replace=False))
measured = [int(i) for i in measured]
best_energy_trace = []
escalations = 0

for step in range(budget - n_seed):
    idx_m = torch.tensor(measured)
    Zm = emb_state[idx_m]
    ym = energy_oracle[idx_m]

    # --- Model: closed-form ridge surrogate on measured crystals ---
    A = torch.cat([Zm, torch.ones(len(measured), 1)], dim=1)
    d = A.shape[1]
    ridge_mat = 1.0 * torch.eye(d); ridge_mat[-1, -1] = 0.0  # don't shadow the CNN regressor `reg` (used again in Block 6)
    w = torch.linalg.solve(A.T @ A + ridge_mat, A.T @ ym.unsqueeze(1))
    resid = (A @ w).squeeze(1) - ym
    sigma = resid.std(unbiased=False).clamp_min(1e-3)        # surrogate noise

    # --- Decide: acquisition over unmeasured crystals ---
    unmeasured = [i for i in range(N) if i not in measured]
    Zu = emb_state[unmeasured]
    pred = (torch.cat([Zu, torch.ones(len(unmeasured), 1)], dim=1) @ w).squeeze(1)
    # exploration bonus: distance to the nearest measured crystal in
    # embedding space (far-from-known => uncertain => worth probing)
    nn_dist = torch.cdist(Zu, Zm).min(dim=1).values
    # we MINIMISE energy, so acquisition = -pred + kappa * novelty
    kappa = 1.5
    acq = -pred + kappa * nn_dist
    pick_local = int(acq.argmax().item())
    pick = unmeasured[pick_local]

    # --- Escalate: conformal-style calibrated band on the surrogate ---
    # band half-width from the measured-residual quantile (alpha=0.1)
    q = torch.quantile(resid.abs(), 0.90)
    band = float(q.item())
    if band > 1.5 * float(sigma.item()):
        escalations += 1                                     # "send to operator"

    # --- Acquire: reveal the true energy, add to pool ---
    measured.append(pick)
    best_energy_trace.append(float(energy_oracle[torch.tensor(measured)].min().item()))

best_idx = int(energy_oracle[torch.tensor(measured)].argmin().item())
best_crystal = measured[int(np.argmin([energy_oracle[m].item() for m in measured]))]
global_best = float(energy_oracle.min().item())
found_best = best_energy_trace[-1]
print(f"Budget: {budget} measurements out of {N} crystals "
      f"({100*budget/N:.0f}% of the library).")
print(f"Global optimum (full enumeration, NOT given to agent): "
      f"{global_best:+.3f} eV/atom")
print(f"Best found by the loop: {found_best:+.3f} eV/atom")
print(f"Operator escalations (conformal band too wide): {escalations}")

# Baseline: random acquisition of the same budget, averaged over seeds.
rand_best = []
for s in range(20):
    r = np.random.default_rng(100 + s)
    sample = r.choice(N, size=budget, replace=False)
    rand_best.append(float(energy_oracle[torch.tensor(sample)].min().item()))
rand_mean = float(np.mean(rand_best))
print(f"Random-acquisition baseline (same budget, mean of 20): "
      f"{rand_mean:+.3f} eV/atom")

fig, ax = plt.subplots(figsize=(7, 4))
xs = range(n_seed + 1, budget + 1)
ax.plot(xs, best_energy_trace, "o-", lw=1.6, label="self-driving-lab loop")
ax.axhline(global_best, ls="--", c="green", label="global optimum (hidden)")
ax.axhline(rand_mean, ls=":", c="grey",
           label=f"random acquisition (mean, n={budget})")
ax.set_xlabel("# crystals measured")
ax.set_ylabel("best formation energy found (eV/atom)")
ax.set_title("Autonomous acquire→model→decide loop on CGNN embeddings")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the loop.** The agent never sees the full library; it spends a
# fixed measurement budget and the active-learning acquisition (exploit
# the surrogate's stability prediction, explore where the embedding is
# sparse) drives the best-found energy down **faster than random
# acquisition** — the self-driving-lab payoff. The conformal-style band is
# the deck's automate-vs-escalate discipline: when the surrogate's
# calibrated band is wide relative to its noise, the step is *escalated*
# to a human rather than silently auto-accepted. This is the same
# "measure, don't assert; refuse when unsure" honesty as Block 6 — here
# applied to *which experiment to run next* instead of *which sample to
# trust*.
#
# **Why this braids cleanly.** The state representation is the *frozen
# Block-5 embedding*: an embedding good enough to retrieve in (Block 5.2)
# is good enough to *steer an autonomous experiment*. MG diagnoses the
# representation; ML-PC puts the diagnosed representation in a closed
# control loop. That is the W11 triad's actual through-line.

# %% [markdown]
# # Block 6 — Honest limitations
#
# Three failure modes worth seeing now, before students reach for a VAE
# in the wild.
#
# 1. **Posterior collapse.** Train at very high β (e.g. β = 8) and watch
#    the KL go to zero — every encoder output collapses to the prior, and
#    the decoder learns to ignore z entirely.
# 2. **Mode collapse / lack of diversity.** Sample many times at the same
#    target energy from a CVAE that didn't see enough training; the
#    samples can be near-duplicates.
# 3. **OOD targets.** Ask the CVAE for an energy *outside* the training
#    range; the decoder hallucinates microstructures whose predicted
#    energy is *closer to the training-range edge* than to the request.
#
# *(see ML-PC §"Failure modes of inverse design", §"Validation discipline
# for generative models")*

# %%
# OOD demo only (the other two are exercise material).
print("OOD-target demo: ask the CVAE for energies outside training range.")
y_min, y_max = float(y_tr.min()), float(y_tr.max())
print(f"Training-set energy range: [{y_min:.0f}, {y_max:.0f}]")
ood_low = (y_min - y_std) - y_mean                         # 1 std below the min
ood_high = (y_max + y_std) - y_mean                        # 1 std above the max
ood_targets_n = [ood_low / y_std, ood_high / y_std]
ood_labels = [f"{(ood_low+y_mean):.0f} (OOD low)", f"{(ood_high+y_mean):.0f} (OOD high)"]

cvae.eval()
with torch.no_grad():
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for r, (yt, lbl) in enumerate(zip(ood_targets_n, ood_labels)):
        z = torch.randn(4, cvae.latent_dim, device=DEVICE)
        y_t = torch.full((4,), yt, device=DEVICE)
        samples = cvae.decode(z, y_t)
        # Score with the regressor.
        achieved = reg(samples) * y_std + y_mean
        for c in range(4):
            axes[r, c].imshow(samples[c, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            axes[r, c].set_title(f"target {lbl}\nregressor says {achieved[c]:.0f}",
                                 fontsize=8)
plt.suptitle("CVAE under OOD targets — the regressor scores never reach the request")
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the OOD plot.** When asked for an energy outside the training
# range, the CVAE produces something — but the regressor's score on that
# something stays near the training-range edge. The model has no *prior*
# for samples outside its data. This is the inverse-design version of
# "extrapolation is not generalisation": *generative models are
# interpolators, not extrapolators*. The discipline is to *measure* the
# achieved property (with an independent regressor) and refuse to
# advertise OOD candidates as valid.

# %% [markdown]
# # Block 7 — Student exercises (~15 min)

# %% [markdown]
# ## Exercise 1 (core) — Diversity-vs-accuracy in CVAE generation
#
# **Setup.** A good inverse-design generator should produce *diverse*
# candidates with the *same* target property. Diverse + on-target = good.
# Same-image-cloned + on-target = mode collapse. Different-images + off-target
# = bad targeting.
#
# **Task.** For each of the 3 target energies in Block 2 (low / median /
# high), generate 20 CVAE samples. For each cohort:
#
# 1. Compute mean ‖regressor(sample) − y_target‖ — the **accuracy**.
# 2. Compute the within-cohort image-pixel variance (`samples.var(dim=0).mean()`)
#    — the **diversity**.
#
# Plot accuracy vs diversity for the 3 cohorts. Where on the
# diversity–accuracy plane does the CVAE land for each target?

# %% [markdown]
# ## Exercise 2 (core) — Latent-GD vs CVAE under OOD targets
#
# **Setup.** Block 6 showed that the CVAE silently misses OOD targets.
# Latent-GD has a different failure mode: it can *visibly diverge* in
# loss space (you see the curve plateau or oscillate), giving a more
# honest "the model can't do this" signal.
#
# **Task.** Pick a target energy outside the training range (use one of
# `ood_targets_n` from Block 6). Run both methods:
#
# 1. CVAE: sample 8 candidates conditioned on the OOD target. Score them.
# 2. Latent-GD: run for 500 steps, plot the loss curve. Score the final z.
#
# Report which method's *output* is closer to the target, and which
# method's *failure signal* is louder.

# %% [markdown]
# ## Exercise 3 (core) — Does pretraining actually help? The probe verdict
#
# **Setup.** Block 5.1 produced a four-row probe table on a *single*
# held-out prototype (`perovskite`). The MG deck (§F slide 42) insists the
# **random-init row** is the comparison that decides whether *training*
# contributed or whether the *architecture* did the work — and it warns
# that a single split can mislead.
#
# **Task.** Turn the single split into a verdict:
#
# 1. Loop over all 5 prototypes as the held-out set in turn (reuse
#    `linear_probe`, `emb_trained`, `emb_random`, `feat_magpie`). For each
#    fold record $R^2$ for the trained encoder, the random-init encoder,
#    and the Magpie-style baseline.
# 2. Report the mean ± std $R^2$ across the 5 folds for each
#    representation.
# 3. Answer in one sentence: on this dataset, did the *supervised
#    training* of the CGNN buy a meaningful probe improvement over the
#    *random-init* architecture, and does either beat the engineered
#    Magpie-style baseline?
#
# **Expected.** If trained ≈ random-init, the message-passing
# architecture (not the training) carries the signal — exactly the
# "most-omitted comparison" the deck builds §F around. If the Magpie-style
# row is competitive, you have measured the deck's "always use the
# foundation model is wrong" claim (MG §G slide 47) on real numbers, not
# a slide.

# %% [markdown]
# ## Exercise 5 (stretch, optional) — Consistency distillation
#
# **Setup.** The flow-matching teacher in Block 4 needs **10 ODE steps**
# per sample (NFE = 10). A *consistency model* [@song_2023_consistency]
# distils that teacher into a **one-step student** $f_\theta(x_t, t)$
# that maps *any* point on the trajectory directly to a clean sample.
# The student's training signal is *consistency along the trajectory*:
# adjacent points $(x_{t_1}, t_1)$ and $(x_{t_2}, t_2)$ on the same flow
# must map to the same output.
#
# **Loss.** For a pair of adjacent times $t_1 < t_2$ sampled from the
# trajectory of the trained teacher,
#
# $$
# \mathcal{L}_{\text{consistency}}
# = \big\| f_\theta(x_{t_1}, t_1)
#        - \mathrm{stop\_grad}\big(f_\theta(x_{t_2}, t_2)\big) \big\|^2.
# $$
#
# The stop-gradient on the second term turns the *later* student call into
# a fixed regression target, the way a target network is used in deep
# RL. After training, set $t = 1$ at inference and sample in **one**
# forward pass (NFE = 1).
#
# **Task.**
#
# 1. Reuse the trained flow-matching `unet` from Block 4 as the *teacher*.
#    Build a `TinyUNet` student with the same architecture.
# 2. For each batch:
#    - sample $x_1$ from `X_tr`, $x_0 \sim \mathcal{N}(0, I)$, and two
#      times $t_1 < t_2$ in $(0, 1)$;
#    - form $x_{t_1}$ and $x_{t_2}$ on the *teacher trajectory* — either
#      with the linear interpolant or, for a stronger signal, with a few
#      Heun steps of the teacher between $t_1$ and $t_2$;
#    - compute the consistency loss above and update the student.
# 3. **Compare** the 1-step student vs the 10-step teacher on a fresh
#    batch of `n=64` samples:
#    - **visual** quality (a 4x4 grid of each);
#    - **2-Wasserstein distance** between the empirical distribution of
#      *mean-pixel intensities* (or any 1-D summary statistic) of the
#      generated samples and of the data — use
#      `scipy.stats.wasserstein_distance` on the 1-D summary;
#    - **NFE** (function evaluations per sample): student = 1, teacher = 10.
#
# **Expected.** The 1-step student should reach *most* of the teacher's
# quality at 10x lower NFE — modest visual degradation but big speed-up.
# If the student's W₂ distance is dramatically worse than the teacher's,
# you've reproduced the real-world finding that *one-step distillation is
# easy at the start of training and hard at the end* (the trajectory is
# straighter where flow matching predicts a near-constant velocity).
#
# **Skeleton.**
#
# ```python
# if HAS_GPU and unet is not None:
#     teacher = unet                                         # from Block 4
#     for p in teacher.parameters(): p.requires_grad_(False)
#     student = TinyUNet().to(DEVICE)
#     opt_s = torch.optim.Adam(student.parameters(), lr=2e-4)
#     loader_cm = DataLoader(TensorDataset(X_tr), batch_size=32, shuffle=True)
#     for ep in range(8):
#         for (xb,) in loader_cm:
#             x1 = xb; x0 = torch.randn_like(x1)
#             # two adjacent times t1 < t2
#             ...
#             # student outputs and stop-grad target
#             ...
#             loss = ((s1 - s2.detach()) ** 2).mean()
#             opt_s.zero_grad(); loss.backward(); opt_s.step()
# ```
#
# *(see MFML §"Consistency models"; [@song_2023_consistency])*

# %% [markdown]
# ---
# **Bridge to Week 12.** Next week MFML moves to *uncertainty
# quantification* (Gaussian processes, MC dropout, ensembles; the
# split-conformal primer was already introduced in MFML Unit 7), MG U12
# turns the *diagnosed* embedding of this week into a *generative*
# inverse-design pipeline (MatterGen / DiffCSP / FlowMM operate on
# exactly the kind of representation Block 5 just verified), and ML-PC
# pairs both with *uncertainty-aware discovery loops* — the conformal
# escalate rule of Block 5b promoted from a guardrail to the steering
# signal. The discipline this week — probe before you project, measure
# achieved properties, refuse OOD candidates, escalate when the
# calibrated band is wide — is the prerequisite for *honest* discovery.
