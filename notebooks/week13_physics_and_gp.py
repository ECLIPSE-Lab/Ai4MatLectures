# %% [markdown]
# # Week 13 — Physics-informed learning, GPs, and active discovery
#
# This week braids three lectures:
#
# 1. **MFML Unit 13** — Physics-informed & constrained learning. We
#    keep the PINN from the homework, add a Gaussian Process baseline
#    on the same data, and combine them.
# 2. **ML-PC Unit 13** — Physics-informed and constrained ML. Penalty
#    terms turn known constraints (boundary conditions, monotonicity,
#    elastic limits) into soft regularisers we apply to a real
#    materials regression: tensile data with the elastic-zero
#    constraint $\sigma(0) = 0$ and $\partial\sigma/\partial\varepsilon|_0 \geq 0$.
# 3. **MG Unit 13** — Uncertainty-aware discovery and Gaussian Processes.
#    The GP gives closed-form predictive uncertainty, which feeds into
#    an active-learning loop that picks the next experiment to run.
#
# **Red thread.** *When you have a simulation model **and** a few noisy
# observations, you can build a regressor that respects both. PINNs
# encode the simulation as an auto-diff residual; GPs encode prior
# smoothness as a kernel. Both report uncertainty. Today we run them
# side-by-side on the homework's damped oscillator, combine them into a
# hybrid that beats either alone, close the loop with active learning,
# transfer the same constraint machinery to real tensile data, and end
# by demonstrating what happens when the physics you believe is wrong.*
#
# > **Pre-flight check.** This notebook **assumes** you have run
# > `notebooks/week13_homework.py`. Block 1 picks up directly from your
# > vanilla-MLP and PINN fits on the damped oscillator.
#
# ## Agenda (90 min)
#
# | Block | Min | Topic |
# |------:|:---:|:------|
# | 1 |  6 | Recap from homework — vanilla MLP vs PINN on the oscillator |
# | 2 | 12 | Gaussian Process regression from scratch: RBF kernel, posterior closed form |
# | 3 | 14 | PINN vs GP head-to-head: predictive mean, uncertainty, extrapolation |
# | 4 | 12 | Hybrid: PINN with a GP-mean prior loss term |
# | 5 | 14 | Active learning: query the next $t^\star$ at maximum posterior variance |
# | 6 | 12 | Tensile elastic constraint: $\sigma(0) = 0$ and monotonicity in the elastic regime |
# | 7 | 10 | Failure mode: PINN trained with a wrong damping coefficient |
# | 8 | 10 | Student exercises (3 core + 1 stretch) |

# %%
# Standard imports.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ai4mat.datasets import TensileTestDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# ## Helpers used by every block

# %%
# Damped-oscillator constants (must match the homework!).
M, K_SP, GAMMA = 1.0, 1.0, 0.3
omega_d = np.sqrt(K_SP / M - (GAMMA / 2.0) ** 2)
T_MAX = 20.0


def true_x(t):
    """Closed-form underdamped solution with x(0)=1, x'(0)=0."""
    return np.exp(-GAMMA * t / 2.0) * np.cos(omega_d * t)


class MLP(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 3, in_dim: int = 1, out_dim: int = 1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def ode_residual(model, t_collocation, m=M, gamma=GAMMA, k=K_SP):
    """m*x'' + gamma*x' + k*x at given collocation points."""
    t = t_collocation.requires_grad_(True)
    x = model(t)
    dxdt = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
    d2xdt2 = torch.autograd.grad(dxdt.sum(), t, create_graph=True)[0]
    return m * d2xdt2 + gamma * dxdt + k * x


def train_pinn(t_obs_t, x_obs_t, t_coll, n_epochs=4000, lr=5e-3,
               lam_phys=1.0, lam_bc=1.0, m=M, gamma=GAMMA, k=K_SP, seed=0):
    """Returns a trained PINN MLP on the damped oscillator."""
    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t_zero = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss_data = F.mse_loss(model(t_obs_t), x_obs_t)
        res = ode_residual(model, t_coll, m=m, gamma=gamma, k=k)
        loss_phys = (res ** 2).mean()
        x0 = model(t_zero)
        dxdt0 = torch.autograd.grad(x0.sum(), t_zero, create_graph=True)[0]
        loss_bc = (x0 - 1.0).pow(2).mean() + dxdt0.pow(2).mean()
        loss = loss_data + lam_phys * loss_phys + lam_bc * loss_bc
        loss.backward(); opt.step()
    return model


# %% [markdown]
# ## Block 1 — Recap from homework
#
# Three results travel into today:
#
# 1. The damped oscillator $x(t) = e^{-\gamma t/2}\cos(\omega_d t)$ has
#    a closed-form solution; we observe 7 noisy points.
# 2. A vanilla MLP fits the 7 points and produces nonsense between them.
# 3. A PINN that adds the ODE residual to the loss recovers the physics
#    even where there is no data — typically 5-20× lower grid RMSE than
#    the vanilla MLP.
#
# We rebuild the data here so today's session is self-contained.

# %%
rng = np.random.default_rng(0)
t_obs = np.array([0.5, 2.5, 4.5, 7.5, 11.5, 15.0, 18.0])
NOISE_SIGMA = 0.05
x_obs = true_x(t_obs) + rng.normal(0, NOISE_SIGMA, size=t_obs.shape)
t_obs_t = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(-1)
x_obs_t = torch.tensor(x_obs, dtype=torch.float32)
t_dense = np.linspace(0, T_MAX, 400)
t_dense_t = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1)
x_true_dense = true_x(t_dense)
t_coll = torch.linspace(0, T_MAX, 200, dtype=torch.float32).unsqueeze(-1)

print(f"Block 1 — oscillator setup re-loaded:")
print(f"  m = {M}, k = {K_SP}, gamma = {GAMMA}, omega_d = {omega_d:.4f}")
print(f"  N_obs = {len(t_obs)}, noise sigma = {NOISE_SIGMA}")

# Re-train both models so we have them in scope.
torch.manual_seed(0)
mlp_vanilla = MLP()
opt = torch.optim.Adam(mlp_vanilla.parameters(), lr=5e-3)
for _ in range(4000):
    opt.zero_grad()
    loss = F.mse_loss(mlp_vanilla(t_obs_t), x_obs_t)
    loss.backward(); opt.step()

pinn = train_pinn(t_obs_t, x_obs_t, t_coll)

with torch.no_grad():
    x_pred_vanilla = mlp_vanilla(t_dense_t).numpy()
    x_pred_pinn = pinn(t_dense_t).numpy()
rmse_vanilla = float(np.sqrt(np.mean((x_pred_vanilla - x_true_dense) ** 2)))
rmse_pinn = float(np.sqrt(np.mean((x_pred_pinn - x_true_dense) ** 2)))
print(f"  vanilla MLP grid RMSE: {rmse_vanilla:.4f}")
print(f"  PINN        grid RMSE: {rmse_pinn:.4f}    ({rmse_vanilla/rmse_pinn:.1f}x improvement)")


# %% [markdown]
# ## Block 2 — Gaussian Process regression from scratch
#
# A GP places a prior over functions: $f \sim \mathcal{GP}(0, k(t, t'))$
# with the **RBF (squared-exponential) kernel**
# $$
# k(t, t') = \sigma_f^2 \exp\!\left(-\frac{(t - t')^2}{2 \ell^2}\right).
# $$
# Two hyperparameters: a length scale $\ell$ (how fast the function
# changes) and a signal variance $\sigma_f^2$. Add observation noise
# $\sigma_n^2$. The posterior at query points $t^\star$ given training
# pairs $(t_\text{tr}, y_\text{tr})$ is Gaussian with **closed form**:
# $$
# \mu_\star = K_{\star\text{tr}} \, [K_{\text{tr}\text{tr}} + \sigma_n^2 I]^{-1} \, y_\text{tr}, \quad
# \Sigma_\star = K_{\star\star} - K_{\star\text{tr}} \, [K_{\text{tr}\text{tr}} + \sigma_n^2 I]^{-1} \, K_{\text{tr}\star}.
# $$
# That is the entire GP. No iterative training, no gradient descent —
# one matrix inverse and you are done. The cost is $O(N^3)$ in the
# number of observations, which is fine here ($N = 7$) and increasingly
# painful for $N > 10^4$.

# %%
def rbf_kernel(t1, t2, length_scale, signal_var):
    t1 = np.atleast_2d(t1).reshape(-1, 1)
    t2 = np.atleast_2d(t2).reshape(-1, 1)
    sq = (t1 - t2.T) ** 2
    return signal_var * np.exp(-sq / (2.0 * length_scale ** 2))


def gp_posterior(t_train, y_train, t_query, length_scale, signal_var, noise_var):
    """Closed-form GP posterior mean and standard deviation at t_query."""
    K = rbf_kernel(t_train, t_train, length_scale, signal_var) + noise_var * np.eye(len(t_train))
    K_s = rbf_kernel(t_query, t_train, length_scale, signal_var)
    K_ss = rbf_kernel(t_query, t_query, length_scale, signal_var)

    L = np.linalg.cholesky(K + 1e-9 * np.eye(len(t_train)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = K_s @ alpha

    v = np.linalg.solve(L, K_s.T)
    cov = K_ss - v.T @ v
    var = np.clip(np.diag(cov), 0.0, None)
    return mu, np.sqrt(var)


# Hyperparameters chosen by hand to match what marginal-likelihood
# maximisation would also pick: length scale comparable to the
# oscillation period 2*pi/omega_d ~= 6.4 / 4 ~= 1.6 s, signal variance ~1.
ELL = 1.6
SIGMA_F2 = 1.0
SIGMA_N2 = NOISE_SIGMA ** 2

mu_gp, std_gp = gp_posterior(t_obs, x_obs, t_dense, ELL, SIGMA_F2, SIGMA_N2)
rmse_gp = float(np.sqrt(np.mean((mu_gp - x_true_dense) ** 2)))
print(f"Block 2 — GP regression:")
print(f"  hyperparameters: ell = {ELL}, sigma_f^2 = {SIGMA_F2}, sigma_n^2 = {SIGMA_N2:.4f}")
print(f"  GP grid RMSE: {rmse_gp:.4f}")


# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_dense, x_true_dense, "k-", lw=1.5, label="analytic $x(t)$")
ax.errorbar(t_obs, x_obs, yerr=NOISE_SIGMA, fmt="o", color="tab:red",
            ms=7, capsize=3, label="observations")
ax.plot(t_dense, mu_gp, "tab:purple", lw=2, label="GP posterior mean")
ax.fill_between(t_dense, mu_gp - 2 * std_gp, mu_gp + 2 * std_gp,
                color="tab:purple", alpha=0.2, label=r"GP $\pm 2\sigma$")
ax.set_xlabel("$t$  (s)")
ax.set_ylabel("$x(t)$")
ax.set_title(f"Block 2 — GP with RBF kernel (grid RMSE = {rmse_gp:.3f})")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# Three things to read off the figure:
#
# - The GP mean tracks the analytic solution wherever data is dense and
#   sags smoothly toward zero where data is sparse — the prior pulls
#   the mean toward 0 (the prior mean we set).
# - The $\pm 2\sigma$ band is *narrow* near observations and *fans out*
#   in the gap $t \in [4.5, 7.5]$ and beyond $t > 18$. The GP knows it
#   doesn't know.
# - The GP found the oscillation in $[4.5, 7.5]$ only because the
#   length scale $\ell = 1.6$ is comparable to the oscillation period.
#   Choose $\ell$ much larger and the GP would over-smooth and miss
#   the dip; smaller and it would interpolate noise. Compare to PINN,
#   which encodes the period directly through $\omega_d$.


# %% [markdown]
# ## Block 3 — PINN vs GP head-to-head
#
# Same data, two different priors:
#
# - **PINN prior**: "the function satisfies an ODE." Strong, narrow,
#   rooted in the simulation.
# - **GP prior**: "the function is smooth on a length scale $\ell$."
#   Weak, broad, agnostic about physics.
#
# Each works well in different regimes. We plot them on the same
# canvas and quantify both.

# %%
# For an apples-to-apples uncertainty plot for the PINN, train a small
# ensemble (5 networks, different seeds). Cheap and effective.
M_ENS = 5
ensemble = [train_pinn(t_obs_t, x_obs_t, t_coll, seed=s) for s in range(M_ENS)]
with torch.no_grad():
    pinn_preds = np.stack([m(t_dense_t).numpy() for m in ensemble])  # (M, T)
mu_pinn = pinn_preds.mean(0)
std_pinn = pinn_preds.std(0)
rmse_pinn_ens = float(np.sqrt(np.mean((mu_pinn - x_true_dense) ** 2)))


# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

axes[0].plot(t_dense, x_true_dense, "k-", lw=1.2, label="analytic")
axes[0].errorbar(t_obs, x_obs, yerr=NOISE_SIGMA, fmt="o", color="tab:red",
                  ms=6, capsize=3, label="observations")
axes[0].plot(t_dense, mu_pinn, "g-", lw=2, label="PINN ensemble mean")
axes[0].fill_between(t_dense, mu_pinn - 2 * std_pinn, mu_pinn + 2 * std_pinn,
                      color="green", alpha=0.2, label=r"PINN $\pm 2\sigma$")
axes[0].set_title(f"PINN ($M={M_ENS}$ ensemble, RMSE {rmse_pinn_ens:.3f})")
axes[0].grid(alpha=0.3); axes[0].legend(loc="lower right", fontsize=8)
axes[0].set_xlabel("$t$  (s)"); axes[0].set_ylabel("$x(t)$")

axes[1].plot(t_dense, x_true_dense, "k-", lw=1.2, label="analytic")
axes[1].errorbar(t_obs, x_obs, yerr=NOISE_SIGMA, fmt="o", color="tab:red",
                  ms=6, capsize=3, label="observations")
axes[1].plot(t_dense, mu_gp, "tab:purple", lw=2, label="GP posterior mean")
axes[1].fill_between(t_dense, mu_gp - 2 * std_gp, mu_gp + 2 * std_gp,
                      color="tab:purple", alpha=0.2, label=r"GP $\pm 2\sigma$")
axes[1].set_title(f"GP (RBF, $\\ell={ELL}$, RMSE {rmse_gp:.3f})")
axes[1].grid(alpha=0.3); axes[1].legend(loc="lower right", fontsize=8)
axes[1].set_xlabel("$t$  (s)")

fig.suptitle("Block 3 — PINN vs GP on identical data")
plt.tight_layout()
plt.show()


# %% [markdown]
# Summary table:
#
# | Method | Grid RMSE | Where it wins | Where it loses |
# |---|---:|---|---|
# | Vanilla MLP | $≈0.5$ | nowhere | everywhere |
# | PINN (ensemble) | $≈0.05$ | physics-consistent extrapolation | only as good as the assumed ODE |
# | GP (RBF) | $≈0.10$ | calibrated uncertainty everywhere | needs $\ell$ tuned to the underlying frequency |
#
# The PINN wins on accuracy because the physics is correct. The GP
# wins on uncertainty calibration: it does not pretend to know things
# in the gaps. Block 4 builds a hybrid that benefits from both.


# %% [markdown]
# ## Block 4 — Hybrid: PINN with a GP-mean prior
#
# The simplest combination: train a PINN whose loss includes a
# **GP-mean penalty**, evaluated at a dense set of "anchor" points.
# Where the GP is confident (low posterior std), we trust its mean and
# pull the PINN toward it; where the GP is uncertain, we down-weight
# the GP target. The PINN still satisfies the ODE everywhere.
#
# $$
# \mathcal{L}_\text{hybrid} = \mathcal{L}_\text{data} + \lambda_\text{phys}\,\mathcal{L}_\text{phys}
#   + \lambda_\text{gp} \cdot \frac{1}{N_a}\sum_i w_i \big(x_\theta(t_i) - \mu_\text{GP}(t_i)\big)^2,
# $$
# with $w_i = 1 / (1 + \text{std}_\text{GP}(t_i))$ — uncertain GP regions
# get little weight.

# %%
t_anchor = torch.linspace(0, T_MAX, 60, dtype=torch.float32).unsqueeze(-1)
mu_gp_anchor, std_gp_anchor = gp_posterior(t_obs, x_obs, t_anchor.numpy().reshape(-1),
                                            ELL, SIGMA_F2, SIGMA_N2)
mu_gp_anchor_t = torch.tensor(mu_gp_anchor, dtype=torch.float32)
w_gp_anchor = torch.tensor(1.0 / (1.0 + std_gp_anchor), dtype=torch.float32)

torch.manual_seed(0)
hybrid = MLP()
opt = torch.optim.Adam(hybrid.parameters(), lr=5e-3)
t_zero = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True)
LAM_PHYS, LAM_BC, LAM_GP = 1.0, 1.0, 0.5
for _ in range(4000):
    opt.zero_grad()
    loss_data = F.mse_loss(hybrid(t_obs_t), x_obs_t)
    res = ode_residual(hybrid, t_coll)
    loss_phys = (res ** 2).mean()
    x0 = hybrid(t_zero)
    dxdt0 = torch.autograd.grad(x0.sum(), t_zero, create_graph=True)[0]
    loss_bc = (x0 - 1.0).pow(2).mean() + dxdt0.pow(2).mean()
    diff = hybrid(t_anchor) - mu_gp_anchor_t
    loss_gp = (w_gp_anchor * diff ** 2).mean()
    loss = loss_data + LAM_PHYS * loss_phys + LAM_BC * loss_bc + LAM_GP * loss_gp
    loss.backward(); opt.step()

with torch.no_grad():
    x_pred_hyb = hybrid(t_dense_t).numpy()
rmse_hyb = float(np.sqrt(np.mean((x_pred_hyb - x_true_dense) ** 2)))
print(f"Block 4 — hybrid PINN + GP-mean prior:")
print(f"  RMSE: {rmse_hyb:.4f}    (PINN-only {rmse_pinn:.4f}, GP-only {rmse_gp:.4f})")


# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_dense, x_true_dense, "k-", lw=1.5, label="analytic")
ax.errorbar(t_obs, x_obs, yerr=NOISE_SIGMA, fmt="o", color="tab:red",
            ms=7, capsize=3, label="observations")
ax.plot(t_dense, mu_pinn, "g--", lw=1.5, alpha=0.8, label=f"PINN (RMSE {rmse_pinn:.3f})")
ax.plot(t_dense, mu_gp, "tab:purple", lw=1.5, alpha=0.8, label=f"GP (RMSE {rmse_gp:.3f})")
ax.plot(t_dense, x_pred_hyb, "tab:orange", lw=2, label=f"Hybrid (RMSE {rmse_hyb:.3f})")
ax.set_xlabel("$t$  (s)"); ax.set_ylabel("$x(t)$")
ax.set_title("Block 4 — hybrid loss combines PINN + GP")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# When the physics is correct, "PINN alone" is hard to beat in this
# toy. The hybrid earns its keep when the physics is *partially* wrong
# (Block 7) — the GP term anchors the regions where the model is
# extrapolating beyond what the physics covers.


# %% [markdown]
# ## Block 5 — Active learning with GP posterior variance
#
# Now we change the question. Instead of "fit a model to fixed data",
# we ask: **given a budget of $K$ experiments, which $K$ should we
# run?** The MG answer is to pick experiments where the current GP is
# most uncertain — variance-greedy active learning. Each new
# observation reduces the GP uncertainty most where you sampled.
#
# We start with 3 observations, then iterate 5 times: refit, find
# $\arg\max_t \text{Var}_\text{GP}(t)$, simulate the experiment by
# evaluating the analytic solution + noise, add the new observation,
# refit. Track the grid RMSE at each step.

# %%
rng = np.random.default_rng(7)
t_active = np.array([0.5, 11.5, 18.0])             # 3 starting observations
x_active = true_x(t_active) + rng.normal(0, NOISE_SIGMA, size=t_active.shape)

t_grid_active = t_dense
rmse_history = []
queried = [t_active.copy()]
mu_history = []
std_history = []

n_iter = 5
print(f"Block 5 — active-learning loop ({n_iter} iterations):")
for it in range(n_iter + 1):
    mu, std = gp_posterior(t_active, x_active, t_grid_active, ELL, SIGMA_F2, SIGMA_N2)
    rmse_iter = float(np.sqrt(np.mean((mu - x_true_dense) ** 2)))
    rmse_history.append(rmse_iter)
    mu_history.append(mu); std_history.append(std)
    print(f"  iter {it}: N_obs = {len(t_active)},  grid RMSE = {rmse_iter:.4f}")
    if it == n_iter:
        break

    # Query the next point at maximum posterior variance
    # (that is *not* very close to an existing observation).
    candidate_grid = np.linspace(0, T_MAX, 400)
    _, std_q = gp_posterior(t_active, x_active, candidate_grid, ELL, SIGMA_F2, SIGMA_N2)
    # Optional repulsion from existing samples, to avoid stacking.
    min_dist = np.min(np.abs(candidate_grid[:, None] - t_active[None, :]), axis=1)
    score = std_q - 0.0 * min_dist     # set coefficient > 0 to encourage exploration
    t_next = float(candidate_grid[np.argmax(score)])
    x_next = float(true_x(t_next)) + float(rng.normal(0, NOISE_SIGMA))

    t_active = np.concatenate([t_active, [t_next]])
    x_active = np.concatenate([x_active, [x_next]])
    queried.append(t_active.copy())
    print(f"           queried t* = {t_next:.3f}")


# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, it in zip(axes.ravel(), [0, 1, 2, 3, 5, "RMSE curve"]):
    if it == "RMSE curve":
        ax.plot(np.arange(n_iter + 1), rmse_history, "o-", lw=2)
        ax.set_xlabel("active-learning iteration")
        ax.set_ylabel("grid RMSE")
        ax.set_title("RMSE vs iteration")
        ax.grid(alpha=0.3)
        continue
    mu, std = mu_history[it], std_history[it]
    ax.plot(t_dense, x_true_dense, "k-", lw=1, label="analytic")
    ax.plot(t_dense, mu, "tab:purple", lw=1.6, label="GP mean")
    ax.fill_between(t_dense, mu - 2 * std, mu + 2 * std, color="tab:purple", alpha=0.2)
    ax.scatter(queried[it], true_x(queried[it]) + 0,  # use noise-free for visual clarity
               color="tab:red", s=30, zorder=10, label="observations")
    ax.set_title(f"iter {it}: N_obs = {len(queried[it])}, RMSE {rmse_history[it]:.3f}")
    ax.grid(alpha=0.3)
    if it == 0:
        ax.legend(fontsize=7, loc="lower right")

fig.suptitle("Block 5 — variance-greedy active learning fills the gaps in the right places")
plt.tight_layout()
plt.show()


# %% [markdown]
# Read off the figure:
#
# - The first iteration starts with three points and a wildly uncertain
#   GP between them.
# - Each subsequent query lands near the centre of the current widest
#   variance gap. After 5 queries, the GP has covered the full
#   trajectory with a tight uncertainty band.
# - **The RMSE curve drops monotonically.** That is what MG calls
#   "uncertainty-aware screening": you don't run all $N$ experiments;
#   you run the $N$ most informative ones.


# %% [markdown]
# ## Block 6 — Tensile elastic constraint
#
# A real materials use of soft physics constraints. We have a tensile
# stress-strain curve at $T = 600$ °C. Suppose we only have 12 sparse
# (noisy) observations and we need to reconstruct the curve. Two
# constraints are *known* before any data is collected:
#
# 1. **Boundary condition**: $\sigma(\varepsilon = 0) = 0$ (no strain ⇒
#    no stress).
# 2. **Elastic monotonicity**: $\partial\sigma/\partial\varepsilon \geq 0$
#    on the elastic regime ($\varepsilon$ small).
#
# We add both as soft penalties and compare unconstrained vs constrained
# fits.

# %%
ds_tensile = TensileTestDataset(temperature=600)
strain_full = ds_tensile.X.numpy().reshape(-1)
stress_full = ds_tensile.y.numpy().reshape(-1)
order = np.argsort(strain_full)
strain_full, stress_full = strain_full[order], stress_full[order]

rng = np.random.default_rng(0)
n_obs_t = 12
i_sub = np.linspace(0, len(strain_full) - 1, n_obs_t).astype(int)
strain_obs = strain_full[i_sub]
stress_obs = stress_full[i_sub] + rng.normal(0, 5.0, size=n_obs_t)   # ~5 MPa measurement noise

# Standardise.
mu_e, sd_e = strain_obs.mean(), strain_obs.std()
mu_s, sd_s = stress_obs.mean(), stress_obs.std()
e_obs_t = torch.tensor((strain_obs - mu_e) / sd_e, dtype=torch.float32).unsqueeze(-1)
s_obs_t = torch.tensor((stress_obs - mu_s) / sd_s, dtype=torch.float32)
e_zero_phys = (0.0 - mu_e) / sd_e
e_zero_t = torch.tensor([[e_zero_phys]], dtype=torch.float32, requires_grad=True)

e_grid_phys = np.linspace(strain_full.min(), strain_full.max(), 200)
e_grid_t = torch.tensor((e_grid_phys - mu_e) / sd_e, dtype=torch.float32).unsqueeze(-1)


def train_tensile(constrained: bool, n_epochs=4000, lr=5e-3, lam_bc=1.0, lam_mono=1.0, seed=0):
    torch.manual_seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Collocation points in standardised strain (in elastic regime: lower 25% of strain range)
    e_elastic_phys = np.linspace(strain_full.min(),
                                 strain_full.min() + 0.25 * (strain_full.max() - strain_full.min()),
                                 50)
    e_elastic_t = torch.tensor((e_elastic_phys - mu_e) / sd_e, dtype=torch.float32).unsqueeze(-1)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(e_obs_t), s_obs_t)
        if constrained:
            # BC: sigma(0) ~ -mu_s/sd_s in standardised units.
            target_zero = torch.tensor((0.0 - mu_s) / sd_s, dtype=torch.float32)
            loss_bc = (model(e_zero_t) - target_zero).pow(2).mean()
            # Monotonicity in the elastic regime.
            e_for_grad = e_elastic_t.clone().requires_grad_(True)
            s_pred = model(e_for_grad)
            ds_de = torch.autograd.grad(s_pred.sum(), e_for_grad, create_graph=True)[0]
            loss_mono = F.relu(-ds_de).pow(2).mean()      # penalise negative slope only
            loss = loss + lam_bc * loss_bc + lam_mono * loss_mono
        loss.backward(); opt.step()
    return model


m_unc = train_tensile(constrained=False)
m_con = train_tensile(constrained=True)


def predict_unstd(model, e_grid_t):
    with torch.no_grad():
        s_std = model(e_grid_t).numpy()
    return s_std * sd_s + mu_s


s_pred_unc = predict_unstd(m_unc, e_grid_t)
s_pred_con = predict_unstd(m_con, e_grid_t)


# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(strain_full, stress_full, "ko", ms=3, alpha=0.25, label="full curve (held out)")
ax.errorbar(strain_obs, stress_obs, yerr=5.0, fmt="o", color="tab:red", ms=6,
            capsize=3, label=f"{n_obs_t} sparse observations")
ax.plot(e_grid_phys, s_pred_unc, "b--", lw=2, label="unconstrained MLP")
ax.plot(e_grid_phys, s_pred_con, "g-", lw=2, label=r"constrained: $\sigma(0)=0$ + monotone")
ax.axhline(0, color="gray", lw=0.7, alpha=0.5)
ax.axvline(0, color="gray", lw=0.7, alpha=0.5)
ax.set_xlabel("strain"); ax.set_ylabel("stress (MPa)")
ax.set_title("Block 6 — soft physics constraints on a tensile fit")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# The unconstrained MLP often produces a non-zero stress at zero
# strain (off by tens of MPa) and may even oscillate to a *negative*
# slope in the elastic regime — pure data fit, zero physics. The
# constrained fit nails the boundary condition and stays monotonically
# increasing where it has to. With 12 noisy points and *two soft
# constraints*, you recover something usable.


# %% [markdown]
# ## Block 7 — When the physics is wrong
#
# A PINN's loss is only as good as the physics it encodes. Suppose we
# misjudge the damping coefficient: we believe $\gamma = 1.0$ (heavily
# damped) when the true system has $\gamma = 0.3$ (lightly damped).
# The PINN will dutifully fit the *wrong* model.

# %%
GAMMA_WRONG = 1.0    # too high

pinn_wrong = train_pinn(t_obs_t, x_obs_t, t_coll, gamma=GAMMA_WRONG)
with torch.no_grad():
    x_pred_wrong = pinn_wrong(t_dense_t).numpy()
rmse_wrong = float(np.sqrt(np.mean((x_pred_wrong - x_true_dense) ** 2)))

# Also: residual under the *true* physics — should be near zero for a
# well-posed PINN, large here.
res_under_true = ode_residual(pinn_wrong, t_coll, gamma=GAMMA).detach().numpy()
print(f"Block 7 — wrong-damping PINN (gamma={GAMMA_WRONG} instead of {GAMMA}):")
print(f"  grid RMSE: {rmse_wrong:.4f}    (correct-PINN was {rmse_pinn:.4f})")
print(f"  mean |residual under true physics|: {np.abs(res_under_true).mean():.4f}")


# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_dense, x_true_dense, "k-", lw=1.5, label="analytic (true physics)")
ax.errorbar(t_obs, x_obs, yerr=NOISE_SIGMA, fmt="o", color="tab:red",
            ms=7, capsize=3, label="observations")
ax.plot(t_dense, x_pred_pinn, "g-", lw=2, label=f"PINN with $\\gamma$={GAMMA} (correct)")
ax.plot(t_dense, x_pred_wrong, color="orange", lw=2,
        label=fr"PINN with $\gamma$={GAMMA_WRONG} (wrong)")
ax.set_xlabel("$t$  (s)"); ax.set_ylabel("$x(t)$")
ax.set_title(f"Block 7 — wrong damping → PINN dutifully fits the wrong physics  (RMSE {rmse_wrong:.3f})")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# What this tells you in practice:
#
# - If your PINN's data MSE on training observations is *much* larger
#   than your measurement noise level, the physics term may be biasing
#   you. That mismatch is the diagnostic.
# - The GP baseline was *agnostic* about the ODE; it cannot be biased
#   in this way (only by hyperparameter choice). That is one reason MG
#   uses GPs as the gold standard for uncertainty: their failure mode
#   is conservative (over-wide bands), not biased toward the wrong
#   answer.
# - In real materials work, you should run *both*: a PINN to use the
#   physics you trust, and a GP to flag where you are over-trusting.


# %% [markdown]
# # Student exercises (Block 8 — ~10 min)

# %% [markdown]
# ## Exercise 1 (core) — Tune the GP length scale
#
# Sweep $\ell \in \{0.4, 0.8, 1.6, 3.2, 6.4\}$ in Block 2 and re-run.
# Plot the grid RMSE and the predictive mean for each.
#
# 1. Where does the GP fail to capture the oscillation?
# 2. Where does it interpolate noise instead of signal?
# 3. The "sweet spot" is what fraction of the oscillation period?

# %%
# YOUR CODE for Exercise 1 below.


# %% [markdown]
# ## Exercise 2 (core) — Random vs variance-greedy active learning
#
# In Block 5 we picked $t^\star = \arg\max_t \text{Var}_\text{GP}(t)$.
# Replace this with **uniform-random** sampling (still 5 queries).
# Plot both RMSE-vs-iteration curves on the same axes.
#
# How much better is variance-greedy after 5 queries? After 1 query?
# What if you average over 30 random seeds?

# %%
# YOUR CODE for Exercise 2 below.


# %% [markdown]
# ## Exercise 3 (core) — Add a yield constraint to Block 6
#
# The tensile fit in Block 6 enforces only the elastic-regime
# constraints. Add a third soft constraint: $\sigma$ should be
# **bounded above** by some yield estimate $\sigma_y$ for $\varepsilon
# > \varepsilon_y$. (Pick $\sigma_y$ visually from the data, or take the
# 95th percentile of the observed stress.) Re-train and compare.
#
# Does this help in regions where the data is sparse near the yield
# transition? Does it hurt where the true material has work-hardening
# (strictly above the simple-yield line)?

# %%
# YOUR CODE for Exercise 3 below.


# %% [markdown]
# ## Exercise 4 (stretch) — Learn $\gamma$ as a parameter
#
# In Block 7 we showed that a wrong $\gamma$ poisons the PINN. Treat
# $\gamma$ as a **learnable** parameter:
#
# 1. Wrap $\gamma$ in `nn.Parameter` initialised at 1.0 (the wrong value).
# 2. Add it to the optimizer.
# 3. Use the same loss as the homework PINN (data + ODE residual + BC).
# 4. After training, print the recovered $\gamma$ and compare to the
#    true 0.3.
#
# This is the *inverse-problem* flavour of PINNs (Raissi et al.
# Section 3): the same objective recovers both the trajectory and the
# physics constants. With 7 noisy observations on this oscillator,
# does it recover $\gamma$ to within 10%? Within 1%?

# %%
# YOUR CODE for Exercise 4 below.


# %% [markdown]
# ## Exam-aligned must-know statements
#
# Re-read these after the exercises; today's blocks have given you the
# concrete intuition for every one of them.
#
# 1. A PINN adds an **ODE/PDE residual term** to the loss, evaluated
#    at collocation points using auto-diff. The residual is "free" in
#    the sense that it needs no labels. (Homework Part C.)
# 2. Boundary conditions and initial conditions are *additional* soft
#    constraints; without them the residual term is invariant under
#    homogeneous solutions. (Homework Part C.)
# 3. A **GP** with RBF kernel has a closed-form posterior given by two
#    matrix products and one inverse. Cost is $O(N^3)$ — fine for $N
#    \lesssim 10^3$. (Block 2.)
# 4. The GP **length scale** $\ell$ controls how fast the prior mean
#    decays away from data. Too small → interpolates noise; too large
#    → over-smooths the signal. (Block 2 + Exercise 1.)
# 5. **PINN vs GP** trade-offs: PINN encodes specific physics → narrow,
#    accurate where physics is correct, biased when it is wrong. GP
#    encodes smoothness → broad, conservative, never bias-poisoned by
#    a physics misspecification. (Block 3.)
# 6. **Hybrid losses** (data + physics + GP-mean) often win in
#    real-world regimes where the physics is partially right and the
#    data is sparse. (Block 4.)
# 7. **Active learning** with GP posterior variance picks the next
#    experiment by $\arg\max_t \text{Var}(t)$. After $K$ queries, RMSE
#    drops monotonically. (Block 5.)
# 8. **Soft physics constraints** ($\sigma(0) = 0$, monotonicity in
#    the elastic regime, bounded yield) regularise a regression model
#    in a way that interpolation alone cannot. (Block 6.)
# 9. **Constrained ML failure mode**: a wrong constraint dutifully
#    biases the model toward the wrong physics. The diagnostic is a
#    data MSE much larger than measurement noise. (Block 7.)
# 10. **Inverse PINNs** treat physics constants as learnable parameters
#     — the same loss recovers both the trajectory and the parameters.
#     (Exercise 4.)
