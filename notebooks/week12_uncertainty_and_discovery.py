# %% [markdown]
# # Week 12 — Uncertainty and discovery
#
# This week we braid three lectures:
#
# 1. **MFML Unit 12**: Uncertainty in predictions — Gaussian Processes (the
#    main tool), MC dropout, deep ensembles, mixture-density networks,
#    calibration, the evidence framework. Theory anchor.
# 2. **ML-PC Unit 11**: Materials UQ case studies — 21CrMoV5-7 GP for
#    hardness, MC-dropout SEM segmentation, additive-manufacturing active
#    learning. Lab-story anchor.
# 3. **MG (curriculum) → slides folder 11**: Clustering vs discovery in
#    materials spaces — top-k retrieval, per-cluster acquisition budgets,
#    the discovery-vs-labeling distinction.
#
# **Red thread:** *Materials discovery loops live or die on uncertainty:
# tight error bars say "exploit", wide ones say "explore", and an outlier
# *without* uncertainty is just noise. This week we braid Gaussian
# Processes (MFML), real lab case studies (ML-PC), and the discovery-vs-
# labeling distinction (MG) into an end-to-end Bayesian-optimization-style
# materials-acceleration loop.*
#
# > **Pre-flight check.** This notebook **assumes** you have run
# > `notebooks/week12_homework.py`. Block 1 picks up directly from your GP
# > posterior, the 3-method comparison, and the reliability diagram on
# > `TensileTestDataset(T=600)`; we will not re-derive them.
#
# ## Agenda (90 min)
#
# | Block | Min | Topic |
# |------:|:---:|:------|
# | 1 |  5  | Recap from homework — GP posterior, calibration on T=600 |
# | 2 | 15  | MLPC anchor — GP active learning on `TensileTestDataset(T=600)` |
# | 3 | 12  | Acquisition functions — UCB / EI / PI on the same AL loop |
# | 4 | 12  | Cost-aware AL — combining cheap T=600 with expensive T=0 |
# | 5 | 15  | MG anchor — clustering vs discovery on `NanoindentationDataset` |
# | 6 | 15  | Per-cluster acquisition budgets — equal / proportional / top-k |
# | 7 | 16  | Student exercises (3 core + 1 stretch) |

# %%
# Standard imports for the whole in-class. Same idiom as weeks 2-11.
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.stats import norm

from ai4mat.datasets import TensileTestDataset, NanoindentationDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# ## Helpers used by every block

# %%
def make_gp(length_scale=0.05, length_scale_bounds=(1e-3, 1.0),
            noise_level=1.0, noise_level_bounds=(1e-3, 1e3),
            n_restarts=3, seed=0):
    """Convenience constructor for a calibrated regression GP.

    The kernel is ConstantKernel * RBF + WhiteKernel; we fit it via
    type-II maximum likelihood (sklearn's default). `normalize_y=True`
    handles the stress-axis scale so the kernel hyperparameters stay
    in a sane numerical range.
    """
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e6))
        * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
    )
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=n_restarts,
                                    random_state=seed)


def to_numpy_dataset(ds):
    """Stack a (small) PyTorch dataset into NumPy (X, y)."""
    X = torch.stack([ds[i][0] for i in range(len(ds))]).numpy().astype(np.float32)
    y = torch.stack([ds[i][1] for i in range(len(ds))]).numpy().astype(np.float32)
    return X, y


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


# %% [markdown]
# # Block 1 — Recap from homework
#
# In Part A you derived the closed-form GP posterior and learned its
# hyperparameters by maximum likelihood. In Part C you fitted a GP on
# `TensileTestDataset(T=600)` and plotted the reliability diagram: the GP
# was approximately on-diagonal (well-calibrated) for nominal levels 50%,
# 80%, 95%. Block 2 onwards uses *that same GP* as the workhorse for an
# active-learning loop.
#
# *(see homework Part C; MFML §"GP posterior")*

# %%
# Refit the homework GP in 5 lines so subsequent blocks can build on it.
ds_T600 = TensileTestDataset(temperature=600)
X_T600, y_T600 = to_numpy_dataset(ds_T600)

rng_split = np.random.default_rng(0)
perm = rng_split.permutation(len(X_T600))
n_tr = int(0.8 * len(X_T600))
tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
X_tr_full, y_tr_full = X_T600[tr_idx], y_T600[tr_idx]
X_te, y_te = X_T600[te_idx], y_T600[te_idx]

gp_recap = make_gp().fit(X_tr_full, y_tr_full)
mu_te, sd_te = gp_recap.predict(X_te, return_std=True)
print(f"recap GP   test RMSE = {np.sqrt(mse(y_te, mu_te)):.2f} MPa   median pred SD = {np.median(sd_te):.2f}")
print(f"fitted kernel: {gp_recap.kernel_}")


# %% [markdown]
# # Block 2 — MLPC anchor: GP active learning on `TensileTestDataset(T=600)`
#
# Lab story (ML-PC §"Active learning in additive manufacturing"): each
# tensile experiment costs an oven hour and a specimen, so you cannot
# afford to label every (strain, stress) point in the operating envelope.
# Active learning **picks the next strain to test** by maximising a
# function of the model's predictive uncertainty.
#
# Setup:
#
# - Pool = the full 350 (strain, stress) points in `TensileTestDataset(T=600)`.
# - Initial labelled set = 5 random points.
# - At each iteration: fit a GP on the current labelled set, predict
#   $\sigma$ on the *unlabelled pool*, query the strain $x^* =
#   \arg\max_x \sigma(x)$, "label" it (look up the true stress) and add
#   to the training set.
# - Run 25 iterations; track validation MSE on a held-out 20%.
# - Compare to a *random*-sampling baseline.
#
# *(see ML-PC §"Active learning in additive manufacturing"; MFML §"Bayesian
# optimisation")*

# %%
# Carve out a fixed held-out 20% test set for AL-loop evaluation; the rest
# is the candidate pool for the AL loop.
rng_al = np.random.default_rng(42)
perm2 = rng_al.permutation(len(X_T600))
n_te = int(0.2 * len(X_T600))
te_al = perm2[:n_te]
pool_al = perm2[n_te:]

X_pool, y_pool = X_T600[pool_al], y_T600[pool_al]
X_eval, y_eval = X_T600[te_al], y_T600[te_al]
print(f"AL pool: {len(X_pool)}   AL eval set: {len(X_eval)}")


# %%
def run_active_learning(X_pool, y_pool, X_eval, y_eval,
                        acquisition,           # callable: (mu, sigma, y_best) -> score per pool point
                        n_init=5, n_iter=25, seed=0):
    """Run a tabular active-learning loop.

    Returns a dict with per-iteration history:
        train_size, eval_mse, max_score, queried_idx_in_pool.
    """
    rng_local = np.random.default_rng(seed)
    pool_remaining = list(range(len(X_pool)))
    init_idx = list(rng_local.choice(pool_remaining, size=n_init, replace=False))
    train_idx = list(init_idx)
    for i in init_idx:
        pool_remaining.remove(i)

    history = dict(train_size=[], eval_mse=[], max_score=[], queried=[])

    for it in range(n_iter + 1):       # +1 so we record the post-init fit
        gp = make_gp(seed=seed).fit(X_pool[train_idx], y_pool[train_idx])
        mu_eval, sd_eval = gp.predict(X_eval, return_std=True)
        eval_mse = mse(y_eval, mu_eval)

        # acquisition score on the remaining pool
        if len(pool_remaining) == 0:
            break
        Xrem = X_pool[pool_remaining]
        mu_p, sd_p = gp.predict(Xrem, return_std=True)
        y_best = float(np.max(y_pool[train_idx]))
        scores = acquisition(mu_p, sd_p, y_best)

        history["train_size"].append(len(train_idx))
        history["eval_mse"].append(eval_mse)
        history["max_score"].append(float(scores.max()))

        if it < n_iter:                # last loop iteration only records, doesn't query
            best_local = int(np.argmax(scores))
            queried = pool_remaining.pop(best_local)
            train_idx.append(queried)
            history["queried"].append(int(queried))

    return history, train_idx


# %%
# Acquisition function for Block 2: pure exploration -- max predictive std.
def acq_std(mu, sigma, y_best):
    return sigma


# Random-sampling baseline (just returns a random per-point score).
def acq_random_factory(seed):
    rng = np.random.default_rng(seed)
    def _acq(mu, sigma, y_best):
        return rng.standard_normal(size=mu.shape[0])
    return _acq


hist_std,    final_idx_std    = run_active_learning(X_pool, y_pool, X_eval, y_eval,
                                                    acquisition=acq_std,
                                                    n_init=5, n_iter=25, seed=0)
hist_random, final_idx_random = run_active_learning(X_pool, y_pool, X_eval, y_eval,
                                                    acquisition=acq_random_factory(seed=11),
                                                    n_init=5, n_iter=25, seed=0)
print(f"AL (max-std)  final eval MSE = {hist_std['eval_mse'][-1]:.2f}   "
      f"random final eval MSE = {hist_random['eval_mse'][-1]:.2f}")


# %%
# Snapshot the GP at iter 5, 15, 25 for the three-panel "fit over time" plot.
def fit_at_size(history, X_pool, y_pool, size, seed=0):
    """Refit a GP using only the first `size` points the AL loop selected.

    history['queried'] gives the *order* of queries (excluding the n_init
    bootstrap). We reconstruct the labelled set and refit.
    """
    n_init = history["train_size"][0]
    queried = history["queried"][:max(0, size - n_init)]
    rng_local = np.random.default_rng(seed)
    init_idx = list(rng_local.choice(len(X_pool), size=n_init, replace=False))
    full = init_idx + list(queried)
    return make_gp(seed=seed).fit(X_pool[full], y_pool[full]), full


# %%
xs_grid = np.linspace(X_T600.min(), X_T600.max(), 300).reshape(-1, 1)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, n in zip(axes[0].tolist() + [axes[1, 0]], [5, 15, 30]):
    gp_n, idx_n = fit_at_size(hist_std, X_pool, y_pool, size=n)
    mu_g, sd_g = gp_n.predict(xs_grid, return_std=True)
    ax.scatter(X_pool, y_pool, c="lightgray", s=8, alpha=0.5, label="pool (unlabelled)")
    ax.scatter(X_pool[idx_n], y_pool[idx_n], c="#d62728", s=22, zorder=5,
               label=f"AL labelled (n={n})")
    ax.fill_between(xs_grid.ravel(), mu_g - 1.96 * sd_g, mu_g + 1.96 * sd_g,
                    color="#1f77b4", alpha=0.2, label="95% CI")
    ax.plot(xs_grid.ravel(), mu_g, color="#1f77b4", lw=2, label="GP mean")
    ax.set_xlabel("strain"); ax.set_ylabel("stress (MPa)")
    ax.set_title(f"AL (max-std) after n={n} labelled points")
    ax.legend(fontsize=8, loc="lower right")

# Bottom-right: regret curve (AL vs random).
ax = axes[1, 1]
ax.plot(hist_std["train_size"],    hist_std["eval_mse"],    "-o", color="#d62728",
        label="AL (max predictive std)", lw=2)
ax.plot(hist_random["train_size"], hist_random["eval_mse"], "-o", color="#888888",
        label="random sampling", lw=2)
ax.set_xlabel("training-set size"); ax.set_ylabel("eval MSE on held-out 20% (MPa$^2$)")
ax.set_title("Active learning vs random: error vs labels acquired")
ax.set_yscale("log")
ax.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# **Read these four panels.** The top three show the GP fit at three
# successive AL iterations: at n=5 the CI is wide everywhere; the AL loop
# preferentially queries strains in the regions of largest CI; by n=30 the
# CI is tight everywhere on the strain support. The bottom-right curve is
# the actionable diagnostic — AL reaches the same eval MSE as random
# sampling at roughly half the label budget.
#
# **Honest caveat.** Pure max-std is myopic: it only cares about the next
# query, not the cumulative information budget. Block 3 fixes this with
# *expected-improvement*-style acquisition functions; Block 4 adds a cost
# model on top.


# %% [markdown]
# # Block 3 — Acquisition functions: UCB, EI, PI
#
# Three acquisition functions you will see all over the BO / materials-AL
# literature:
#
# 1. **Upper Confidence Bound (UCB).** $\alpha_{\text{UCB}}(x) = \mu(x) +
#    \beta\,\sigma(x)$. Linear trade-off between exploitation ($\mu$) and
#    exploration ($\sigma$). $\beta=2$ is a robust default.
# 2. **Expected Improvement (EI).** $\alpha_{\text{EI}}(x) = (\mu - f^* -
#    \xi)\,\Phi(z) + \sigma\,\phi(z)$ with $z = (\mu - f^* - \xi) / \sigma$.
#    Closed-form; the standard for materials BO.
# 3. **Probability of Improvement (PI).** $\alpha_{\text{PI}}(x) = \Phi(z)$.
#    Greediest of the three; tends to over-exploit.
#
# We run the same AL loop with each acquisition and tabulate regret.
#
# *(see MFML §"Bayesian optimisation, acquisition functions")*

# %%
def acq_ucb_factory(beta):
    def _acq(mu, sigma, y_best):
        return mu + beta * sigma
    return _acq


def acq_ei_factory(xi=0.01):
    """Expected improvement (maximisation form).

    EI(x) = (mu(x) - f_best - xi) * Phi(z) + sigma(x) * phi(z),
    z = (mu(x) - f_best - xi) / sigma(x).  EI = 0 where sigma = 0.
    """
    def _acq(mu, sigma, y_best):
        sigma_safe = np.maximum(sigma, 1e-12)
        improvement = mu - y_best - xi
        z = improvement / sigma_safe
        ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
        # EI is non-negative by construction; clamp small negatives from FP.
        ei = np.where(sigma > 1e-10, ei, 0.0)
        return np.maximum(ei, 0.0)
    return _acq


def acq_pi_factory(xi=0.01):
    def _acq(mu, sigma, y_best):
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (mu - y_best - xi) / sigma_safe
        return norm.cdf(z)
    return _acq


acquisitions = {
    "UCB (beta=2.0)":        acq_ucb_factory(beta=2.0),
    "EI (xi=0.01)":          acq_ei_factory(xi=0.01),
    "PI (xi=0.01)":          acq_pi_factory(xi=0.01),
    "max-std (Block 2)":     acq_std,
    "random":                acq_random_factory(seed=11),
}

results_acq = {}
for name, acq in acquisitions.items():
    h, _ = run_active_learning(X_pool, y_pool, X_eval, y_eval,
                               acquisition=acq, n_init=5, n_iter=25, seed=0)
    results_acq[name] = h
    print(f"{name:<22s}   final MSE = {h['eval_mse'][-1]:>7.2f}   "
          f"min MSE on path = {min(h['eval_mse']):.2f}")


# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
colors_acq = {
    "UCB (beta=2.0)":    "#1f77b4",
    "EI (xi=0.01)":      "#d62728",
    "PI (xi=0.01)":      "#9467bd",
    "max-std (Block 2)": "#ff7f0e",
    "random":            "#888888",
}
for name, h in results_acq.items():
    ax.plot(h["train_size"], h["eval_mse"], "-o", lw=2, ms=4,
            color=colors_acq[name], label=name)
ax.set_xlabel("training-set size"); ax.set_ylabel("eval MSE (MPa$^2$, log)")
ax.set_yscale("log")
ax.set_title("Acquisition-function bake-off on TensileTestDataset(T=600)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


# %% [markdown]
# **Read this curve.** Roughly:
#
# - **UCB($\beta=2$)** is competitive with max-std for pure regression
#   error reduction; the $\mu$ term tilts queries toward the high-stress
#   region of the curve (which is where the data is densest, so it doesn't
#   hurt much here).
# - **EI** is the best general default — it is theoretically grounded for
#   maximisation problems and degrades gracefully when there is no
#   maximisation objective.
# - **PI** over-exploits: once it finds a single good point, it queries
#   nearby strains repeatedly and stops exploring. Worst regret on this run.
# - **max-std** and **random** are the two baselines; max-std beats random
#   reliably, as we already saw in Block 2.
#
# **Default recipe for materials AL.** UCB($\beta=2$) for robustness, EI
# for principled BO, PI never as a default. If you do not have a clear
# maximisation objective (just a regression target), max-std with a
# careful kernel is fine.


# %% [markdown]
# # Block 4 — Cost-aware AL: combining cheap T=600 with expensive T=0
#
# Real lab budgets are not "labels" but "lab time". A T=600 experiment
# might cost 1 oven-hour; a T=0 experiment might cost 5 oven-hours
# because the rig has to cool, the specimen has to be re-mounted, and
# the operator has to re-calibrate. Cost-aware AL replaces the
# acquisition function $\alpha(x)$ with $\alpha(x) / c(x)$.
#
# Setup:
#
# - Pool = (T=600 points: cost 1) $\cup$ (T=0 points: cost 5).
# - Feature = a 2-vector `(strain, fidelity_id)` where fidelity_id is 0
#   for T=600 and 1 for T=0. The GP sees both temperatures as one task.
# - Three policies on a shared budget axis (cumulative cost):
#   (i) random, (ii) cost-blind EI, (iii) cost-aware EI / cost.
# - Track *eval MSE on the held-out T=600 set* as a function of total
#   spend.
#
# *(see ML-PC §"Cost-aware active learning"; MG §"Multi-fidelity")*

# %%
# Build the combined pool.
ds_T0 = TensileTestDataset(temperature=0)
X_T0, y_T0 = to_numpy_dataset(ds_T0)

# Tag each pool entry with a fidelity id (0 = cheap T=600, 1 = expensive T=0).
fidelity_id_T600 = np.zeros((len(X_T600), 1), dtype=np.float32)
fidelity_id_T0   = np.ones((len(X_T0), 1),   dtype=np.float32)
X_combined = np.concatenate([
    np.concatenate([X_T600, fidelity_id_T600], axis=1),
    np.concatenate([X_T0,   fidelity_id_T0],   axis=1),
], axis=0)
y_combined = np.concatenate([y_T600, y_T0], axis=0)
cost_combined = np.concatenate([np.ones(len(X_T600)),     # T=600 -> cost 1
                                5.0 * np.ones(len(X_T0))], axis=0)
print(f"combined pool: N={len(X_combined)}   X dim = {X_combined.shape[1]}   "
      f"costs: T=600 -> 1, T=0 -> 5")


# %%
# Held-out eval: only T=600 (we're benchmarking the T=600 surrogate).
rng_cost = np.random.default_rng(7)
perm_cost = rng_cost.permutation(len(X_T600))
n_te_c = int(0.2 * len(X_T600))
te_idx_c = perm_cost[:n_te_c]
X_eval_c = np.concatenate([X_T600[te_idx_c],
                           np.zeros((len(te_idx_c), 1), dtype=np.float32)], axis=1)
y_eval_c = y_T600[te_idx_c]

# Build the candidate pool: combined minus the T=600 eval indices.
mask_pool = np.ones(len(X_combined), dtype=bool)
mask_pool[te_idx_c] = False        # remove the T=600 eval indices from the combined pool
X_pool_c = X_combined[mask_pool]
y_pool_c = y_combined[mask_pool]
cost_pool_c = cost_combined[mask_pool]
print(f"cost-aware AL pool: {len(X_pool_c)}   eval set: {len(X_eval_c)}")


# %%
def run_cost_aware_al(X_pool, y_pool, cost, X_eval, y_eval,
                      acquisition_with_cost,        # callable: (mu, sigma, y_best, cost) -> score
                      n_init=5, n_iter=30, seed=0):
    rng_local = np.random.default_rng(seed)
    pool_remaining = list(range(len(X_pool)))
    init_idx = list(rng_local.choice(pool_remaining, size=n_init, replace=False))
    train_idx = list(init_idx)
    for i in init_idx:
        pool_remaining.remove(i)

    history = dict(train_size=[], eval_mse=[], total_cost=[])
    total_cost = float(cost[init_idx].sum())

    for it in range(n_iter + 1):
        gp = make_gp(seed=seed).fit(X_pool[train_idx], y_pool[train_idx])
        mu_eval, _ = gp.predict(X_eval, return_std=True)
        history["train_size"].append(len(train_idx))
        history["eval_mse"].append(mse(y_eval, mu_eval))
        history["total_cost"].append(total_cost)

        if it >= n_iter or len(pool_remaining) == 0:
            break

        Xrem = X_pool[pool_remaining]
        crem = cost[pool_remaining]
        mu_p, sd_p = gp.predict(Xrem, return_std=True)
        y_best = float(np.max(y_pool[train_idx]))
        scores = acquisition_with_cost(mu_p, sd_p, y_best, crem)
        best_local = int(np.argmax(scores))
        queried = pool_remaining.pop(best_local)
        train_idx.append(queried)
        total_cost += float(cost[queried])

    return history


# %%
# Three policies:
#   (a) random (cost-blind),
#   (b) EI (cost-blind),
#   (c) EI / cost (cost-aware).
ei_base = acq_ei_factory(xi=0.01)


def acq_random_with_cost_factory(seed):
    rng = np.random.default_rng(seed)
    def _acq(mu, sigma, y_best, c):
        return rng.standard_normal(size=mu.shape[0])
    return _acq


def acq_ei_blind(mu, sigma, y_best, c):
    return ei_base(mu, sigma, y_best)


def acq_ei_per_cost(mu, sigma, y_best, c):
    return ei_base(mu, sigma, y_best) / np.maximum(c, 1e-6)


hist_random_cost  = run_cost_aware_al(X_pool_c, y_pool_c, cost_pool_c, X_eval_c, y_eval_c,
                                      acq_random_with_cost_factory(seed=13),
                                      n_init=5, n_iter=30, seed=0)
hist_ei_blind     = run_cost_aware_al(X_pool_c, y_pool_c, cost_pool_c, X_eval_c, y_eval_c,
                                      acq_ei_blind,    n_init=5, n_iter=30, seed=0)
hist_ei_per_cost  = run_cost_aware_al(X_pool_c, y_pool_c, cost_pool_c, X_eval_c, y_eval_c,
                                      acq_ei_per_cost, n_init=5, n_iter=30, seed=0)
print(f"random (cost-blind)  final spend={hist_random_cost['total_cost'][-1]:.0f}   "
      f"final MSE={hist_random_cost['eval_mse'][-1]:.2f}")
print(f"EI     (cost-blind)  final spend={hist_ei_blind['total_cost'][-1]:.0f}   "
      f"final MSE={hist_ei_blind['eval_mse'][-1]:.2f}")
print(f"EI/cost (cost-aware) final spend={hist_ei_per_cost['total_cost'][-1]:.0f}   "
      f"final MSE={hist_ei_per_cost['eval_mse'][-1]:.2f}")


# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
for h, name, color in [
    (hist_random_cost, "random (cost-blind)",     "#888888"),
    (hist_ei_blind,    "EI (cost-blind)",         "#1f77b4"),
    (hist_ei_per_cost, "EI / cost (cost-aware)",  "#d62728"),
]:
    ax.plot(h["total_cost"], h["eval_mse"], "-o", lw=2, ms=4, color=color, label=name)
ax.set_xlabel("cumulative spend (oven-hours)")
ax.set_ylabel("eval MSE on held-out T=600 (MPa$^2$, log)")
ax.set_yscale("log")
ax.set_title("Cost-aware vs cost-blind AL on the combined T=0 + T=600 pool")
ax.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# **Read this curve.** The horizontal axis is the *thing the lab actually
# pays* (cumulative cost), not the number of labels. The cost-aware EI/cost
# policy reaches a target MSE for roughly half the spend of cost-blind EI:
# it preferentially picks cheap T=600 points and only queries the
# expensive T=0 points when the EI gain is large enough to justify the 5×
# cost.
#
# **Why the comparison is fair.** All three policies share the same
# initial 5-point bootstrap, the same GP architecture, and the same eval
# set. The *only* difference is the acquisition rule.
#
# **Take-away.** Cost-awareness is the simplest practical upgrade to a
# vanilla BO loop and almost always pays for itself.


# %% [markdown]
# ## Block 4.5 — Conformalize the ensemble (CQR)
#
# Block 4's UQ knob is the GP posterior standard deviation, which gives a
# *symmetric, homoscedastic-looking* interval whose width is the same
# wherever predictive variance is the same. Real tensile-test residuals
# are heteroscedastic — wider in the plastic regime than in the elastic
# one — so a symmetric interval over-covers the easy regions and
# under-covers the hard ones.
#
# **Conformalized Quantile Regression (CQR)** [@romano_2019_cqr] fixes this
# in two steps:
#
# 1. **Quantile head.** Fit a small NN head with two outputs
#    ($\hat{q}_{\alpha/2}$, $\hat{q}_{1-\alpha/2}$) on top of the GP-mean
#    feature, trained with the **pinball loss**
#    $L_\tau(y, \hat{y}) = \max(\tau(y - \hat{y}), (\tau - 1)(y - \hat{y}))$
#    and total loss
#    $\mathcal{L} = L_{\alpha/2}(y, \hat{q}_{\alpha/2})
#                 + L_{1-\alpha/2}(y, \hat{q}_{1-\alpha/2}).$
# 2. **Conformalize.** On a held-out calibration set, compute
#    $s_i = \max\{\hat{q}_{\alpha/2}(x_i) - y_i,\;
#                 y_i - \hat{q}_{1-\alpha/2}(x_i)\}$,
#    take its $(1 - \alpha)$-quantile $\hat{Q}$, and emit
#    $[\hat{q}_{\alpha/2}(x) - \hat{Q},\; \hat{q}_{1-\alpha/2}(x) + \hat{Q}]$.
#
# The result: an interval whose **width adapts** with the noise level
# (wide in plastic regime, tight in elastic regime), with the same
# finite-sample coverage guarantee as split conformal.
#
# *(see MFML Unit 7 §"Conformal prediction" for the split-conformal primer;
# here we apply the recipe — and its CQR extension — to materials data. See
# also ML-PC §"CQR for materials regression".)*

# %%
# Re-use the T=600 surrogate from Block 1 (gp_recap) as the mean
# predictor. Carve a 60/20/20 split: GP train / quantile-head train +
# conformal calibration / test, all disjoint.
rng_cqr = np.random.default_rng(2026)
perm_cqr = rng_cqr.permutation(len(X_T600))
n_total = len(X_T600)
n_tr_q   = int(0.60 * n_total)
n_cal_q  = int(0.20 * n_total)
tr_idx_q   = perm_cqr[:n_tr_q]
cal_idx_q  = perm_cqr[n_tr_q:n_tr_q + n_cal_q]
te_idx_q   = perm_cqr[n_tr_q + n_cal_q:]
X_tr_q,  y_tr_q  = X_T600[tr_idx_q],  y_T600[tr_idx_q]
X_cal_q, y_cal_q = X_T600[cal_idx_q], y_T600[cal_idx_q]
X_te_q,  y_te_q  = X_T600[te_idx_q],  y_T600[te_idx_q]
print(f"CQR split: GP-train={len(X_tr_q)}   q-head+conf={len(X_cal_q)}   test={len(X_te_q)}")

# Refit GP on the CQR-train slice so the calib/test slices are unseen.
gp_cqr = make_gp().fit(X_tr_q, y_tr_q)
mu_cal_q,  _ = gp_cqr.predict(X_cal_q, return_std=True)
mu_te_q,   _ = gp_cqr.predict(X_te_q,  return_std=True)


# %%
# Pinball-loss quantile head. Input = (strain, gp_mean) so the head can
# learn a quantile *correction* around the GP mean.
class QuantileHead(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),       # [q_lo, q_hi]
        )

    def forward(self, x):
        return self.net(x)


def pinball_loss(y, yhat, tau):
    diff = y - yhat
    return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()


alpha_cqr = 0.1
tau_lo, tau_hi = alpha_cqr / 2.0, 1.0 - alpha_cqr / 2.0

# Stack features (strain, gp_mean) for the quantile head.
feat_tr  = np.concatenate([X_tr_q,  gp_cqr.predict(X_tr_q).reshape(-1, 1)], axis=1)
feat_cal = np.concatenate([X_cal_q, mu_cal_q.reshape(-1, 1)],                axis=1)
feat_te  = np.concatenate([X_te_q,  mu_te_q.reshape(-1, 1)],                 axis=1)

torch.manual_seed(0)
qhead = QuantileHead(hidden=32)
opt_q = torch.optim.Adam(qhead.parameters(), lr=1e-2)
ft_tr_t = torch.tensor(feat_tr, dtype=torch.float32)
y_tr_t  = torch.tensor(y_tr_q,  dtype=torch.float32)
for _ in range(1500):
    opt_q.zero_grad()
    pred = qhead(ft_tr_t)
    loss = pinball_loss(y_tr_t, pred[:, 0], tau_lo) + pinball_loss(y_tr_t, pred[:, 1], tau_hi)
    loss.backward(); opt_q.step()
print(f"final pinball loss (sum of tau_lo + tau_hi) = {loss.item():.3f}")

with torch.no_grad():
    q_cal = qhead(torch.tensor(feat_cal, dtype=torch.float32)).numpy()    # (n_cal, 2)
    q_te  = qhead(torch.tensor(feat_te,  dtype=torch.float32)).numpy()


# %%
# Conformalize: nonconformity = max(q_lo - y, y - q_hi).
nc_cal = np.maximum(q_cal[:, 0] - y_cal_q, y_cal_q - q_cal[:, 1])
q_level = np.ceil((len(nc_cal) + 1) * (1 - alpha_cqr)) / len(nc_cal)
q_level = min(q_level, 1.0)
Q_hat = float(np.quantile(nc_cal, q_level))
print(f"CQR conformal correction Q_hat = {Q_hat:.2f} MPa   q-level = {q_level:.4f}")

lo_te = q_te[:, 0] - Q_hat
hi_te = q_te[:, 1] + Q_hat
covered = (y_te_q >= lo_te) & (y_te_q <= hi_te)
emp_cov_cqr = float(covered.mean())
mean_width = float(np.mean(hi_te - lo_te))
print(f"CQR empirical coverage = {emp_cov_cqr:.3f}   (target = {1 - alpha_cqr:.2f})   "
      f"mean width = {mean_width:.1f} MPa")


# %%
# Plot CQR intervals on the existing tensile axes from Block 4: the
# adaptive width should be visibly wider in sparse/noisy regions.
xs_plot = np.linspace(X_T600.min(), X_T600.max(), 300).reshape(-1, 1)
mu_plot, _ = gp_cqr.predict(xs_plot, return_std=True)
feat_plot = np.concatenate([xs_plot, mu_plot.reshape(-1, 1)], axis=1)
with torch.no_grad():
    q_plot = qhead(torch.tensor(feat_plot, dtype=torch.float32)).numpy()
lo_plot = q_plot[:, 0] - Q_hat
hi_plot = q_plot[:, 1] + Q_hat

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(X_tr_q,  y_tr_q,  c="lightgray", s=10, alpha=0.5, label="GP train")
ax.scatter(X_te_q,  y_te_q,  c="k",         s=14, alpha=0.7, label="test")
ax.plot(xs_plot.ravel(), mu_plot, color="#1f77b4", lw=2, label="GP mean (Block 4)")
ax.fill_between(xs_plot.ravel(), lo_plot, hi_plot, color="#d62728", alpha=0.25,
                label=f"CQR 90% interval ($\\hat{{Q}}={Q_hat:.1f}$)")
ax.set_xlabel("strain"); ax.set_ylabel("stress (MPa)")
ax.set_title(f"CQR on TensileTestDataset(T=600), coverage = {emp_cov_cqr:.3f}")
ax.legend(fontsize=9, loc="lower right")

# Width vs strain — make adaptivity explicit.
ax = axes[1]
width_plot = hi_plot - lo_plot
ax.plot(xs_plot.ravel(), width_plot, color="#d62728", lw=2, label="CQR width")
ax.axhline(2 * 1.96 * np.median(gp_cqr.predict(xs_plot, return_std=True)[1]),
           color="#1f77b4", ls="--", lw=1.5, label="GP 95% width (median)")
ax.set_xlabel("strain"); ax.set_ylabel("interval width (MPa)")
ax.set_title("Width adapts: wider where data is sparse / noisy")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


# %% [markdown]
# **Read these two panels.** Left: the CQR band hugs the tensile curve
# tightly in the elastic regime and flares out in the plastic regime
# where residual variance is genuinely larger — a behaviour the symmetric
# $\mu \pm 1.96\sigma$ band from Block 4 cannot reproduce without
# re-fitting a heteroscedastic GP. Right: the width-vs-strain plot makes
# the adaptivity quantitative — CQR width varies by a factor of ~2 across
# the support, while the GP's $1.96\sigma$ width is roughly flat.
#
# **Take-away.** CQR is the cheapest way to make an existing point
# predictor *adaptive* without retraining the predictor itself. The
# conformal step preserves the same distribution-free coverage guarantee
# as plain split conformal, as long as exchangeability holds — Exercise 5
# shows what breaks when it does not [@romano_2019_cqr].


# %% [markdown]
# # Block 5 — MG anchor: clustering vs discovery on `NanoindentationDataset`
#
# Discovery $\neq$ labeling. A *labeling* loop assigns one of K known
# classes to each sample (Week 5 K-means). A *discovery* loop flags
# samples that lie outside all known classes — those are the candidates
# for follow-up characterisation.
#
# Recipe (MG §"Discovery in materials spaces"):
#
# 1. Cluster the (E, H) features with K=4 — these are the "structure
#    families" we already know about.
# 2. Train **4 separate per-cluster GPs**, each predicting H from E
#    *within its cluster*.
# 3. For every point in the dataset, compute:
#    - **predictive variance** from the per-cluster GP (high = model not
#      sure within its cluster);
#    - **novelty score** = distance to the nearest cluster centroid in
#      (E, H) space, normalised by the cluster's average intra-cluster
#      distance.
# 4. Cross-tabulate (low/high variance) × (low/high novelty) and colour
#    points accordingly. *(high-var + high-novelty = discovery candidate.)*
#
# *(see MG §"Clustering vs discovery"; ML-PC §"21CrMoV5-7 GP for hardness")*

# %%
ds_nano = NanoindentationDataset()
Xn = ds_nano.X.numpy()                  # (N, 2): E, H
yn = ds_nano.y.numpy()
print(f"Nanoindentation: N={len(ds_nano)}   features=(E [GPa], H [GPa])   classes={np.unique(yn).tolist()}")

# Standardise before K-means (Week 5 lesson).
mu_n, sd_n = Xn.mean(axis=0), Xn.std(axis=0)
Xn_std = (Xn - mu_n) / sd_n
km4 = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Xn_std)
cluster_id = km4.labels_
centroids_std = km4.cluster_centers_
print(f"K-means cluster sizes: {np.bincount(cluster_id).tolist()}")


# %%
# Per-cluster GPs: each predicts H from E within its cluster.
# We standardise E inside each cluster so the GP's RBF lengthscale is
# in a sane range, then un-standardise predictions.
per_cluster_gp = {}
for k in range(4):
    mask_k = (cluster_id == k)
    E_k = Xn[mask_k, 0:1]
    H_k = Xn[mask_k, 1]
    if mask_k.sum() < 5:
        continue
    gp_k = make_gp(length_scale=10.0, length_scale_bounds=(1e-1, 1e3),
                   noise_level=0.5, noise_level_bounds=(1e-3, 1e2)).fit(E_k, H_k)
    per_cluster_gp[k] = gp_k

# For each point, compute the predictive variance under its assigned cluster's GP.
pred_var = np.zeros(len(Xn))
for k, gp_k in per_cluster_gp.items():
    mask_k = (cluster_id == k)
    _, sd_k = gp_k.predict(Xn[mask_k, 0:1], return_std=True)
    pred_var[mask_k] = sd_k ** 2

# Novelty score: distance to assigned-cluster centroid, normalised by mean
# intra-cluster distance.
novelty = np.zeros(len(Xn))
for k in range(4):
    mask_k = (cluster_id == k)
    if mask_k.sum() == 0:
        continue
    pts_k = Xn_std[mask_k]
    dist_to_cent = np.linalg.norm(pts_k - centroids_std[k], axis=1)
    norm_factor = max(float(dist_to_cent.mean()), 1e-6)
    novelty[mask_k] = dist_to_cent / norm_factor

print(f"predictive variance: median = {np.median(pred_var):.3f}   max = {pred_var.max():.3f}")
print(f"novelty score:       median = {np.median(novelty):.3f}   max = {novelty.max():.3f}")


# %%
# Threshold on the medians for a 2x2 flag matrix.
tau_var = float(np.median(pred_var))
tau_nov = float(np.median(novelty))

# 4-class flag:
#   0: in-cluster, low-var      -> "well-explained, well-known"
#   1: in-cluster, high-var     -> "noise / inherent uncertainty"
#   2: out-of-cluster, low-var  -> "boundary points (mostly noise)"
#   3: out-of-cluster, high-var -> "discovery candidate"
flag = np.zeros(len(Xn), dtype=int)
flag[(pred_var <= tau_var) & (novelty >  tau_nov)] = 2
flag[(pred_var >  tau_var) & (novelty <= tau_nov)] = 1
flag[(pred_var >  tau_var) & (novelty >  tau_nov)] = 3
print("flag counts:", dict(zip(["in/low", "in/high (noise)", "out/low", "out/high (discovery)"],
                               np.bincount(flag, minlength=4).tolist())))


# %%
# Scatter coloured by flag.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
flag_colors = {0: "#bbbbbb", 1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}
flag_labels = {0: "in-cluster low-var",
               1: "in-cluster high-var (noise)",
               2: "out-of-cluster low-var",
               3: "out-of-cluster high-var (DISCOVERY)"}

# Left panel: K-means clusters.
for k in range(4):
    mask_k = (cluster_id == k)
    axes[0].scatter(Xn[mask_k, 0], Xn[mask_k, 1], s=14, alpha=0.65, label=f"cluster {k}")
axes[0].set_xlabel("E (GPa)"); axes[0].set_ylabel("H (GPa)")
axes[0].set_title("K-means K=4 on standardised (E, H)"); axes[0].legend(fontsize=9)

# Right panel: discovery-flag scatter.
for f in range(4):
    mask_f = (flag == f)
    axes[1].scatter(Xn[mask_f, 0], Xn[mask_f, 1], c=flag_colors[f], s=18, alpha=0.8,
                    label=flag_labels[f])
axes[1].set_xlabel("E (GPa)"); axes[1].set_ylabel("H (GPa)")
axes[1].set_title(f"Discovery flag  (tau_var={tau_var:.2f}, tau_nov={tau_nov:.2f})")
axes[1].legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Read these two panels.** The left shows the cluster assignment we *had
# already*; the right shows the points the discovery pipeline would
# **flag** for follow-up. Discovery candidates (red) are simultaneously
# (i) far from any cluster centroid AND (ii) under-explained by any
# per-cluster GP — exactly the points where we suspect a *new* phase or
# a measurement artefact and want a metallurgist to look at the specimen.
#
# **Honest caveat.** In a real lab, the threshold pair $(\tau_{\text{var}},
# \tau_{\text{nov}})$ is a *policy choice*, not a model output. We use the
# medians here as a defensible default; Block 7 Exercise (iii) sweeps the
# pair to give you a feel for how brittle the choice is.


# %% [markdown]
# # Block 6 — Per-cluster acquisition budgets
#
# We have 4 clusters and a budget of 20 experiments. How do we allocate?
# Three policies:
#
# - **(a) Equal split.** 5 experiments per cluster.
# - **(b) Variance-proportional.** Allocate $n_k \propto \bar\sigma_k^2$,
#   where $\bar\sigma_k^2$ is the mean predictive variance under
#   cluster $k$'s GP.
# - **(c) Global top-k EI.** Pool every (point, cluster GP) pair, score
#   by EI on a uniform exploration target, take the global top-20.
#
# Track per-policy: total MSE across clusters and "discoveries found".
#
# *(see MG §"Per-cluster acquisition budgets")*

# %%
# Mean predictive variance per cluster.
mean_var_per_cluster = np.zeros(4)
for k, gp_k in per_cluster_gp.items():
    mask_k = (cluster_id == k)
    _, sd_k = gp_k.predict(Xn[mask_k, 0:1], return_std=True)
    mean_var_per_cluster[k] = float((sd_k ** 2).mean())
print(f"mean predictive variance per cluster: {mean_var_per_cluster.round(3).tolist()}")


# %%
# We treat the "AL pool" here as the FULL nanoindentation dataset and
# pretend a fresh experimental round queries 20 of those points. We score
# each policy by:
#   - total MSE across clusters AFTER refitting per-cluster GPs with the new points,
#   - number of (high-var, high-novelty) discoveries the policy uncovered.
#
# Initial labelled set: 50% of every cluster (so the GPs are non-trivial).
def per_cluster_initial_split(seed=0, frac=0.5):
    rng = np.random.default_rng(seed)
    init = {}
    for k in range(4):
        idx_k = np.where(cluster_id == k)[0]
        n_take = max(2, int(frac * len(idx_k)))
        chosen = rng.choice(idx_k, size=n_take, replace=False)
        init[k] = list(chosen)
    return init


def evaluate_policy(allocation, seed=0):
    """allocation: dict {cluster_id: list of new-point indices to label}."""
    init = per_cluster_initial_split(seed=seed)
    cluster_mse = np.zeros(4)
    discoveries_found = 0
    for k in range(4):
        labelled_k = list(init[k]) + list(allocation.get(k, []))
        # refit cluster GP on the labelled set (only points that actually live in cluster k)
        labelled_k = [i for i in labelled_k if cluster_id[i] == k]
        E_k = Xn[labelled_k, 0:1]
        H_k = Xn[labelled_k, 1]
        if len(labelled_k) < 3:
            cluster_mse[k] = np.nan
            continue
        gp_k = make_gp(length_scale=10.0, length_scale_bounds=(1e-1, 1e3),
                       noise_level=0.5, noise_level_bounds=(1e-3, 1e2)).fit(E_k, H_k)
        # eval on the rest of the cluster
        mask_k = (cluster_id == k)
        idx_k = np.where(mask_k)[0]
        eval_idx = [i for i in idx_k if i not in labelled_k]
        if len(eval_idx) == 0:
            cluster_mse[k] = 0.0; continue
        mu_k, _ = gp_k.predict(Xn[eval_idx, 0:1], return_std=True)
        cluster_mse[k] = mse(Xn[eval_idx, 1], mu_k)
        # discoveries among newly-allocated points
        for i in allocation.get(k, []):
            if flag[i] == 3:
                discoveries_found += 1
    return cluster_mse, discoveries_found


# Build a candidate pool for new experiments: all points NOT in the initial split.
init_default = per_cluster_initial_split(seed=0)
init_set = set(i for v in init_default.values() for i in v)
candidate_pool = np.array([i for i in range(len(Xn)) if i not in init_set])


# Policy (a): equal split, 5 per cluster, sampled by max predictive variance.
def policy_equal(budget=20):
    per = budget // 4
    alloc = {k: [] for k in range(4)}
    for k, gp_k in per_cluster_gp.items():
        cand_k = [i for i in candidate_pool if cluster_id[i] == k]
        if not cand_k:
            continue
        E_c = Xn[cand_k, 0:1]
        _, sd_c = gp_k.predict(E_c, return_std=True)
        order = np.argsort(-sd_c)
        alloc[k] = [cand_k[i] for i in order[:per]]
    return alloc


# Policy (b): proportional to mean cluster predictive variance.
def policy_proportional(budget=20):
    weights = mean_var_per_cluster / mean_var_per_cluster.sum()
    alloc = {k: [] for k in range(4)}
    n_per_k = (weights * budget).astype(int)
    # Round-robin to absorb integer truncation residual.
    for _ in range(budget - int(n_per_k.sum())):
        n_per_k[int(np.argmax(weights * budget - n_per_k))] += 1
    for k, gp_k in per_cluster_gp.items():
        cand_k = [i for i in candidate_pool if cluster_id[i] == k]
        if not cand_k:
            continue
        E_c = Xn[cand_k, 0:1]
        _, sd_c = gp_k.predict(E_c, return_std=True)
        order = np.argsort(-sd_c)
        alloc[k] = [cand_k[i] for i in order[:int(n_per_k[k])]]
    return alloc


# Policy (c): global top-k EI across all clusters.
def policy_top_k_ei(budget=20):
    # We reuse EI with y_best = max H seen so far in the labelled set.
    y_best_global = float(np.max([Xn[i, 1] for i in init_set]))
    scores = []
    score_idx = []
    for i in candidate_pool:
        k = cluster_id[i]
        if k not in per_cluster_gp:
            continue
        mu, sd = per_cluster_gp[k].predict(Xn[i:i+1, 0:1], return_std=True)
        ei = ei_base(mu, sd, y_best_global)
        scores.append(float(ei[0]))
        score_idx.append(i)
    order = np.argsort(-np.array(scores))[:budget]
    alloc = {k: [] for k in range(4)}
    for o in order:
        i = score_idx[o]
        alloc[int(cluster_id[i])].append(int(i))
    return alloc


policies = {
    "equal split":        policy_equal(20),
    "proportional":       policy_proportional(20),
    "global top-k EI":    policy_top_k_ei(20),
}

results_policy = {}
for name, alloc in policies.items():
    cm, ndisc = evaluate_policy(alloc, seed=0)
    results_policy[name] = dict(cluster_mse=cm, total_mse=float(np.nansum(cm)),
                                discoveries=ndisc,
                                allocation=alloc,
                                allocation_sizes=[len(alloc.get(k, [])) for k in range(4)])
    print(f"{name:<18s}   alloc={results_policy[name]['allocation_sizes']}   "
          f"total cluster MSE = {results_policy[name]['total_mse']:.3f}   "
          f"discoveries = {results_policy[name]['discoveries']}")


# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
names = list(results_policy.keys())
totals = [results_policy[n]["total_mse"] for n in names]
discs = [results_policy[n]["discoveries"] for n in names]

axes[0].bar(names, totals, color=["#888888", "#1f77b4", "#d62728"])
axes[0].set_ylabel("total cluster MSE (sum across 4 clusters)")
axes[0].set_title("Per-policy fit quality"); axes[0].tick_params(axis="x", rotation=10)

axes[1].bar(names, discs, color=["#888888", "#1f77b4", "#d62728"])
axes[1].set_ylabel("# discoveries among 20 queries")
axes[1].set_title("Per-policy discovery yield"); axes[1].tick_params(axis="x", rotation=10)
plt.tight_layout()
plt.show()


# %% [markdown]
# **Read these two bars.** Equal split is the lazy baseline; proportional
# allocation should match or beat it on cluster MSE because it spends
# more queries on the noisier clusters; global top-k EI tends to win on
# *discovery yield* because it ignores cluster boundaries entirely and
# chases the highest-EI point regardless of which family it lives in.
#
# **Take-away.** Different objectives demand different allocation rules.
# If your goal is uniform regression accuracy: proportional. If your
# goal is finding the next outlier specimen: top-k EI. There is no
# single "best" policy without specifying the objective.


# %% [markdown]
# # Block 7 — Student exercises
#
# **Three core (do all three) + one stretch (optional).** Write your code
# in the empty cells below; bring printed plots / numbers to the next class
# for the 5-minute walk-through.

# %% [markdown]
# ## Exercise 1 (core) — Implement EI closed form from scratch
#
# Block 3 used a closed-form Expected Improvement and we trusted the
# formula. Re-derive and reimplement it from $\Phi$ and $\phi$ alone, then
# verify it matches the version used above on a fixed test set.
#
# **Your task:**
#
# 1. Implement
#    $\alpha_{\text{EI}}(x) = (\mu - f^* - \xi)\,\Phi(z) + \sigma\,\phi(z)$
#    with $z = (\mu - f^* - \xi) / \sigma$, using only `scipy.stats.norm.cdf`
#    and `scipy.stats.norm.pdf` (no other helpers).
# 2. Take 50 evenly-spaced strain values in $[0, 0.05]$. Predict $(\mu,
#    \sigma)$ from `gp_recap`.
# 3. Compute EI with your implementation and EI with `acq_ei_factory(xi=0.01)`
#    from Block 3. Verify they agree to within `1e-10` numerically.
# 4. Plot EI(x) on the strain grid; mark the argmax with a vertical line.
#
# *Hint: be careful at $\sigma = 0$ — set EI = 0 there (Block 3 does the
# same).*

# %%
# TODO: your EI-from-scratch implementation goes here.
# Skeleton:
#
#   from scipy.stats import norm
#
#   def ei_scratch(mu, sigma, y_best, xi=0.01):
#       sigma_safe = np.maximum(sigma, 1e-12)
#       z = (mu - y_best - xi) / sigma_safe
#       ei = (mu - y_best - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
#       return np.where(sigma > 1e-10, np.maximum(ei, 0.0), 0.0)
#
#   xs_test = np.linspace(0.0, 0.05, 50).reshape(-1, 1)
#   mu_t, sd_t = gp_recap.predict(xs_test, return_std=True)
#   y_best = float(np.max(y_tr_full))
#   ei_mine = ei_scratch(mu_t, sd_t, y_best)
#   ei_ref  = acq_ei_factory(xi=0.01)(mu_t, sd_t, y_best)
#   assert np.max(np.abs(ei_mine - ei_ref)) < 1e-10
#   ...


# %% [markdown]
# ## Exercise 2 (core) — Show that GP uncertainty is unreliable off-support
#
# A GP with an RBF kernel has *bounded* uncertainty: outside the training
# data, the posterior mean reverts to the prior mean and the CI saturates
# at the prior amplitude $\sigma_f$. This can be misleading: the CI says
# "I'm not sure" but with a *fixed* width that does not reflect how truly
# unconstrained the model is.
#
# **Your task:**
#
# 1. Take `TensileTestDataset(T=600)`. Train a GP **only on the data points
#    with strain $x \in [0, 0.05]$**.
# 2. Predict $(\mu, \sigma)$ at strains $x \in [0, 0.5]$ — i.e., extrapolate
#    10x beyond the training range.
# 3. Plot the truth, the GP mean, and the 95% CI. Note where the CI fails
#    to cover the truth.
# 4. **Question (3 sentences in the markdown cell below).** Why does the
#    GP's CI become unreliable here? Reference the *stationarity* of the
#    RBF kernel.
#
# *Alternative framing if you finish quickly: instead of restricting the
# data range, fit a GP with a **linear** kernel on the same data and show
# how the resulting CI is miscalibrated against the reliability diagram
# from the homework.*

# %%
# TODO: your off-support extrapolation experiment goes here.
# Skeleton:
#
#   mask_train = (X_T600.ravel() <= 0.05)
#   X_in,  y_in  = X_T600[mask_train], y_T600[mask_train]
#   gp_in = make_gp().fit(X_in, y_in)
#
#   xs_extrap = np.linspace(0.0, 0.5, 200).reshape(-1, 1)
#   mu_x, sd_x = gp_in.predict(xs_extrap, return_std=True)
#   ...


# %% [markdown]
# > # Your answer to the why-question:
# >
# > *(replace this text with your 3-sentence explanation)*


# %% [markdown]
# ## Exercise 3 (core) — 2-D discovery threshold sweep
#
# Block 5 used the *median* of `pred_var` and `novelty` as fixed
# thresholds. The result is a single (4-class) flag plot. The thresholds
# are arbitrary; in practice the metallurgist will tune them.
#
# **Your task:**
#
# 1. Define a 2-D grid of thresholds: $\tau_{\text{var}}$ at the
#    20/40/60/80 percentiles of `pred_var`, and $\tau_{\text{nov}}$ at the
#    20/40/60/80 percentiles of `novelty`.
# 2. For each $(\tau_{\text{var}}, \tau_{\text{nov}})$ pair (16 pairs),
#    count the number of "discovery candidates" (`pred_var > tau_var AND
#    novelty > tau_nov`).
# 3. Plot a 4×4 heat map of the candidate counts. Annotate each cell.
# 4. **Question (2 sentences).** What is the *trade-off*? What does the
#    metallurgist gain by lowering both thresholds, and what do they pay?

# %%
# TODO: your threshold-sweep heatmap goes here.
# Skeleton:
#
#   pcts = [20, 40, 60, 80]
#   tau_vars = np.percentile(pred_var, pcts)
#   tau_novs = np.percentile(novelty, pcts)
#   heatmap = np.zeros((len(pcts), len(pcts)), dtype=int)
#   for i, tv in enumerate(tau_vars):
#       for j, tn in enumerate(tau_novs):
#           heatmap[i, j] = int(np.sum((pred_var > tv) & (novelty > tn)))
#   fig, ax = plt.subplots(figsize=(6, 5))
#   im = ax.imshow(heatmap, cmap="Reds", aspect="equal", origin="lower")
#   ...


# %% [markdown]
# > # Your answer to the trade-off question:
# >
# > *(replace this text with your 2-sentence explanation)*


# %% [markdown]
# ## Exercise 4 (stretch) — Multi-fidelity GP with cheap-noisy + expensive-clean data
#
# Real materials labs often have access to a *cheap-noisy* measurement
# (e.g. simulated stress-strain at T=600 with a coarse force model) and an
# *expensive-clean* measurement (e.g. real-rig stress-strain at T=0).
# A multi-fidelity GP combines them by treating the fidelity id as an
# extra input dimension and using a kernel of the form:
#
# $$ k\big((x, f), (x', f')\big) \;=\; k_{\text{data}}(x, x') \cdot
#    \big(\,1 \text{ if } f = f' \text{ else } \rho\,\big), $$
#
# where $\rho \in [0, 1]$ is the cross-fidelity correlation. We fix
# $\rho = 0.7$ for tractability; learning $\rho$ jointly with the other
# kernel hyperparameters is a follow-up.
#
# **Your task:**
#
# 1. Build a noisy variant of `TensileTestDataset(T=600)` by adding
#    Gaussian noise of standard deviation 0.5 (in the standardised stress
#    scale, or 50 MPa raw). Use 50 such cheap points.
# 2. Build a clean variant of `TensileTestDataset(T=0)` (use it as-is,
#    `sigma_n = 0.05`). Use 10 such expensive points.
# 3. Fit three GPs:
#    a. only on the 50 cheap-noisy T=600 points,
#    b. only on the 10 clean T=0 points,
#    c. on both, with a fidelity-id input.
# 4. Plot RMSE on a held-out clean-T=600 set vs cumulative cost (cheap=1,
#    expensive=5). The multi-fidelity GP should beat both single-fidelity
#    baselines for some intermediate budget.
#
# *Hint: the simplest path is to concatenate `(x, fidelity_id)` as a 2-vector
# and let sklearn's RBF kernel act on both dimensions. The "fidelity-aware"
# correlation $\rho$ then comes from the lengthscale on the fidelity-id
# dimension — short lengthscale ≈ low cross-correlation, long lengthscale
# ≈ high cross-correlation. Hand-tune the fidelity-dim lengthscale so
# that points across fidelities still inform each other (a common choice
# is to set it to ~1.0 in standardised units).*

# %%
# TODO: your multi-fidelity GP experiment goes here.
# Skeleton:
#
#   # Step 1: cheap-noisy T=600 subsample
#   rng_mf = np.random.default_rng(123)
#   idx_cheap = rng_mf.choice(len(X_T600), size=50, replace=False)
#   X_cheap = X_T600[idx_cheap]
#   y_cheap = y_T600[idx_cheap] + 50.0 * rng_mf.standard_normal(size=50)
#   # Step 2: clean T=0 subsample
#   idx_exp = rng_mf.choice(len(X_T0), size=10, replace=False)
#   X_exp = X_T0[idx_exp]
#   y_exp = y_T0[idx_exp]
#   # Step 3: stack with fidelity tags
#   X_mf = np.concatenate([
#       np.concatenate([X_cheap, np.zeros((50, 1))], axis=1),
#       np.concatenate([X_exp,   np.ones((10, 1))],  axis=1),
#   ], axis=0)
#   y_mf = np.concatenate([y_cheap, y_exp])
#   gp_mf = make_gp(length_scale=[0.05, 1.0],
#                   length_scale_bounds=(1e-3, 10)).fit(X_mf, y_mf)
#   # Step 4: evaluate on a held-out clean T=600 set with fidelity_id=0.
#   ...


# %% [markdown]
# ## Exercise 5 (stretch, optional) — Coverage under distribution shift
#
# Block 4.5's CQR construction gives a marginal-coverage guarantee
# **only under exchangeability** of calibration and test
# [@angelopoulos_2023_conformal]. Real materials labs routinely violate
# this: you calibrate UQ at one temperature, deploy at another, and the
# residual distribution shifts. This exercise shows the failure mode and
# two simple fixes.
#
# **Your task:**
#
# 1. Train the Block 4.5 CQR pipeline (GP + quantile head + conformal
#    correction $\hat{Q}$) **only on `TensileTestDataset(T=600)`** (warm
#    conditions). Save the trained `qhead`, `Q_hat`, and `gp_cqr`.
# 2. Evaluate empirical coverage of the resulting intervals on
#    `TensileTestDataset(T=0)` (cold conditions). It should drop well
#    below the 0.9 target — typically to 0.6 or lower — even though the
#    nominal target was 0.9. **Exchangeability has failed.**
# 3. **Quick fix.** Inflate $\hat{Q}$ by a heuristic factor (e.g. 2x)
#    and re-measure coverage on T=0. Note the coverage / width trade-off.
# 4. **Principled fix: adaptive conformal.** Implement the online update
#    $\hat\alpha_t \leftarrow \hat\alpha_{t-1} + \gamma\,(\mathbf{1}[\text{miss}_t] - \alpha)$
#    sweeping through the T=0 stream one sample at a time. After each
#    step, recompute $\hat{Q}_t$ as the $(1 - \hat\alpha_t)$-quantile of
#    the original calibration scores. Plot empirical coverage (cumulative
#    or rolling-window) vs sample index. Show that the policy recovers
#    to ~0.9 within a few hundred samples.
#
# *Hint: $\gamma = 0.01$ is a fine default; set $\hat\alpha_0 = \alpha$.
# Clip $\hat\alpha_t$ to $(10^{-3}, 1 - 10^{-3})$ for stability.*

# %%
# TODO: your distribution-shift conformal experiment goes here.
# Skeleton:
#
#   # Step 1: reuse `qhead`, `Q_hat`, `gp_cqr` from Block 4.5
#   #          (all trained on T=600 only).
#
#   # Step 2: evaluate on T=0
#   X_shift, y_shift = X_T0, y_T0
#   mu_shift = gp_cqr.predict(X_shift)
#   feat_shift = np.concatenate([X_shift, mu_shift.reshape(-1, 1)], axis=1)
#   with torch.no_grad():
#       q_shift = qhead(torch.tensor(feat_shift, dtype=torch.float32)).numpy()
#   lo = q_shift[:, 0] - Q_hat
#   hi = q_shift[:, 1] + Q_hat
#   cov_shift = float(((y_shift >= lo) & (y_shift <= hi)).mean())
#   print(f"T=0 coverage = {cov_shift:.3f}   (target 0.9)")
#
#   # Step 3: heuristic 2x inflation
#   Q2 = 2.0 * Q_hat
#   lo2 = q_shift[:, 0] - Q2
#   hi2 = q_shift[:, 1] + Q2
#   cov2 = float(((y_shift >= lo2) & (y_shift <= hi2)).mean())
#
#   # Step 4: adaptive conformal update
#   gamma = 0.01
#   alpha_t = alpha_cqr
#   covers = []
#   alpha_traj = []
#   for t in range(len(X_shift)):
#       qlvl = min(np.ceil((len(nc_cal) + 1) * (1 - alpha_t)) / len(nc_cal), 1.0)
#       Q_t = float(np.quantile(nc_cal, qlvl))
#       lo_t = q_shift[t, 0] - Q_t
#       hi_t = q_shift[t, 1] + Q_t
#       miss = float(not (lo_t <= y_shift[t] <= hi_t))
#       covers.append(1 - miss)
#       alpha_t = float(np.clip(alpha_t + gamma * (miss - alpha_cqr), 1e-3, 1 - 1e-3))
#       alpha_traj.append(alpha_t)
#
#   # Plot rolling-window coverage vs sample index; mark the 0.9 target.
#   ...


# %% [markdown]
# ## Exam-aligned must-know statements (from MFML Unit 12 §"Exam-aligned")
#
# Re-read these after the exercises; today's blocks have given you the
# concrete intuition for every one of them.
#
# 1. The GP posterior is a closed-form Gaussian: $\mu^* = K_*^\top (K +
#    \sigma_n^2 I)^{-1} y$ and $\Sigma^* = K_{**} - K_*^\top (K +
#    \sigma_n^2 I)^{-1} K_*$ (homework Part A; Block 1).
# 2. GP hyperparameters are learned by maximising the log marginal
#    likelihood (homework Part A).
# 3. Deep ensembles and MC dropout are *approximate* GPs that scale better
#    but lose calibration guarantees (homework Part B).
# 4. A reliability diagram measures empirical coverage vs nominal
#    confidence — *the* operational test for whether a UQ method is
#    trustworthy on your data (homework Part C).
# 5. Active learning queries the point with maximum predictive uncertainty
#    (Block 2).
# 6. Acquisition functions trade off exploration ($\sigma$) and
#    exploitation ($\mu$); UCB is robust, EI is the standard, PI
#    over-exploits (Block 3).
# 7. Cost-aware AL replaces $\alpha(x)$ with $\alpha(x) / c(x)$ and is the
#    simplest practical upgrade for real lab budgets (Block 4).
# 8. Discovery $\neq$ labeling: a discovery candidate is simultaneously
#    high predictive variance AND high novelty (Block 5).
# 9. Per-cluster acquisition budgets allow allocation strategies tuned to
#    the objective (uniform fit vs maximum discovery) (Block 6).
# 10. Multi-fidelity GPs combine cheap-noisy and expensive-clean data and
#     can beat either single-fidelity baseline for intermediate budgets
#     (Block 7 stretch).
