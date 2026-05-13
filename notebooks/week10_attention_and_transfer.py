# %% [markdown]
# # Week 10 — Attention, ViT, and cross-system transfer
#
# This week braids three lectures:
#
# 1. **MFML Unit 10** — Attention & Transformers. We turn the from-scratch
#    multi-head attention from the homework into a tiny ViT (positional
#    encoding + transformer blocks + classification head) and compare it
#    against a CNN baseline of the same parameter budget.
# 2. **MG Week 10** — Representation learning and feature discovery. We
#    freeze the ViT trained on Ising, transfer its embedding to a
#    Cahn-Hilliard regression task, and shoot it out against PCA and
#    raw-pixel features.
# 3. **ML-PC Unit 10** — ML for characterization signals. We end the
#    session by running the *same* multi-head attention on tensile
#    curves treated as 1D sequences — the spectral-compression analogue.
#
# **Red thread.** *Self-attention does not care whether its tokens are
# image patches, spectral channels, or atoms in a crystal — the same
# $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ operation builds a representation
# from any sequence of tokens. Today we build a tiny ViT, watch it learn
# attention maps that beat or tie a parameter-matched CNN, transfer
# the ViT-Ising embedding to a regression task on Cahn-Hilliard, and
# finish by running the same architecture on 1D tensile-curve sequences.*
#
# > **Pre-flight check.** This notebook **assumes** you have run
# > `notebooks/week10_homework.py`. Block 1 picks up directly from your
# > scaled-dot-product attention function and your `MultiHeadSelfAttention`
# > module; we will not re-derive them.
#
# ## Agenda (90 min)
#
# | Block | Min | Topic |
# |------:|:---:|:------|
# | 1 |  6 | Recap from homework — patch attention on Ising |
# | 2 | 12 | Positional encoding: why permutation-equivariance is a bug, not a feature |
# | 3 | 14 | Tiny ViT: stack two transformer blocks, classify Ising; compare to a CNN |
# | 4 | 12 | Attention maps as interpretability vs CNN input-gradient saliency |
# | 5 | 14 | Cross-system transfer: ViT-Ising frozen → Cahn-Hilliard regression |
# | 6 | 12 | Engineered vs learned features: 3-way bake-off on Cahn-Hilliard |
# | 7 | 10 | Same architecture, 1D input: attention on tensile curves |
# | 8 | 10 | Student exercises (3 core + 1 stretch) |

# %%
# Standard imports.
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from ai4mat.datasets import IsingDataset, CahnHilliardDataset, TensileTestDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# ## Helpers used by every block

# %%
def patchify(img: torch.Tensor, patch: int = 4) -> torch.Tensor:
    """(B, C, H, W) -> (B, T, C * patch * patch). No assumptions about C."""
    if img.dim() == 3:                                # add batch
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    nh, nw = H // patch, W // patch
    p = img.unfold(2, patch, patch).unfold(3, patch, patch)        # (B, C, nh, nw, patch, patch)
    p = p.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, nh * nw, -1)  # (B, T, C*patch^2)
    return p


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class MultiHeadSelfAttention(nn.Module):
    """Same module as the homework, kept here for self-containedness."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X, return_weights: bool = False):
        B, T, D = X.shape
        H, dh = self.n_heads, self.d_head
        Q = self.W_Q(X).view(B, T, H, dh).transpose(1, 2)
        K = self.W_K(X).view(B, T, H, dh).transpose(1, 2)
        V = self.W_V(X).view(B, T, H, dh).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(dh)
        weights = F.softmax(scores, dim=-1)
        per_head = weights @ V
        per_head = per_head.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_O(per_head)
        if return_weights:
            return out, weights
        return out


# %% [markdown]
# ## Block 1 — Recap from homework
#
# Three results travel into today:
#
# 1. **Scaled dot-product attention** is one line of softmax + matmul.
#    It works on any sequence of tokens. (Part A)
# 2. **A 16×16 Ising image is a sequence of 16 patch tokens** under the
#    `patchify` operation. The model sees it the same way it sees the
#    toy 1D signal. (Part B)
# 3. **Multi-head attention** is just $H$ independent heads on subspaces
#    of $d_\text{model}/H$, concatenated and projected. (Part C)
#
# Today we add positional encoding, build a transformer block, train
# end-to-end, and then run the trained encoder on a totally different
# materials task.

# %%
# Reload datasets.
ising = IsingDataset(size="light")
print(f"Block 1 — IsingDataset(size='light'): {len(ising)} samples, X {ising.X.shape}, y {ising.y.shape}")
print(f"  class balance: {torch.bincount(ising.y).tolist()}")


# %% [markdown]
# ## Block 2 — Positional encoding
#
# Self-attention is *permutation-equivariant*: shuffle the input tokens
# and the output tokens are shuffled in exactly the same way. For a
# **set** that is the right behaviour. For an image, it is a bug —
# patch (0, 0) and patch (3, 3) are physically distinguishable, and the
# model needs to know that. We add a fixed sinusoidal positional
# encoding (Vaswani 2017): for token index $t$ and channel $c$,
# $$
# PE_{t, 2k}   = \sin(t / 10000^{2k/d_\mathrm{model}}), \qquad
# PE_{t, 2k+1} = \cos(t / 10000^{2k/d_\mathrm{model}}).
# $$
# We first *demonstrate* that without PE, shuffling patches leaves
# attention output identical (up to permutation), then add PE and show
# it does not.

# %%
def sinusoidal_pe(T: int, d_model: int) -> torch.Tensor:
    """(T, d_model) sinusoidal positional encoding."""
    pe = torch.zeros(T, d_model)
    pos = torch.arange(T).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


d_model = 32
patch = 4
T_seq = (16 // patch) ** 2          # 16 tokens
patch_embed = nn.Linear(patch * patch, d_model, bias=False)
mha = MultiHeadSelfAttention(d_model=d_model, n_heads=4)

# One Ising sample for the demo.
img = ising.X[0:1]                  # (1, 1, 16, 16)
tokens = patch_embed(patchify(img, patch=patch))   # (1, 16, d_model)

# Demo 1: no PE -> permutation-equivariant.
torch.manual_seed(0)
perm = torch.randperm(T_seq)
out_orig = mha(tokens)
out_perm = mha(tokens[:, perm])
out_perm_unshuffled = out_perm[:, torch.argsort(perm)]
diff_no_pe = (out_orig - out_perm_unshuffled).abs().max().item()
print(f"Block 2 — without positional encoding:")
print(f"  ||out(X) - perm^-1(out(perm(X)))||_inf = {diff_no_pe:.2e}    (~ 0: permutation-equivariant)")

# Demo 2: add PE -> equivariance broken.
pe = sinusoidal_pe(T_seq, d_model).unsqueeze(0)    # (1, T, d_model)
out_orig_pe = mha(tokens + pe)
out_perm_pe = mha(tokens[:, perm] + pe)            # PE NOT permuted
out_perm_pe_unshuffled = out_perm_pe[:, torch.argsort(perm)]
diff_with_pe = (out_orig_pe - out_perm_pe_unshuffled).abs().max().item()
print(f"  with positional encoding:")
print(f"  ||out(X+PE) - perm^-1(out(perm(X)+PE))||_inf = {diff_with_pe:.2e}    (large: position now matters)")


# %%
# Visualise the PE itself: it is just a fixed lookup table.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(pe.squeeze(0).numpy(), aspect="auto", cmap="RdBu_r")
axes[0].set_xlabel("channel $c$")
axes[0].set_ylabel("token index $t$")
axes[0].set_title(f"Sinusoidal PE  (T = {T_seq}, d_model = {d_model})")
axes[1].plot(pe.squeeze(0)[:, :4].numpy())
axes[1].set_xlabel("token index $t$")
axes[1].set_ylabel(r"$PE_{t,c}$")
axes[1].set_title("first 4 channels of the PE")
axes[1].grid(alpha=0.3)
axes[1].legend(["c=0", "c=1", "c=2", "c=3"])
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Block 3 — Tiny ViT vs parameter-matched CNN
#
# We now assemble a full transformer block:
# $$
# \mathbf{z}' = \mathbf{z} + \mathrm{MHA}(\mathrm{LN}(\mathbf{z})), \qquad
# \mathbf{z}'' = \mathbf{z}' + \mathrm{MLP}(\mathrm{LN}(\mathbf{z}')),
# $$
# stack two of them, prepend a learnable `[CLS]` token whose final
# representation goes to a 2-class linear head, and train on Ising-light.
# Then we train a CNN of *roughly the same parameter count* on the same
# data and compare.

# %%
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyViT(nn.Module):
    def __init__(self, img_size: int = 16, patch: int = 4, d_model: int = 32,
                 n_heads: int = 4, n_blocks: int = 2, n_classes: int = 2):
        super().__init__()
        self.patch = patch
        T = (img_size // patch) ** 2
        self.patch_embed = nn.Linear(patch * patch, d_model, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("pe", sinusoidal_pe(T + 1, d_model).unsqueeze(0))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, d_model) CLS embedding -- used by Block 5/6."""
        B = x.size(0)
        toks = self.patch_embed(patchify(x, patch=self.patch))
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, toks], dim=1) + self.pe
        for blk in self.blocks:
            z = blk(z)
        return self.ln_f(z[:, 0])

    def forward(self, x):
        return self.head(self.encode(x))

    @torch.no_grad()
    def cls_attention_to_patches(self, x: torch.Tensor) -> np.ndarray:
        """Average attention from CLS token to each image patch, last block."""
        B = x.size(0)
        toks = self.patch_embed(patchify(x, patch=self.patch))
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, toks], dim=1) + self.pe
        for blk in self.blocks[:-1]:
            z = blk(z)
        # last block: take attention weights from CLS row
        last = self.blocks[-1]
        z_norm = last.ln1(z)
        _, w = last.mha(z_norm, return_weights=True)    # (B, H, T+1, T+1)
        cls_attn = w[:, :, 0, 1:].mean(dim=1)            # (B, T) average over heads
        return cls_attn.cpu().numpy()


class TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


vit = TinyViT()
cnn = TinyCNN()
print(f"Block 3 — parameter counts:")
print(f"  TinyViT: {count_params(vit):>6} params")
print(f"  TinyCNN: {count_params(cnn):>6} params")


# %%
# Train both on Ising-light. ~5 epochs is enough for a clear comparison.
def train_classifier(model, X, y, n_epochs=5, lr=3e-3, batch=128, val_frac=0.1, seed=0):
    torch.manual_seed(seed)
    ds = TensorDataset(X, y)
    n_val = int(val_frac * len(ds))
    n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(seed))
    tr_dl = DataLoader(tr, batch_size=batch, shuffle=True)
    va_dl = DataLoader(va, batch_size=batch, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = {"train_loss": [], "val_acc": []}
    for ep in range(n_epochs):
        model.train()
        ep_loss = 0.0; n = 0
        for xb, yb in tr_dl:
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward(); opt.step()
            ep_loss += loss.item() * len(yb); n += len(yb)
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for xb, yb in va_dl:
                pred = model(xb).argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        history["train_loss"].append(ep_loss / n)
        history["val_acc"].append(correct / total)
        print(f"  epoch {ep+1}: train loss {history['train_loss'][-1]:.4f} | val acc {history['val_acc'][-1]:.3f}")
    return history


print(f"\nTraining TinyViT...")
hist_vit = train_classifier(vit, ising.X, ising.y)
print(f"\nTraining TinyCNN...")
hist_cnn = train_classifier(cnn, ising.X, ising.y)


# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ep_axis = np.arange(1, len(hist_vit["val_acc"]) + 1)
ax.plot(ep_axis, hist_vit["val_acc"], "o-", label=f"TinyViT ({count_params(vit)} params)")
ax.plot(ep_axis, hist_cnn["val_acc"], "s-", label=f"TinyCNN ({count_params(cnn)} params)")
ax.set_xlabel("epoch")
ax.set_ylabel("val accuracy")
ax.set_title("Block 3 — ViT vs CNN on Ising-light at matched parameter budget")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()


# %% [markdown]
# **Reading the result.** ViT and CNN end up close on Ising-light. ViT
# usually trails CNN for a few epochs (no built-in locality bias), then
# catches up. On larger datasets ViT *passes* CNN — that is the
# Dosovitskiy 2020 (ViT) data-efficiency curve we discussed in MFML
# Unit 10. With 5000 images we are nowhere near the ViT-favoured regime,
# yet both work.


# %% [markdown]
# ### Block 3b — Naive MHA vs `F.scaled_dot_product_attention` (Flash SDPA)
#
# Our `MultiHeadSelfAttention` above is the textbook implementation: it
# explicitly materialises the $T \times T$ attention matrix
# $\mathrm{softmax}(QK^\top / \sqrt{d_k})$ before multiplying by $V$. That
# matrix has $O(T^2)$ memory cost; at $T = 1024$ it dominates GPU memory.
#
# PyTorch's `F.scaled_dot_product_attention(q, k, v)` is a drop-in
# replacement that *auto-dispatches* to one of three fused kernels:
#
# 1. **Flash Attention** [@dao_2022_flashattention] on Ampere or newer
#    (sm_80+: A100, RTX 30/40 series, H100). True IO-aware, $O(T)$
#    memory.
# 2. **Memory-efficient attention** (xFormers-style) on older GPUs like
#    Pascal (sm_60, GTX 10-series including the **1080 Ti** in our lab).
#    Still tiled, still ~2-3x faster than naive MHA, but not the full
#    Flash kernel.
# 3. **Math** (a plain einsum fallback) on CPU or when the inputs do not
#    fit any fused kernel's constraints.
#
# `nn.MultiheadAttention` *also* dispatches through SDPA under the hood
# in modern PyTorch, but its public API still materialises a few extra
# tensors (key padding masks, batch-first reshaping). The cleanest
# comparison is **naive textbook MHA vs raw `F.scaled_dot_product_attention`**;
# we add `nn.MultiheadAttention` as a third row for completeness.

# %%
class SDPABlock(nn.Module):
    """Thin wrapper that calls F.scaled_dot_product_attention directly."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        B, T, D = X.shape
        H, dh = self.n_heads, self.d_head
        Q = self.W_Q(X).view(B, T, H, dh).transpose(1, 2)   # (B, H, T, dh)
        K = self.W_K(X).view(B, T, H, dh).transpose(1, 2)
        V = self.W_V(X).view(B, T, H, dh).transpose(1, 2)
        # F.scaled_dot_product_attention auto-dispatches to Flash / mem-efficient / math.
        out = F.scaled_dot_product_attention(Q, K, V)        # (B, H, T, dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(out)


def bench_attention(module, B, T, D, device, n_warm=3, n_iter=10):
    """Return (mean wall-clock seconds per fwd, peak GPU bytes)."""
    module = module.to(device).eval()
    x = torch.randn(B, T, D, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(n_warm):
            _ = module(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = module(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    return (t1 - t0) / n_iter, peak


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Block 3b — benchmarking attention kernels on device = {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}  (sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})")

B_bench, H_bench, dh_bench = 8, 4, 64
D_bench = H_bench * dh_bench                    # 256
seq_lens = [64, 256, 1024]

# Three modules to compare.
naive = MultiHeadSelfAttention(D_bench, H_bench)
sdpa = SDPABlock(D_bench, H_bench)
mha_nn = nn.MultiheadAttention(D_bench, H_bench, bias=False, batch_first=True)


class NNMHAWrapper(nn.Module):
    """Wrap nn.MultiheadAttention so its call signature matches the others."""
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        out, _ = self.mod(x, x, x, need_weights=False)
        return out


print(f"\n  seq_len |   naive MHA (time / peak)  |   nn.MultiheadAttention   |   F.SDPA (Flash/mem-eff)")
print(f"  -------- + -------------------------- + -------------------------- + --------------------------")
for T_bench in seq_lens:
    t_naive, m_naive = bench_attention(naive, B_bench, T_bench, D_bench, device)
    t_nn,    m_nn    = bench_attention(NNMHAWrapper(mha_nn), B_bench, T_bench, D_bench, device)
    t_sdpa,  m_sdpa  = bench_attention(sdpa, B_bench, T_bench, D_bench, device)
    fmt = lambda t, m: f"{t*1e3:6.2f} ms / {m/1024**2:6.1f} MB" if device.type == "cuda" else f"{t*1e3:6.2f} ms /     —  "
    print(f"  {T_bench:>6}   |  {fmt(t_naive, m_naive):<24}  |  {fmt(t_nn, m_nn):<24}  |  {fmt(t_sdpa, m_sdpa):<24}")


# %% [markdown]
# **Reading the table.** Two things to notice.
#
# 1. **Wall-clock.** Naive MHA scales as $O(T^2)$ in both time and memory.
#    SDPA's tiled kernel hides the $O(T^2)$ behind streaming
#    multiprocessor blocks and shared memory — the wall-clock gap widens
#    as $T$ grows. At $T = 1024$, expect roughly 2-3x speedup on the lab's
#    1080 Ti and 5-10x on Ampere/Hopper.
# 2. **Peak memory.** Naive MHA stores the full $B H T T$ attention
#    matrix (at $T = 1024$, $B = 8$, $H = 4$ that is 128 MB of fp32).
#    SDPA never materialises it; peak memory grows roughly linearly in
#    $T$, not quadratically.
#
# **1080 Ti caveat (sm_61, Pascal).** Flash Attention v1/v2 require
# sm_80+ (Ampere). On Pascal, `F.scaled_dot_product_attention` falls
# back to the *memory-efficient* kernel — still tiled and substantially
# faster than naive MHA, but it is **not** true Flash. The qualitative
# story (sub-quadratic memory, faster wall-clock) holds; the absolute
# speedup will be the 2-3x kind, not the 10x you read in papers benched
# on A100s. The pedagogical point — *write SDPA, not hand-rolled MHA* —
# is the same on either GPU.
#
# **What this means for the rest of the notebook.** Our `MultiHeadSelfAttention`
# stays in place for *teaching* (you can read every line of it). In any
# production model you would write `F.scaled_dot_product_attention(Q, K, V)`
# instead and ship the same six-line wrapper as `SDPABlock` above.


# %% [markdown]
# ### Block 3c — Mixture of Experts: lecture concept, not exercised today
#
# MFML Unit 10 introduces **Mixture of Experts (MoE)** as the third
# modern scaling trick alongside Flash Attention and state-space models.
# The idea is simple to state and expensive to implement: replace the
# dense MLP in each transformer block with a *router* that picks
# $k \ll N$ of $N$ expert MLPs per token. You get the parameter count of
# a 100-billion-parameter model with the FLOPs of a 10-billion-parameter
# one, because only $k/N$ experts fire per token.
#
# **We do not implement MoE in this notebook.** A faithful MoE block
# needs (a) a load-balancing auxiliary loss to stop the router from
# collapsing onto one expert, (b) all-to-all expert dispatch primitives
# for multi-GPU training, and (c) capacity-factor bookkeeping to handle
# variable tokens-per-expert. That is one more lecture's worth of code
# and well outside today's 90-minute budget.
#
# **Forward links if you want the production-scale story:**
#
# - PyTorch's `torch.distributed.tensor.parallel` for the all-to-all
#   primitives.
# - The **DeepSeek-MoE** paper (and the closely related Mixtral 8x7B,
#   DBRX, and Qwen-MoE technical reports) for the routing and
#   load-balancing recipes that actually ship.


# %% [markdown]
# ## Block 4 — Attention maps vs CNN saliency
#
# A trained ViT carries an interpretable artefact for free: the attention
# from the `[CLS]` token to each image patch tells you which patches the
# classifier looked at. We compare to a CNN saliency baseline (input
# times gradient) on the same samples.

# %%
n_show = 4
torch.manual_seed(1)
idx = torch.randperm(len(ising))[:n_show]
imgs = ising.X[idx]                  # (n_show, 1, 16, 16)
labels = ising.y[idx]
attn = vit.cls_attention_to_patches(imgs)        # (n_show, 16) over patches
attn_map = attn.reshape(n_show, 4, 4)             # (n_show, 4, 4)
attn_map_up = np.repeat(np.repeat(attn_map, 4, axis=1), 4, axis=2)  # (n_show, 16, 16)


# CNN saliency: |x * dlogit_class/dx|, on the predicted class.
imgs_grad = imgs.clone().requires_grad_(True)
logits = cnn(imgs_grad)
pred_class = logits.argmax(dim=-1)
selected = logits.gather(1, pred_class.unsqueeze(1)).sum()
selected.backward()
saliency = (imgs_grad.grad * imgs_grad.detach()).abs().squeeze(1).numpy()
saliency = saliency / (saliency.max(axis=(1, 2), keepdims=True) + 1e-12)


# %%
fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 8))
for i in range(n_show):
    axes[0, i].imshow(imgs[i, 0].numpy(), cmap="gray")
    axes[0, i].set_title(f"label={labels[i].item()}")
    axes[0, i].axis("off")
    axes[1, i].imshow(imgs[i, 0].numpy(), cmap="gray")
    axes[1, i].imshow(attn_map_up[i], cmap="hot", alpha=0.5)
    axes[1, i].axis("off")
    axes[2, i].imshow(imgs[i, 0].numpy(), cmap="gray")
    axes[2, i].imshow(saliency[i], cmap="hot", alpha=0.5)
    axes[2, i].axis("off")

axes[0, 0].set_ylabel("input")
axes[1, 0].set_ylabel("ViT CLS attention")
axes[2, 0].set_ylabel("CNN |x·grad| saliency")
fig.suptitle("Block 4 — what each model looks at on the same Ising samples")
plt.tight_layout()
plt.show()


# %% [markdown]
# ViT attention is *coarser* (per-patch) and CNN saliency is *finer*
# (per-pixel). The ViT cannot, by construction, produce a sub-patch
# explanation; the CNN cannot, by construction, look at distant patches
# at the same time. Both views are useful, and a domain expert reading
# microstructure attribution should know they exist.


# %% [markdown]
# ## Block 5 — Cross-system transfer: ViT-Ising → Cahn-Hilliard
#
# We now ask MG's Week 10 question directly: do the embeddings the ViT
# learned on the Ising classification task carry useful structure to a
# *different* materials regression task? We freeze the ViT encoder, attach
# a fresh linear regression head (one per scalar target), and fine-tune
# only the head on a small Cahn-Hilliard subset that predicts free energy.
#
# To match input shape, we down-pool the 64×64 CH images to 16×16. This
# loses high-frequency detail but preserves the domain morphology — the
# physics that mattered for Ising classification is still there.

# %%
ch = CahnHilliardDataset(simulation_number=0)
print(f"Block 5 — CahnHilliardDataset(simulation_number=0): {len(ch)} samples, X {ch.X.shape}, y {ch.y.shape}")

# Downsample 64 -> 16 by 4x4 average pooling.
ch_X16 = F.avg_pool2d(ch.X, kernel_size=4)        # (N, 1, 16, 16)
ch_y = ch.y.numpy()
print(f"  downsampled X: {ch_X16.shape}, free-energy range = [{ch_y.min():.3f}, {ch_y.max():.3f}]")

# 80/20 split.
N = len(ch)
rng = np.random.default_rng(0)
perm = rng.permutation(N)
ntr = int(0.8 * N)
tr_idx, te_idx = perm[:ntr], perm[ntr:]
X_tr_t = ch_X16[tr_idx]
X_te_t = ch_X16[te_idx]
y_tr = ch_y[tr_idx]; y_te = ch_y[te_idx]


# %%
def vit_embed(model, X_t, batch=256):
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            embs.append(model.encode(X_t[i : i + batch]).cpu().numpy())
    return np.concatenate(embs, axis=0)


# (a) ViT-Ising frozen encoder + ridge head
emb_tr = vit_embed(vit, X_tr_t)
emb_te = vit_embed(vit, X_te_t)
print(f"  ViT-Ising embedding: {emb_tr.shape}    (frozen)")
ridge_vit = Ridge(alpha=1.0).fit(emb_tr, y_tr)
rmse_vit_ising = float(np.sqrt(mean_squared_error(y_te, ridge_vit.predict(emb_te))))
r2_vit_ising = float(r2_score(y_te, ridge_vit.predict(emb_te)))

# (b) ViT trained from scratch on Cahn-Hilliard (regression head)
class TinyViTRegressor(TinyViT):
    def __init__(self, **kw):
        super().__init__(**kw, n_classes=1)


vit_ch = TinyViTRegressor()
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.float32)


def train_regressor(model, X, y, n_epochs=5, lr=3e-3, batch=128, seed=0):
    torch.manual_seed(seed)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(n_epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = F.mse_loss(pred, yb)
            loss.backward(); opt.step()
    return model


train_regressor(vit_ch, X_tr_t, y_tr_t)
vit_ch.eval()
with torch.no_grad():
    y_pred_scratch = vit_ch(X_te_t).squeeze(-1).cpu().numpy()
rmse_vit_scratch = float(np.sqrt(mean_squared_error(y_te, y_pred_scratch)))
r2_vit_scratch = float(r2_score(y_te, y_pred_scratch))

# (c) PCA on raw flattened CH pixels + ridge head
flat_tr = X_tr_t.numpy().reshape(len(X_tr_t), -1)
flat_te = X_te_t.numpy().reshape(len(X_te_t), -1)
pca = PCA(n_components=32).fit(flat_tr)
ridge_pca = Ridge(alpha=1.0).fit(pca.transform(flat_tr), y_tr)
rmse_pca = float(np.sqrt(mean_squared_error(y_te, ridge_pca.predict(pca.transform(flat_te)))))
r2_pca = float(r2_score(y_te, ridge_pca.predict(pca.transform(flat_te))))

print(f"\nBlock 5 — Cahn-Hilliard free-energy regression (downsampled 16×16):")
print(f"  {'method':<35} {'RMSE':>8}   {'R^2':>6}")
print(f"  {'(a) ViT-Ising frozen + Ridge':<35} {rmse_vit_ising:>8.4f}   {r2_vit_ising:>6.3f}")
print(f"  {'(b) ViT trained on CH from scratch':<35} {rmse_vit_scratch:>8.4f}   {r2_vit_scratch:>6.3f}")
print(f"  {'(c) PCA(32) + Ridge':<35} {rmse_pca:>8.4f}   {r2_pca:>6.3f}")


# %% [markdown]
# **Reading the result.** "ViT-Ising frozen" is a *zero-fine-tune* baseline:
# the encoder was trained on a binary classification of a *different*
# microstructure family, and we are now asking it to support a *regression*
# on a different physical system. That it does not collapse to chance is
# the entire MG transferability story — embeddings carry chemistry/physics
# across systems, *to the extent the systems share underlying texture*.
# Often "scratch on CH" wins; sometimes "frozen ViT-Ising" wins, especially
# when the CH training set is small.


# %% [markdown]
# ## Block 6 — Engineered vs learned features (3-way bake-off)
#
# We now isolate the *feature* discussion from the *learning algorithm*
# discussion. Same ridge regression head everywhere. Three feature
# extractors, all on the same Cahn-Hilliard target:
#
# 1. **Raw flattened pixels** (256-d for the 16×16 image): the simplest
#    "features", every pixel is a feature.
# 2. **PCA(32)** on raw pixels: classical engineered linear dimension
#    reduction.
# 3. **ViT-Ising frozen embedding** (32-d): non-linear learned
#    representation transferred from a different system.

# %%
# (a) raw pixels
ridge_raw = Ridge(alpha=1.0).fit(flat_tr, y_tr)
rmse_raw = float(np.sqrt(mean_squared_error(y_te, ridge_raw.predict(flat_te))))
r2_raw = float(r2_score(y_te, ridge_raw.predict(flat_te)))

# (b) and (c) reuse rmse_pca, rmse_vit_ising from Block 5
print(f"Block 6 — bake-off on the same target (CH free energy, Ridge head):")
print(f"  {'feature':<35} {'dim':>6} {'RMSE':>8} {'R^2':>6}")
print(f"  {'(a) raw flattened pixels':<35} {flat_tr.shape[1]:>6} {rmse_raw:>8.4f} {r2_raw:>6.3f}")
print(f"  {'(b) PCA(32)':<35} {32:>6} {rmse_pca:>8.4f} {r2_pca:>6.3f}")
print(f"  {'(c) ViT-Ising frozen (32-d)':<35} {32:>6} {rmse_vit_ising:>8.4f} {r2_vit_ising:>6.3f}")


# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
labels_bar = ["raw pixels (256-d)", "PCA(32)", "ViT-Ising frozen (32-d)"]
rmses = [rmse_raw, rmse_pca, rmse_vit_ising]
ax.bar(labels_bar, rmses, color=["#94a3b8", "#3b82f6", "#8b5cf6"])
ax.set_ylabel("test RMSE (free-energy units)")
ax.set_title("Block 6 — same Ridge head, three feature extractors")
for i, v in enumerate(rmses):
    ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.show()


# %% [markdown]
# Three observations to take with you:
#
# - **256-d raw is not necessarily worst.** Ridge can handle 256
#   features against ~800 training samples and may pick up texture that
#   a 32-d compression throws away.
# - **PCA(32) is a *linear* compressor.** It captures the highest-variance
#   linear directions in CH images. If those happen to correlate with
#   energy, PCA wins.
# - **ViT-Ising(32) is a *non-linear, transferred* compressor.** Whether
#   it beats PCA depends on whether Ising-physics-relevant features
#   overlap with CH-energy-relevant features. The honest answer for this
#   16×16 setup is "they often do, because both are texture-driven".


# %% [markdown]
# ## Block 7 — Same architecture, 1D input: attention on tensile curves
#
# We now show that *nothing* in the transformer block requires the input
# to be an image. We tokenise each tensile curve at temperature
# $T \in \{0, 400, 600\}$ °C into 10 strain bins, take the mean stress
# per bin as the (1-d) per-token feature, embed each token to
# $d_\text{model}$, attach a CLS token, and classify $T$.
#
# This is the *spectral compression* analogue ML-PC asked for in Week 10:
# replace "tensile curve" with "XRD pattern" or "EELS edge", and the
# architecture is identical.

# %%
T_TENSILE = 10                         # number of tokens per curve
def tokenise_tensile(temperature: int, n_tokens: int = T_TENSILE) -> torch.Tensor:
    ds = TensileTestDataset(temperature=temperature)
    strain = ds.X.numpy().reshape(-1)
    stress = ds.y.numpy().reshape(-1)
    edges = np.linspace(strain.min(), strain.max(), n_tokens + 1)
    edges[-1] += 1e-9
    bins = np.digitize(strain, edges) - 1
    bins = np.clip(bins, 0, n_tokens - 1)
    feat = np.zeros((n_tokens,), dtype=np.float32)
    for b in range(n_tokens):
        m = (bins == b)
        feat[b] = stress[m].mean() if m.any() else 0.0
    return torch.tensor(feat).unsqueeze(-1)        # (T, 1)


X_tens, y_tens = [], []
for ti, T in enumerate([0, 400, 600]):
    # We have 350 samples at each temperature; build several
    # bootstrap-bagged curves to get a non-trivial training set.
    rng = np.random.default_rng(ti)
    base = TensileTestDataset(temperature=T)
    s_all = base.X.numpy().reshape(-1)
    y_all = base.y.numpy().reshape(-1)
    for _ in range(150):
        m = rng.choice(len(s_all), size=int(0.7 * len(s_all)), replace=False)
        s = s_all[m]; st = y_all[m]
        edges = np.linspace(s_all.min(), s_all.max(), T_TENSILE + 1)
        edges[-1] += 1e-9
        bins = np.digitize(s, edges) - 1
        bins = np.clip(bins, 0, T_TENSILE - 1)
        feat = np.zeros((T_TENSILE,), dtype=np.float32)
        for b in range(T_TENSILE):
            mb = (bins == b)
            feat[b] = st[mb].mean() if mb.any() else 0.0
        X_tens.append(feat)
        y_tens.append(ti)

X_tens = torch.tensor(np.array(X_tens), dtype=torch.float32).unsqueeze(-1)   # (N, T, 1)
y_tens = torch.tensor(np.array(y_tens), dtype=torch.long)
# Standardise the per-token stress values.
mu, sd = X_tens.mean(), X_tens.std()
X_tens = (X_tens - mu) / sd
print(f"Block 7 — 1D-attention dataset:")
print(f"  X_tens {tuple(X_tens.shape)}, y_tens {tuple(y_tens.shape)} ({y_tens.bincount().tolist()})")


# %%
class Tiny1DTransformer(nn.Module):
    def __init__(self, T: int, in_dim: int = 1, d_model: int = 16, n_heads: int = 4,
                 n_blocks: int = 2, n_classes: int = 3):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("pe", sinusoidal_pe(T + 1, d_model).unsqueeze(0))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.size(0)
        toks = self.embed(x)
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, toks], dim=1) + self.pe
        for blk in self.blocks:
            z = blk(z)
        return self.head(self.ln_f(z[:, 0]))


m1d = Tiny1DTransformer(T=T_TENSILE)
print(f"  Tiny1DTransformer params: {count_params(m1d)}")
hist_1d = train_classifier(m1d, X_tens, y_tens, n_epochs=8, lr=3e-3, batch=64)
print(f"  final val accuracy: {hist_1d['val_acc'][-1]:.3f}    (chance = 1/3 = 0.333)")


# %% [markdown]
# Same architecture, 1D input — and it learns to classify the three
# process conditions from the *shape* of the stress-strain curve. The
# message lands as soon as you replace "stress" with "intensity", "strain"
# with "wavenumber", and "temperature" with "phase ID": this is exactly
# the ML-PC Week 10 spectral pipeline, expressed without ever using a
# convolution or a hand-engineered peak finder.


# %% [markdown]
# # Student exercises (Block 8 — ~10 min)

# %% [markdown]
# ## Exercise 1 (core) — Halve d_model and n_heads
#
# Re-train `TinyViT` with `d_model = 16` and `n_heads = 2`. Re-plot the
# Block 3 ViT-vs-CNN figure. How does the parameter sweet spot shift?
# At equal parameter budget, which architecture wins on Ising-light, and
# by how much?

# %%
# YOUR CODE for Exercise 1 below.


# %% [markdown]
# ## Exercise 2 (core) — Unfreeze the last block
#
# Block 5's frozen-ViT-Ising baseline used the encoder unchanged. Now
# unfreeze *only the last transformer block* (and the `ln_f` layer) and
# fine-tune for 3 epochs on the CH regression task. Compare RMSE to the
# frozen baseline and to the from-scratch ViT-CH. Does fine-tuning help
# the most when the source (Ising) and target (CH) are similar, or
# different?

# %%
# YOUR CODE for Exercise 2 below.


# %% [markdown]
# ## Exercise 3 (core) — Mean pool instead of CLS
#
# Replace the `[CLS]`-token readout in `TinyViT` with mean pooling over
# the patch tokens (drop the `cls` parameter; in `forward`, take
# `z.mean(dim=1)` instead of `z[:, 0]` after the final block). Re-train
# on Ising. How does the Block 4 attention figure change? In particular,
# does the model still produce a single attention map, or do you have
# to plot one per patch?

# %%
# YOUR CODE for Exercise 3 below.


# %% [markdown]
# ## Exercise 4 (stretch) — Rotary positional encoding
#
# Replace the `sinusoidal_pe` table with **rotary positional encoding**
# (RoPE; Su et al. 2021). RoPE rotates each Q and K vector in
# 2D-subspace pairs by an angle proportional to position, so attention
# scores depend on relative position only. Implement RoPE inside
# `MultiHeadSelfAttention` and re-train `TinyViT`.
#
# **Your task:**
#
# 1. Add a `apply_rope(Q, K)` step inside `forward` that rotates the
#    head-dim pairs by `pos * theta_i` with $\theta_i = 10000^{-2i/d_h}$.
# 2. Re-train. Compare validation accuracy to the sinusoidal-PE baseline.
# 3. Does RoPE help on the small 16-token Ising input, or is it overkill?
#    Why?
#
# *Pedagogical pointer: RoPE is what every modern LLM (LLaMA, Qwen,
# Mistral) uses. It is the standard answer to "how do we encode position
# in a way that generalises to longer sequences than training saw".*

# %%
# YOUR CODE for Exercise 4 below.


# %% [markdown]
# ## Exercise 5 (stretch, optional) — Tiny Mamba on tensile-curve regression
#
# Block 7 used a `Tiny1DTransformer` on tokenised tensile curves. Modern
# alternatives to attention drop the $O(T^2)$ self-attention block in
# favour of a **state-space model (SSM)** with $O(T)$ recurrence. The
# best-known instance is **Mamba** [@gu_2023_mamba]: a selective SSM
# whose state-transition matrix depends on the input, giving it the
# *content-aware* gating of attention with the *linear-time* recurrence
# of an RNN. On long sequences (DNA, audio, long context) Mamba matches
# or beats Transformers at a fraction of the FLOPs.
#
# **Heavyweight dependency.** Mamba ships as a fused CUDA kernel:
#
# ```bash
# pip install mamba-ssm causal-conv1d
# ```
#
# **This will only install on a CUDA-equipped machine** with a working
# nvcc toolchain. CPU-only builds will fail at `pip install` time — that
# is expected. The `try/except` block below skips the exercise cleanly
# in that case.
#
# **Your task:**
#
# 1. Replace the transformer backbone of Block 7 with a 1-2 layer Mamba
#    block at the same `d_model = 16` and the same per-curve token
#    budget. Keep the CLS-token readout (or switch to mean pooling —
#    your call).
# 2. Train for the same epoch budget as `Tiny1DTransformer` (8 epochs,
#    same optimiser).
# 3. Plot the two validation-accuracy curves side-by-side. Comment on
#    parameter count, wall-clock per epoch, and final accuracy.
# 4. At $T = 10$ tokens, do you expect Mamba's linear scaling to matter?
#    Why or why not? (Hint: re-read the Block 3b table.)
#
# *Pedagogical pointer: SSMs are the most-credible non-attention sequence
# model on the table as of 2026; the materials-informatics use case is
# long sensor traces (full XRD line scans, full nano-indentation curves)
# where $T \gg 10^3$ and the $O(T^2)$ attention bill bites hard.*

# %%
# YOUR CODE for Exercise 5 below. The reference implementation is
# provided as a runnable scaffold — wrap the import in try/except so the
# rest of the notebook keeps working without mamba-ssm installed.
try:
    from mamba_ssm import Mamba

    class TinyMamba1D(nn.Module):
        def __init__(self, T: int, in_dim: int = 1, d_model: int = 16,
                     n_blocks: int = 2, n_classes: int = 3):
            super().__init__()
            self.embed = nn.Linear(in_dim, d_model, bias=False)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.register_buffer("pe", sinusoidal_pe(T + 1, d_model).unsqueeze(0))
            # Each Mamba block is a content-aware selective SSM.
            self.blocks = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(n_blocks)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_classes)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def forward(self, x):
            B = x.size(0)
            toks = self.embed(x)
            cls = self.cls.expand(B, -1, -1)
            z = torch.cat([cls, toks], dim=1) + self.pe
            for blk in self.blocks:
                z = blk(z)
            return self.head(self.ln_f(z[:, 0]))

    if torch.cuda.is_available():
        mamba_dev = torch.device("cuda")
        m_mamba = TinyMamba1D(T=T_TENSILE).to(mamba_dev)
        print(f"Exercise 5 — TinyMamba1D params: {count_params(m_mamba)}")
        hist_mamba = train_classifier(
            m_mamba, X_tens.to(mamba_dev), y_tens.to(mamba_dev),
            n_epochs=8, lr=3e-3, batch=64,
        )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ep_axis = np.arange(1, len(hist_1d["val_acc"]) + 1)
        ax.plot(ep_axis, hist_1d["val_acc"], "o-", label=f"Tiny1DTransformer ({count_params(m1d)} params)")
        ax.plot(ep_axis, hist_mamba["val_acc"], "s-", label=f"TinyMamba1D ({count_params(m_mamba)} params)")
        ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy")
        ax.set_title("Exercise 5 — Transformer vs Mamba on tensile-curve classification")
        ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout(); plt.show()
    else:
        print("Exercise 5 — skipping: mamba-ssm requires CUDA.")
except ImportError as e:
    print(f"Exercise 5 — mamba-ssm not installed ({e}); install with `pip install mamba-ssm causal-conv1d` on a CUDA machine.")


# %% [markdown]
# ## Exam-aligned must-know statements
#
# Re-read these after the exercises; today's blocks have given you the
# concrete intuition for every one of them.
#
# 1. Scaled dot-product attention is
#    $\mathrm{softmax}(QK^\top / \sqrt{d_k}) V$. The softmax row sums to 1.
#    (Homework Part A.)
# 2. A $H \times W$ image becomes a sequence of $T = (H/p)(W/p)$ tokens
#    under patch size $p$. Attention treats it the same as any other
#    sequence. (Homework Part B + Block 1.)
# 3. Multi-head attention runs $H$ independent heads on $d_\text{model}/H$
#    subspaces, then concatenates and projects. Different heads can
#    specialise on different relationships. (Homework Part C.)
# 4. Self-attention is **permutation-equivariant**. To make position
#    matter, add positional encoding (sinusoidal, learned, or RoPE).
#    (Block 2.)
# 5. A transformer block is residual MHA + residual MLP, with LayerNorm.
#    Stack many. (Block 3.)
# 6. ViT and CNN are comparable at small scale (Ising-light, 5000
#    images); ViT only pulls ahead at much larger data scales
#    (Dosovitskiy 2020). (Block 3.)
# 7. The `[CLS]` token's attention to image patches is a free
#    interpretability artefact; CNN saliency is per-pixel. Both views
#    are partial. (Block 4.)
# 8. **Cross-system transfer** asks whether embeddings learned on one
#    materials task carry useful structure to another. Frozen-encoder
#    + new linear head is the cleanest baseline. (Block 5.)
# 9. At fixed regression head, the **feature extractor** matters: raw
#    pixels, PCA, and a transferred ViT often disagree by >2x in RMSE.
#    Which wins is empirical, not pre-determined. (Block 6.)
# 10. The same transformer architecture works on 1D sequences; replacing
#     "tensile curve" with "XRD pattern" gives you the ML-PC Week 10
#     pipeline for spectral compression and phase ID, with no
#     convolutions and no hand-engineered peak finder. (Block 7.)
