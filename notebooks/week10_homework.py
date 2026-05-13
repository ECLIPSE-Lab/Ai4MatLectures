# %% [markdown]
# # Week 10 — Homework (do BEFORE the Thursday exercise)
#
# This week braids three lectures' Week 10 content onto one mechanism:
# **scaled dot-product attention**. We build it from scratch on a toy 1D
# signal, apply it to Ising microstructure patches, then extend it to
# multi-head — *exactly* the building blocks Thursday will assemble into
# a tiny ViT.
#
# 1. **MFML Unit 10** — Attention & Transformers. Self-attention,
#    multi-head, positional encoding, the transformer block, ViT.
# 2. **MG Week 10** — Representation learning and feature discovery.
#    Engineered vs learned features; transferability.
# 3. **ML-PC parallel-track this week.** Unit 9b (Transformers for
#    Materials) is the natural pair for this homework. ML-PC Unit 10
#    (Automation in microscopy) is the calendar-W10 lecture but uses
#    different material; we keep this homework on attention/ViT and apply
#    the same machinery to 1D sequences of intensities (Thursday Block 7).
#
# **Red thread.** *Self-attention does not care whether its tokens are
# image patches, spectral channels, or atoms in a crystal — the same
# $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ operation builds a representation
# from any sequence of tokens. Today's homework writes that operation in
# raw NumPy/PyTorch on a toy signal you can read off the page, then
# applies it to 16-patch Ising images so Thursday can stack it into a
# transformer block.*
#
# **Time:** ~75 minutes.
#
# ## What this homework is
#
# | Part | Min | Topic | Lecture anchor |
# |---|---:|---|---|
# | A | 20 | Scaled dot-product attention from scratch on a toy 1D sequence | MFML §"Self-attention formula" |
# | B | 25 | Patchify Ising-light → 16 tokens; single-head attention over patches | MFML §"Image as a sequence of patches" |
# | C | 20 | Multi-head attention from scratch: H=4 heads, concat + project | MFML §"Multi-head attention" |
# | D | 10 | Reflection — attention vs convolution: what does each have built-in? | bridge to Thursday |
#
# ## What you must hand in (or be able to show on Thursday)
#
# 1. **Part A:** the toy-sequence attention matrix as a heatmap; verify
#    rows sum to 1 (printed check).
# 2. **Part B:** two attention-matrix heatmaps side-by-side — one
#    above-Curie sample (random microstructure), one below-Curie sample
#    (clustered domains).
# 3. **Part C:** a 4-panel grid of attention maps from the four heads on
#    the same Ising sample.
# 4. **Part D:** your written reflection paragraph (4-6 sentences).

# %%
# Standard imports for the whole homework.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ai4mat.datasets import IsingDataset

np.random.seed(0)
torch.manual_seed(0)


# %% [markdown]
# # Part A — Scaled dot-product attention from scratch
#
# Self-attention takes a sequence of tokens $\mathbf{x}_1, \dots, \mathbf{x}_T$
# and produces a *new* sequence of tokens of the same length, where each
# output is a content-weighted average of the input *value* vectors:
# $$
# \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.
# $$
# The matrices $Q = X W_Q$, $K = X W_K$, $V = X W_V$ are linear
# projections of the input. For each row $i$, the attention weight on
# row $j$ is proportional to the dot-product similarity of $Q_i$ and
# $K_j$, normalised by softmax.
#
# **Your task.** Implement attention as a single 8-line function on a
# toy signal where you can read attention off the figure: a length-16
# signal $f(t) = \sin(2\pi t / 16) + \cos(2\pi t / 4)$. The slow-mode
# component repeats every 16 samples; the fast-mode component every 4.
# A randomly initialised attention layer (no training!) already shows
# this periodicity, because positions with similar $K$ vectors land
# similar attention weights.

# %%
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                  return_weights: bool = False):
    """The whole transformer in one function.

    Q, K, V : tensors of shape (..., T, d).
    Returns: output of shape (..., T, d) and (optionally) the attention
    weight matrix of shape (..., T, T).
    """
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k)         # (..., T, T)
    weights = F.softmax(scores, dim=-1)                        # (..., T, T)
    out = weights @ V                                          # (..., T, d)
    if return_weights:
        return out, weights
    return out


# %%
# Build the toy signal and lift each sample into a 1-D "token"
T_seq = 16
t = np.arange(T_seq)
signal = np.sin(2 * np.pi * t / T_seq) + np.cos(2 * np.pi * t / 4)

# Each token is a tiny feature vector. We use [signal_value, sin_phase, cos_phase]
# so the attention layer has something non-trivial to compare on.
X_np = np.stack([signal,
                 np.sin(2 * np.pi * t / T_seq),
                 np.cos(2 * np.pi * t / 4)], axis=1).astype(np.float32)
X = torch.tensor(X_np)                                         # (16, 3)
print(f"Part A — input X shape: {X.shape}")

# Random projections: Q, K, V each in R^{d_k}. We pick d_k=3 so we can read off
# what is happening. In a real transformer, d_k might be 64 per head.
d_k = 3
W_Q = torch.randn(3, d_k)
W_K = torch.randn(3, d_k)
W_V = torch.randn(3, d_k)
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

out, weights = scaled_dot_product_attention(Q, K, V, return_weights=True)
print(f"  output shape: {out.shape}    (same length, projected dim)")
print(f"  weights shape: {weights.shape}  (T x T attention matrix)")

# Diagnostic: each row of the attention matrix should sum to 1 (softmax).
row_sums = weights.sum(dim=-1)
print(f"  row sums (should all be ~1): min = {row_sums.min():.6f}, max = {row_sums.max():.6f}")


# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(t, signal, "o-")
axes[0].set_xlabel("position $t$")
axes[0].set_ylabel("signal value")
axes[0].set_title(r"Toy signal: $\sin(2\pi t/16) + \cos(2\pi t/4)$")
axes[0].grid(alpha=0.3)

im = axes[1].imshow(weights.numpy(), cmap="magma", aspect="auto")
axes[1].set_xlabel("attended position $j$")
axes[1].set_ylabel("query position $i$")
axes[1].set_title("Attention matrix (random projections)")
plt.colorbar(im, ax=axes[1])
fig.suptitle("Part A — scaled dot-product attention on a 1D toy sequence")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Part A deliverable:** the attention-matrix heatmap and the printed
# row-sum check.
#
# What to notice:
#
# - Even with *random* projections, the heatmap is not uniform. Some
#   positions land high attention from many queries; others land none.
# - With trained projections (Thursday), the structure becomes
#   meaningful: position $i$ learns *which* other positions are
#   informative for its representation.


# %% [markdown]
# # Part B — Patchify Ising-light, attend over the 16 patches
#
# A 16×16 Ising image becomes a sequence of 16 tokens by carving it
# into a 4×4 grid of 4×4-pixel patches. Each patch (16 pixels) is
# linearly projected to $d_\text{model}$. The output is a sequence of
# shape $(16, d_\text{model})$ — *exactly* the same shape attention
# expects in Part A. From the model's point of view, image and toy
# signal are the same kind of object.
#
# Below-Curie samples have clustered black/white domains; above-Curie
# samples are speckled. Even with random attention projections, you
# should see the attention structure differ between the two classes —
# patches from one domain attend strongly to patches from the same
# domain.

# %%
ising = IsingDataset(size="light")
print(f"Part B — IsingDataset(size='light'): {len(ising)} samples, X shape {ising.X.shape}, y shape {ising.y.shape}")

# Pick one above-Curie sample (label 1) and one below-Curie sample (label 0).
labels_np = ising.y.numpy()
i_below = int(np.where(labels_np == 0)[0][0])
i_above = int(np.where(labels_np == 1)[0][0])
img_below = ising.X[i_below, 0].numpy()   # (16, 16)
img_above = ising.X[i_above, 0].numpy()


# %%
def patchify(img: torch.Tensor, patch: int = 4) -> torch.Tensor:
    """(C, H, W) -> (T, C * patch * patch). Here C=1, H=W=16, patch=4 -> T=16."""
    if img.dim() == 2:
        img = img.unsqueeze(0)
    C, H, W = img.shape
    assert H % patch == 0 and W % patch == 0
    nh, nw = H // patch, W // patch
    # (C, nh, patch, nw, patch) -> (nh, nw, C, patch, patch) -> (nh*nw, C*patch*patch)
    p = img.unfold(1, patch, patch).unfold(2, patch, patch)        # (C, nh, nw, patch, patch)
    p = p.permute(1, 2, 0, 3, 4).contiguous().view(nh * nw, -1)    # (T, C*patch*patch)
    return p


d_in = 4 * 4 * 1   # patch_size * patch_size * channels
d_model = 32
W_embed = torch.randn(d_in, d_model) * 0.1   # patch embedding
W_Q = torch.randn(d_model, d_model) * 0.1
W_K = torch.randn(d_model, d_model) * 0.1
W_V = torch.randn(d_model, d_model) * 0.1


def attention_on_image(img_np: np.ndarray):
    img = torch.tensor(img_np, dtype=torch.float32)
    tokens = patchify(img, patch=4)          # (16, 16)
    embed = tokens @ W_embed                 # (16, d_model)
    Q = embed @ W_Q
    K = embed @ W_K
    V = embed @ W_V
    out, w = scaled_dot_product_attention(Q, K, V, return_weights=True)
    return embed, out, w


_, _, w_below = attention_on_image(img_below)
_, _, w_above = attention_on_image(img_above)
print(f"  attention matrix shapes: {w_below.shape}, {w_above.shape}")


# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

axes[0, 0].imshow(img_below, cmap="gray")
axes[0, 0].set_title(f"Below Curie (sample {i_below}, label 0)")
axes[0, 0].axis("off")
axes[0, 1].imshow(img_above, cmap="gray")
axes[0, 1].set_title(f"Above Curie (sample {i_above}, label 1)")
axes[0, 1].axis("off")

vmin = min(w_below.min().item(), w_above.min().item())
vmax = max(w_below.max().item(), w_above.max().item())
im0 = axes[1, 0].imshow(w_below.numpy(), cmap="magma", vmin=vmin, vmax=vmax)
axes[1, 0].set_title("attention matrix (below-Curie sample)")
axes[1, 0].set_xlabel("attended patch $j$")
axes[1, 0].set_ylabel("query patch $i$")
plt.colorbar(im0, ax=axes[1, 0])
im1 = axes[1, 1].imshow(w_above.numpy(), cmap="magma", vmin=vmin, vmax=vmax)
axes[1, 1].set_title("attention matrix (above-Curie sample)")
axes[1, 1].set_xlabel("attended patch $j$")
axes[1, 1].set_ylabel("query patch $i$")
plt.colorbar(im1, ax=axes[1, 1])
plt.tight_layout()
plt.show()


# %% [markdown]
# **Part B deliverable:** the four-panel figure (two images, two
# attention matrices) above.
#
# Even with random projections, the two attention matrices look
# different because the underlying patch embeddings differ: clustered
# domains produce groups of similar patches that attend to each other,
# while a random microstructure produces a more uniform attention
# pattern. After training (Thursday Block 3) the contrast will be
# dramatic.


# %% [markdown]
# # Part C — Multi-head attention from scratch
#
# A single attention head sees only one similarity geometry; it cannot
# simultaneously notice "patches from the same domain" *and* "patches
# at the same row". Multi-head attention runs $H$ independent attention
# operations on different projected subspaces, then concatenates and
# linearly mixes the outputs:
# $$
# \mathrm{MHA}(X) = \big[\mathrm{head}_1, \ldots, \mathrm{head}_H\big] W_O,
# \qquad
# \mathrm{head}_h = \mathrm{Attention}(X W_Q^h, X W_K^h, X W_V^h).
# $$
# Each head has its own $d_\text{model}/H$-dimensional projection. Output
# shape stays $(T, d_\text{model})$.
#
# **Your task.** Implement multi-head attention with $H = 4$ heads. Apply
# it to the same below-Curie Ising sample as Part B. Plot the four
# attention matrices side-by-side. Different heads should produce
# *visibly different* attention patterns.

# %%
class MultiHeadSelfAttention(nn.Module):
    """A from-scratch multi-head self-attention layer (no PyTorch shortcuts).

    The whole point of this exercise is that there are no surprises:
    the only operations are linear projections, einsum reshapes, and
    softmax. Read every line.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly into n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X, return_weights: bool = False):
        # X: (B, T, d_model). Always operate with a leading batch dim.
        if X.dim() == 2:
            X = X.unsqueeze(0)
        B, T, D = X.shape
        H, dh = self.n_heads, self.d_head

        Q = self.W_Q(X).view(B, T, H, dh).transpose(1, 2)   # (B, H, T, dh)
        K = self.W_K(X).view(B, T, H, dh).transpose(1, 2)
        V = self.W_V(X).view(B, T, H, dh).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(dh)    # (B, H, T, T)
        weights = F.softmax(scores, dim=-1)                  # (B, H, T, T)
        per_head = weights @ V                               # (B, H, T, dh)
        per_head = per_head.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_O(per_head)                             # (B, T, D)
        if return_weights:
            return out, weights
        return out


# %%
torch.manual_seed(0)
mha = MultiHeadSelfAttention(d_model=32, n_heads=4)

# Re-embed the below-Curie image with a learnable Linear (still untrained).
patch_embed = nn.Linear(d_in, d_model, bias=False)
torch.manual_seed(0)
patch_embed.weight.data.mul_(0.5)   # smaller init for nicer-looking attention

with torch.no_grad():
    tokens_below = patch_embed(patchify(torch.tensor(img_below, dtype=torch.float32)))
    tokens_below = tokens_below.unsqueeze(0)             # add batch dim
    out_mha, weights_mha = mha(tokens_below, return_weights=True)
print(f"Part C — multi-head output shape: {out_mha.shape}    (B, T, d_model)")
print(f"  weights shape: {weights_mha.shape}    (B, H, T, T)")


# %%
# Visualise the four heads' attention on the same sample.
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
W = weights_mha[0].detach().numpy()      # (4, 16, 16)
vmin, vmax = W.min(), W.max()
for h in range(4):
    im = axes[h].imshow(W[h], cmap="magma", vmin=vmin, vmax=vmax)
    axes[h].set_title(f"head {h}")
    axes[h].set_xlabel("attended patch")
axes[0].set_ylabel("query patch")
fig.suptitle("Part C — different heads attend to different patch relationships (untrained!)")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Part C deliverable:** the four-panel attention-head grid above.
#
# Different heads already differ visibly because each one's $W_Q^h$,
# $W_K^h$, $W_V^h$ are independently random. After training, each head
# learns to specialise. ML interpretability research (Vaswani 2017
# Figs. 3-5; Vig 2019 BertViz) lives off looking at these.


# %% [markdown]
# # Part D — Reflection: attention vs convolution
#
# A convolutional layer has *locality* and *translation equivariance*
# baked in: a 3×3 kernel always looks at a 3×3 neighbourhood, and the
# same kernel is applied at every position. Attention has neither.
# Position-$i$'s representation is a content-weighted average of *all*
# positions, regardless of distance, and there is no built-in notion
# of "which two positions are neighbours" — the model learns that from
# data, with help from positional encoding.
#
# **Your task (~10 min, write 4-6 sentences):**
#
# Pick one of the following pairs and answer two questions:
#
# - Microstructure image classification (Ising / Cahn-Hilliard) vs
#   spectral classification (XRD / EELS).
# - Crystal-structure property prediction (graph of atoms) vs sequence
#   prediction in linguistics.
# - Microscopy denoising (lots of pixels, lots of training data) vs
#   property prediction (few hundred materials).
#
# Questions:
#
# 1. Which inductive bias — attention's *learn-everything-from-data* or
#    convolution's *built-in locality* — is better suited to the
#    problem, and why?
# 2. What is the smallest experiment you would run to test your answer?
#
# Bring the paragraph to Thursday; we will pick two volunteers to read
# theirs aloud at the start of Block 1.
#
# **Hand in:** your written paragraph (Markdown cell below).

# %% [markdown]
# > *(your reflection paragraph here)*
