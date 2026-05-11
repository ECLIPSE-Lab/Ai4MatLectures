# NEU-DET clustering notebook (MLPC Unit 5) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a per-course MLPC week 5 Quarto notebook that clusters NEU-DET steel-surface-defect images with K-means and GMM on two feature pipelines (raw pixels+PCA, ResNet18 embeddings), plus the supporting `NEUDETDataset` class, dataset test, dep bump, and index entry.

**Architecture:** Add one `NEUDETDataset(Dataset)` to `ai4mat/datasets/neu_det.py` matching the existing dataset convention but with on-disk loading. Build one self-contained Quarto notebook in `notebooks/MLPC/` that uses the dataset, generates and caches ResNet18 embeddings, runs clustering, and renders every figure named in the spec.

**Tech Stack:** PyTorch, torchvision (new), scikit-learn (KMeans, GaussianMixture, PCA, TSNE, metrics), scipy (Hungarian), matplotlib, pandas, imageio, tqdm, Quarto.

**Spec:** `docs/superpowers/specs/2026-05-11-neu-det-clustering-mlpc-unit5-design.md`

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `requirements.txt` | Add `torchvision>=0.16` |
| Create | `ai4mat/datasets/neu_det.py` | `NEUDETDataset` class + `download_if_missing()` helper |
| Modify | `ai4mat/datasets/__init__.py` | Re-export `NEUDETDataset` |
| Create | `tests/datasets/test_neu_det.py` | Smoke tests, skipped if data dir is empty |
| Create | `notebooks/MLPC/week05_clustering_neu_det.qmd` | Lecture notebook |
| Modify | `index.qmd` | Add link to the new notebook under MLPC table |

---

## Task 1: Add torchvision to requirements

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dep**

Append one line so the file ends with `torchvision`:

```diff
 scikit-learn>=1.4
 xgboost>=2.0
+torchvision>=0.16
```

- [ ] **Step 2: Install locally**

Run: `pip install -r requirements.txt`
Expected: torchvision is installed; `python -c "import torchvision; print(torchvision.__version__)"` prints `0.16.x` or newer.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add torchvision for pretrained CNN embeddings"
```

---

## Task 2: Failing dataset test (TDD red)

**Files:**
- Create: `tests/datasets/test_neu_det.py`

- [ ] **Step 1: Write the failing tests**

Mirror the style of `tests/datasets/test_ising.py`. Place exactly this content:

```python
import os
import pytest
import torch

DATA_DIR = "data/NEU-DET"


def _data_present() -> bool:
    return os.path.isdir(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0


skip_if_no_data = pytest.mark.skipif(
    not _data_present(), reason="NEU-DET data not present in data/NEU-DET"
)


def test_neu_det_importable():
    from ai4mat.datasets import NEUDETDataset  # noqa: F401


@skip_if_no_data
def test_neu_det_contract():
    from ai4mat.datasets import NEUDETDataset
    from tests.conftest import assert_dataset_contract

    ds = NEUDETDataset(download=False)
    assert_dataset_contract(
        ds, expected_x_shape=[1, 200, 200], expected_y_dtype=torch.long, min_len=1000
    )


@skip_if_no_data
def test_neu_det_length_and_classes():
    from ai4mat.datasets import NEUDETDataset

    ds = NEUDETDataset(download=False)
    assert len(ds) == 1800, f"expected 1800, got {len(ds)}"
    assert set(ds.y.tolist()) == {0, 1, 2, 3, 4, 5}
    assert len(ds.class_names) == 6
    joined = " ".join(ds.class_names).lower()
    for token in ["crazing", "inclusion", "patches", "pitted", "rolled", "scratches"]:
        assert token in joined, f"missing class token: {token!r}"


@skip_if_no_data
def test_neu_det_normalised():
    from ai4mat.datasets import NEUDETDataset

    ds = NEUDETDataset(download=False)
    for i in [0, len(ds) // 2, len(ds) - 1]:
        x, _ = ds[i]
        assert x.dtype == torch.float32
        assert 0.0 <= x.min().item() and x.max().item() <= 1.0


@skip_if_no_data
def test_neu_det_split_train_only():
    from ai4mat.datasets import NEUDETDataset

    ds_train = NEUDETDataset(download=False, split="train")
    ds_val = NEUDETDataset(download=False, split="validation")
    assert len(ds_train) == 1440  # 240 per class * 6
    assert len(ds_val) == 360     # 60 per class * 6


def test_neu_det_download_false_errors_on_empty(tmp_path):
    # Doesn't depend on the dataset being downloaded — runs in CI too.
    from ai4mat.datasets import NEUDETDataset

    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        NEUDETDataset(root=empty_root, download=False)


@skip_if_no_data
def test_neu_det_image_paths_populated():
    from ai4mat.datasets import NEUDETDataset

    ds = NEUDETDataset(download=False)
    assert len(ds.image_paths) == len(ds)
    assert all(p.endswith(".jpg") for p in ds.image_paths[:5])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/datasets/test_neu_det.py -v`
Expected: `test_neu_det_importable` fails with `ImportError: cannot import name 'NEUDETDataset'`. The `skip_if_no_data` tests either skip (data absent) or also fail at import.

- [ ] **Step 3: Commit**

```bash
git add tests/datasets/test_neu_det.py
git commit -m "test(neu_det): add failing dataset contract tests (TDD red)"
```

---

## Task 3: Implement `download_if_missing()` helper

**Files:**
- Create: `ai4mat/datasets/neu_det.py` (skeleton + helper only)

- [ ] **Step 1: Write the skeleton + helper**

```python
"""NEU-DET steel-surface-defect dataset."""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

_KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "kaustubhdikshit/neu-surface-defect-database"
)


def _find_train_root(root: str, max_depth: int = 2) -> Optional[str]:
    """Return the dir that directly contains `train/`; None if not found.

    Kaggle archives unzip with varying nesting (`NEU-DET/train/...`,
    `NEU-DET/NEU-DET/train/...`, or `NEU Metal Surface Defects Data/.../train/...`),
    so search up to `max_depth` levels below `root`.
    """
    if not os.path.isdir(root):
        return None
    if os.path.isdir(os.path.join(root, "train")):
        return root

    stack: list[tuple[str, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            entries = sorted(os.listdir(current))
        except OSError:
            continue
        for entry in entries:
            candidate = os.path.join(current, entry)
            if not os.path.isdir(candidate):
                continue
            if os.path.isdir(os.path.join(candidate, "train")):
                return candidate
            stack.append((candidate, depth + 1))
    return None


def download_if_missing(root: str) -> None:
    """Download + unzip the NEU-DET archive into `root` if not already present.

    Idempotent: returns immediately if `root` already contains a `train/`
    directory anywhere up to one level deep.
    """
    os.makedirs(root, exist_ok=True)
    if _find_train_root(root) is not None:
        return

    tmp_zip = os.path.join(os.path.dirname(root) or ".", "_neu_det_tmp.zip")
    try:
        with urlopen(_KAGGLE_URL) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(tmp_zip, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc="NEU-DET"
            ) as pbar:
                while True:
                    chunk = response.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    fh.write(chunk)
                    pbar.update(len(chunk))

        with open(tmp_zip, "rb") as fh:
            magic = fh.read(4)
        if magic != b"PK\x03\x04":
            raise RuntimeError(
                "Kaggle endpoint did not return a zip — likely an auth "
                "redirect. Download manually and place the archive contents "
                f"in {root}/."
            )

        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(root)
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
```

- [ ] **Step 2: Sanity-check it parses**

Run: `python -c "from ai4mat.datasets.neu_det import download_if_missing; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add ai4mat/datasets/neu_det.py
git commit -m "feat(neu_det): add download_if_missing helper"
```

---

## Task 4: Implement `NEUDETDataset` class

**Files:**
- Modify: `ai4mat/datasets/neu_det.py` (append class below the helper)

- [ ] **Step 1: Append the class**

```python
class NEUDETDataset(Dataset):
    """NEU-DET steel-surface-defect classification dataset.

    1800 grayscale 200x200 images across 6 classes (300 per class).
    Pixel values normalised to [0, 1].

    Args:
        root: base directory. Default 'data/NEU-DET'.
        split: 'all' (default), 'train', or 'validation'.
        download: if True and root is empty, fetch and unzip the Kaggle archive.
        transform / target_transform: optional callables, applied per item.

    X shape: (1, 200, 200)  dtype: float32  range: [0, 1]
    y shape: ()  dtype: long  values: {0..5}

    Public attributes:
        X (Tensor [N, 1, 200, 200]), y (Tensor [N]),
        class_names (list[str], length 6), image_paths (list[str], length N).
    """

    def __init__(
        self,
        root: str = "data/NEU-DET",
        split: str = "all",
        download: bool = True,
        transform=None,
        target_transform=None,
    ):
        if split not in {"all", "train", "validation"}:
            raise ValueError(f"split must be 'all'|'train'|'validation', got {split!r}")

        if download:
            download_if_missing(root)

        train_root = _find_train_root(root)
        if train_root is None:
            raise FileNotFoundError(
                f"No 'train/' directory found under {root}. "
                "Set download=True or place the unzipped NEU-DET archive there."
            )

        split_dirs = (
            ["train", "validation"] if split == "all" else [split]
        )

        # Discover class names from the train/images dir (sorted, stable).
        class_dir = os.path.join(train_root, "train", "images")
        class_names = sorted(
            d for d in os.listdir(class_dir)
            if os.path.isdir(os.path.join(class_dir, d))
        )
        if len(class_names) != 6:
            raise RuntimeError(
                f"Expected 6 NEU-DET classes, found {len(class_names)}: {class_names}"
            )
        self.class_names: list[str] = class_names
        cls_to_idx = {c: i for i, c in enumerate(class_names)}

        # Walk and collect image paths + labels.
        image_paths: list[str] = []
        labels: list[int] = []
        for sd in split_dirs:
            for cls in class_names:
                cls_path = os.path.join(train_root, sd, "images", cls)
                if not os.path.isdir(cls_path):
                    continue
                for fname in sorted(os.listdir(cls_path)):
                    if fname.lower().endswith(".jpg"):
                        image_paths.append(os.path.join(cls_path, fname))
                        labels.append(cls_to_idx[cls])

        if not image_paths:
            raise FileNotFoundError(f"No .jpg files found under {train_root}")

        # Eager load into one tensor.
        import imageio.v3 as iio
        import numpy as np

        N = len(image_paths)
        X = np.empty((N, 200, 200), dtype=np.float32)
        for i, p in enumerate(image_paths):
            img = iio.imread(p)
            if img.ndim == 3:
                img = img.mean(axis=-1)  # NEU-DET ships grayscale-as-RGB sometimes
            if img.shape != (200, 200):
                raise ValueError(f"Unexpected image shape {img.shape} at {p}")
            X[i] = img.astype(np.float32) / 255.0

        self.X = torch.from_numpy(X).unsqueeze(1)  # (N, 1, 200, 200)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.image_paths = image_paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
```

- [ ] **Step 2: Re-export from package**

Edit `ai4mat/datasets/__init__.py` to add the new entries (alphabetical-ish placement, after `nanoindentation`):

```diff
 from .nanoindentation import NanoindentationDataset
+from .neu_det import NEUDETDataset
 from .crystal_graphs import CrystalGraphsDataset

 __all__ = [
@@
     "NanoindentationDataset",
+    "NEUDETDataset",
     "CrystalGraphsDataset",
 ]
```

- [ ] **Step 3: Run the import-only test**

Run: `pytest tests/datasets/test_neu_det.py::test_neu_det_importable -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add ai4mat/datasets/neu_det.py ai4mat/datasets/__init__.py
git commit -m "feat(neu_det): implement NEUDETDataset class"
```

---

## Task 5: Trigger the download and verify dataset end-to-end

**Files:** (none changed; this is a verification step that produces `data/NEU-DET/...`)

- [ ] **Step 1: Download data via the dataset itself**

Run (one line, expect ~30 s):

```bash
python -c "from ai4mat.datasets import NEUDETDataset; ds = NEUDETDataset(); print(len(ds), ds.class_names)"
```

Expected: prints `1800 ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']` (class names may differ slightly in punctuation; length must be 1800).

If the print shows a wrong length, inspect `data/NEU-DET/` layout (`find data/NEU-DET -maxdepth 3 -type d`) and refine `_find_train_root` or the walk logic before continuing.

- [ ] **Step 2: Run the full dataset test suite**

Run: `pytest tests/datasets/test_neu_det.py -v`
Expected: 7 passed (one of them — `test_neu_det_download_false_errors_on_empty` — runs without needing the dataset; the rest pass now that data is present).

- [ ] **Step 3: Confirm .gitignore covers data/**

Run: `grep -E '^/?data/?' .gitignore || echo MISSING`
Expected: prints either a matching line or `MISSING`. If `MISSING`, append `data/` to `.gitignore`, `git add .gitignore`, and commit with `chore: gitignore data/`. **Do not commit the 26 MB NEU-DET archive.**

- [ ] **Step 4: Commit (only the .gitignore edit if it was needed)**

```bash
git status   # should show nothing else staged
```

---

## Task 6: Notebook scaffold — frontmatter, imports, data preview

**Files:**
- Create: `notebooks/MLPC/week05_clustering_neu_det.qmd`

- [ ] **Step 1: Write the file**

```markdown
---
title: "MLPC Week 5: Clustering steel-surface defects (NEU-DET)"
subtitle: "K-means vs GMM on raw pixels and ResNet18 embeddings"
jupyter: python3
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ECLIPSE-Lab/Ai4MatLectures/blob/main/notebooks/MLPC/week05_clustering_neu_det.ipynb)

## Learning Objectives

- Recognise when clustering recovers real structure vs spurious groups.
- Compare raw-pixel features to pretrained CNN embeddings on the same task.
- Read t-SNE scatters, contingency matrices and ARI/NMI together.
- Connect K-means and GMM/EM (MFML Unit 5) to a real defect-classification setting.

## Setup

```{python}
#| eval: false
!pip install git+https://github.com/ECLIPSE-Lab/Ai4MatLectures.git "mdsdata>=0.1.5" "torchvision>=0.16"
```

```{python}
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.optimize import linear_sum_assignment

from ai4mat.datasets import NEUDETDataset

np.random.seed(0)
torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
```

## 1. Data preview

```{python}
ds = NEUDETDataset()
print(f"N = {len(ds)}, X shape = {tuple(ds.X.shape)}, classes = {ds.class_names}")
```

```{python}
fig, axes = plt.subplots(6, 6, figsize=(10, 10))
rng = np.random.default_rng(0)
for c_idx, cls in enumerate(ds.class_names):
    idxs_in_class = np.where(ds.y.numpy() == c_idx)[0]
    picks = rng.choice(idxs_in_class, 6, replace=False)
    for j, k in enumerate(picks):
        ax = axes[c_idx, j]
        ax.imshow(ds.X[k, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if j == 0:
            ax.set_ylabel(cls, fontsize=9, rotation=0, labelpad=40, ha="right")
plt.suptitle("NEU-DET — 6 random examples per defect class", y=0.92)
plt.tight_layout()
plt.show()
```
```

- [ ] **Step 2: Render this notebook only**

Run: `quarto render notebooks/MLPC/week05_clustering_neu_det.qmd`
Expected: exits 0; `_site/notebooks/MLPC/week05_clustering_neu_det.html` exists with the class grid figure visible.

- [ ] **Step 3: Commit**

```bash
git add notebooks/MLPC/week05_clustering_neu_det.qmd
git commit -m "docs(mlpc-w5): scaffold NEU-DET clustering notebook with data preview"
```

---

## Task 7: Notebook Section 2 — raw pixels + PCA features

**Files:**
- Modify: `notebooks/MLPC/week05_clustering_neu_det.qmd`

- [ ] **Step 1: Append Section 2 to the notebook**

```markdown
## 2. Feature pipeline A — raw pixels + PCA(50)

```{python}
X_flat = ds.X.reshape(len(ds), -1).numpy()           # (1800, 40000)
X_flat_std = (X_flat - X_flat.mean(axis=0)) / (X_flat.std(axis=0) + 1e-8)

pca = PCA(n_components=50, random_state=0)
Z_pca = pca.fit_transform(X_flat_std)                # (1800, 50)
print(f"Z_pca shape: {Z_pca.shape}")
print(f"Cumulative explained variance @ 50 comps: {pca.explained_variance_ratio_.sum():.3f}")
```

```{python}
plt.figure(figsize=(6, 3.5))
plt.plot(np.arange(1, 51), np.cumsum(pca.explained_variance_ratio_), marker="o", ms=3)
plt.xlabel("PCA component")
plt.ylabel("cumulative explained variance")
plt.title("Scree — raw pixels (40000-d) → PCA")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```
```

- [ ] **Step 2: Render**

Run: `quarto render notebooks/MLPC/week05_clustering_neu_det.qmd`
Expected: exits 0; HTML now shows the scree plot.

- [ ] **Step 3: Commit**

```bash
git add notebooks/MLPC/week05_clustering_neu_det.qmd
git commit -m "docs(mlpc-w5): add PCA feature pipeline section"
```

---

## Task 8: Notebook Section 3 — ResNet18 embeddings (with cache)

**Files:**
- Modify: `notebooks/MLPC/week05_clustering_neu_det.qmd`

- [ ] **Step 1: Append Section 3**

```markdown
## 3. Feature pipeline B — pretrained ResNet18 embeddings

```{python}
from torchvision import models
from torchvision.models import ResNet18_Weights

CACHE_PATH = "data/NEU-DET/embeddings_resnet18.npz"

def compute_resnet_embeddings(ds, device=DEVICE, batch_size=64):
    """Return (N, 512) float32 ResNet18 embeddings; ImageNet preprocessing."""
    backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    backbone.fc = nn.Identity()
    backbone.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    out = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            x = ds.X[i : i + batch_size].to(device)        # (B, 1, 200, 200)
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            x = x.expand(-1, 3, -1, -1)                    # gray -> 3 channels
            x = (x - mean) / std
            z = backbone(x)                                # (B, 512)
            out.append(z.cpu())
    return torch.cat(out).numpy()

if os.path.exists(CACHE_PATH):
    Z_resnet = np.load(CACHE_PATH)["Z"]
    print(f"Loaded cached embeddings: {Z_resnet.shape}")
else:
    Z_resnet = compute_resnet_embeddings(ds)
    np.savez(CACHE_PATH, Z=Z_resnet)
    print(f"Computed + cached embeddings: {Z_resnet.shape}")

Z_resnet_std = (Z_resnet - Z_resnet.mean(axis=0)) / (Z_resnet.std(axis=0) + 1e-8)
```
```

- [ ] **Step 2: Render**

Run: `quarto render notebooks/MLPC/week05_clustering_neu_det.qmd`
Expected: exits 0; on first run prints "Computed + cached embeddings: (1800, 512)" and creates `data/NEU-DET/embeddings_resnet18.npz`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/MLPC/week05_clustering_neu_det.qmd
git commit -m "docs(mlpc-w5): add ResNet18 embedding pipeline with on-disk cache"
```

---

## Task 9: Notebook plot helpers

**Files:**
- Modify: `notebooks/MLPC/week05_clustering_neu_det.qmd`

- [ ] **Step 1: Append a helpers section before any clustering block**

Insert this section between Section 3 and (the future) Section 4. It defines three reusable plotting functions used by Section 6 four times each.

```markdown
## Helpers — plot routines used across both features × both algorithms

```{python}
def hungarian_remap(y_true, y_pred, K):
    """Return a permutation `perm` of cluster ids so that the diagonal of
    crosstab(y_true, perm[y_pred]) is maximal. Used for visual alignment
    only; metrics like ARI/NMI are permutation-invariant.

    Assumes the number of predicted clusters equals the number of true
    classes (the case throughout this notebook). For unequal counts you'd
    need to handle leftover rows/columns separately.
    """
    K_true = int(y_true.max()) + 1
    assert K == K_true, (
        f"hungarian_remap assumes K==K_true; got K={K}, K_true={K_true}"
    )
    C = np.zeros((K, K_true), dtype=int)
    for k in range(K):
        for c in range(K_true):
            C[k, c] = int(((y_pred == k) & (y_true == c)).sum())
    # Maximise diagonal -> minimise -C
    row_ind, col_ind = linear_sum_assignment(-C)
    perm = np.zeros(K, dtype=int)
    perm[row_ind] = col_ind
    return perm

def plot_tsne_dual(Z2, y_true, y_pred, class_names, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, lbl, name in zip(axes, [y_true, y_pred], ["true class", "predicted cluster"]):
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=lbl, cmap="tab10", s=8, alpha=0.7)
        ax.set_title(f"{name}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(title, y=1.02)
    if len(class_names) <= 10:
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                              color=plt.cm.tab10(i)) for i in range(len(class_names))]
        fig.legend(handles, class_names, loc="lower center", ncol=len(class_names),
                   bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_cluster_tiles(ds, y_pred, score_per_sample, n_per=6, title=""):
    """For each cluster, plot the n_per samples with the highest score
    (e.g. -dist-to-centroid for KMeans, or component responsibility for GMM)."""
    K = int(y_pred.max()) + 1
    fig, axes = plt.subplots(K, n_per, figsize=(1.1 * n_per, 1.1 * K))
    for k in range(K):
        in_cluster = np.where(y_pred == k)[0]
        order = in_cluster[np.argsort(-score_per_sample[in_cluster])]
        picks = order[:n_per]
        for j in range(n_per):
            ax = axes[k, j] if K > 1 else axes[j]
            if j < len(picks):
                ax.imshow(ds.X[picks[j], 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"c{k}", fontsize=8, rotation=0, labelpad=14)
    plt.suptitle(title, y=1.0)
    plt.tight_layout()
    plt.show()

def plot_contingency(y_true, y_pred, class_names, title=""):
    K = int(y_pred.max()) + 1
    perm = hungarian_remap(y_true, y_pred, K)
    y_pred_aligned = perm[y_pred]
    df = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred_aligned, name="cluster (aligned)"),
    )
    df.index = [class_names[i] for i in df.index]
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(df.values, cmap="viridis")
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns)
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, int(df.values[i, j]), ha="center", va="center",
                    color="white" if df.values[i, j] < df.values.max() / 2 else "black",
                    fontsize=8)
    ax.set_title(f"{title}\nARI={ari:.3f}  NMI={nmi:.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    return ari, nmi
```
```

- [ ] **Step 2: Render to check the helpers parse**

Run: `quarto render notebooks/MLPC/week05_clustering_neu_det.qmd`
Expected: exits 0; helpers section has no figures (no calls yet).

- [ ] **Step 3: Commit**

```bash
git add notebooks/MLPC/week05_clustering_neu_det.qmd
git commit -m "docs(mlpc-w5): add plot helpers (tsne dual, tiles, contingency)"
```

---

## Task 10: Notebook Section 4 — K-means on both features + K-sweep

**Files:**
- Modify: `notebooks/MLPC/week05_clustering_neu_det.qmd`

- [ ] **Step 1: Append Section 4**

```markdown
## 4. K-means

```{python}
def run_kmeans_sweep(Z, y_true, K_range=range(2, 11), seed=0):
    sil, ari = [], []
    for K in K_range:
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Z)
        sil.append(silhouette_score(Z, km.labels_))
        ari.append(adjusted_rand_score(y_true, km.labels_))
    return list(K_range), sil, ari

y_true = ds.y.numpy()

for name, Z in [("raw+PCA", Z_pca), ("ResNet18", Z_resnet_std)]:
    Ks, sil, ari = run_kmeans_sweep(Z, y_true)
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(Ks, sil, "o-", color="C0", label="silhouette")
    ax1.set_xlabel("K"); ax1.set_ylabel("silhouette", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(Ks, ari, "s--", color="C3", label="ARI vs truth")
    ax2.set_ylabel("ARI", color="C3")
    ax1.axvline(6, color="gray", linestyle=":", alpha=0.6)
    ax1.set_title(f"K-means sweep — {name}")
    plt.tight_layout()
    plt.show()

# Fit K=6 once for each feature set and stash results.
kmeans_results = {}
for name, Z in [("raw+PCA", Z_pca), ("ResNet18", Z_resnet_std)]:
    km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(Z)
    kmeans_results[name] = {
        "labels": km.labels_,
        "centroids": km.cluster_centers_,
        "Z": Z,
        # score for tile plot: higher = closer to centroid
        "score": -np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1),
    }
    ari = adjusted_rand_score(y_true, km.labels_)
    nmi = normalized_mutual_info_score(y_true, km.labels_)
    print(f"K-means K=6 on {name:9s}: ARI={ari:.3f}  NMI={nmi:.3f}")
```
```

- [ ] **Step 2: Render**

Run: `quarto render notebooks/MLPC/week05_clustering_neu_det.qmd`
Expected: exits 0; HTML now shows two K-sweep plots and prints ARI/NMI for both K=6 fits.

- [ ] **Step 3: Commit**

```bash
git add notebooks/MLPC/week05_clustering_neu_det.qmd
git commit -m "docs(mlpc-w5): add K-means clustering with K-sweep"
```

---

## Task 11: Notebook Section 5 — GMM on both features + BIC sweep

**Files:**
- Modify: `notebooks/MLPC/week05_clustering_neu_det.qmd`

- [ ] **Step 1: Append Section 5**

```markdown
## 5. Gaussian Mixture (EM)

```{python}
def run_gmm_sweep(Z, K_range=range(2, 11), seed=0):
    bics = []
    for K in K_range:
        gm = GaussianMixture(
            n_components=K, covariance_type="diag", random_state=seed, n_init=3
        ).fit(Z)
        bics.append(gm.bic(Z))
    return list(K_range), bics

for name, Z in [("raw+PCA", Z_pca), ("ResNet18", Z_resnet_std)]:
    Ks, bics = run_gmm_sweep(Z)
    plt.figure(figsize=(6, 3.5))
    plt.plot(Ks, bics, "o-")
    plt.axvline(6, color="gray", linestyle=":", alpha=0.6)
    plt.xlabel("K"); plt.ylabel("BIC (lower = better)")
    plt.title(f"GMM BIC sweep — {name}")
    plt.tight_layout()
    plt.show()

gmm_results = {}
for name, Z in [("raw+PCA", Z_pca), ("ResNet18", Z_resnet_std)]:
    gm = GaussianMixture(
        n_components=6, covariance_type="diag", random_state=0, n_init=3
    ).fit(Z)
    labels = gm.predict(Z)
    resp = gm.predict_proba(Z)
    gmm_results[name] = {
        "labels": labels,
        "Z": Z,
        # score for tile plot: max responsibility (assignment confidence)
        "score": resp.max(axis=1),
    }
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    ent = -(resp * np.log(resp + 1e-12)).sum(axis=1).mean()
    print(f"GMM K=6 on {name:9s}: ARI={ari:.3f}  NMI={nmi:.3f}  mean-entropy={ent:.3f}")
```
```

- [ ] **Step 2: Render**

Run: `quarto render notebooks/MLPC/week05_clustering_neu_det.qmd`
Expected: exits 0; HTML shows two BIC plots and prints GMM ARI/NMI/entropy.

- [ ] **Step 3: Commit**

```bash
git add notebooks/MLPC/week05_clustering_neu_det.qmd
git commit -m "docs(mlpc-w5): add GMM clustering with BIC sweep"
```

---

## Task 12: Notebook Section 6 — Evaluation panel (4 combinations)

**Files:**
- Modify: `notebooks/MLPC/week05_clustering_neu_det.qmd`

- [ ] **Step 1: Append Section 6**

```markdown
## 6. Evaluation panel — t-SNE, tiles, contingency for each (features × algorithm)

```{python}
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)

# Cache t-SNE projections per feature set so we don't re-run for KMeans and GMM.
tsne_proj = {}
for name, Z in [("raw+PCA", Z_pca), ("ResNet18", Z_resnet_std)]:
    print(f"running t-SNE on {name} ...")
    tsne_proj[name] = TSNE(
        n_components=2, perplexity=30, init="pca", random_state=0
    ).fit_transform(Z)
```

```{python}
summary_rows = []
for algo_name, results in [("KMeans", kmeans_results), ("GMM", gmm_results)]:
    for feat_name in ["raw+PCA", "ResNet18"]:
        r = results[feat_name]
        labels = r["labels"]
        Z2 = tsne_proj[feat_name]

        plot_tsne_dual(
            Z2, y_true, labels, ds.class_names,
            title=f"{algo_name} on {feat_name}",
        )
        plot_cluster_tiles(
            ds, labels, r["score"], n_per=6,
            title=f"{algo_name} on {feat_name} — exemplar tiles per cluster",
        )
        ari, nmi = plot_contingency(
            y_true, labels, ds.class_names,
            title=f"{algo_name} on {feat_name}",
        )
        summary_rows.append(
            dict(features=feat_name, algorithm=algo_name, ARI=ari, NMI=nmi)
        )

summary = pd.DataFrame(summary_rows)
summary
```
```

- [ ] **Step 2: Render**

Run: `quarto render notebooks/MLPC/week05_clustering_neu_det.qmd`
Expected: exits 0; HTML shows 4 t-SNE dual scatters, 4 tile grids, 4 contingency heatmaps, and the summary dataframe table.

- [ ] **Step 3: Commit**

```bash
git add notebooks/MLPC/week05_clustering_neu_det.qmd
git commit -m "docs(mlpc-w5): add 4-way evaluation panel (tsne+tiles+contingency)"
```

---

## Task 13: Notebook Section 7 — Wrap-up summary plot + discussion

**Files:**
- Modify: `notebooks/MLPC/week05_clustering_neu_det.qmd`

- [ ] **Step 1: Append Section 7**

```markdown
## 7. Wrap-up

```{python}
fig, ax = plt.subplots(figsize=(7, 4))
width = 0.35
x = np.arange(len(summary))
ax.bar(x - width / 2, summary["ARI"], width, label="ARI")
ax.bar(x + width / 2, summary["NMI"], width, label="NMI")
ax.set_xticks(x)
ax.set_xticklabels([f"{r.algorithm}\n{r.features}" for r in summary.itertuples()],
                   fontsize=9)
ax.set_ylabel("score (higher = better)")
ax.set_title("Clustering quality — 4 combinations on NEU-DET")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
```

### Takeaways

- **Representation beats algorithm.** For both KMeans and GMM, ResNet18
  embeddings cluster substantially better than raw-pixel features. The
  features carry most of the signal; the choice between hard and soft
  assignment is secondary.
- **Some defects are easy, others are not.** Scratches and patches form
  visually-coherent clusters; crazing and rolled-in_scale tend to be
  conflated regardless of features. The contingency heatmaps make this
  failure mode visible at a glance.
- **GMM's soft assignments add interpretability, not accuracy here.**
  Mean assignment entropy is informative ("which samples is the model
  least sure about?") but ARI/NMI track KMeans closely.
```

- [ ] **Step 2: Render the full notebook**

Run: `quarto render notebooks/MLPC/week05_clustering_neu_det.qmd`
Expected: exits 0; the rendered HTML contains the summary bar chart and the takeaways markdown.

- [ ] **Step 3: Sanity-check figure count**

Run: `grep -c '<img' _site/notebooks/MLPC/week05_clustering_neu_det.html`
Expected: returns a count ≥ 14. (1 class grid + 1 scree + 2 KMeans sweeps + 2 GMM sweeps + 4 t-SNE + 4 tile grids + 4 contingency + 1 summary = 19 nominal; some may render as `<svg>` inline, so we test the conservative lower bound.)

- [ ] **Step 4: Commit**

```bash
git add notebooks/MLPC/week05_clustering_neu_det.qmd
git commit -m "docs(mlpc-w5): add summary plot and takeaways section"
```

---

## Task 14: Index page entry

**Files:**
- Modify: `index.qmd`

- [ ] **Step 1: Update the MLPC week-5 row**

Find the existing line in `index.qmd` under the **MLPC** section (currently `| 5 | Unsupervised learning in materials | Ising / Cahn-Hilliard | **braided:** ...; per-course: [week11_anomaly_cahn_hilliard]...`) and edit it so the dataset cell and the per-course list both mention NEU-DET:

```diff
-| 5 | Unsupervised learning in materials | Ising / Cahn-Hilliard | **braided:** [week5_clustering_and_autoencoders.py](notebooks/week5_clustering_and_autoencoders.py); per-course: [week11_anomaly_cahn_hilliard](notebooks/MLPC/week11_anomaly_cahn_hilliard.html) |
+| 5 | Unsupervised learning in materials | Ising / Cahn-Hilliard / NEU-DET | **braided:** [week5_clustering_and_autoencoders.py](notebooks/week5_clustering_and_autoencoders.py); per-course: [week05_clustering_neu_det](notebooks/MLPC/week05_clustering_neu_det.html), [week11_anomaly_cahn_hilliard](notebooks/MLPC/week11_anomaly_cahn_hilliard.html) |
```

- [ ] **Step 2: Verify the link appears**

Run: `grep "week05_clustering_neu_det" index.qmd`
Expected: prints exactly one match.

- [ ] **Step 3: Re-render the site index**

Run: `quarto render index.qmd`
Expected: exits 0; `_site/index.html` contains the new link.

- [ ] **Step 4: Commit**

```bash
git add index.qmd
git commit -m "docs(index): link MLPC week 5 NEU-DET clustering notebook"
```

---

## Task 15: Final verification pass

**Files:** (none)

- [ ] **Step 1: Run every dataset test**

Run: `pytest tests/datasets/test_neu_det.py -v`
Expected: 7 passed, 0 skipped. (The `tmp_path` test runs regardless of
whether NEU-DET is downloaded; the rest skip when data is absent.)

- [ ] **Step 2: Confirm clean state**

Run: `git status`
Expected: `nothing to commit, working tree clean`. The `data/NEU-DET/`
directory and the embedding cache stay un-tracked (covered by `.gitignore`).

- [ ] **Step 3: List the commits this branch added**

Run: `git log --oneline main..HEAD` (if working on a feature branch) or `git log --oneline -20` to spot-check.
Expected: ~14 commits, one per task above; messages match the patterns used in the plan.

- [ ] **Step 4: Optional — clean re-run from scratch**

If time permits, validate the "clean machine" path:

```bash
rm -rf data/NEU-DET _freeze/notebooks/MLPC/week05_clustering_neu_det
python -c "from ai4mat.datasets import NEUDETDataset; ds = NEUDETDataset(); print(len(ds))"
quarto render notebooks/MLPC/week05_clustering_neu_det.qmd
```

Expected: download runs, dataset reports 1800, notebook renders to HTML without errors. This is the verification command from the spec's Definition of Done.

---

## Notes for the implementer

- **Frequent commits.** Every task ends with one commit. Do not batch commits across tasks.
- **Render after every notebook section.** Quarto errors are easier to bisect when the diff is small.
- **Do not commit data or embeddings.** `data/NEU-DET/` and `data/NEU-DET/embeddings_resnet18.npz` are intentionally outside the repo.
- **GPU is optional.** The notebook auto-detects CUDA. On CPU, Task 8 will take ~30 s instead of ~10 s; everything else is unchanged.
- **The Hungarian remap is for visualisation only.** ARI and NMI are permutation-invariant; the heatmap diagonal alignment is purely a readability convenience.
- **YAGNI reminders.** No XML annotation parsing. No autoencoder. No UMAP. No DBSCAN. No augmentation. No train/val split for clustering (we use everything).
