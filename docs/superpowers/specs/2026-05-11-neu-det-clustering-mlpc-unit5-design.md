# NEU-DET clustering notebook — MLPC Unit 5

**Date:** 2026-05-11
**Author:** Philipp Pelz (with Claude)
**Course slot:** MLPC Week 5 — "Unsupervised learning in materials"

## Goal

Add a per-course MLPC notebook that clusters the NEU-DET steel-surface-defect
dataset using two feature pipelines (raw pixels + PCA vs ImageNet-pretrained
ResNet18 embeddings) and two algorithms (K-means and GMM/EM), with a full
evaluation panel against the 6 ground-truth defect classes. The narrative
punchline is: *learned features beat raw pixels for unsupervised structure
discovery, and the algorithm choice matters less than the representation.*

This is the per-course companion to the braided in-class notebook
`notebooks/week5_clustering_and_autoencoders.py`, which already covers
clustering and autoencoders on the Ising and Nanoindentation datasets. The
new notebook applies the same toolkit to a real microscopy dataset.

## Scope

In scope:

- A new `NEUDETDataset` class in `ai4mat/datasets/neu_det.py`, re-exported
  from `ai4mat.datasets.__init__`.
- A new Quarto notebook `notebooks/MLPC/week05_clustering_neu_det.qmd`.
- A `download_if_missing()` helper that fetches the public Kaggle URL into
  `data/NEU-DET/` and unzips it.
- An entry in `index.qmd` under the MLPC table.
- One smoke test in `tests/test_neu_det_dataset.py`.
- Adding `torchvision>=0.16` to `requirements.txt`.

Out of scope (YAGNI):

- Parsing the NEU-DET XML object-detection annotations (we only need class
  labels, which are encoded in directory names).
- Train/val splits — clustering is unsupervised, we use all 1800 images.
- Data augmentation.
- Autoencoders on NEU-DET (covered for Ising in the braided file).
- UMAP, DBSCAN, hierarchical clustering — explicitly deferred.

## Data layer — `ai4mat/datasets/neu_det.py`

A new `NEUDETDataset(torch.utils.data.Dataset)` modelled on the existing
`IsingDataset` (same `X` / `y` / `transform` / `target_transform` shape),
but extending the API with `root`, `split`, `download`, `class_names`, and
`image_paths` because the data lives on disk rather than coming from
`mdsdata`. The extra attributes are called out explicitly below — this
class is *not* a drop-in API match to IsingDataset.

**Expected on-disk layout** (after unzipping the Kaggle archive):

```
data/NEU-DET/
├── train/
│   ├── images/
│   │   ├── crazing/         (240 .jpg)
│   │   ├── inclusion/
│   │   ├── patches/
│   │   ├── pitted_surface/
│   │   ├── rolled-in_scale/
│   │   └── scratches/
│   └── annotations/         (ignored)
└── validation/
    ├── images/<class>/      (60 .jpg per class)
    └── annotations/         (ignored)
```

The exact split sizes (240/60) are confirmed by the Kaggle dataset card;
total = 1800 images, 300 per class.

**Class names** (alphabetical, sorted by directory name; label indices 0..5):
`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`,
`scratches`. The class names live on the instance as `ds.class_names`.

**Constructor signature:**

```python
class NEUDETDataset(Dataset):
    def __init__(self,
                 root: str = "data/NEU-DET",
                 split: str = "all",      # "train" | "validation" | "all"
                 download: bool = True,
                 transform=None,
                 target_transform=None):
```

**Behaviour:**

- If `download=True` and `root` is missing or empty, call
  `download_if_missing(root)` (specified below); this populates the
  expected layout. If `download=False` and the data is absent, raise
  `FileNotFoundError` with a one-line hint to set `download=True`.
- After download, the dataset finds the `train/` directory: if
  `<root>/train` is missing, it walks one level deeper looking for the
  single subdirectory that contains `train/`, and rebinds `root` to that
  subdirectory. (Kaggle archives sometimes unzip with a nested top-level
  folder like `NEU-DET/NEU-DET/...` or `NEU Metal Surface Defects Data/`.)
  If no such directory is found, raise a clear `FileNotFoundError`.
- Walks `<root>/<split-dirs>/images/<class>/*.jpg` (class subdirs
  discovered by `sorted(os.listdir(...))` so label indexing is stable
  across machines) and loads every image with `imageio.v3.imread` (already
  a project dep).
- **Class-name normalisation**: discovered directory names are
  lowercased; any `_` in the name is preserved, any `-` is preserved
  (`rolled-in_scale` and `rolled_in_scale` are both recognised but the
  string is stored verbatim). The dataset asserts exactly 6 classes; the
  canonical class strings are exposed on `ds.class_names` as discovered
  (no rewriting). Tests must check `len(ds.class_names) == 6` and set
  containment of the lowercase tokens (see test section).
- Asserts the image is 200×200 grayscale (NEU-DET nominal); converts to
  `float32` tensor of shape `(1, 200, 200)` in `[0, 1]` (matches IsingDataset).
- Stores `self.X` as one `torch.float32` tensor of shape `(N, 1, 200, 200)`
  and `self.y` as `torch.long` of shape `(N,)`.
- Eager loading: ~1800 × 1 × 200 × 200 × 4 B ≈ 274 MB; acceptable for a
  lecture machine. Matches the load-everything-up-front idiom used by every
  other ai4mat dataset.
- Exposes `self.class_names: list[str]`, `self.image_paths: list[str]`,
  `self.X`, `self.y` as **public attributes** — the notebook reads
  `ds.X` / `ds.y` directly for the clustering steps rather than iterating
  `__getitem__`.

**`download_if_missing(root)` — spec:**

- Streams the public URL
  `https://www.kaggle.com/api/v1/datasets/download/kaustubhdikshit/neu-surface-defect-database`
  with `urllib.request.urlopen`, writing chunks of 1 MB to
  `<root>.tmp.zip` (in the parent dir of `root`).
- Shows a `tqdm` progress bar (project already depends on `tqdm`) sized
  from the `Content-Length` header when present, otherwise unbounded.
- Before unzipping, sniffs the first 4 bytes: if they are not `PK\x03\x04`
  (zip magic), raises `RuntimeError("Kaggle endpoint did not return a
  zip — likely an auth redirect; download manually and place files in
  data/NEU-DET/")` and deletes the temp file. Guards against the silent
  "Kaggle returned an HTML login page" failure mode.
- On success: unzips into `root` with `zipfile.ZipFile`, removes the temp
  zip. Does not move nested directories — the dataset class handles
  nesting on read (see above).
- Idempotent: if `root` exists and contains a `train/` folder anywhere
  one level deep, this function is a no-op.

**Public API:**

```python
ds = NEUDETDataset()                 # 1800 imgs, downloads if missing
ds = NEUDETDataset(split="train")    # 1440 imgs
ds = NEUDETDataset(download=False)   # FileNotFoundError if data dir is empty
len(ds), ds[0][0].shape, ds[0][1]    # 1800, torch.Size([1,200,200]), tensor(0)
ds.X.shape                           # torch.Size([1800, 1, 200, 200])
ds.y.shape                           # torch.Size([1800])
ds.class_names                       # ["crazing", "inclusion", ...]
ds.image_paths                       # ["data/NEU-DET/train/images/crazing/...", ...]
```

**Re-export:** append `"NEUDETDataset"` to the existing `__all__` list at
the bottom of `ai4mat/datasets/__init__.py` (lines 10–19), and add
`from .neu_det import NEUDETDataset` alongside the other imports.

## Notebook — `notebooks/MLPC/week05_clustering_neu_det.qmd`

Quarto format, frontmatter and structure match the sibling
`week05_cnn_ising_full.qmd`:

```yaml
---
title: "MLPC Week 5: Clustering steel-surface defects (NEU-DET)"
subtitle: "K-means vs GMM on raw pixels and ResNet18 embeddings"
jupyter: python3
---
```

A Colab badge under the title, then learning objectives:

- Recognise when clustering recovers real structure vs spurious groups
- Compare raw-pixel features to pretrained CNN embeddings on the same task
- Read t-SNE scatters, contingency matrices and ARI/NMI together
- Connect K-means and GMM/EM (covered in MFML Unit 5) to a real defect-classification setting

### Section layout

1. **Setup & data preview.** Imports; `NEUDETDataset()`; print shapes; a
   `6 × 6` grid (one row per class, 6 random examples).

2. **Feature pipeline A — raw pixels + PCA(50).**
   - Flatten `X` to `(N, 40000)`.
   - Standardise per-feature (zero mean, unit variance).
   - `PCA(n_components=50, random_state=0)`; print cumulative explained
     variance; plot scree.
   - Save the resulting `Z_pca` (50-d) for clustering.

3. **Feature pipeline B — pretrained ResNet18 embeddings.**
   - `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`.
   - Strip the final `fc` (replace with `nn.Identity()`); `.eval()`; freeze.
   - Move to CUDA if available.
   - Preprocessing per batch: `F.interpolate` 200→224, replicate grayscale
     channel to 3, normalise with ImageNet `mean=[0.485, 0.456, 0.406]`,
     `std=[0.229, 0.224, 0.225]`.
   - Forward in batches of 128, no_grad, collect 512-d vectors.
   - Cache to `data/NEU-DET/embeddings_resnet18.npz` so reruns are instant;
     load from cache if present.
   - Standardise; keep `Z_resnet`.

4. **K-means on both feature sets.**
   - For each feature set: `KMeans(n_clusters=6, n_init=10, random_state=0)`.
   - K-sweep `K=2..10`: fit, record silhouette and ARI. Plot silhouette and
     ARI on twin axes vs K; annotate `K=6`.
   - Print ARI / NMI for the K=6 fit.

5. **GMM / EM on both feature sets.**
   - `GaussianMixture(n_components=6, covariance_type="diag", random_state=0, n_init=3)`.
     `diag` covariance keeps it tractable at 512-d.
   - BIC sweep `K=2..10`; plot BIC vs K.
   - Print ARI / NMI for the K=6 fit; also print mean predicted-assignment
     entropy as a soft-clustering sanity readout.

6. **Evaluation panel.** For each of the four `(features, algo)` combinations
   produce:
   - **t-SNE scatter.** `TSNE(n_components=2, perplexity=30, init="pca", random_state=0)`
     on the same `Z` used for clustering. Two side-by-side panels: left
     coloured by ground-truth label, right coloured by predicted cluster.
     `tab10` palette throughout.
   - **Per-cluster example tiles.** For KMeans: the 6 nearest-to-centroid
     images per cluster. For GMM: the 6 highest-responsibility images per
     component. Plotted as a 6 × 6 grid with cluster index as row label.
   - **Contingency matrix.** `pd.crosstab(true, pred)`, permute predicted
     columns with `scipy.optimize.linear_sum_assignment` (Hungarian) to
     maximise the diagonal. Render with `imshow`, true classes as rows
     (with names), predicted clusters as columns; print ARI / NMI in the
     title.

7. **Wrap-up.** One summary figure: grouped bar chart of {ARI, NMI} × the
   four combinations. Closing markdown paragraph stating the two takeaways:
   - Representation > algorithm. Pretrained ResNet18 features cluster
     substantially better than raw-pixel features for both KMeans and GMM.
   - GMM and KMeans give similar ARI on this dataset; the win comes from
     features, not from soft vs hard assignment.

### Plot helpers (notebook-local)

Three functions defined once near the top, used four times each:

```python
def plot_tsne_dual(Z2, y_true, y_pred, class_names, title): ...
def plot_cluster_tiles(images, labels_pred, centroids_or_resp, n_per=6): ...
def plot_contingency(y_true, y_pred, class_names, hungarian=True): ...
```

Each returns a `matplotlib.figure.Figure`. They are kept inline (not in
`ai4mat`) so students reading the lecture see exactly how each plot is built.

### Runtime budget (GPU available)

| Step | Time |
|------|------|
| Dataset load | <1 s (already on disk after first run) |
| PCA + standardise | ~2 s |
| ResNet18 embed (cached after 1st run) | ~10 s (first), ~0.3 s (cached) |
| KMeans/GMM K-sweep × 2 features | ~10 s |
| t-SNE × 4 | ~30–60 s each (the bottleneck) |
| **Total cold start** | **~4–5 min** |
| **Total warm** | **~3 min** |

## Supporting changes

### `requirements.txt`

Add one line: `torchvision>=0.16`. Notebook-only dep; no change to
`pyproject.toml` (the importable `ai4mat` package stays minimal).

### `index.qmd`

Under the MLPC table, replace the Week 5 row so it lists the new notebook
alongside the existing references:

```
| 5 | Unsupervised learning in materials | Ising / Cahn-Hilliard / NEU-DET |
**braided:** [week5_clustering_and_autoencoders.py](...); per-course:
[week05_clustering_neu_det](notebooks/MLPC/week05_clustering_neu_det.html),
[week11_anomaly_cahn_hilliard](...) |
```

### Test — `tests/test_neu_det_dataset.py`

A single file with one smoke test, skipped when the data isn't present:

```python
import os, pytest, torch
from ai4mat.datasets import NEUDETDataset

DATA_DIR = "data/NEU-DET"

@pytest.mark.skipif(
    not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR),
    reason="NEU-DET data not present"
)
def test_neu_det_basic():
    ds = NEUDETDataset(download=False)
    assert len(ds) == 1800
    x, y = ds[0]
    assert x.shape == (1, 200, 200)
    assert x.dtype == torch.float32
    assert 0.0 <= x.min().item() and x.max().item() <= 1.0
    assert set(ds.y.tolist()) == {0, 1, 2, 3, 4, 5}
    # Robust to '-' vs '_' in class dir names ("rolled-in_scale" vs
    # "rolled_in_scale"); only check core tokens are present.
    assert len(ds.class_names) == 6
    joined = " ".join(ds.class_names).lower()
    for token in ["crazing", "inclusion", "patches",
                  "pitted", "rolled", "scratches"]:
        assert token in joined
```

CI won't have the dataset, so the test is a no-op there; it provides a
local fast check after running the data download.

## Risks / open questions

- **Kaggle URL stability.** The unauthenticated public-URL download
  (`api/v1/datasets/download/...`) worked today but isn't formally
  supported. If it stops working we fall back to "user downloads manually
  and unzips into `data/NEU-DET/`" — the dataset class already supports
  that via `download=False`.
- **Class directory naming.** Kaggle archives occasionally re-zip with
  slightly different folder names (`rolled-in_scale` vs `rolled_in_scale`).
  The dataset class sorts whatever it finds and asserts there are exactly
  6 classes, so naming drift surfaces as a clear assertion error.
- **ResNet18 size mismatch.** NEU-DET is 200×200 grayscale; ResNet18 wants
  224×224 RGB. We resize + channel-replicate at preprocessing time. This
  is a standard recipe and gives ~95% acc on the supervised version of
  this task in published baselines, so the features are clearly strong
  enough to support a good clustering story.
- **GMM at 512-d.** Full covariance is intractable; `diag` covariance with
  3 inits is the standard choice and what we'll use. We note this in the
  notebook narrative.

## Definition of done

Each item below has a verification command in brackets — run it before
calling the work complete.

- `ai4mat/datasets/neu_det.py` written; re-exported.
  [`python -c "from ai4mat.datasets import NEUDETDataset; print(NEUDETDataset.__doc__[:60])"`]
- `NEUDETDataset()` downloads and loads 1800 images on a clean machine.
  [`rm -rf data/NEU-DET && python -c "from ai4mat.datasets import NEUDETDataset; ds=NEUDETDataset(); assert len(ds)==1800, len(ds); print('OK', len(ds))"`]
- `notebooks/MLPC/week05_clustering_neu_det.qmd` renders end-to-end via
  Quarto and produces all named figures.
  [`quarto render notebooks/MLPC/week05_clustering_neu_det.qmd` exits 0,
  AND the rendered HTML contains at least one image per evaluation panel:
  `grep -c '<img' _site/notebooks/MLPC/week05_clustering_neu_det.html`
  returns ≥ 14 (1 class grid + 1 scree + 4 sweeps + 4 t-SNE + 4 tile grids
  + 4 contingency + 1 summary; ≥14 is the conservative lower bound after
  Quarto inlines some as `<svg>`).]
- `index.qmd` updated.
  [`grep "week05_clustering_neu_det" index.qmd` returns a match.]
- `tests/test_neu_det_dataset.py` passes when the dataset is present,
  skips otherwise.
  [`pytest tests/test_neu_det_dataset.py -v` passes locally;
  `pytest tests/test_neu_det_dataset.py -v` in a fresh tmpdir with no
  data prints "skipped".]
- `requirements.txt` includes `torchvision>=0.16`.
  [`grep '^torchvision' requirements.txt` returns a match.]
- Committed on a feature branch.
