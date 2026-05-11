# ESTM thermoelectric clustering notebook — MLPC Unit 5

**Date:** 2026-05-11
**Author:** Philipp Pelz (with Claude)
**Course slot:** MLPC Week 5 — "Unsupervised learning in materials"
**Sibling spec:** `2026-05-11-neu-det-clustering-mlpc-unit5-design.md`

## Goal

Add a per-course MLPC notebook that clusters the ESTM thermoelectric dataset
(Na & Chang, *npj Comput. Mater.* 8 (2022) 214,
[doi:10.1038/s41524-022-00897-2](https://doi.org/10.1038/s41524-022-00897-2);
5 205 experimental observations over 880 unique compounds with Seebeck
coefficient, electrical conductivity, thermal conductivity, power factor and
ZT at known temperature) using **two parallel feature pipelines** (element
fractions + T, and matminer/Magpie + T) and a **single algorithm**
(K-means, with elbow + silhouette diagnostics). The pedagogical punchline is:
*on real composition data the choice of featurization changes which clusters
emerge more than the choice of k, and clustering by composition descriptors
reproduces canonical thermoelectric families (PbTe-like, Bi₂Te₃-like,
SnSe-like, skutterudites, half-Heuslers, …) — which then visibly enrich
for high ZT.*

This is the per-course companion to the braided in-class notebook
`notebooks/week5_clustering_and_autoencoders.py` and sits alongside the
NEU-DET week 5 notebook described in the sibling spec. NEU-DET answers
"clustering an image dataset"; ESTM answers "clustering a tabular materials
dataset by composition".

## Scope

In scope:

- A new `ESTMDataset` class in `ai4mat/datasets/estm.py`, re-exported from
  `ai4mat.datasets.__init__`.
- A new Quarto notebook `notebooks/MLPC/week05_clustering_estm.qmd`.
- A `download_if_missing()` helper that fetches the public ESTM CSV from the
  Na & Chang GitHub release into `data/estm/`.
- A featurization step that caches Magpie / fraction features to
  `data/estm/features_<mode>.npz` so reruns are instant.
- The notebook *also* explicitly `fig.savefig(...)`s six headline figures
  into `_public_presentations/ml_for_characterization_and_processing/unit05_unsupervised_learning/images/estm/`
  so the lecture deck has slide-ready PNGs from one render.
- An entry in `index.qmd` under the MLPC week-5 row.
- One smoke test in `tests/datasets/test_estm.py`.
- Adding `matminer>=0.9`, `pymatgen>=2024.3` to `requirements.txt`
  (scikit-learn, matplotlib, pandas, numpy are already pinned there).

Out of scope (YAGNI):

- t-SNE / UMAP / hierarchical / DBSCAN / GMM — only PCA + K-means.
- The full SIMD descriptor of Na & Chang. We use generic featurization
  (fractions, Magpie). SIMD is mentioned in the wrap-up as further reading.
- Train/val/test splits — clustering is unsupervised, all rows used.
- Autoencoder on ESTM — autoencoder content stays in the braided file.
- Anything that touches the slide deck source (`01_intro.qmd`). We drop PNGs
  into `images/estm/` only; integration into slides is Philipp's job.
- A separate test that exercises matminer featurization (heavy dependency
  chain; smoke test skips when CSV is missing and the featurization path is
  exercised by the notebook render in CI-equivalent local runs).

## Data layer — `ai4mat/datasets/estm.py`

A new `ESTMDataset(torch.utils.data.Dataset)`. Unlike the `mdsdata`-backed
datasets (Iris, Digits, ChemicalElements), this one — like
`NEUDETDataset` — owns its on-disk layout and download path. Constructor:

```python
class ESTMDataset(Dataset):
    def __init__(self,
                 root: str = "data/estm",
                 features: str = "fraction",   # "fraction" | "magpie"
                 target: str = "ZT",           # "ZT" | "S" | "sigma" | "kappa" | "PF" | "all"
                 download: bool = True,
                 standardize: bool = False,
                 transform=None,
                 target_transform=None):
```

**On-disk layout** (after download + first featurization):

```
data/estm/
├── ESTM.csv                       # raw, untouched (5205 rows × ~8 cols)
├── README.md                      # source URL, license, citation
├── features_fraction.npz          # X_frac, columns, formulas, T, props
└── features_magpie.npz            # X_magpie, columns, formulas, T, props
```

**Expected CSV columns** (verified against the Na & Chang repo; if the
release ships them under different names we map them in
`_canonicalise_columns()` and update this spec):

| canonical name | meaning | unit |
|----------------|---------|------|
| `formula`      | chemical formula string                | —     |
| `T`            | measurement temperature                 | K     |
| `S`            | Seebeck coefficient                     | µV/K  |
| `sigma`        | electrical conductivity                 | S/m   |
| `kappa`        | thermal conductivity                    | W/m·K |
| `PF`           | power factor (= S²σ)                    | W/m·K² |
| `ZT`           | figure of merit                          | —     |

**Behaviour:**

- If `download=True` and `<root>/ESTM.csv` is missing, call
  `download_if_missing(root)` (specified below). If `download=False` and the
  CSV is absent, raise `FileNotFoundError` with a one-line hint.
- Load the CSV with `pandas.read_csv`, call `_canonicalise_columns(df)` to
  rename/select the seven canonical columns above. Drop any row with NaN in
  `T`, `formula`, or the chosen target(s).
- Build features lazily by mode:
  - `features="fraction"`: parse each formula with `pymatgen.core.Composition`,
    compute the fractional composition over the full 118-element periodic
    table, and stack into an `(N, 119)` matrix (118 elements + `T` as last
    column). Rows whose formula fails to parse are dropped, with the count
    logged.
  - `features="magpie"`: use
    `matminer.featurizers.composition.ElementProperty.from_preset("magpie")`
    on the `Composition` objects, append `T` as the last feature. ~133
    columns. Replace any remaining NaNs with the column median.
- Cache the resulting matrix and column names to `<root>/features_<mode>.npz`
  (keys: `X`, `columns`, `formulas`, `T`, `S`, `sigma`, `kappa`, `PF`, `ZT`).
  On second instantiation, load from the cache and skip featurization
  entirely — so a fresh notebook run that has both caches present does not
  import matminer.
- If `standardize=True`, fit a `sklearn.preprocessing.StandardScaler` on `X`
  and store it as `self.scaler`. Default `False` so the notebook controls
  standardization explicitly.
- `__getitem__(i)` returns `(x: Tensor, y: Tensor)`:
  - `x` is `torch.float32` of shape `(F,)` where `F = 119` or `~134`.
  - `y` is `torch.float32` of shape `()` if `target ∈ {"ZT","S",...}`, or
    `(5,)` if `target="all"`.
- Public attributes (set after `__init__`):
  - `self.X: torch.float32 (N, F)` — features (standardized iff requested).
  - `self.y: torch.float32 (N,)` or `(N, 5)` — targets.
  - `self.feature_names: list[str]` — `F` entries.
  - `self.formulas: list[str]` — `N` entries; same order as `X`.
  - `self.T: torch.float32 (N,)` — measurement temperatures.
  - `self.properties: pandas.DataFrame` of shape `(N, 5)` with columns
    `["ZT", "S", "sigma", "kappa", "PF"]` — used by the notebook for
    per-cluster property enrichment plots regardless of which `target` was
    requested.

**`download_if_missing(root)` — spec:**

- Primary source: the raw CSV in the Na & Chang GitHub release for the SIMD
  paper. URL candidates to try in order (first that returns a CSV ≥ 100 kB
  with the expected columns wins):
  1. `https://raw.githubusercontent.com/ngs00/SIMD/main/dataset/ESTM.csv`
  2. `https://raw.githubusercontent.com/ngs00/SIMD/master/dataset/ESTM.csv`
  3. `https://github.com/ngs00/SIMD/raw/main/dataset/ESTM.csv`
- `urllib.request.urlopen` with a `tqdm` progress bar driven by
  `Content-Length` when present.
- Atomic write: download to `<root>/ESTM.csv.partial`, validate (`pandas`
  can parse it, has ≥ 5 000 rows, contains the formula + temperature
  columns we can canonicalize), then rename to `ESTM.csv`. On failure,
  delete the partial file and try the next URL.
- If all URLs fail, raise `RuntimeError` with a one-line manual-download
  hint pointing at the Zenodo/GitHub mirror and the expected layout.
- Also writes a `README.md` next to the CSV that records (i) the URL that
  succeeded, (ii) the SHA-256 of the downloaded file, (iii) the paper
  citation, (iv) the date.
- Idempotent: if `<root>/ESTM.csv` exists, no-op.

**Public API:**

```python
ds = ESTMDataset()                                 # 5205-ish rows, fraction feats, ZT target
ds = ESTMDataset(features="magpie")                # Magpie feats
ds = ESTMDataset(target="all", standardize=True)   # all 5 props, z-scored X
len(ds), ds[0][0].shape, ds[0][1].shape            # ~5205, torch.Size([119]), torch.Size([])
ds.X.shape, ds.y.shape                             # (N, 119), (N,)
ds.formulas[:3]                                    # ["PbTe", "Bi2Te3", "SnSe", ...]
ds.properties.head()                               # DataFrame with the 5 props
ds.T[:3]                                           # tensor([300., 400., 500.])
```

**Re-export:** append `"ESTMDataset"` to `__all__` in
`ai4mat/datasets/__init__.py`, and add `from .estm import ESTMDataset`
alongside the others.

## Notebook — `notebooks/MLPC/week05_clustering_estm.qmd`

Quarto format, frontmatter matches `week05_clustering_neu_det.qmd`:

```yaml
---
title: "MLPC Week 5: Clustering thermoelectric materials (ESTM)"
subtitle: "PCA + K-means on element-fraction and Magpie features"
jupyter: python3
---
```

Colab badge under the title, then learning objectives:

- Apply K-means to a tabular materials dataset with quantitative property targets.
- Understand how featurization choice (raw fractions vs physics-aware Magpie descriptors) changes the cluster geometry.
- Use elbow + silhouette together to motivate a cluster count.
- Read per-cluster property distributions as a sanity check that clusters mean something.
- Connect this workflow to materials discovery (Na & Chang 2022 / SIMD).

### Section layout

1. **Setup & dataset preview.** Imports; instantiate `ESTMDataset()` for
   both `features="fraction"` and `features="magpie"`; print shapes; show a
   `pd.crosstab` of the 15 most common dominant elements vs. coarse
   temperature bins (300/500/700/900 K) to give a feel for the dataset.

2. **Feature pipeline A — element fractions + T.**
   - Standardise (`StandardScaler`) on `ds_frac.X`.
   - `PCA(n_components=10, random_state=0)` for clustering input;
     `PCA(n_components=2)` for plotting; print cumulative explained variance.
   - Save `Z_frac10` and `Z_frac2`.

3. **Feature pipeline B — Magpie descriptors + T.**
   - Standardise on `ds_mag.X`.
   - Same PCA structure → `Z_mag10`, `Z_mag2`. Print first 5 loadings of PC1
     and PC2 to read off which Magpie descriptors dominate (e.g. mean
     atomic mass vs mean electronegativity).

4. **K-means + elbow/silhouette on both feature sets.**
   - K-sweep `K=2..12`, `KMeans(n_init=10, random_state=0)`.
   - Plot inertia (elbow) and silhouette score vs K on a side-by-side
     panel — **figure 1** (`elbow_silhouette.png`).
   - Pick `K*` as the argmax silhouette per featurization (typically 4–7 on
     this data, exact value reported in the rendered notebook).

5. **Cluster visualisation in PCA space.**
   - Two PCA-2D scatters (fraction / Magpie), each coloured by K-means
     cluster id with cluster centroids overlaid. Centroid annotations show
     the single most common dominant element across cluster members
     (helper: `_dominant_element(formulas) → str`). → **figure 2**
     (`pca_scatter_fraction.png`) and **figure 3** (`pca_scatter_magpie.png`).
   - Same two scatters again, coloured continuously by ZT
     (viridis, vmin=0, vmax≈2.5). → **figure 4**
     (`pca_scatter_by_zt.png`) — a two-panel composite (fraction left,
     Magpie right) using the same axes as fig 2/3.

6. **Per-cluster material families & property enrichment.**
   - Table: for each Magpie cluster, list the top 5 most common dominant
     elements + the median ZT, S, σ, κ. Printed via
     `df.style.background_gradient` (rendered inline).
   - **Figure 5** (`property_box_per_cluster.png`): 1×4 grid of box plots
     (ZT, S, σ, κ), x-axis = Magpie cluster id, log y where helpful.
   - **Figure 6** (`zt_by_cluster_vs_T.png`): line plot of median ZT vs
     temperature, one line per cluster — exposes high-temperature vs
     low-temperature families.

7. **Wrap-up.** Markdown paragraph stating the takeaways:
   - Featurization choice dominates: Magpie clusters separate chalcogenides
     from skutterudites/half-Heuslers cleanly, fraction-feature clusters
     mostly track "which element is dominant" and miss subtler structure.
   - Cluster identity correlates with high-ZT: a small number of Magpie
     clusters concentrate the top-decile ZT entries.
   - Forward link to Na & Chang's SIMD descriptor as a learned, materials-
     aware featurization that pushes this further.

### Slide-figure save helper (notebook-local)

`_quarto.yml` sets `execute-dir: project`, so the notebook's runtime cwd is
the `Ai4MatLectures/` project root, not `notebooks/MLPC/`. The slide deck
lives at `/home/philipp/projects/_public_presentations/...`, and
`SS26/_public_presentations` is a symlink to it (verified via `readlink -f`
at spec time); from the project root that path is one level up. The helper
must therefore:

1. Resolve target via env var `ESTM_SLIDE_IMG_DIR` if set, else fall back to
   the relative path `../_public_presentations/ml_for_characterization_and_processing/unit05_unsupervised_learning/images/estm`.
2. **Only `mkdir` the final `estm/` leaf, and only if the parent
   `images/` already exists.** This is what guards against the
   "phantom `_public_presentations` tree silently materialises inside
   `Ai4MatLectures/`" failure mode the spec reviewer flagged.
3. If the parent does not exist, fall back to a notebook-local
   `figs/estm/` directory (created), and print a one-line warning with
   the resolved absolute path of both candidates so the human sees why
   PNGs aren't reaching the slide repo.

```python
import os, warnings
from pathlib import Path

def _resolve_slide_img_dir() -> Path:
    env = os.environ.get("ESTM_SLIDE_IMG_DIR")
    target = Path(env) if env else Path(
        "../_public_presentations/ml_for_characterization_and_processing/"
        "unit05_unsupervised_learning/images/estm"
    )
    if target.parent.exists():
        target.mkdir(exist_ok=True)
        return target
    fallback = Path("figs/estm")
    fallback.mkdir(parents=True, exist_ok=True)
    warnings.warn(
        f"slide-deck images parent {target.parent.resolve()} not found; "
        f"writing slide PNGs to {fallback.resolve()} instead"
    )
    return fallback

SLIDE_IMG_DIR = _resolve_slide_img_dir()
def save_slide_fig(fig, name):
    fig.savefig(SLIDE_IMG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
```

Each headline figure is rendered inline (Quarto handles this) *and*
explicitly saved through `save_slide_fig(fig, "elbow_silhouette")` etc., so
one `quarto render` produces both HTML and the six slide PNGs (or, in the
fallback case, six local PNGs and an audible warning).

### Runtime budget (CPU)

| Step | Time |
|------|------|
| CSV load + canonicalise | <1 s |
| Fraction featurization (cold) | ~5 s |
| Magpie featurization (cold) | ~40 s (matminer is slow) |
| Featurization (cached) | <0.5 s |
| PCA + standardise × 2 | ~1 s |
| K-means K-sweep × 2 | ~5 s |
| **Cold start** | **~50 s** |
| **Warm (caches present)** | **~10 s** |

## Supporting changes

### `requirements.txt`

Append two lines:

```
matminer>=0.9
pymatgen>=2024.3
```

scikit-learn, matplotlib, pandas, numpy, tqdm are already pinned. No change
to `pyproject.toml` — the importable `ai4mat` package stays minimal; matminer
+ pymatgen are notebook-time deps.

### `index.qmd`

Under the MLPC table, update the Week 5 row to list both new notebooks:

```
| 5 | Unsupervised learning in materials | Ising / Cahn-Hilliard / NEU-DET / ESTM |
**braided:** [week5_clustering_and_autoencoders.py](...); per-course:
[week05_clustering_neu_det](notebooks/MLPC/week05_clustering_neu_det.html),
[week05_clustering_estm](notebooks/MLPC/week05_clustering_estm.html),
[week11_anomaly_cahn_hilliard](...) |
```

(The NEU-DET row edit may already be in flight from the sibling spec —
this spec assumes that row exists when it lands and adds the ESTM entry
next to it; if NEU-DET hasn't landed yet, this row gets created here and
NEU-DET appends to it later. Resolution lives in whichever PR lands second.)

### Test — `tests/datasets/test_estm.py`

A single file with two smoke tests, both skipped when the CSV isn't
present:

```python
import os, pytest, torch
from ai4mat.datasets import ESTMDataset

DATA_DIR = "data/estm"
CSV = os.path.join(DATA_DIR, "ESTM.csv")

@pytest.mark.skipif(not os.path.isfile(CSV), reason="ESTM CSV not present")
def test_estm_fraction():
    ds = ESTMDataset(features="fraction", download=False)
    assert len(ds) > 4500
    x, y = ds[0]
    assert x.shape == (119,)
    assert x.dtype == torch.float32
    assert y.shape == ()
    assert len(ds.formulas) == len(ds)
    assert ds.properties.shape == (len(ds), 5)

@pytest.mark.skipif(not os.path.isfile(CSV), reason="ESTM CSV not present")
@pytest.mark.slow
def test_estm_magpie():
    ds = ESTMDataset(features="magpie", download=False)
    x, _ = ds[0]
    assert x.shape[0] > 100      # ~134 Magpie + T features
    assert torch.isfinite(x).all()
```

The `slow` mark is registered already in `pyproject.toml`'s
`[tool.pytest.ini_options]`. CI without the CSV is a no-op; locally it's a
fast sanity check after the first notebook run has populated the cache.

## Risks / open questions

- **GitHub URL stability.** The Na & Chang SIMD repo is the canonical
  source. If `raw.githubusercontent.com/ngs00/SIMD/.../dataset/ESTM.csv`
  has moved by the time this runs, the fallback chain of three URLs
  catches the common renames; if all fail, the function dies loud with a
  manual-download hint. We will verify the live URL during implementation
  and pin the canonical one as URL #1.
- **Column-name drift.** Na & Chang's CSV uses field names like
  `temperature`, `electrical conductivity` with units in headers. The
  `_canonicalise_columns(df)` helper handles the mapping; if a column name
  cannot be matched to one of the seven canonical names, the function
  raises a `KeyError` listing what it found and what it expected.
- **Formula parsing failures.** Some ESTM entries are non-stoichiometric
  (e.g. `Bi0.5Sb1.5Te3`). `pymatgen.Composition` handles fractional
  subscripts fine in standard cases but may reject other oddities. The
  loader counts and logs dropped rows; we expect ≤ 1 % loss.
- **Magpie NaN columns.** Some Magpie descriptors are undefined for noble
  gases / actinides; column-median imputation is the standard
  matminer-side recipe. If a *row* still has all-NaN features after
  imputation it is dropped with a warning.
- **`pymatgen` install footprint.** pymatgen is ~150 MB. We accept that for
  the notebook-time deps but do not push it into the base `ai4mat` install.
- **Number of clusters.** The silhouette-argmax may land at K=2 (one
  large + one outlier cluster) on the raw fractions; the notebook reports
  the K it picks and visualises it honestly rather than fudging to a
  prettier K. If K=2 is uninformative, the wrap-up calls that out — it
  *is* the lesson about featurization mattering.

## Definition of done

Each item below has a verification command in brackets — run it before
calling the work complete.

- `ai4mat/datasets/estm.py` written; re-exported.
  [`python -c "from ai4mat.datasets import ESTMDataset; print(ESTMDataset.__doc__[:80])"`]
- `ESTMDataset()` downloads and loads ≥ 4500 rows on a clean machine.
  [`rm -rf data/estm && python -c "from ai4mat.datasets import ESTMDataset; ds=ESTMDataset(); assert len(ds)>4500, len(ds); print('OK', len(ds))"`]
- `notebooks/MLPC/week05_clustering_estm.qmd` renders end-to-end via
  Quarto and produces all named figures inline *and* writes the six PNGs
  to the slide-deck image folder (verified by resolving via the symlink,
  not against a phantom local tree).
  [`quarto render notebooks/MLPC/week05_clustering_estm.qmd` exits 0,
  AND
  `ls /home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit05_unsupervised_learning/images/estm/*.png | wc -l`
  is ≥ 6,
  AND `find Ai4MatLectures/_public_presentations -type d 2>/dev/null` is
  empty (no phantom tree was created inside the repo).]
- `index.qmd` updated.
  [`grep "week05_clustering_estm" index.qmd` returns a match.]
- `tests/datasets/test_estm.py` passes when the CSV is present, skips
  otherwise.
  [`pytest tests/datasets/test_estm.py -v` passes locally with the data
  in place; `pytest tests/datasets/test_estm.py -v` in a fresh tmpdir
  with no data prints "skipped".]
- `requirements.txt` includes `matminer` and `pymatgen`.
  [`grep -E '^(matminer|pymatgen)' requirements.txt | wc -l` returns 2.]
- Committed on `main` (matching the precedent set by the NEU-DET work).
