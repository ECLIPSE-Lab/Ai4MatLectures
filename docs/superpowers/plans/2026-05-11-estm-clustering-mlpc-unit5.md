# ESTM thermoelectric clustering notebook (MLPC Unit 5) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a per-course MLPC week 5 Quarto notebook that clusters the ESTM thermoelectric dataset with K-means on two composition feature pipelines (element-fraction and matminer/Magpie), plus the supporting `ESTMDataset` class, dataset tests, dep bumps, and index entry. The notebook also writes six headline figures into the lecture slide deck.

**Architecture:** Add one `ESTMDataset(Dataset)` to `ai4mat/datasets/estm.py` modelled on `NEUDETDataset` (on-disk loading, `download_if_missing()` helper, `download=` flag), but tabular: it parses a CSV, canonicalises columns, and lazily builds either element-fraction features or matminer/Magpie features which are cached as `.npz`. Build one self-contained Quarto notebook in `notebooks/MLPC/week05_clustering_estm.qmd` that uses the dataset, runs PCA + K-means, and saves both inline figures and slide-ready PNGs to the symlinked slide-deck folder via a guarded helper that cannot create phantom directories.

**Tech Stack:** PyTorch, pandas, numpy, scikit-learn (StandardScaler, PCA, KMeans, silhouette), pymatgen (Composition parsing — new dep), matminer (ElementProperty/Magpie featurizer — new dep), matplotlib, tqdm, Quarto.

**Spec:** `docs/superpowers/specs/2026-05-11-estm-clustering-mlpc-unit5-design.md`

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `requirements.txt` | Add `pymatgen>=2024.3`, `matminer>=0.9` |
| Create | `ai4mat/datasets/estm.py` | `ESTMDataset` class, `download_if_missing()`, `_canonicalise_columns()`, featurizers |
| Modify | `ai4mat/datasets/__init__.py` | Re-export `ESTMDataset` |
| Create | `tests/datasets/test_estm.py` | Smoke tests, gated on CSV presence |
| Create | `notebooks/MLPC/week05_clustering_estm.qmd` | Lecture notebook |
| Modify | `index.qmd` | Add link under MLPC Week 5 row |

Slide PNGs land via the notebook's render in:
`/home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit05_unsupervised_learning/images/estm/` (resolved through the `SS26/_public_presentations` symlink at `../_public_presentations/...` from the project root).

---

## Task 1: Add `pymatgen` + `matminer` to requirements

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the deps**

Append two lines so the file ends with the new pins:

```diff
 xgboost>=2.0
 torchvision>=0.16
+pymatgen>=2024.3
+matminer>=0.9
```

- [ ] **Step 2: Install locally**

Run: `pip install -r requirements.txt`
Expected: pymatgen + matminer install (matminer pulls a lot — accept ~150 MB).
Verify: `python -c "import pymatgen, matminer; print(pymatgen.__version__, matminer.__version__)"` prints two version strings.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add pymatgen + matminer for ESTM composition featurization"
```

---

## Task 2: Failing dataset tests (TDD red)

**Files:**
- Create: `tests/datasets/test_estm.py`

- [ ] **Step 1: Write the failing tests**

Mirror the style of `tests/datasets/test_neu_det.py`. Place exactly this content:

```python
import os
import pytest
import torch

DATA_DIR = "data/estm"
CSV = os.path.join(DATA_DIR, "ESTM.csv")


def _data_present() -> bool:
    return os.path.isfile(CSV)


skip_if_no_data = pytest.mark.skipif(
    not _data_present(), reason="ESTM CSV not present at data/estm/ESTM.csv"
)


def test_estm_importable():
    from ai4mat.datasets import ESTMDataset  # noqa: F401


@skip_if_no_data
def test_estm_fraction_contract():
    from ai4mat.datasets import ESTMDataset
    from tests.conftest import assert_dataset_contract

    ds = ESTMDataset(features="fraction", download=False)
    assert_dataset_contract(
        ds,
        expected_x_shape=[119],          # 118 elements + T
        expected_y_dtype=torch.float32,
        min_len=4500,
    )


@skip_if_no_data
def test_estm_fraction_attributes():
    from ai4mat.datasets import ESTMDataset

    ds = ESTMDataset(features="fraction", download=False)
    assert len(ds.formulas) == len(ds)
    assert ds.T.shape == (len(ds),)
    assert ds.properties.shape == (len(ds), 5)
    assert list(ds.properties.columns) == ["ZT", "S", "sigma", "kappa", "PF"]
    assert len(ds.feature_names) == 119


@skip_if_no_data
@pytest.mark.slow
def test_estm_magpie_features():
    from ai4mat.datasets import ESTMDataset

    ds = ESTMDataset(features="magpie", download=False)
    x, _ = ds[0]
    assert x.shape[0] > 100, f"expected >100 Magpie features, got {x.shape[0]}"
    assert torch.isfinite(x).all(), "Magpie features must be finite after imputation"


@skip_if_no_data
def test_estm_target_all():
    from ai4mat.datasets import ESTMDataset

    ds = ESTMDataset(features="fraction", target="all", download=False)
    _, y = ds[0]
    assert y.shape == (5,)


def test_estm_download_false_errors_on_empty(tmp_path):
    # Runs in CI too — no live data needed.
    from ai4mat.datasets import ESTMDataset

    with pytest.raises(FileNotFoundError):
        ESTMDataset(root=str(tmp_path / "empty"), download=False)


@skip_if_no_data
def test_estm_cache_roundtrip(tmp_path):
    """Second instantiation hits the .npz cache and matches the first."""
    from ai4mat.datasets import ESTMDataset
    import shutil

    src = "data/estm/ESTM.csv"
    root = tmp_path / "estm"
    root.mkdir()
    shutil.copy(src, root / "ESTM.csv")

    ds1 = ESTMDataset(root=str(root), features="fraction", download=False)
    ds2 = ESTMDataset(root=str(root), features="fraction", download=False)
    assert torch.allclose(ds1.X, ds2.X)
    assert ds1.formulas == ds2.formulas
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/datasets/test_estm.py -v`
Expected: `test_estm_importable` fails with `ImportError: cannot import name 'ESTMDataset'`. The `skip_if_no_data` tests either skip (CSV absent) or also fail at import.

- [ ] **Step 3: Commit**

```bash
git add tests/datasets/test_estm.py
git commit -m "test(estm): add failing dataset contract tests (TDD red)"
```

---

## Task 3: Implement `_canonicalise_columns()` + module skeleton

**Files:**
- Create: `ai4mat/datasets/estm.py`

- [ ] **Step 1: Write the skeleton + column canonicaliser**

```python
"""ESTM thermoelectric dataset (Na & Chang 2022, npj Comput. Mater. 8, 214)."""
from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Try in order; first that returns a parseable CSV with ≥4500 rows wins.
_ESTM_URLS = [
    "https://raw.githubusercontent.com/ngs00/SIMD/main/dataset/ESTM.csv",
    "https://raw.githubusercontent.com/ngs00/SIMD/master/dataset/ESTM.csv",
    "https://github.com/ngs00/SIMD/raw/main/dataset/ESTM.csv",
]

_CANONICAL_COLUMNS = ["formula", "T", "S", "sigma", "kappa", "PF", "ZT"]

# Header-name aliases from the published CSV → canonical names.
# Lowercased + stripped before matching.
_COLUMN_ALIASES = {
    "formula": "formula",
    "composition": "formula",
    "chemical formula": "formula",
    "t": "T",
    "temperature": "T",
    "temperature (k)": "T",
    "s": "S",
    "seebeck": "S",
    "seebeck coefficient": "S",
    "seebeck coefficient (uv/k)": "S",
    "seebeck coefficient (µv/k)": "S",
    "sigma": "sigma",
    "electrical conductivity": "sigma",
    "electrical conductivity (s/m)": "sigma",
    "kappa": "kappa",
    "thermal conductivity": "kappa",
    "thermal conductivity (w/mk)": "kappa",
    "thermal conductivity (w/m-k)": "kappa",
    "pf": "PF",
    "power factor": "PF",
    "zt": "ZT",
    "figure of merit": "ZT",
}


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ESTM CSV columns to the canonical set and select them.

    Raises KeyError listing what was found and what was expected if any
    canonical column cannot be matched.
    """
    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    df = df.rename(columns=rename)
    missing = [c for c in _CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"ESTM CSV is missing canonical columns {missing}. "
            f"Found headers: {list(df.columns)}"
        )
    return df[_CANONICAL_COLUMNS].copy()
```

(The rest of the file lands in later tasks.)

- [ ] **Step 2: Sanity-check the helper in isolation**

Run:
```bash
python - <<'PY'
import pandas as pd
from ai4mat.datasets.estm import _canonicalise_columns
df = pd.DataFrame({
    "Formula": ["PbTe"],
    "Temperature (K)": [300.0],
    "Seebeck coefficient (uV/K)": [-100.0],
    "Electrical conductivity (S/m)": [1e5],
    "Thermal conductivity (W/mK)": [2.0],
    "PF": [1e-3],
    "ZT": [0.5],
})
out = _canonicalise_columns(df)
print(out.columns.tolist())
print(out.shape)
PY
```
Expected: prints `['formula', 'T', 'S', 'sigma', 'kappa', 'PF', 'ZT']` then `(1, 7)`.

- [ ] **Step 3: Run the importable test**

Run: `pytest tests/datasets/test_estm.py::test_estm_importable -v`
Expected: this still fails because `ESTMDataset` isn't exported yet — but the module imports cleanly. Verify with: `python -c "from ai4mat.datasets.estm import _canonicalise_columns; print('ok')"` → `ok`.

- [ ] **Step 4: Commit**

```bash
git add ai4mat/datasets/estm.py
git commit -m "feat(estm): module skeleton + column canonicaliser"
```

---

## Task 4: Implement `download_if_missing()`

**Files:**
- Modify: `ai4mat/datasets/estm.py`

- [ ] **Step 1: Append the downloader**

```python
def _try_download_one(url: str, dest: Path) -> bool:
    """Download `url` to `dest`. Return True if dest now exists & looks like ESTM."""
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urlopen(url, timeout=30) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(tmp, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=f"ESTM ({url.split('/')[-3]})"
            ) as pbar:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    pbar.update(len(chunk))
    except Exception as exc:  # noqa: BLE001 — try the next URL
        warnings.warn(f"ESTM download from {url} failed: {exc!r}")
        tmp.unlink(missing_ok=True)
        return False

    # Validate: parseable CSV with the columns we need and ≥ 4500 rows.
    try:
        df = pd.read_csv(tmp)
        df = _canonicalise_columns(df)
        if len(df) < 4500:
            raise ValueError(f"only {len(df)} rows in downloaded CSV")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"ESTM CSV from {url} failed validation: {exc!r}")
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(dest)
    return True


def _write_readme(root: Path, source_url: str, sha256: str) -> None:
    citation = (
        "Na, G. S. & Chang, H. A public database of thermoelectric materials "
        "and system-identified material representation for data-driven "
        "discovery. npj Comput. Mater. 8, 214 (2022). "
        "https://doi.org/10.1038/s41524-022-00897-2"
    )
    (root / "README.md").write_text(
        f"# ESTM dataset\n\n"
        f"- Source URL: {source_url}\n"
        f"- SHA-256: {sha256}\n"
        f"- Downloaded: via `ai4mat.datasets.estm.download_if_missing`\n\n"
        f"## Citation\n\n{citation}\n"
    )


def download_if_missing(root: str) -> None:
    """Download the ESTM CSV into `<root>/ESTM.csv` if not already present."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    dest = root_path / "ESTM.csv"
    if dest.exists():
        return

    for url in _ESTM_URLS:
        if _try_download_one(url, dest):
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            _write_readme(root_path, url, sha)
            return

    raise RuntimeError(
        "Could not download ESTM dataset from any known URL. "
        "Download manually from https://github.com/ngs00/SIMD "
        f"(dataset/ESTM.csv) and place it at {dest}"
    )
```

- [ ] **Step 2: Exercise it once**

Run:
```bash
rm -rf data/estm
python -c "from ai4mat.datasets.estm import download_if_missing; download_if_missing('data/estm'); import os; print('rows:', sum(1 for _ in open('data/estm/ESTM.csv')) - 1)"
```
Expected: tqdm bar; prints `rows: 5205` (or similar; must be ≥ 4500). `data/estm/README.md` exists and contains a SHA-256.

If all three URLs fail at runtime, the candidate list itself is the bug — pause here, hand back to the human with the exception output and the suggested manual download command from the error message. Do NOT silently fall back to manual download or proceed past Task 4 with no CSV.

- [ ] **Step 3: Confirm idempotence**

Run again: `python -c "from ai4mat.datasets.estm import download_if_missing; download_if_missing('data/estm'); print('ok')"`
Expected: no download bar, prints `ok` immediately.

- [ ] **Step 4: Commit**

```bash
git add ai4mat/datasets/estm.py
git commit -m "feat(estm): downloader with URL fallbacks and validation"
```

---

## Task 5: Implement `ESTMDataset` core (CSV load + fraction featurization)

**Files:**
- Modify: `ai4mat/datasets/estm.py`
- Modify: `ai4mat/datasets/__init__.py`

- [ ] **Step 1: Append the class and the fraction featurizer**

```python
# pymatgen's full element list, used for fraction featurization.
def _periodic_table_symbols() -> list[str]:
    from pymatgen.core import Element
    return [el.symbol for el in Element]


def _featurize_fraction(formulas: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (X, columns, kept_mask) where X is (N_kept, 118) of element fractions.

    Formulas that fail pymatgen parsing are recorded as False in `kept_mask`.
    """
    from pymatgen.core import Composition

    symbols = _periodic_table_symbols()
    idx = {s: i for i, s in enumerate(symbols)}
    X = np.zeros((len(formulas), len(symbols)), dtype=np.float32)
    kept = np.ones(len(formulas), dtype=bool)
    for i, f in enumerate(formulas):
        try:
            comp = Composition(f)
            frac = comp.fractional_composition.as_dict()
            for sym, v in frac.items():
                X[i, idx[sym]] = float(v)
        except Exception:  # noqa: BLE001 — drop unparseable rows
            kept[i] = False
    return X[kept], list(symbols), kept


class ESTMDataset(Dataset):
    """ESTM thermoelectric dataset (Na & Chang 2022) as a PyTorch Dataset.

    Loads 5205 experimental observations across ~880 compounds with
    measurement temperature and five thermoelectric properties.

    Features are computed lazily on first instantiation and cached as
    `<root>/features_<mode>.npz`. Subsequent loads skip featurization
    entirely.

    X shape: (F,) where F = 119 (fraction+T) or ~134 (magpie+T)
    y shape: () (single target) or (5,) (target="all")
    """

    def __init__(
        self,
        root: str = "data/estm",
        features: str = "fraction",
        target: str = "ZT",
        download: bool = True,
        standardize: bool = False,
        transform=None,
        target_transform=None,
    ):
        if features not in {"fraction", "magpie"}:
            raise ValueError(f"features must be 'fraction' or 'magpie', got {features!r}")
        allowed_targets = {"ZT", "S", "sigma", "kappa", "PF", "all"}
        if target not in allowed_targets:
            raise ValueError(f"target must be one of {allowed_targets}, got {target!r}")

        self.root = Path(root)
        self.features = features
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

        csv = self.root / "ESTM.csv"
        if not csv.exists():
            if download:
                download_if_missing(str(self.root))
            else:
                raise FileNotFoundError(
                    f"{csv} not found. Set download=True or place the file manually."
                )

        cache = self.root / f"features_{features}.npz"
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            X = data["X"].astype(np.float32)
            self.feature_names = list(data["columns"])
            self.formulas = list(data["formulas"])
            self.T = torch.from_numpy(data["T"].astype(np.float32))
            self.properties = pd.DataFrame(
                {k: data[k] for k in ["ZT", "S", "sigma", "kappa", "PF"]}
            )
        else:
            X, feature_names, formulas, T, props = self._build_features(csv)
            self.feature_names = feature_names
            self.formulas = formulas
            self.T = T
            self.properties = props
            np.savez(
                cache,
                X=X,
                columns=np.array(feature_names),
                formulas=np.array(formulas),
                T=T.numpy(),
                **{c: props[c].to_numpy() for c in props.columns},
            )

        if standardize:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X).astype(np.float32)
        else:
            self.scaler = None

        self.X = torch.from_numpy(X).float()
        self.y = self._build_target()

    def _build_features(self, csv: Path):
        df = pd.read_csv(csv)
        df = _canonicalise_columns(df).dropna(subset=["formula", "T", "ZT"])
        df = df.reset_index(drop=True)

        formulas_in = df["formula"].astype(str).tolist()
        if self.features == "fraction":
            X_no_T, columns, kept = _featurize_fraction(formulas_in)
        else:
            X_no_T, columns, kept = _featurize_magpie(formulas_in)

        df = df.loc[kept].reset_index(drop=True)
        n_dropped = int((~kept).sum())
        if n_dropped:
            warnings.warn(
                f"Dropped {n_dropped} ESTM rows with unparseable formulas "
                f"({n_dropped / len(kept):.1%} of the dataset)"
            )

        T = df["T"].to_numpy(dtype=np.float32)
        X = np.concatenate([X_no_T, T[:, None]], axis=1).astype(np.float32)
        columns = columns + ["T"]
        formulas = df["formula"].tolist()
        props = df[["ZT", "S", "sigma", "kappa", "PF"]].astype(np.float32).copy()
        return X, columns, formulas, torch.from_numpy(T), props

    def _build_target(self) -> torch.Tensor:
        if self.target == "all":
            return torch.from_numpy(
                self.properties[["ZT", "S", "sigma", "kappa", "PF"]]
                .to_numpy(dtype=np.float32)
            )
        return torch.from_numpy(self.properties[self.target].to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


def _featurize_magpie(formulas: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Stub — implemented in Task 6."""
    raise NotImplementedError("Magpie featurization lands in Task 6")
```

- [ ] **Step 2: Re-export**

Edit `ai4mat/datasets/__init__.py` so the imports and `__all__` include the new class:

```diff
 from .crystal_graphs import CrystalGraphsDataset
+from .estm import ESTMDataset

 __all__ = [
     "IrisDataset",
     "DigitsDataset",
     "TensileTestDataset",
     "IsingDataset",
     "CahnHilliardDataset",
     "ChemicalElementsDataset",
     "NanoindentationDataset",
     "CrystalGraphsDataset",
+    "ESTMDataset",
 ]
```

- [ ] **Step 3: Run the fraction tests**

Run: `pytest tests/datasets/test_estm.py -v -m "not slow"`
Expected:
- `test_estm_importable` PASSES
- `test_estm_fraction_contract` PASSES
- `test_estm_fraction_attributes` PASSES
- `test_estm_target_all` PASSES
- `test_estm_download_false_errors_on_empty` PASSES
- `test_estm_cache_roundtrip` PASSES
- `test_estm_magpie_features` SKIPS (the `slow` marker filters it out)

If `test_estm_fraction_attributes` fails on `len(ds.feature_names) == 119`, that means pymatgen's `Element` list isn't 118 — adjust to `len(_periodic_table_symbols()) + 1` and update the spec/test together.

- [ ] **Step 4: Commit**

```bash
git add ai4mat/datasets/estm.py ai4mat/datasets/__init__.py
git commit -m "feat(estm): ESTMDataset class with element-fraction featurization"
```

---

## Task 6: Implement Magpie featurization

**Files:**
- Modify: `ai4mat/datasets/estm.py`

- [ ] **Step 1: Replace the `_featurize_magpie` stub**

Replace the `NotImplementedError` stub at the bottom of the file with the real implementation:

```python
def _featurize_magpie(formulas: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (X, columns, kept_mask) where X is (N_kept, ~133) of Magpie features.

    Rows with unparseable formulas → kept_mask=False.
    Column-median imputation for NaNs; rows still all-NaN → also dropped.
    """
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    featurizer = ElementProperty.from_preset("magpie")
    columns = featurizer.feature_labels()

    rows: list[np.ndarray] = []
    kept = np.zeros(len(formulas), dtype=bool)
    for i, f in enumerate(tqdm(formulas, desc="Magpie", unit="formula")):
        try:
            comp = Composition(f)
            row = featurizer.featurize(comp)
        except Exception:  # noqa: BLE001
            continue
        rows.append(np.asarray(row, dtype=np.float32))
        kept[i] = True

    X = np.vstack(rows) if rows else np.empty((0, len(columns)), dtype=np.float32)

    # Column-median imputation.
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X = np.where(nan_mask, col_medians, X).astype(np.float32)

    # Drop rows that are still all-NaN (column median was NaN too).
    still_bad = np.isnan(X).any(axis=1)
    if still_bad.any():
        # Need to map this back to the original index space: kept-True positions
        # currently in row order — flip the corresponding kept entries off.
        kept_indices = np.flatnonzero(kept)
        kept[kept_indices[still_bad]] = False
        X = X[~still_bad]

    return X, list(columns), kept
```

- [ ] **Step 2: Run the slow Magpie test**

Run: `pytest tests/datasets/test_estm.py::test_estm_magpie_features -v -m slow`
Expected: PASS. First run takes ~40 s (matminer is slow), populates `data/estm/features_magpie.npz`.

- [ ] **Step 3: Verify the cache is honoured**

Re-run: `pytest tests/datasets/test_estm.py::test_estm_magpie_features -v -m slow`
Expected: PASS in < 5 s (cache hit, no matminer import path exercised inside the class).

- [ ] **Step 4: Commit**

```bash
git add ai4mat/datasets/estm.py
git commit -m "feat(estm): Magpie featurization via matminer ElementProperty"
```

---

## Task 7: Notebook — setup, data preview, fraction PCA

**Files:**
- Create: `notebooks/MLPC/week05_clustering_estm.qmd`

- [ ] **Step 1: Write the frontmatter + first three sections**

```markdown
---
title: "MLPC Week 5: Clustering thermoelectric materials (ESTM)"
subtitle: "PCA + K-means on element-fraction and Magpie features"
jupyter: python3
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ECLIPSE-Lab/Ai4MatLectures/blob/main/notebooks/MLPC/week05_clustering_estm.ipynb)

## Learning Objectives

- Apply K-means to a tabular materials dataset with quantitative property targets.
- Understand how featurization choice (raw fractions vs Magpie physics-aware descriptors) changes cluster geometry.
- Use elbow + silhouette together to motivate a cluster count.
- Read per-cluster property distributions as a sanity check that clusters mean something.
- Connect this workflow to materials discovery (Na & Chang 2022 / SIMD).

## Setup

```{python}
#| eval: false
!pip install git+https://github.com/ECLIPSE-Lab/Ai4MatLectures.git "pymatgen>=2024.3" "matminer>=0.9"
```

```{python}
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ai4mat.datasets import ESTMDataset

np.random.seed(0)
warnings.filterwarnings("ignore", category=UserWarning, module="matminer")
```

### Slide-figure save helper

```{python}
def _resolve_slide_img_dir() -> Path:
    env = os.environ.get("ESTM_SLIDE_IMG_DIR")
    target = Path(env) if env else Path(
        "../_public_presentations/ml_for_characterization_and_processing/"
        "unit05_unsupervised_learning/images/estm"
    )
    # parents=False on purpose: do NOT silently create _public_presentations/.
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

print(f"Slide images → {SLIDE_IMG_DIR.resolve()}")
```

## 1. Dataset preview

```{python}
ds_frac = ESTMDataset(features="fraction", standardize=True)
ds_mag = ESTMDataset(features="magpie", standardize=True)
print(f"fraction: X={tuple(ds_frac.X.shape)}, y={tuple(ds_frac.y.shape)}")
print(f"magpie:   X={tuple(ds_mag.X.shape)}, y={tuple(ds_mag.y.shape)}")
print("temperature range:", ds_frac.T.min().item(), "-", ds_frac.T.max().item(), "K")
```

```{python}
def _dominant_element(formula: str) -> str:
    from pymatgen.core import Composition
    comp = Composition(formula).fractional_composition.as_dict()
    return max(comp, key=comp.get)

dom = [_dominant_element(f) for f in ds_frac.formulas]
top_elements = [e for e, _ in Counter(dom).most_common(15)]
T_bins = pd.cut(ds_frac.T.numpy(), bins=[0, 400, 600, 800, 1500],
                labels=["≤400 K", "400-600 K", "600-800 K", ">800 K"])
preview = (
    pd.DataFrame({"dom": dom, "Tbin": T_bins})
      .query("dom in @top_elements")
      .pipe(lambda d: pd.crosstab(d["dom"], d["Tbin"]))
      .loc[top_elements]
)
preview
```

## 2. Feature pipeline A — element fractions + T

```{python}
X_frac = ds_frac.X.numpy()
pca_frac_10 = PCA(n_components=10, random_state=0).fit(X_frac)
pca_frac_2  = PCA(n_components=2,  random_state=0).fit(X_frac)
Z_frac10 = pca_frac_10.transform(X_frac)
Z_frac2  = pca_frac_2.transform(X_frac)

cumvar = np.cumsum(pca_frac_10.explained_variance_ratio_)
print(f"fraction PCA-10 cumulative variance: {cumvar[-1]:.2%}")
```
```

- [ ] **Step 2: Render the notebook so far**

Run: `quarto render notebooks/MLPC/week05_clustering_estm.qmd`
Expected: exits 0. The first instantiation will download the CSV (~5 s) and build both feature caches (Magpie pass takes ~40 s); subsequent renders are fast.

Sanity-check: `ls data/estm/` shows `ESTM.csv`, `README.md`, `features_fraction.npz`, `features_magpie.npz`.
Sanity-check: `grep -q "fraction PCA-10 cumulative variance" _site/notebooks/MLPC/week05_clustering_estm.html` exits 0.

- [ ] **Step 3: Commit**

```bash
git add notebooks/MLPC/week05_clustering_estm.qmd
git commit -m "docs(mlpc-w5): scaffold ESTM notebook with data preview and fraction PCA"
```

---

## Task 8: Notebook — Magpie PCA + K-means K-sweep on both featurizations (figure 1)

**Files:**
- Modify: `notebooks/MLPC/week05_clustering_estm.qmd`

- [ ] **Step 1: Append sections 3 and 4**

Append after the section 2 cell:

````markdown
## 3. Feature pipeline B — Magpie descriptors + T

```{python}
X_mag = ds_mag.X.numpy()
pca_mag_10 = PCA(n_components=10, random_state=0).fit(X_mag)
pca_mag_2  = PCA(n_components=2,  random_state=0).fit(X_mag)
Z_mag10 = pca_mag_10.transform(X_mag)
Z_mag2  = pca_mag_2.transform(X_mag)

cumvar = np.cumsum(pca_mag_10.explained_variance_ratio_)
print(f"magpie PCA-10 cumulative variance: {cumvar[-1]:.2%}")

# Inspect top loadings of PC1/PC2 to see which Magpie descriptors dominate.
loadings = pd.DataFrame(
    pca_mag_2.components_.T,
    index=ds_mag.feature_names,
    columns=["PC1", "PC2"],
)
print("Top 5 by |PC1|:\n", loadings["PC1"].abs().nlargest(5))
print("Top 5 by |PC2|:\n", loadings["PC2"].abs().nlargest(5))
```

## 4. K-means + elbow / silhouette on both feature sets

```{python}
def k_sweep(Z, k_values):
    inertias, sils = [], []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Z, km.labels_))
    return np.array(inertias), np.array(sils)

K_VALUES = np.arange(2, 13)
frac_inertia, frac_sil = k_sweep(Z_frac10, K_VALUES)
mag_inertia,  mag_sil  = k_sweep(Z_mag10,  K_VALUES)

K_star_frac = int(K_VALUES[np.argmax(frac_sil)])
K_star_mag  = int(K_VALUES[np.argmax(mag_sil)])
print(f"K* (fraction features) = {K_star_frac}")
print(f"K* (Magpie features)   = {K_star_mag}")
```

```{python}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, inertia, sil, label, K_star in [
    (axes[0], frac_inertia, frac_sil, "fraction", K_star_frac),
    (axes[1], mag_inertia,  mag_sil,  "magpie",   K_star_mag),
]:
    ax.plot(K_VALUES, inertia / inertia.max(), "o-", label="inertia (norm.)")
    ax.plot(K_VALUES, sil / sil.max(),         "s-", label="silhouette (norm.)")
    ax.axvline(K_star, color="k", linestyle=":", alpha=0.5, label=f"K* = {K_star}")
    ax.set_title(f"{label} features")
    ax.set_xlabel("K")
    ax.set_xticks(K_VALUES)
    ax.legend(loc="best", fontsize=9)
axes[0].set_ylabel("score (max-normalised)")
fig.tight_layout()
save_slide_fig(fig, "elbow_silhouette")
plt.show()
```

```{python}
km_frac = KMeans(n_clusters=K_star_frac, n_init=10, random_state=0).fit(Z_frac10)
km_mag  = KMeans(n_clusters=K_star_mag,  n_init=10, random_state=0).fit(Z_mag10)
labels_frac = km_frac.labels_
labels_mag  = km_mag.labels_
```
````

- [ ] **Step 2: Render**

Run: `quarto render notebooks/MLPC/week05_clustering_estm.qmd`
Expected: exits 0. PNG `elbow_silhouette.png` appears at the resolved slide-dir path.

Sanity-check: `ls /home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit05_unsupervised_learning/images/estm/elbow_silhouette.png` exists.

- [ ] **Step 3: Commit**

```bash
git add notebooks/MLPC/week05_clustering_estm.qmd
git commit -m "docs(mlpc-w5): add Magpie PCA + K-means K-sweep with elbow/silhouette figure"
```

---

## Task 9: Notebook — PCA scatter figures (figures 2, 3, 4)

**Files:**
- Modify: `notebooks/MLPC/week05_clustering_estm.qmd`

- [ ] **Step 1: Append section 5**

````markdown
## 5. Cluster visualisation in PCA space

```{python}
def cluster_dominant_elements(formulas, labels) -> dict[int, str]:
    out: dict[int, str] = {}
    for c in sorted(set(labels)):
        members = [f for f, l in zip(formulas, labels) if l == c]
        elems = [_dominant_element(f) for f in members]
        out[c] = Counter(elems).most_common(1)[0][0]
    return out

dom_frac = cluster_dominant_elements(ds_frac.formulas, labels_frac)
dom_mag  = cluster_dominant_elements(ds_mag.formulas,  labels_mag)
print("fraction cluster → dominant element:", dom_frac)
print("magpie cluster   → dominant element:", dom_mag)
```

```{python}
def scatter_clusters(ax, Z2, labels, centroids_2d, dom_map, title):
    cmap = plt.cm.tab10
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(Z2[m, 0], Z2[m, 1], s=6, alpha=0.5,
                   color=cmap(c % 10), label=f"{c}: {dom_map[c]}")
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
               s=180, marker="*", color="black", edgecolors="white", linewidths=1.2)
    for c, (cx, cy) in enumerate(centroids_2d):
        ax.annotate(dom_map[c], (cx, cy), textcoords="offset points",
                    xytext=(6, 6), fontsize=10, weight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, markerscale=2)

# Centroids of the 10-D clusters projected into 2-D PCA space.
cent_frac_2d = pca_frac_2.transform(
    pca_frac_10.inverse_transform(km_frac.cluster_centers_)
)
cent_mag_2d = pca_mag_2.transform(
    pca_mag_10.inverse_transform(km_mag.cluster_centers_)
)

fig, ax = plt.subplots(figsize=(7, 6))
scatter_clusters(ax, Z_frac2, labels_frac, cent_frac_2d, dom_frac,
                 "Element-fraction features — K-means clusters")
fig.tight_layout()
save_slide_fig(fig, "pca_scatter_fraction")
plt.show()
```

```{python}
fig, ax = plt.subplots(figsize=(7, 6))
scatter_clusters(ax, Z_mag2, labels_mag, cent_mag_2d, dom_mag,
                 "Magpie features — K-means clusters")
fig.tight_layout()
save_slide_fig(fig, "pca_scatter_magpie")
plt.show()
```

```{python}
ZT_vals = ds_frac.properties["ZT"].to_numpy()
vmin, vmax = 0.0, float(np.nanquantile(ZT_vals, 0.99))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, Z2, title in [
    (axes[0], Z_frac2, "fraction features"),
    (axes[1], Z_mag2,  "magpie features"),
]:
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=ZT_vals, s=8, alpha=0.7,
                    cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
fig.colorbar(sc, ax=axes, label="ZT", shrink=0.85)
save_slide_fig(fig, "pca_scatter_by_zt")
plt.show()
```
````

- [ ] **Step 2: Render and verify three PNGs**

Run: `quarto render notebooks/MLPC/week05_clustering_estm.qmd`
Expected: exits 0.

Sanity-check:
```bash
ls /home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit05_unsupervised_learning/images/estm/ | sort
```
Expected output includes `elbow_silhouette.png`, `pca_scatter_by_zt.png`, `pca_scatter_fraction.png`, `pca_scatter_magpie.png` (4 PNGs so far).

- [ ] **Step 3: Commit**

```bash
git add notebooks/MLPC/week05_clustering_estm.qmd
git commit -m "docs(mlpc-w5): add PCA cluster scatter figures (fraction, magpie, by ZT)"
```

---

## Task 10: Notebook — property enrichment figures + wrap-up (figures 5, 6)

**Files:**
- Modify: `notebooks/MLPC/week05_clustering_estm.qmd`

- [ ] **Step 1: Append section 6 and the wrap-up**

````markdown
## 6. Per-cluster material families and property enrichment

```{python}
def top_elements_per_cluster(formulas, labels, n=5):
    rows = []
    for c in sorted(set(labels)):
        members = [f for f, l in zip(formulas, labels) if l == c]
        elems = Counter(_dominant_element(f) for f in members).most_common(n)
        rows.append({"cluster": c, "n": len(members),
                     "top_elements": ", ".join(f"{e}({k})" for e, k in elems)})
    return pd.DataFrame(rows)

summary = top_elements_per_cluster(ds_mag.formulas, labels_mag, n=5).set_index("cluster")
for col in ["ZT", "S", "sigma", "kappa", "PF"]:
    summary[f"median_{col}"] = (
        pd.Series(ds_mag.properties[col].to_numpy())
          .groupby(labels_mag).median().values
    )
summary
```

```{python}
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
prop_names = ["ZT", "S", "sigma", "kappa"]
log_axes = {"sigma": True, "kappa": True}
for ax, p in zip(axes, prop_names):
    data = [
        ds_mag.properties[p].to_numpy()[labels_mag == c]
        for c in sorted(set(labels_mag))
    ]
    ax.boxplot(data, labels=sorted(set(labels_mag)), showfliers=False)
    ax.set_title(p)
    ax.set_xlabel("magpie cluster")
    if log_axes.get(p):
        ax.set_yscale("log")
fig.tight_layout()
save_slide_fig(fig, "property_box_per_cluster")
plt.show()
```

```{python}
T_vals = ds_mag.T.numpy()
T_bins = np.array([300, 400, 500, 600, 700, 800, 900, 1100])
T_centres = 0.5 * (T_bins[:-1] + T_bins[1:])

fig, ax = plt.subplots(figsize=(8, 5))
ZT = ds_mag.properties["ZT"].to_numpy()
for c in sorted(set(labels_mag)):
    m = labels_mag == c
    if m.sum() < 20:
        continue
    bin_id = np.digitize(T_vals[m], T_bins) - 1
    medians = [
        np.nanmedian(ZT[m][bin_id == b]) if (bin_id == b).any() else np.nan
        for b in range(len(T_centres))
    ]
    ax.plot(T_centres, medians, "o-", label=f"cluster {c} ({dom_mag[c]})")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("median ZT")
ax.set_title("Median ZT vs T, per Magpie cluster")
ax.legend(fontsize=9)
fig.tight_layout()
save_slide_fig(fig, "zt_by_cluster_vs_T")
plt.show()
```

## 7. Wrap-up

**Takeaways**

- *Featurization choice dominates the cluster structure.* Magpie descriptors
  separate chalcogenides from skutterudites / half-Heuslers cleanly; raw
  element-fraction features mostly recover "which element is dominant" and
  miss subtler family structure.
- *Cluster identity correlates with high ZT.* A small number of Magpie
  clusters concentrate the top-decile ZT entries — exactly the materials-
  discovery signal we want from unsupervised methods.
- *This is one step short of Na & Chang's SIMD.* Their representation
  learns a clustering-aware projection from a graph over similar materials.
  Magpie + K-means is the unsupervised baseline; SIMD is the natural
  follow-up if you want to extrapolate beyond seen families.

**References**

- Na, G. S. & Chang, H. *npj Comput. Mater.* **8**, 214 (2022). DOI: 10.1038/s41524-022-00897-2
- Ward, L. et al. *npj Comput. Mater.* **2**, 16028 (2016) — Magpie descriptors.
````

- [ ] **Step 2: Render**

Run: `quarto render notebooks/MLPC/week05_clustering_estm.qmd`
Expected: exits 0.

- [ ] **Step 3: Verify all six slide PNGs exist**

Run:
```bash
ls /home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit05_unsupervised_learning/images/estm/*.png | wc -l
```
Expected: `6`.

Run:
```bash
ls /home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit05_unsupervised_learning/images/estm/ | sort
```
Expected: `elbow_silhouette.png`, `pca_scatter_by_zt.png`, `pca_scatter_fraction.png`, `pca_scatter_magpie.png`, `property_box_per_cluster.png`, `zt_by_cluster_vs_T.png`.

Run:
```bash
find . -path "*/_public_presentations/*" -prune -o -path "./_public_presentations*" -print 2>/dev/null
```
Expected: prints nothing (no phantom `_public_presentations` tree inside the repo).

- [ ] **Step 4: Commit**

```bash
git add notebooks/MLPC/week05_clustering_estm.qmd
git commit -m "docs(mlpc-w5): add per-cluster property enrichment and wrap-up"
```

---

## Task 11: Update `index.qmd`

**Files:**
- Modify: `index.qmd`

- [ ] **Step 1: Inspect current MLPC week 5 row**

Run: `grep -n "Unsupervised learning in materials" index.qmd`
Expected: matches one row, currently:

```
| 5 | Unsupervised learning in materials | Ising / Cahn-Hilliard | **braided:** ... per-course: [week11_anomaly_cahn_hilliard](...) |
```

(Or, if the NEU-DET PR has landed, that row already mentions NEU-DET.)

- [ ] **Step 2: Add the ESTM entry**

If the row currently says `Ising / Cahn-Hilliard`:

```diff
-| 5 | Unsupervised learning in materials | Ising / Cahn-Hilliard | **braided:** [week5_clustering_and_autoencoders.py](notebooks/week5_clustering_and_autoencoders.py); per-course: [week11_anomaly_cahn_hilliard](notebooks/MLPC/week11_anomaly_cahn_hilliard.html) |
+| 5 | Unsupervised learning in materials | Ising / Cahn-Hilliard / ESTM | **braided:** [week5_clustering_and_autoencoders.py](notebooks/week5_clustering_and_autoencoders.py); per-course: [week05_clustering_estm](notebooks/MLPC/week05_clustering_estm.html), [week11_anomaly_cahn_hilliard](notebooks/MLPC/week11_anomaly_cahn_hilliard.html) |
```

If the row already mentions NEU-DET (e.g. `Ising / Cahn-Hilliard / NEU-DET`), append ` / ESTM` to the Dataset cell and add ` [week05_clustering_estm](notebooks/MLPC/week05_clustering_estm.html),` immediately after `per-course:` in the Notebook cell — preserving the alphabetical-by-week-suffix order: `clustering_estm`, `clustering_neu_det`, then `anomaly_cahn_hilliard`.

- [ ] **Step 3: Verify**

Run: `grep "week05_clustering_estm" index.qmd`
Expected: one match.

Run: `quarto render index.qmd`
Expected: exits 0; `_site/index.html` exists.

- [ ] **Step 4: Commit**

```bash
git add index.qmd
git commit -m "docs(index): link MLPC week-5 ESTM notebook"
```

---

## Task 12: Final verification

- [ ] **Step 1: Full notebook test**

Run from project root:
```bash
pytest tests/datasets/test_estm.py -v
```
Expected: 6 pass + 1 skipped (the `slow` Magpie test, unless `-m slow` is used). With `-m slow`: 7 pass.

- [ ] **Step 2: Cold-start sanity (don't actually wipe unless you have time to re-download)**

This is the DoD command from the spec, run only if you want a true cold check:
```bash
rm -rf data/estm
python -c "from ai4mat.datasets import ESTMDataset; ds=ESTMDataset(); assert len(ds)>4500, len(ds); print('OK', len(ds))"
```
Expected: prints `OK <n>` where `n > 4500`. Then re-render the notebook once to repopulate Magpie cache.

- [ ] **Step 3: Final figure inventory**

```bash
ls /home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit05_unsupervised_learning/images/estm/*.png | wc -l
```
Expected: `6`.

```bash
find . -name "_public_presentations" -type d 2>/dev/null
```
Expected: no output (no phantom tree inside the repo).

- [ ] **Step 4: Final commit (only if anything is uncommitted)**

```bash
git status
```
If clean, nothing to do. Otherwise stage and commit any straggler files with a `chore(estm):` message.

---

## Out-of-scope reminders

- Do **not** edit `_public_presentations/ml_for_characterization_and_processing/unit05_unsupervised_learning/01_intro.qmd`. Plots land in `images/estm/`; integration into the deck is Philipp's job.
- Do **not** add t-SNE / UMAP / GMM / hierarchical clustering. The spec is explicit: PCA + K-means only.
- Do **not** add matminer/pymatgen to `pyproject.toml`. They are notebook-time deps via `requirements.txt`.
