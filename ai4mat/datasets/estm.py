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

# Try in order; first that returns a parseable file with ≥4500 rows wins.
_ESTM_URLS = [
    "https://raw.githubusercontent.com/ngs00/SIMD/main/dataset/estm.xlsx",
    "https://github.com/ngs00/SIMD/raw/main/dataset/estm.xlsx",
    # Historical CSV fallbacks — kept in case upstream re-publishes CSV.
    "https://raw.githubusercontent.com/ngs00/SIMD/main/dataset/ESTM.csv",
    "https://raw.githubusercontent.com/ngs00/SIMD/master/dataset/ESTM.csv",
]

_CANONICAL_COLUMNS = ["formula", "T", "S", "sigma", "kappa", "PF", "ZT"]

# Header-name aliases (post-normalisation form: lowercased, ASCII u, no
# trailing unit parens). `_canonicalise_columns` normalises before lookup,
# so e.g. "Seebeck coefficient (µV/K)", "seebeck_coefficient(μV/K)", and
# "seebeck coefficient" all match the "seebeck coefficient" entry.
_COLUMN_ALIASES = {
    "formula": "formula",
    "composition": "formula",
    "chemical formula": "formula",
    "t": "T",
    "temperature": "T",
    "s": "S",
    "seebeck": "S",
    "seebeck coefficient": "S",
    "sigma": "sigma",
    "electrical conductivity": "sigma",
    "kappa": "kappa",
    "thermal conductivity": "kappa",
    "pf": "PF",
    "power factor": "PF",
    "zt": "ZT",
    "figure of merit": "ZT",
}


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ESTM CSV/xlsx columns to the canonical set and select them.

    Robust to header drift: collapses underscores → spaces, unifies both
    micro-sign characters (U+00B5 µ and U+03BC μ) to ASCII `u`, and strips
    a trailing `(unit)` annotation before alias lookup.

    Raises KeyError listing what was found and what was expected if any
    canonical column cannot be matched.
    """
    import re

    def _normalise(key: str) -> str:
        key = str(key).strip().lower()
        key = key.replace("_", " ")
        key = key.replace("μ", "u").replace("µ", "u")  # μ, µ → u
        key = re.sub(r"\s*\([^)]*\)\s*$", "", key)  # drop trailing (unit)
        return " ".join(key.split())

    rename: dict[str, str] = {}
    for col in df.columns:
        for cand in (str(col).strip().lower(), _normalise(col)):
            if cand in _COLUMN_ALIASES:
                rename[col] = _COLUMN_ALIASES[cand]
                break
    df = df.rename(columns=rename)
    missing = [c for c in _CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"ESTM CSV is missing canonical columns {missing}. "
            f"Found headers: {list(df.columns)}"
        )
    return df[_CANONICAL_COLUMNS].copy()


def _read_estm_file(path: Path) -> pd.DataFrame:
    """Read raw ESTM file (xlsx or csv) into a DataFrame.

    Strips a possible `.partial` suffix before checking the extension so
    that temporary filenames like `foo.xlsx.partial` are recognised as xlsx.
    """
    name = path.name
    if name.endswith(".partial"):
        name = name[: -len(".partial")]
    ext = Path(name).suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _try_download_one(url: str, dest: Path) -> bool:
    """Download `url`, validate, canonicalise, and save as CSV at `dest`.

    Returns True iff `dest` (a `.csv`) now exists and looks like ESTM.
    Source file format may be xlsx or csv depending on URL; the on-disk
    destination is always the canonical CSV.
    """
    url_ext = Path(url).suffix.lower() or ".csv"
    tmp = dest.with_suffix(url_ext + ".partial")
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

    try:
        df = _read_estm_file(tmp)
        df = _canonicalise_columns(df)
        if len(df) < 4500:
            raise ValueError(f"only {len(df)} rows in downloaded file")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"ESTM file from {url} failed validation: {exc!r}")
        tmp.unlink(missing_ok=True)
        return False

    df.to_csv(dest, index=False)
    tmp.unlink(missing_ok=True)
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
