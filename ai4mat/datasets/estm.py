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
