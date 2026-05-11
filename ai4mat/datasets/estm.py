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
