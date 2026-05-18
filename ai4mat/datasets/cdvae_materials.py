"""CDVAE generative-materials benchmark datasets (Xie et al., ICLR 2022).

Wraps the three official benchmark datasets curated for the Crystal
Diffusion Variational Autoencoder (CDVAE) paper, published with their
canonical train/val/test splits in the `txie-93/cdvae` GitHub repo:

- ``perov_5``  : ~19k cubic perovskites (Castelli et al., 2012). Same
  structure family, varying composition.
- ``carbon_24``: ~10k carbon allotropes (Pickard, 2020, AIRSS). Same
  composition (pure C), varying structure.
- ``mp_20``    : ~45k general inorganic materials with <=20 atoms/cell
  (Materials Project; Jain et al., 2013).

The raw CSVs contain a ``cif`` column (full pymatgen-generated CIF string)
plus subset-specific scalar property columns. This loader keeps the raw
CIF string accessible per item (``self.cif``) and builds a lightweight
118-dimensional element-fraction feature vector ``x`` *without* pymatgen
or ase, by hand-parsing the chemical formula (from a formula column when
present, else from the CIF's ``_chemical_formula_sum`` / ``_chemical_
formula_structural`` tag).

Args:
    subset: one of ``"perov_5"`` (default, smallest), ``"carbon_24"``,
        ``"mp_20"``.
    split: one of ``"train"``, ``"val"``, ``"test"``.
    root: base cache directory. Default ``"data/cdvae"``. Each CSV is
        cached under ``<root>/<subset>/<split>.csv``.
    target: name of the numeric property column to use as the regression
        target. Defaults to ``"formation_energy_per_atom"`` (valid for
        ``mp_20``); per-subset auto-detected numeric columns are listed
        in the error message if an invalid target is given. Per-subset
        sensible defaults are substituted when the global default is not
        present in the chosen subset (``heat_ref`` for perov_5,
        ``energy_per_atom`` for carbon_24).
    download: if True, fetch the CSV from raw GitHub if not cached.
    transform / target_transform: optional callables, applied per item.

Shapes:
    x : (118,)  float32  — element fractions (sum to 1 per row, or all
        zeros if the formula could not be parsed).
    y : ()      float32  — selected scalar property.

Public attributes:
    df (pd.DataFrame, the parsed table for this split),
    cif (list[str], length N, raw CIF strings),
    ids (list[str], length N, material ids),
    feature_names (list[str], length 118, element symbols),
    target (str, resolved target column),
    X (Tensor [N, 118]), y (Tensor [N]).

Source URLs (static, no key required):
    https://raw.githubusercontent.com/txie-93/cdvae/main/data/<subset>/<split>.csv

Citation:
    Xie, T., Fu, X., Ganea, O.-E., Barzilay, R. & Jaakkola, T. "Crystal
    Diffusion Variational Autoencoder for Periodic Material Generation."
    ICLR 2022. arXiv:2110.06197. Please also cite the upstream sources:
    perov_5 from the Castelli et al. (2012) computational perovskite set,
    carbon_24 from Pickard (2020) AIRSS, mp_20 from the Materials Project
    (Jain et al., 2013).

License:
    The CDVAE code repository is MIT licensed (Copyright (c) 2021 Tian
    Xie, Xiang Fu). The datasets are redistributed from the upstream DFT
    studies cited above; cite the original papers when using them.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

_RAW_BASE = "https://raw.githubusercontent.com/txie-93/cdvae/main/data"

_SUBSETS = ("perov_5", "carbon_24", "mp_20")
_SPLITS = ("train", "val", "test")

# Columns that are never valid regression targets even though they may be
# numeric-looking (identifiers / the unnamed pandas index).
_NON_TARGET_COLS = {"", "Unnamed: 0", "material_id", "cif", "formula",
                     "pretty_formula", "elements"}

# Per-subset fallback default target when the global default
# ("formation_energy_per_atom") is absent from the subset.
_DEFAULT_TARGET = {
    "perov_5": "heat_ref",
    "carbon_24": "energy_per_atom",
    "mp_20": "formation_energy_per_atom",
}

# Formula column name per subset (None => parse from the CIF text).
_FORMULA_COL = {
    "perov_5": "formula",
    "carbon_24": None,        # pure carbon; no formula column
    "mp_20": "pretty_formula",
}

# 118-element periodic table (symbol order = atomic number). Hardcoded so
# the core path needs no pymatgen/ase.
_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
_ELEM_IDX = {s: i for i, s in enumerate(_ELEMENTS)}

# Matches "Fe2 O3" / "FeO" / "Cu1 Cd1 N3" style formula tokens.
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)")


def _csv_url(subset: str, split: str) -> str:
    return f"{_RAW_BASE}/{subset}/{split}.csv"


def _download_one(url: str, dest: Path) -> None:
    """Stream `url` to `dest` via a `.partial` temp file (atomic-ish)."""
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(tmp, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=f"CDVAE {url.split('/')[-2]}/{url.split('/')[-1]}",
            ) as pbar:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    pbar.update(len(chunk))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)


def _write_readme(root: Path, entries: list[tuple[str, str, str]]) -> None:
    """Write/refresh `<root>/README.md` with sources, SHA-256, citation.

    `entries` is a list of (relative_path, source_url, sha256).
    """
    citation = (
        "Xie, T., Fu, X., Ganea, O.-E., Barzilay, R. & Jaakkola, T. "
        "\"Crystal Diffusion Variational Autoencoder for Periodic Material "
        "Generation.\" ICLR 2022. arXiv:2110.06197.\n\n"
        "Upstream dataset sources (please also cite):\n"
        "- perov_5: Castelli, I. E. et al. New cubic perovskites for one- "
        "and two-photon water splitting. Energy Environ. Sci. 5, 9034 "
        "(2012); and the related computational screening study "
        "(Energy Environ. Sci. 5, 5814, 2012).\n"
        "- carbon_24: Pickard, C. J. AIRSS data for carbon at 10 GPa "
        "(Materials Cloud, 2020). doi:10.24435/MATERIALSCLOUD:2020.0026/V1.\n"
        "- mp_20: Jain, A. et al. Commentary: The Materials Project. "
        "APL Mater. 1, 011002 (2013)."
    )
    lines = ["# CDVAE benchmark datasets\n",
             "Downloaded via `ai4mat.datasets.cdvae_materials."
             "download_if_missing`.\n",
             "## Cached files\n"]
    for rel, url, sha in sorted(entries):
        lines.append(f"- `{rel}`\n  - Source: {url}\n  - SHA-256: {sha}\n")
    lines.append("\n## Citation\n\n" + citation + "\n")
    lines.append(
        "\n## License\n\nThe CDVAE code repository "
        "(https://github.com/txie-93/cdvae) is MIT licensed "
        "(Copyright (c) 2021 Tian Xie, Xiang Fu). The benchmark data is "
        "redistributed from the upstream DFT studies above; cite the "
        "original papers when using these datasets.\n"
    )
    (root / "README.md").write_text("".join(lines))


def _readme_entries(root: Path) -> list[tuple[str, str, str]]:
    """Scan `<root>` for cached `<subset>/<split>.csv` and hash them."""
    entries: list[tuple[str, str, str]] = []
    for subset in _SUBSETS:
        for split in _SPLITS:
            f = root / subset / f"{split}.csv"
            if f.exists():
                sha = hashlib.sha256(f.read_bytes()).hexdigest()
                entries.append(
                    (f"{subset}/{split}.csv", _csv_url(subset, split), sha)
                )
    return entries


def download_if_missing(root: str, subset: str, split: str) -> Path:
    """Ensure `<root>/<subset>/<split>.csv` exists; return its path.

    Idempotent: returns immediately if the CSV is already cached. The
    `<root>/README.md` is regenerated to cover all currently cached files.
    """
    root_path = Path(root)
    dest = root_path / subset / f"{split}.csv"
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _csv_url(subset, split)
    try:
        _download_one(url, dest)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download CDVAE {subset}/{split} from {url}: {exc!r}. "
            f"Download it manually and place it at {dest}."
        ) from exc

    _write_readme(root_path, _readme_entries(root_path))
    return dest


def _parse_formula_from_cif(cif: str) -> str:
    """Extract a chemical formula string from a pymatgen-generated CIF.

    Prefers `_chemical_formula_sum` (e.g. ``'Cu1 Cd1 N3'``); falls back to
    `_chemical_formula_structural`. Returns ``""`` if none found.
    """
    for tag in ("_chemical_formula_sum", "_chemical_formula_structural"):
        m = re.search(
            rf"{tag}\s+'?\"?([^'\"\n]+)'?\"?", cif
        )
        if m:
            return m.group(1).strip()
    return ""


def _formula_to_fractions(formula: str) -> np.ndarray:
    """Hand-parse a formula into a (118,) element-fraction vector.

    Unparseable / empty formulas yield an all-zero vector. Unknown element
    symbols are skipped. Counts are normalised to sum to 1.
    """
    vec = np.zeros(len(_ELEMENTS), dtype=np.float32)
    if not formula:
        return vec
    for sym, num in _FORMULA_TOKEN.findall(formula):
        if sym not in _ELEM_IDX:
            continue
        count = float(num) if num not in ("", ".") else 1.0
        vec[_ELEM_IDX[sym]] += count
    s = vec.sum()
    if s > 0:
        vec /= s
    return vec


def _detect_numeric_targets(df: pd.DataFrame) -> list[str]:
    """Numeric columns of `df` usable as regression targets (sorted)."""
    targets = []
    for col in df.columns:
        if col in _NON_TARGET_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            targets.append(col)
    return sorted(targets)


class CDVAEMaterialsDataset(Dataset):
    """CDVAE generative-materials benchmark datasets (Xie et al., 2022).

    See module docstring for the full description, shapes, citation and
    license. Subset/split/target are validated with explicit errors.
    """

    def __init__(
        self,
        subset: str = "perov_5",
        split: str = "train",
        root: str = "data/cdvae",
        target: str = "formation_energy_per_atom",
        download: bool = True,
        transform=None,
        target_transform=None,
    ):
        if subset not in _SUBSETS:
            raise ValueError(
                f"subset must be one of {_SUBSETS}, got {subset!r}"
            )
        if split not in _SPLITS:
            raise ValueError(
                f"split must be one of {_SPLITS}, got {split!r}"
            )

        self.subset = subset
        self.split = split
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        csv = self.root / subset / f"{split}.csv"
        if not csv.exists():
            if download:
                csv = download_if_missing(str(self.root), subset, split)
            else:
                raise FileNotFoundError(
                    f"{csv} not found. Set download=True or place the CDVAE "
                    f"{subset}/{split}.csv there manually."
                )

        df = pd.read_csv(csv)
        # Drop the unnamed pandas/CSV index column if present.
        df = df.drop(columns=[c for c in df.columns
                              if c == "" or c.startswith("Unnamed:")],
                     errors="ignore")
        self.df = df.reset_index(drop=True)

        numeric_targets = _detect_numeric_targets(self.df)
        # Resolve target: explicit > requested global default mapped to a
        # per-subset sensible default if absent.
        resolved = target
        if resolved not in numeric_targets:
            if target == "formation_energy_per_atom":
                resolved = _DEFAULT_TARGET[subset]
        if resolved not in numeric_targets:
            raise ValueError(
                f"target {target!r} is not a numeric column of subset "
                f"{subset!r}. Available numeric targets: {numeric_targets}"
            )
        self.target = resolved

        # Per-item raw CIF strings and ids.
        self.cif: list[str] = self.df["cif"].astype(str).tolist()
        if "material_id" in self.df.columns:
            self.ids = self.df["material_id"].astype(str).tolist()
        else:
            self.ids = [str(i) for i in range(len(self.df))]

        # Feature source: formula column if this subset has one, else parse
        # the formula out of the CIF text.
        fcol = _FORMULA_COL[subset]
        if fcol is not None and fcol in self.df.columns:
            formulas = self.df[fcol].astype(str).tolist()
        else:
            formulas = [_parse_formula_from_cif(c) for c in self.cif]
        self.formulas = formulas

        X = np.vstack(
            [_formula_to_fractions(f) for f in formulas]
        ).astype(np.float32) if formulas else np.zeros((0, 118), np.float32)
        self.feature_names = list(_ELEMENTS)
        self.X = torch.from_numpy(X).float()

        y = self.df[self.target].to_numpy(dtype=np.float32)
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
