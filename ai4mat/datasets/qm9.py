"""QM9 quantum-chemistry dataset (Ramakrishnan et al. 2014, Sci. Data 1, 140022).

133,885 small organic molecules (up to 9 heavy atoms: C, N, O, F) with DFT
(B3LYP/6-31G(2df,p)) geometric, energetic, electronic and thermodynamic
properties. This loader uses the clean CSV redistribution hosted by DeepChem.

Citation:
    Ramakrishnan, R., Dral, P. O., Rupp, M. & von Lilienfeld, O. A.
    Quantum chemistry structures and properties of 134 kilo molecules.
    Scientific Data 1, 140022 (2014). https://doi.org/10.1038/sdata.2014.22
    Original deposit: figshare collection, DOI 10.6084/m9.figshare.c.978904

License:
    CC0 1.0 (public domain dedication, per the original figshare deposit).
"""
from __future__ import annotations

import hashlib
import tempfile
import warnings
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

_QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

# Verified header (first KB fetched 2026-05-18):
#   mol_id, smiles, A, B, C, mu, alpha, homo, lumo, gap, r2, zpve,
#   u0, u298, h298, g298, cv, u0_atom, u298_atom, h298_atom, g298_atom
# Non-property identifier columns:
_ID_COLUMNS = ["mol_id", "smiles"]

# Selectable numeric target properties (every column that is not an id).
_PROPERTY_NAMES = [
    "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
    "u0", "u298", "h298", "g298", "cv",
    "u0_atom", "u298_atom", "h298_atom", "g298_atom",
]

# Elements covered by the hand-rolled featuriser. QM9 molecules contain only
# H, C, N, O, F; we keep this explicit list as the feature basis.
_FEATURE_ELEMENTS = ["C", "H", "N", "O", "F"]


def _write_readme(root: Path, source_url: str, sha256: str) -> None:
    citation = (
        "Ramakrishnan, R., Dral, P. O., Rupp, M. & von Lilienfeld, O. A. "
        "Quantum chemistry structures and properties of 134 kilo molecules. "
        "Scientific Data 1, 140022 (2014). https://doi.org/10.1038/sdata.2014.22"
    )
    (root / "README.md").write_text(
        f"# QM9 dataset\n\n"
        f"- Source URL: {source_url}\n"
        f"- SHA-256: {sha256}\n"
        f"- Downloaded: via `ai4mat.datasets.qm9.download_if_missing`\n\n"
        f"## Citation\n\n{citation}\n\n"
        f"Original deposit: figshare collection, "
        f"DOI 10.6084/m9.figshare.c.978904\n\n"
        f"## License\n\n"
        f"CC0 1.0 (public domain dedication, per the original figshare "
        f"deposit).\n"
    )


def download_if_missing(root: str) -> None:
    """Download the QM9 CSV into ``<root>/qm9.csv`` if not already present.

    Idempotent: returns immediately if the file already exists. Downloads to
    a temporary ``.partial`` file in ``root`` and atomically renames it on
    success, so an interrupted download never leaves a truncated CSV in place.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    dest = root_path / "qm9.csv"
    if dest.exists():
        return

    with tempfile.NamedTemporaryFile(
        dir=root_path, prefix="_qm9_", suffix=".csv.partial", delete=False
    ) as tmp_handle:
        tmp = Path(tmp_handle.name)
    try:
        with urlopen(_QM9_URL, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(tmp, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc="QM9"
            ) as pbar:
                while True:
                    chunk = response.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    fh.write(chunk)
                    pbar.update(len(chunk))

        # Sanity-check the header before committing the file.
        header = pd.read_csv(tmp, nrows=1)
        missing = [c for c in _ID_COLUMNS if c not in header.columns]
        if missing:
            raise RuntimeError(
                f"Downloaded QM9 CSV is missing expected columns {missing}. "
                f"Found: {list(header.columns)}"
            )
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    _write_readme(root_path, _QM9_URL, sha)


def _tokenize_smiles_atoms(smiles: str) -> list[str]:
    """Hand-rolled SMILES atom tokenizer (teaching baseline, NOT chemistry-grade).

    Scans the SMILES string left to right and extracts element symbols:

    * Two-letter organic-subset symbols ``Cl`` and ``Br`` are matched first.
    * Bracket atoms ``[...]`` (e.g. ``[NH4+]``, ``[O-]``, ``[nH]``) contribute
      the element symbol immediately after the ``[`` (and an optional second
      lowercase letter), e.g. ``[Si]`` -> ``Si``; the bracket's explicit
      ``H``-count digits are NOT parsed (this is a deliberate simplification).
    * Single upper-case organic-subset atoms ``B C N O P S F I`` and aromatic
      lower-case ``b c n o p s`` are mapped to their element (aromatic ->
      upper-case).
    * Structural characters (digits, bonds ``= # - : / \\``, branch parens,
      ring-closure digits, ``%``, ``.``, ``@``, ``+``) are ignored.

    Implicit hydrogens are *not* inferred here; they are added afterwards in
    ``_featurize_smiles`` via a crude valence model. This is intentionally a
    simple, dependency-free feature for teaching, not a real descriptor.
    """
    atoms: list[str] = []
    i = 0
    n = len(smiles)
    aromatic_lower = {"b", "c", "n", "o", "p", "s"}
    organic_upper = {"B", "C", "N", "O", "P", "S", "F", "I"}
    while i < n:
        ch = smiles[i]
        if ch == "[":
            j = smiles.find("]", i)
            if j == -1:
                break  # malformed; stop parsing
            inner = smiles[i + 1 : j]
            # Skip a leading isotope number, then read the element symbol.
            k = 0
            while k < len(inner) and inner[k].isdigit():
                k += 1
            if k < len(inner) and inner[k].isalpha():
                sym = inner[k].upper()
                if k + 1 < len(inner) and inner[k + 1].islower():
                    sym += inner[k + 1]
                atoms.append(sym)
            i = j + 1
            continue
        # Two-letter organic-subset halogens.
        if smiles[i : i + 2] == "Cl":
            atoms.append("Cl")
            i += 2
            continue
        if smiles[i : i + 2] == "Br":
            atoms.append("Br")
            i += 2
            continue
        if ch in organic_upper:
            atoms.append(ch)
        elif ch in aromatic_lower:
            atoms.append(ch.upper())
        # else: structural char (digits, bonds, parens, etc.) — ignore.
        i += 1
    return atoms


# Crude default valences for implicit-H estimation (teaching baseline only).
_DEFAULT_VALENCE = {"C": 4, "N": 3, "O": 2, "F": 1, "H": 1}


def _featurize_smiles(smiles: str) -> np.ndarray:
    """Map one SMILES string to a small fixed-length numeric feature vector.

    Features (7 dims), all dependency-free and derived from the hand-rolled
    tokenizer above:

        [0..4] counts of C, H, N, O, F
        [5]    heavy-atom count (all non-H atoms found)
        [6]    total atom count (heavy + estimated implicit H)

    Hydrogen handling: explicit H tokens (rare in QM9 SMILES) are counted
    directly; implicit hydrogens are estimated per heavy atom as
    ``max(default_valence - degree_in_SMILES, 0)`` using a fixed default
    valence table. This is a deliberately crude valence model — it ignores
    bond orders, charges and aromaticity — and exists only as a teaching
    feature baseline, not a serious molecular descriptor.
    """
    atoms = _tokenize_smiles_atoms(smiles)
    counts = {el: 0 for el in _FEATURE_ELEMENTS}
    explicit_h = 0
    heavy = 0
    for a in atoms:
        if a == "H":
            explicit_h += 1
            continue
        heavy += 1
        if a in counts:
            counts[a] += 1
        # Atoms outside the C/H/N/O/F basis still count toward `heavy`
        # but contribute no element-count feature (QM9 has none of these).

    # Crude implicit-H estimate: every heavy atom in the feature basis with
    # a known default valence donates (valence - 1) hydrogens as a flat
    # approximation (no bond-order accounting). This is intentionally simple.
    implicit_h = 0
    for el in ("C", "N", "O", "F"):
        implicit_h += counts[el] * max(_DEFAULT_VALENCE[el] - 1, 0)
    total_h = explicit_h + implicit_h
    counts["H"] = total_h

    total_atoms = heavy + total_h
    return np.array(
        [
            counts["C"],
            counts["H"],
            counts["N"],
            counts["O"],
            counts["F"],
            heavy,
            total_atoms,
        ],
        dtype=np.float32,
    )


_FEATURE_NAMES = [
    "n_C", "n_H", "n_N", "n_O", "n_F", "n_heavy", "n_atoms_total",
]


class QM9Dataset(Dataset):
    """QM9 quantum-chemistry dataset (Ramakrishnan et al. 2014) as a Dataset.

    133,885 small organic molecules. Features ``x`` are a 7-dim hand-derived
    numeric vector parsed from each molecule's SMILES string *without* rdkit
    or any chemistry dependency (element counts + atom-count proxies — a
    teaching baseline, not a real descriptor). The target ``y`` is a single
    selectable scalar DFT property.

    Args:
        root: cache directory. Default ``"data/qm9"``.
        target: property column to predict. One of
            ``A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, u0, u298,
            h298, g298, cv, u0_atom, u298_atom, h298_atom, g298_atom``.
            Default ``"gap"`` (HOMO-LUMO gap, Hartree).
        download: if True and the CSV is absent, fetch it.
        transform / target_transform: optional callables, applied per item.

    Shapes:
        x: (7,)  dtype float32  — [n_C, n_H, n_N, n_O, n_F, n_heavy, n_atoms]
        y: ()    dtype float32  — selected scalar property

    Public attributes:
        df (pandas.DataFrame, the full QM9 table),
        smiles (list[str], length N, raw SMILES),
        target (str, the selected property name),
        property_names (list[str], all selectable targets),
        feature_names (list[str], length 7),
        X (Tensor [N, 7]), y (Tensor [N]).

    Citation:
        Ramakrishnan, Dral, Rupp & von Lilienfeld, Sci. Data 1, 140022 (2014).
        https://doi.org/10.1038/sdata.2014.22

    License: CC0 1.0 (public domain dedication).
    """

    property_names = list(_PROPERTY_NAMES)

    def __init__(
        self,
        root: str = "data/qm9",
        target: str = "gap",
        download: bool = True,
        transform=None,
        target_transform=None,
    ):
        if target not in _PROPERTY_NAMES:
            raise ValueError(
                f"target must be one of {_PROPERTY_NAMES}, got {target!r}"
            )

        self.root = Path(root)
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

        csv = self.root / "qm9.csv"
        if not csv.exists():
            if download:
                download_if_missing(str(self.root))
            else:
                raise FileNotFoundError(
                    f"{csv} not found. Set download=True or place the file "
                    f"manually."
                )

        df = pd.read_csv(csv)
        missing = [c for c in _ID_COLUMNS + _PROPERTY_NAMES
                   if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"QM9 CSV at {csv} is missing expected columns {missing}. "
                f"Found: {list(df.columns)}"
            )

        df = df.dropna(subset=["smiles", target]).reset_index(drop=True)
        self.df = df
        self.smiles: list[str] = df["smiles"].astype(str).tolist()
        self.feature_names = list(_FEATURE_NAMES)

        X = np.empty((len(self.smiles), len(_FEATURE_NAMES)), dtype=np.float32)
        for i, s in enumerate(
            tqdm(self.smiles, desc="QM9 SMILES features", unit="mol")
        ):
            X[i] = _featurize_smiles(s)

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(
            df[target].to_numpy(dtype=np.float32)
        ).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
