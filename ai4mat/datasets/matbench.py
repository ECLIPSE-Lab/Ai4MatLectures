"""Matbench v0.1 single-task regression datasets (Dunn et al. 2020).

Thin teaching loader over the Figshare-hosted Matbench v0.1 task files
(matminer's dataset registry; project page
https://figshare.com/projects/Matbench_v0_1_Datasets/67337). Each task
ships as a gzip-compressed pandas DataFrame JSON (``orient="split"``),
encoded with monty/``MontyEncoder`` so pymatgen ``Structure`` objects
survive as plain nested dicts.

The core path is dependency-light: it never imports pymatgen or matminer.
Composition features are hand-rolled element-fraction vectors over a fixed
118-element table, derived either from a chemical-formula string
(composition tasks) or from the structure dict's per-site species
(structure tasks). The raw structure dicts stay accessible via
``self.structures`` for richer downstream lessons.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Figshare file URLs + SHA-256, lifted verbatim from matminer's public
# dataset registry (matminer/datasets/dataset_metadata.json). These are
# the canonical per-task Matbench v0.1 files; the host below is the stable
# Figshare-backed mirror used by matminer. We reuse the URLs only — we do
# NOT depend on matminer.
#
# Schema per task: (url, sha256, feature_column, target_column, n_entries)
# feature_column is "composition" (a formula string) or "structure"
# (a pymatgen Structure dict).
_TASKS: dict[str, dict] = {
    "matbench_jdft2d": {
        "url": "https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz",
        "sha256": "26057dc4524e193e32abffb296ce819b58b6e11d1278cae329a2f97817a4eddf",
        "feature": "structure",
        "target": "exfoliation_en",
        "n": 636,
    },
    "matbench_steels": {
        "url": "https://ml.materialsproject.org/projects/matbench_steels.json.gz",
        "sha256": "473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252",
        "feature": "composition",
        "target": "yield strength",
        "n": 312,
    },
    "matbench_phonons": {
        "url": "https://ml.materialsproject.org/projects/matbench_phonons.json.gz",
        "sha256": "4db551f21ec5f577e6202725f10e34dfc509aa7df3a6bdaac497da7f6dbbb9b3",
        "feature": "structure",
        "target": "last phdos peak",
        "n": 1265,
    },
    "matbench_dielectric": {
        "url": "https://ml.materialsproject.org/projects/matbench_dielectric.json.gz",
        "sha256": "83befa09bc2ec2f4b6143afc413157827a90e5e2e42c1eb507ccfa01bf26a1d6",
        "feature": "structure",
        "target": "n",
        "n": 4764,
    },
    "matbench_log_gvrh": {
        "url": "https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz",
        "sha256": "098af941f4c663270f1fe21abf20ffad6fb85ecbfcba5786ceac03983ac29da7",
        "feature": "structure",
        "target": "log10(G_VRH)",
        "n": 10987,
    },
    "matbench_log_kvrh": {
        "url": "https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz",
        "sha256": "44b113ddb7e23aa18731a62c74afa7e5aa654199e0db5f951c8248a00955c9cd",
        "feature": "structure",
        "target": "log10(K_VRH)",
        "n": 10987,
    },
    "matbench_perovskites": {
        "url": "https://ml.materialsproject.org/projects/matbench_perovskites.json.gz",
        "sha256": "4641e2417f8ec8b50096d2230864468dfa08278dc9d257c327f65d0305278483",
        "feature": "structure",
        "target": "e_form",
        "n": 18928,
    },
    "matbench_expt_gap": {
        "url": "https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz",
        "sha256": "783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2",
        "feature": "composition",
        "target": "gap expt",
        "n": 4604,
    },
}

_DEFAULT_TASK = "matbench_steels"

# Fixed 118-element symbol table (Z = 1..118), used for the element-fraction
# featurization. Hand-rolled so the core path needs no pymatgen/mendeleev.
_ELEMENTS: list[str] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
_ELEMENT_IDX: dict[str, int] = {s: i for i, s in enumerate(_ELEMENTS)}

# Element token, possibly followed by a (possibly fractional) count.
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def supported_tasks() -> list[str]:
    """Sorted list of Matbench task names this loader understands."""
    return sorted(_TASKS)


def _readme_path(root: Path, task: str) -> Path:
    return root / f"{task}.README.md"


def _raw_path(root: Path, task: str) -> Path:
    return root / f"{task}.json.gz"


def _write_readme(root: Path, task: str, source_url: str, sha256: str) -> None:
    citation = (
        "Dunn, A., Wang, Q., Ganose, A., Dopp, D. & Jain, A. Benchmarking "
        "materials property prediction methods: the Matbench test suite. "
        "npj Comput. Mater. 6, 138 (2020). "
        "https://doi.org/10.1038/s41524-020-00406-3"
    )
    _readme_path(root, task).write_text(
        f"# Matbench v0.1 — {task}\n\n"
        f"- Source URL: {source_url}\n"
        f"- SHA-256: {sha256}\n"
        f"- Downloaded via `ai4mat.datasets.matbench.download_if_missing`\n"
        f"- Matbench v0.1 datasets project: "
        f"https://figshare.com/projects/Matbench_v0_1_Datasets/67337\n\n"
        f"## Citation\n\n{citation}\n\n"
        f"## License\n\n"
        f"The Matbench datasets are redistributed by their authors under "
        f"the MIT License (see the Matbench repository, "
        f"https://github.com/materialsproject/matbench). Individual source "
        f"datasets carry their own original attributions; consult the "
        f"Matbench paper above for per-task provenance.\n"
    )


def download_if_missing(task: str, root: str) -> None:
    """Download ``<task>`` Figshare file into ``<root>/<task>.json.gz``.

    Idempotent: returns immediately if the raw file already exists and its
    SHA-256 matches the registry entry. A mismatching cached file is
    re-downloaded.
    """
    if task not in _TASKS:
        raise ValueError(
            f"Unknown Matbench task {task!r}. Supported tasks: "
            f"{supported_tasks()}"
        )
    meta = _TASKS[task]
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    dest = _raw_path(root_path, task)

    if dest.exists():
        have = hashlib.sha256(dest.read_bytes()).hexdigest()
        if have == meta["sha256"]:
            return
        warnings.warn(
            f"Cached {dest} SHA-256 {have} != expected {meta['sha256']}; "
            f"re-downloading."
        )

    url = meta["url"]
    tmp = dest.with_suffix(".gz.partial")
    try:
        with urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(tmp, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=task
            ) as pbar:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    pbar.update(len(chunk))
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download Matbench task {task!r} from {url}: {exc!r}. "
            f"Download it manually and place it at {dest}."
        ) from exc

    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if sha != meta["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {task!r} SHA-256 {sha} != expected "
            f"{meta['sha256']}. Refusing to use a corrupt/altered file."
        )
    tmp.replace(dest)
    _write_readme(root_path, task, url, sha)


def _load_split_json(path: Path) -> tuple[list, list]:
    """Read a gzip pandas ``orient='split'`` JSON; return (columns, data).

    ``data`` is the row-major list of ``[feature, target]`` pairs.
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        obj = json.load(fh)
    if not (isinstance(obj, dict) and {"columns", "data"} <= set(obj)):
        raise ValueError(
            f"{path} is not a pandas orient='split' JSON "
            f"(got keys {list(obj) if isinstance(obj, dict) else type(obj)})."
        )
    return obj["columns"], obj["data"]


def _frac_from_counts(counts: dict[str, float]) -> np.ndarray:
    """Element-symbol→amount dict → length-118 normalised fraction vector."""
    vec = np.zeros(len(_ELEMENTS), dtype=np.float32)
    total = float(sum(counts.values()))
    if total <= 0:
        return vec
    for sym, amt in counts.items():
        j = _ELEMENT_IDX.get(sym)
        if j is not None:
            vec[j] += float(amt) / total
    return vec


def _counts_from_formula(formula: str) -> dict[str, float]:
    """Parse a flat chemical formula string into element→amount counts.

    Handles plain formulas and simple non-nested decimals/integers
    (e.g. ``Fe0.8Ni0.2``, ``Al2O3``). Bracket groups are flattened by
    ignoring the brackets, which is adequate for Matbench composition
    tasks (steels / expt_gap formulas are bracket-free).
    """
    counts: dict[str, float] = {}
    cleaned = formula.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    for sym, num in _FORMULA_TOKEN.findall(cleaned):
        if not sym:
            continue
        amt = float(num) if num not in ("", ".") else 1.0
        counts[sym] = counts.get(sym, 0.0) + amt
    return counts


def _counts_from_structure(struct: dict) -> dict[str, float]:
    """Sum element amounts over a pymatgen Structure dict's sites.

    Uses only ``struct["sites"][i]["species"]`` (list of
    ``{"element": sym, "occu": x}``); no pymatgen import.
    """
    counts: dict[str, float] = {}
    for site in struct.get("sites", []):
        for sp in site.get("species", []):
            sym = sp.get("element")
            occu = float(sp.get("occu", 1.0))
            if sym:
                counts[sym] = counts.get(sym, 0.0) + occu
    return counts


def _kfold_indices(n: int, n_splits: int = 5, seed: int = 18012019) -> list:
    """Deterministic shuffled K-fold (train_idx, test_idx) tuples.

    Matbench's *official* CV indices ship in the separate ``matbench``
    package, NOT in the Figshare data files, so they cannot be recovered
    from the download alone. We therefore expose a reproducible surrogate
    split (fixed seed) and document the discrepancy. Use the official
    ``matbench`` package if exact benchmark parity is required.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_splits)
    out = []
    for k in range(n_splits):
        test = np.sort(folds[k])
        train = np.sort(np.concatenate([folds[j] for j in range(n_splits) if j != k]))
        out.append((train.astype(np.int64), test.astype(np.int64)))
    return out


class MatBenchDataset(Dataset):
    """Matbench v0.1 single-task regression dataset (Dunn et al. 2020).

    A teaching loader for the Matbench test suite. Each task is a Figshare
    file (matminer's registry) holding a gzip pandas ``orient='split'``
    JSON. The core path uses no pymatgen/matminer: the numeric ``x`` is a
    hand-rolled length-118 element-fraction vector, derived from a formula
    string (composition tasks) or from the structure dict's per-site
    species (structure tasks). Raw structure dicts remain available via
    ``self.structures`` for richer lessons.

    Args:
        task: Matbench task name. One of ``supported_tasks()``; default
            ``"matbench_steels"`` (312 rows, composition-only).
        root: cache directory. Default ``"data/matbench"``.
        download: if True, fetch the Figshare file when absent (verifies
            SHA-256).
        transform / target_transform: optional per-item callables.

    X shape: (118,)  dtype: float32  (per-element fractions, sum ≈ 1)
    y shape: ()       dtype: float32  (the task's target property)

    Public attributes:
        X (Tensor [N, 118]), y (Tensor [N]),
        task (str), target_name (str), feature_kind (str:
            "composition"|"structure"),
        element_names (list[str], length 118),
        formulas (list[str] | None — composition tasks only),
        structures (list[dict] | None — structure tasks only; raw
            pymatgen Structure dicts),
        folds (list of (train_idx, test_idx) int64 ndarray pairs; a
            deterministic seeded 5-fold surrogate — see note below),
        official_folds (False — Matbench's official indices live in the
            separate `matbench` package, not the Figshare files).

    Note on folds:
        The Figshare data files do NOT carry Matbench's official 5-fold CV
        indices (those ship in the `matbench` PyPI package). ``self.folds``
        is therefore a reproducible *surrogate* (fixed-seed shuffled
        KFold), suitable for split-design teaching but NOT for reporting
        official Matbench leaderboard numbers.

    Citation:
        Dunn, A., Wang, Q., Ganose, A., Dopp, D. & Jain, A. Benchmarking
        materials property prediction methods: the Matbench test suite.
        npj Comput. Mater. 6, 138 (2020).
        https://doi.org/10.1038/s41524-020-00406-3

    License:
        Matbench datasets are redistributed by their authors under the MIT
        License (https://github.com/materialsproject/matbench). See the
        per-task README written under ``root`` for provenance.
    """

    def __init__(
        self,
        task: str = _DEFAULT_TASK,
        root: str = "data/matbench",
        download: bool = True,
        transform=None,
        target_transform=None,
    ):
        if task not in _TASKS:
            raise ValueError(
                f"Unknown Matbench task {task!r}. Supported tasks: "
                f"{supported_tasks()}"
            )
        meta = _TASKS[task]
        self.task = task
        self.root = Path(root)
        self.target_name: str = meta["target"]
        self.feature_kind: str = meta["feature"]
        self.element_names: list[str] = list(_ELEMENTS)
        self.transform = transform
        self.target_transform = target_transform
        self.official_folds = False

        raw = _raw_path(self.root, task)
        if not raw.exists():
            if download:
                download_if_missing(task, str(self.root))
            else:
                raise FileNotFoundError(
                    f"{raw} not found. Set download=True or place the "
                    f"Matbench file there manually."
                )

        columns, data = _load_split_json(raw)
        try:
            f_col = columns.index(meta["feature"])
            t_col = columns.index(meta["target"])
        except ValueError as exc:
            raise ValueError(
                f"{raw} columns {columns} do not contain expected "
                f"{meta['feature']!r}/{meta['target']!r}."
            ) from exc

        X = np.zeros((len(data), len(_ELEMENTS)), dtype=np.float32)
        y = np.empty(len(data), dtype=np.float32)
        formulas: Optional[list[str]] = (
            [] if self.feature_kind == "composition" else None
        )
        structures: Optional[list[dict]] = (
            [] if self.feature_kind == "structure" else None
        )

        for i, row in enumerate(data):
            feat = row[f_col]
            if self.feature_kind == "composition":
                formula = str(feat)
                formulas.append(formula)
                counts = _counts_from_formula(formula)
            else:
                structures.append(feat)
                counts = _counts_from_structure(feat)
            X[i] = _frac_from_counts(counts)
            y[i] = float(row[t_col])

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.formulas = formulas
        self.structures = structures
        self.folds = _kfold_indices(len(self.X))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
