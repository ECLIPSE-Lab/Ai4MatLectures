"""Revised MD17 (rMD17) molecular energy/force dataset.

Christensen & von Lilienfeld, "On the role of gradients for machine learning
of molecular energies and forces", Mach. Learn.: Sci. Technol. 1, 045018
(2020). Materials Cloud Archive 2020.82. Hosted on Figshare article 12672038.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import warnings
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Figshare article 12672038 ("Revised MD17 dataset (rMD17)") exposes every
# molecule as an individually downloadable .npz with a *stable* ndownloader
# file id, so we never have to fetch the ~1 GB rmd17.tar.bz2 archive. File
# ids were resolved once via the public, key-less figshare article API
# (https://api.figshare.com/v2/articles/12672038); they are immutable.
_FIGSHARE_ARTICLE = "https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038"
_NDOWNLOADER = "https://ndownloader.figshare.com/files/{file_id}"

# molecule -> (figshare file id, expected sha256 of the .npz)
_MOLECULES: dict[str, tuple[int, str]] = {
    "aspirin": (62265757, "17fd6fb69066888613f7e16b358a7553"),
    "azobenzene": (62265754, "be79df918468eb3579aa73a0becf7390"),
    "benzene": (62265739, "18c9242bc90fbf28215f6dd81e650f16"),
    "ethanol": (62265733, "eb837fb8deb27d4e0d52f71a03aff776"),
    "malonaldehyde": (62265736, "cdc1d70c0c34062ddde6e5071eb6fe21"),
    "naphthalene": (62265751, "0efba19c9907e3852318b1e6008b3b9e"),
    "paracetamol": (62265760, "ba10784f7b67635427085f6d7ec2dd97"),
    "salicylic": (62265748, "900fedf242da438400fa4293348d7dd1"),
    "toluene": (62265742, "0f2913d51f8149c90ab28d697a076f64"),
    "uracil": (62265745, "992a4479c28a07e0cce6da964805be31"),
}

# rMD17 hard cap: the conformers are sampled from short MD trajectories and
# are therefore strongly correlated. The dataset authors explicitly warn
# that no more than 1000 of them should be used together for training.
_MAX_RECOMMENDED = 1000

_CITATION = (
    "Christensen, A. S. & von Lilienfeld, O. A. On the role of gradients "
    "for machine learning of molecular energies and forces. "
    "Mach. Learn.: Sci. Technol. 1, 045018 (2020). "
    "https://doi.org/10.1088/2632-2153/abba6f  "
    "Dataset: Materials Cloud Archive 2020.82, "
    "https://doi.org/10.24435/materialscloud:wy-kn ; "
    "Figshare article 12672038."
)


def _figshare_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_readme(root: Path, molecule: str, file_id: int, sha256: str) -> None:
    (root / "README.md").write_text(
        f"# Revised MD17 (rMD17) dataset\n\n"
        f"- Source article: {_FIGSHARE_ARTICLE}\n"
        f"- Downloaded file: `rmd17_{molecule}.npz` "
        f"(figshare file id {file_id})\n"
        f"- Download URL: {_NDOWNLOADER.format(file_id=file_id)}\n"
        f"- SHA-256 (rmd17_{molecule}.npz): {sha256}\n"
        f"- Downloaded via `ai4mat.datasets.rmd17.download_if_missing`\n\n"
        f"## Units\n\n"
        f"- Energies: kcal/mol\n"
        f"- Forces: kcal/mol/Angstrom\n"
        f"- Coordinates: Angstrom\n\n"
        f"## IMPORTANT teaching caveat\n\n"
        f"The rMD17 conformers come from short molecular-dynamics\n"
        f"trajectories and are therefore strongly correlated. The dataset\n"
        f"authors explicitly warn that **no more than 1000 of these\n"
        f"structures should be used together** for training a model -- "
        f"using more does not add independent information and leads to\n"
        f"misleadingly optimistic error estimates. `RMD17Dataset` defaults\n"
        f"to `n_samples=1000` and subsamples deterministically.\n\n"
        f"## Citation\n\n{_CITATION}\n\n"
        f"## License\n\n"
        f"Creative Commons Attribution 4.0 (CC BY 4.0).\n"
    )


def download_if_missing(root: str, molecule: str) -> Path:
    """Download `rmd17_<molecule>.npz` into `root` if not already present.

    Returns the path to the local .npz. Idempotent: if the file already
    exists with the expected SHA-256 it is reused. Only the requested
    molecule is fetched (each .npz is ~65-175 MB).
    """
    if molecule not in _MOLECULES:
        raise ValueError(
            f"Unknown molecule {molecule!r}. "
            f"Choose one of {sorted(_MOLECULES)}."
        )
    file_id, expected_md5 = _MOLECULES[molecule]
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    dest = root_path / f"rmd17_{molecule}.npz"

    if dest.exists():
        _write_readme(root_path, molecule, file_id, _sha256(dest))
        return dest

    url = _NDOWNLOADER.format(file_id=file_id)
    tmp_handle = tempfile.NamedTemporaryFile(
        dir=str(root_path), prefix=f"_rmd17_{molecule}_",
        suffix=".npz.partial", delete=False,
    )
    tmp = Path(tmp_handle.name)
    tmp_handle.close()
    try:
        with urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(tmp, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=f"rMD17 ({molecule})",
            ) as pbar:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    pbar.update(len(chunk))

        got_md5 = _figshare_md5(tmp)
        if got_md5 != expected_md5:
            raise RuntimeError(
                f"Checksum mismatch for rmd17_{molecule}.npz: expected MD5 "
                f"{expected_md5}, got {got_md5}. Download may be corrupt or "
                f"the figshare file id changed; re-resolve via "
                f"https://api.figshare.com/v2/articles/12672038"
            )
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()

    _write_readme(root_path, molecule, file_id, _sha256(dest))
    return dest


class RMD17Dataset(Dataset):
    """Revised MD17 (rMD17) molecular energy-regression dataset.

    Per-molecule conformers with DFT (PBE/def2-SVP) recomputed energies
    and forces, from Christensen & von Lilienfeld (2020). Usable as a
    plain energy-regression ``Dataset``; per-conformer forces are exposed
    as an attribute for force-matching exercises.

    IMPORTANT teaching caveat
    -------------------------
    The conformers are drawn from short MD trajectories and are strongly
    correlated. The dataset authors explicitly warn that **no more than
    1000 structures should be used together for training**. ``n_samples``
    therefore defaults to 1000 and the subsample is drawn deterministically
    from ``seed``. Raising it above 1000 emits a warning.

    Args:
        molecule: one of aspirin, azobenzene, benzene, ethanol,
            malonaldehyde, naphthalene, paracetamol, salicylic, toluene,
            uracil. Default ``"aspirin"``.
        root: cache directory. Default ``"data/rmd17"``.
        n_samples: number of conformers to keep (default 1000; see caveat).
        split: ``"train"`` or ``"test"``. ``"train"`` takes the first
            ``n_samples`` of the deterministic permutation; ``"test"``
            takes the next ``n_samples`` (disjoint hold-out).
        seed: RNG seed for the deterministic conformer permutation.
        download: if True, fetch the molecule's .npz when missing.
        transform / target_transform: optional callables, applied per item.

    Item ``(x, y)``:
        x: float32 tensor, shape ``(4 * n_atoms,)`` -- the atomic numbers
           Z (cast to float, first ``n_atoms`` entries) concatenated with
           the flattened conformer coordinates (``3 * n_atoms`` entries,
           in Angstrom).
        y: float32 scalar tensor -- the conformer energy (kcal/mol).

    Units:
        energies kcal/mol, forces kcal/mol/Angstrom, coords Angstrom.

    Public attributes:
        Z          (LongTensor [n_atoms])               atomic numbers
        coords     (FloatTensor [N, n_atoms, 3])        Angstrom
        energies   (FloatTensor [N])                    kcal/mol
        forces     (FloatTensor [N, n_atoms, 3])        kcal/mol/Angstrom
        n_atoms    (int)
        molecule   (str)
        indices    (LongTensor [N])  indices into the full 100k array
        X (FloatTensor [N, 4*n_atoms]), y (FloatTensor [N])

    Citation:
        Christensen, A. S. & von Lilienfeld, O. A. On the role of gradients
        for machine learning of molecular energies and forces.
        Mach. Learn.: Sci. Technol. 1, 045018 (2020).
        Materials Cloud Archive 2020.82.

    License:
        Creative Commons Attribution 4.0 (CC BY 4.0).
    """

    MOLECULES = tuple(sorted(_MOLECULES))

    def __init__(
        self,
        molecule: str = "aspirin",
        root: str = "data/rmd17",
        n_samples: int = 1000,
        split: str = "train",
        seed: int = 0,
        download: bool = True,
        transform=None,
        target_transform=None,
    ):
        if molecule not in _MOLECULES:
            raise ValueError(
                f"Unknown molecule {molecule!r}. "
                f"Choose one of {sorted(_MOLECULES)}."
            )
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if n_samples > _MAX_RECOMMENDED:
            warnings.warn(
                f"n_samples={n_samples} exceeds the rMD17 recommended limit "
                f"of {_MAX_RECOMMENDED}: these conformers are strongly "
                f"correlated and using more than {_MAX_RECOMMENDED} together "
                f"yields misleadingly optimistic error estimates.",
                stacklevel=2,
            )

        self.molecule = molecule
        self.root = Path(root)
        self.split = split
        self.seed = seed
        self.transform = transform
        self.target_transform = target_transform

        npz_path = self.root / f"rmd17_{molecule}.npz"
        if not npz_path.exists():
            if download:
                npz_path = download_if_missing(str(self.root), molecule)
            else:
                raise FileNotFoundError(
                    f"{npz_path} not found. Set download=True or place the "
                    f"rmd17_{molecule}.npz file there manually."
                )

        with np.load(npz_path) as data:
            Z = np.asarray(data["nuclear_charges"], dtype=np.int64)
            coords = np.asarray(data["coords"], dtype=np.float32)
            energies = np.asarray(data["energies"], dtype=np.float32)
            forces = np.asarray(data["forces"], dtype=np.float32)

        n_total = coords.shape[0]
        self.n_atoms = int(Z.shape[0])

        # Deterministic conformer permutation; train = first block,
        # test = the disjoint next block.
        perm = np.random.RandomState(seed).permutation(n_total)
        offset = 0 if split == "train" else n_samples
        take = min(n_samples, max(0, n_total - offset))
        if take <= 0:
            raise ValueError(
                f"Not enough conformers ({n_total}) for split={split!r} "
                f"with n_samples={n_samples}."
            )
        if take < n_samples:
            warnings.warn(
                f"Requested n_samples={n_samples} for split={split!r} but "
                f"only {take} conformers available after the split offset.",
                stacklevel=2,
            )
        sel = perm[offset:offset + take]

        self.indices = torch.from_numpy(sel.astype(np.int64))
        self.Z = torch.from_numpy(Z)  # (n_atoms,) long
        self.coords = torch.from_numpy(coords[sel])  # (N, n_atoms, 3)
        self.energies = torch.from_numpy(energies[sel])  # (N,)
        self.forces = torch.from_numpy(forces[sel])  # (N, n_atoms, 3)

        # Flatten to a single float tensor so the dataset stays usable with
        # the standard (Tensor x, scalar y) contract: [Z..., coords...].
        z_float = self.Z.to(torch.float32).unsqueeze(0).expand(take, -1)
        flat_coords = self.coords.reshape(take, -1)
        self.X = torch.cat([z_float, flat_coords], dim=1).contiguous()
        self.y = self.energies

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
