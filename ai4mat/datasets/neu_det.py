"""NEU-DET steel-surface-defect dataset."""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

_KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "kaustubhdikshit/neu-surface-defect-database"
)


def _find_train_root(root: str, max_depth: int = 2) -> Optional[str]:
    """Return the dir that directly contains `train/`; None if not found.

    Kaggle archives unzip with varying nesting (`NEU-DET/train/...`,
    `NEU-DET/NEU-DET/train/...`, or `NEU Metal Surface Defects Data/.../train/...`),
    so search up to `max_depth` levels below `root`.
    """
    if not os.path.isdir(root):
        return None
    if os.path.isdir(os.path.join(root, "train")):
        return root

    stack: list[tuple[str, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            entries = sorted(os.listdir(current))
        except OSError:
            continue
        for entry in entries:
            candidate = os.path.join(current, entry)
            if not os.path.isdir(candidate):
                continue
            if os.path.isdir(os.path.join(candidate, "train")):
                return candidate
            stack.append((candidate, depth + 1))
    return None


def download_if_missing(root: str) -> None:
    """Download + unzip the NEU-DET archive into `root` if not already present.

    Idempotent: returns immediately if `root` already contains a `train/`
    directory anywhere up to one level deep.
    """
    os.makedirs(root, exist_ok=True)
    if _find_train_root(root) is not None:
        return

    # Use NamedTemporaryFile so concurrent downloads don't collide on a
    # fixed filename. The file is closed immediately; we reopen by path below.
    tmp_parent = os.path.dirname(root) or "."
    with tempfile.NamedTemporaryFile(
        dir=tmp_parent, prefix="_neu_det_", suffix=".zip", delete=False
    ) as tmp_handle:
        tmp_zip = tmp_handle.name
    try:
        with urlopen(_KAGGLE_URL) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(tmp_zip, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc="NEU-DET"
            ) as pbar:
                while True:
                    chunk = response.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    fh.write(chunk)
                    pbar.update(len(chunk))

        with open(tmp_zip, "rb") as fh:
            magic = fh.read(4)
        if magic != b"PK\x03\x04":
            raise RuntimeError(
                "Kaggle endpoint did not return a zip — likely an auth "
                "redirect. Download manually and place the archive contents "
                f"in {root}/."
            )

        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(root)
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)


class NEUDETDataset(Dataset):
    """NEU-DET steel-surface-defect classification dataset.

    1800 grayscale 200x200 images across 6 classes (300 per class).
    Pixel values normalised to [0, 1].

    Args:
        root: base directory. Default 'data/NEU-DET'.
        split: 'all' (default), 'train', or 'validation'.
        download: if True and root is empty, fetch and unzip the Kaggle archive.
        transform / target_transform: optional callables, applied per item.

    X shape: (1, 200, 200)  dtype: float32  range: [0, 1]
    y shape: ()  dtype: long  values: {0..5}

    Public attributes:
        X (Tensor [N, 1, 200, 200]), y (Tensor [N]),
        class_names (list[str], length 6), image_paths (list[str], length N).
    """

    def __init__(
        self,
        root: str = "data/NEU-DET",
        split: str = "all",
        download: bool = True,
        transform=None,
        target_transform=None,
    ):
        if split not in {"all", "train", "validation"}:
            raise ValueError(f"split must be 'all'|'train'|'validation', got {split!r}")

        if download:
            download_if_missing(root)

        train_root = _find_train_root(root)
        if train_root is None:
            raise FileNotFoundError(
                f"No 'train/' directory found under {root}. "
                "Set download=True or place the unzipped NEU-DET archive there."
            )

        split_dirs = (
            ["train", "validation"] if split == "all" else [split]
        )

        # Discover class names from the train/images dir (sorted, stable).
        class_dir = os.path.join(train_root, "train", "images")
        class_names = sorted(
            d for d in os.listdir(class_dir)
            if os.path.isdir(os.path.join(class_dir, d))
        )
        if len(class_names) != 6:
            raise RuntimeError(
                f"Expected 6 NEU-DET classes, found {len(class_names)}: {class_names}"
            )
        self.class_names: list[str] = class_names
        cls_to_idx = {c: i for i, c in enumerate(class_names)}

        # Walk and collect image paths + labels.
        image_paths: list[str] = []
        labels: list[int] = []
        for sd in split_dirs:
            for cls in class_names:
                cls_path = os.path.join(train_root, sd, "images", cls)
                if not os.path.isdir(cls_path):
                    continue
                for fname in sorted(os.listdir(cls_path)):
                    if fname.lower().endswith(".jpg"):
                        image_paths.append(os.path.join(cls_path, fname))
                        labels.append(cls_to_idx[cls])

        if not image_paths:
            raise FileNotFoundError(f"No .jpg files found under {train_root}")

        # Eager load into one tensor.
        import imageio.v3 as iio
        import numpy as np

        N = len(image_paths)
        X = np.empty((N, 200, 200), dtype=np.float32)
        for i, p in enumerate(image_paths):
            img = iio.imread(p)
            if img.ndim == 3:
                img = img.mean(axis=-1)  # NEU-DET ships grayscale-as-RGB sometimes
            if img.shape != (200, 200):
                raise ValueError(f"Unexpected image shape {img.shape} at {p}")
            X[i] = img.astype(np.float32) / 255.0

        self.X = torch.from_numpy(X).unsqueeze(1)  # (N, 1, 200, 200)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.image_paths = image_paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
