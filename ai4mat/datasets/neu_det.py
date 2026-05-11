"""NEU-DET steel-surface-defect dataset."""
from __future__ import annotations

import os
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

    tmp_zip = os.path.join(os.path.dirname(root) or ".", "_neu_det_tmp.zip")
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
