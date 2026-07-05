"""MetalDAM SEM semantic-segmentation dataset (ArcelorMittal / DaSCI)."""
from __future__ import annotations

import os
import tempfile
import zipfile
from typing import Optional
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

_RELEASE_URL = (
    "https://github.com/ari-dasci/OD-MetalDAM/releases/download/1.0/"
    "MetalDAM_labeled.zip"
)

CLASS_NAMES = [
    "matrix",
    "austenite",
    "martensite_austenite",
    "precipitate",
    "defect",
]


def _find_images_root(root: str, max_depth: int = 2) -> Optional[str]:
    """Return the dir that directly contains `images/` and `labels/`.

    The release zip unpacks as `MetalDAM/{images,labels,coloured_labels}`,
    but tolerate other nestings up to `max_depth` levels below `root`.
    """
    if not os.path.isdir(root):
        return None

    def has_both(d: str) -> bool:
        return os.path.isdir(os.path.join(d, "images")) and os.path.isdir(
            os.path.join(d, "labels")
        )

    if has_both(root):
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
            if has_both(candidate):
                return candidate
            stack.append((candidate, depth + 1))
    return None


def download_if_missing(root: str) -> None:
    """Download + unzip the MetalDAM labeled archive into `root` if absent.

    Idempotent: returns immediately if `root` already contains an
    `images/` + `labels/` pair anywhere up to two levels deep.
    """
    os.makedirs(root, exist_ok=True)
    if _find_images_root(root) is not None:
        return

    tmp_parent = os.path.dirname(root) or "."
    with tempfile.NamedTemporaryFile(
        dir=tmp_parent, prefix="_metaldam_", suffix=".zip", delete=False
    ) as tmp_handle:
        tmp_zip = tmp_handle.name
    try:
        with urlopen(_RELEASE_URL) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            with open(tmp_zip, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc="MetalDAM"
            ) as pbar:
                while True:
                    chunk = response.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    fh.write(chunk)
                    pbar.update(len(chunk))

        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(root)
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)


class MetalDAMDataset(Dataset):
    """MetalDAM per-pixel segmentation of SEM micrographs of AM steel.

    42 grayscale scanning-electron micrographs of additively manufactured
    steel, expert-annotated with 5 classes (heavily imbalanced):
    0 matrix (~32%), 1 austenite (~58%), 2 martensite/austenite (~9%),
    3 precipitate (~0.2%, mostly *unannotated* per the dataset README —
    treat as label noise), 4 defect (~0.7%). Micrographs come in two
    native resolutions (1024x703 and 1280x895 after the info band at the
    bottom is cropped away; this loader crops images to their label size).

    Source: ArcelorMittal / DaSCI, https://github.com/ari-dasci/OD-MetalDAM
    (also see Luengo et al. 2022, Information Fusion 78, 232-253).

    Two modes:

    * ``tile_size=None`` (default): one item per micrograph, variable
      spatial size. x: (1, H, W) float32 in [0, 1]; y: (H, W) long.
      Use ``batch_size=1`` (or a custom collate_fn) in a DataLoader.
    * ``tile_size=int`` (e.g. 256): micrographs are eagerly cut into
      square tiles with ``stride`` (default: ``tile_size``); edge tiles
      are shifted inward so every pixel is covered. All tiles share one
      shape, so standard batching works. ``tile_image_index[i]`` gives
      the source-micrograph index of tile ``i`` — split by *micrograph*,
      never by tile, to avoid leakage across overlapping tiles.

    Args:
        root: base directory. Default 'data/MetalDAM'.
        download: if True and root is empty, fetch the GitHub release zip.
        tile_size: None for full micrographs, or square tile edge length.
        stride: tile stride; defaults to tile_size (non-overlapping).
        transform / target_transform: optional callables, applied per item.

    Public attributes:
        images (list[Tensor (1,H,W)]), masks (list[Tensor (H,W)]),
        names (list[str]), class_names (list[str], length 5);
        in tile mode additionally X (Tensor [N,1,t,t]), y (Tensor [N,t,t]),
        tile_image_index (Tensor [N], long).
    """

    def __init__(
        self,
        root: str = "data/MetalDAM",
        download: bool = True,
        tile_size: Optional[int] = None,
        stride: Optional[int] = None,
        transform=None,
        target_transform=None,
    ):
        if download:
            download_if_missing(root)

        data_root = _find_images_root(root)
        if data_root is None:
            raise FileNotFoundError(
                f"No 'images/' + 'labels/' pair found under {root}. "
                "Set download=True or unzip MetalDAM_labeled.zip there."
            )

        import imageio.v3 as iio
        import numpy as np

        img_dir = os.path.join(data_root, "images")
        lab_dir = os.path.join(data_root, "labels")

        self.class_names: list[str] = list(CLASS_NAMES)
        self.names: list[str] = []
        self.images: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []

        for fname in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lab_path = os.path.join(lab_dir, stem + ".png")
            if not os.path.isfile(lab_path):
                continue
            lab = iio.imread(lab_path)
            if lab.ndim == 3:  # one file ships as RGB with identical channels
                lab = lab[..., 0]
            img = iio.imread(os.path.join(img_dir, fname))
            if img.ndim == 3:
                img = img.mean(axis=-1)
            # crop the instrument info band at the bottom (label height rules)
            img = img[: lab.shape[0], : lab.shape[1]]
            if img.shape != lab.shape:
                raise ValueError(
                    f"image/label shape mismatch for {stem}: "
                    f"{img.shape} vs {lab.shape}"
                )
            self.names.append(stem)
            self.images.append(
                torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            )
            self.masks.append(torch.from_numpy(lab.astype(np.int64)))

        if not self.images:
            raise FileNotFoundError(f"No image/label pairs found under {data_root}")

        self.tile_size = tile_size
        self.transform = transform
        self.target_transform = target_transform

        if tile_size is not None:
            stride = stride or tile_size
            tiles_x, tiles_y, owner = [], [], []
            for i, (im, lb) in enumerate(zip(self.images, self.masks)):
                H, W = lb.shape
                if H < tile_size or W < tile_size:
                    raise ValueError(
                        f"tile_size {tile_size} exceeds micrograph "
                        f"{self.names[i]} of size {H}x{W}"
                    )
                ys = list(range(0, H - tile_size + 1, stride))
                xs = list(range(0, W - tile_size + 1, stride))
                if ys[-1] != H - tile_size:
                    ys.append(H - tile_size)
                if xs[-1] != W - tile_size:
                    xs.append(W - tile_size)
                for yy in ys:
                    for xx in xs:
                        tiles_x.append(im[:, yy : yy + tile_size, xx : xx + tile_size])
                        tiles_y.append(lb[yy : yy + tile_size, xx : xx + tile_size])
                        owner.append(i)
            self.X = torch.stack(tiles_x)  # (N, 1, t, t)
            self.y = torch.stack(tiles_y)  # (N, t, t)
            self.tile_image_index = torch.tensor(owner, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X) if self.tile_size is not None else len(self.images)

    def __getitem__(self, idx):
        if self.tile_size is not None:
            x, y = self.X[idx], self.y[idx]
        else:
            x, y = self.images[idx], self.masks[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
