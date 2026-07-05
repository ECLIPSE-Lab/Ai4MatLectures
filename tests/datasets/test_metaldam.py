import os
from pathlib import Path

import pytest
import torch

# Anchor to repo root so the check works regardless of CWD when pytest runs.
DATA_DIR = str(Path(__file__).resolve().parents[2] / "data" / "MetalDAM")


def _data_present() -> bool:
    return os.path.isdir(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0


skip_if_no_data = pytest.mark.skipif(
    not _data_present(), reason="MetalDAM data not present in data/MetalDAM"
)


def test_metaldam_importable():
    from ai4mat.datasets import MetalDAMDataset  # noqa: F401


@skip_if_no_data
def test_metaldam_full_micrographs():
    from ai4mat.datasets import MetalDAMDataset

    ds = MetalDAMDataset(root=DATA_DIR, download=False)
    assert len(ds) == 42
    assert ds.class_names == [
        "matrix",
        "austenite",
        "martensite_austenite",
        "precipitate",
        "defect",
    ]
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32 and y.dtype == torch.long
    assert x.ndim == 3 and x.shape[0] == 1
    assert y.shape == x.shape[1:]
    assert 0.0 <= x.min() and x.max() <= 1.0
    # every mask uses only the 5 documented classes (incl. the RGB-shipped one)
    for m in ds.masks:
        assert m.min() >= 0 and m.max() <= 4


@skip_if_no_data
def test_metaldam_tiles_and_leakage_free_split():
    from ai4mat.datasets import MetalDAMDataset
    from torch.utils.data import DataLoader

    ds = MetalDAMDataset(root=DATA_DIR, download=False, tile_size=256)
    assert ds.X.shape[1:] == torch.Size([1, 256, 256])
    assert ds.y.shape[1:] == torch.Size([256, 256])
    assert len(ds.X) == len(ds.y) == len(ds.tile_image_index) == len(ds)
    assert len(ds) >= 300  # 42 micrographs -> hundreds of 256^2 tiles
    assert int(ds.tile_image_index.max()) == 41

    # image-level split: no micrograph contributes tiles to both sides
    train_imgs = set(range(0, 30))
    train_mask = torch.tensor([int(i) in train_imgs for i in ds.tile_image_index])
    assert 0 < int(train_mask.sum()) < len(ds)

    batch_x, batch_y = next(iter(DataLoader(ds, batch_size=4)))
    assert batch_x.shape == torch.Size([4, 1, 256, 256])
    assert batch_y.shape == torch.Size([4, 256, 256])
