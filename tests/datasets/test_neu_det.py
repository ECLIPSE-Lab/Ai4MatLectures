import os
from pathlib import Path

import pytest
import torch

# Anchor to repo root so the check works regardless of CWD when pytest runs.
DATA_DIR = str(Path(__file__).resolve().parents[2] / "data" / "NEU-DET")


def _data_present() -> bool:
    return os.path.isdir(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0


skip_if_no_data = pytest.mark.skipif(
    not _data_present(), reason="NEU-DET data not present in data/NEU-DET"
)


def test_neu_det_importable():
    from ai4mat.datasets import NEUDETDataset  # noqa: F401


@skip_if_no_data
def test_neu_det_contract():
    from ai4mat.datasets import NEUDETDataset
    from tests.conftest import assert_dataset_contract

    ds = NEUDETDataset(download=False)
    assert_dataset_contract(
        ds, expected_x_shape=[1, 200, 200], expected_y_dtype=torch.long, min_len=1000
    )


@skip_if_no_data
def test_neu_det_length_and_classes():
    from ai4mat.datasets import NEUDETDataset

    ds = NEUDETDataset(download=False)
    assert len(ds) == 1800, f"expected 1800, got {len(ds)}"
    assert set(ds.y.tolist()) == {0, 1, 2, 3, 4, 5}
    assert len(ds.class_names) == 6
    joined = " ".join(ds.class_names).lower()
    for token in ["crazing", "inclusion", "patches", "pitted", "rolled", "scratches"]:
        assert token in joined, f"missing class token: {token!r}"


@skip_if_no_data
def test_neu_det_normalised():
    from ai4mat.datasets import NEUDETDataset

    ds = NEUDETDataset(download=False)
    for i in [0, len(ds) // 2, len(ds) - 1]:
        x, _ = ds[i]
        assert x.dtype == torch.float32
        assert 0.0 <= x.min().item() and x.max().item() <= 1.0


@skip_if_no_data
def test_neu_det_split_train_only():
    from ai4mat.datasets import NEUDETDataset

    ds_train = NEUDETDataset(download=False, split="train")
    ds_val = NEUDETDataset(download=False, split="validation")
    assert len(ds_train) == 1440  # 240 per class * 6
    assert len(ds_val) == 360     # 60 per class * 6


def test_neu_det_download_false_errors_on_empty(tmp_path):
    # Doesn't depend on the dataset being downloaded — runs in CI too.
    from ai4mat.datasets import NEUDETDataset

    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        NEUDETDataset(root=empty_root, download=False)


@skip_if_no_data
def test_neu_det_image_paths_populated():
    from ai4mat.datasets import NEUDETDataset

    ds = NEUDETDataset(download=False)
    assert len(ds.image_paths) == len(ds)
    assert all(p.endswith(".jpg") for p in ds.image_paths[:5])
