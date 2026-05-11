import os
import pytest
import torch

DATA_DIR = "data/estm"
CSV = os.path.join(DATA_DIR, "ESTM.csv")


def _data_present() -> bool:
    return os.path.isfile(CSV)


skip_if_no_data = pytest.mark.skipif(
    not _data_present(), reason="ESTM CSV not present at data/estm/ESTM.csv"
)


def test_estm_importable():
    from ai4mat.datasets import ESTMDataset  # noqa: F401


@skip_if_no_data
def test_estm_fraction_contract():
    from ai4mat.datasets import ESTMDataset
    from tests.conftest import assert_dataset_contract

    ds = ESTMDataset(features="fraction", download=False)
    assert_dataset_contract(
        ds,
        expected_x_shape=[119],          # 118 elements + T
        expected_y_dtype=torch.float32,
        min_len=4500,
    )


@skip_if_no_data
def test_estm_fraction_attributes():
    from ai4mat.datasets import ESTMDataset

    ds = ESTMDataset(features="fraction", download=False)
    assert len(ds.formulas) == len(ds)
    assert ds.T.shape == (len(ds),)
    assert ds.properties.shape == (len(ds), 5)
    assert list(ds.properties.columns) == ["ZT", "S", "sigma", "kappa", "PF"]
    assert len(ds.feature_names) == 119


@skip_if_no_data
@pytest.mark.slow
def test_estm_magpie_features():
    from ai4mat.datasets import ESTMDataset

    ds = ESTMDataset(features="magpie", download=False)
    x, _ = ds[0]
    assert x.shape[0] > 100, f"expected >100 Magpie features, got {x.shape[0]}"
    assert torch.isfinite(x).all(), "Magpie features must be finite after imputation"


@skip_if_no_data
def test_estm_target_all():
    from ai4mat.datasets import ESTMDataset

    ds = ESTMDataset(features="fraction", target="all", download=False)
    _, y = ds[0]
    assert y.shape == (5,)


def test_estm_download_false_errors_on_empty(tmp_path):
    # Runs in CI too — no live data needed.
    from ai4mat.datasets import ESTMDataset

    with pytest.raises(FileNotFoundError):
        ESTMDataset(root=str(tmp_path / "empty"), download=False)


@skip_if_no_data
def test_estm_cache_roundtrip(tmp_path):
    """Second instantiation hits the .npz cache and matches the first."""
    from ai4mat.datasets import ESTMDataset
    import shutil

    src = "data/estm/ESTM.csv"
    root = tmp_path / "estm"
    root.mkdir()
    shutil.copy(src, root / "ESTM.csv")

    ds1 = ESTMDataset(root=str(root), features="fraction", download=False)
    ds2 = ESTMDataset(root=str(root), features="fraction", download=False)
    assert torch.allclose(ds1.X, ds2.X)
    assert ds1.formulas == ds2.formulas
