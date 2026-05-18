import os
import pytest
import torch

DATA_DIR = "data/rmd17"
# benzene = 12 atoms; smallest-but-one .npz (~89 MB).
_MOL = "benzene"
NPZ = os.path.join(DATA_DIR, f"rmd17_{_MOL}.npz")


def _data_present() -> bool:
    return os.path.isfile(NPZ)


skip_if_no_data = pytest.mark.skipif(
    not _data_present(),
    reason=f"rMD17 npz not present at {NPZ}",
)


def test_rmd17_importable():
    from ai4mat.datasets.rmd17 import RMD17Dataset  # noqa: F401


def test_rmd17_unknown_molecule_errors():
    from ai4mat.datasets.rmd17 import RMD17Dataset

    with pytest.raises(ValueError):
        RMD17Dataset(molecule="caffeine", download=False)


def test_rmd17_download_false_errors_when_missing(tmp_path):
    from ai4mat.datasets.rmd17 import RMD17Dataset

    with pytest.raises(FileNotFoundError):
        RMD17Dataset(
            molecule="benzene", root=str(tmp_path / "empty"),
            download=False,
        )


def test_rmd17_oversampling_warns(tmp_path, monkeypatch):
    """n_samples > 1000 must warn about the rMD17 correlation cap.

    Uses a tiny synthetic npz mimicking the rMD17 key schema so the check
    runs offline / in CI.
    """
    import numpy as np
    from ai4mat.datasets.rmd17 import RMD17Dataset

    root = tmp_path / "rmd17"
    root.mkdir()
    n, na = 3000, 6
    rng = np.random.RandomState(1)
    np.savez(
        root / "rmd17_benzene.npz",
        nuclear_charges=np.array([6, 6, 6, 6, 6, 6], dtype=np.uint8),
        coords=rng.randn(n, na, 3).astype(np.float64),
        energies=rng.randn(n).astype(np.float64),
        forces=rng.randn(n, na, 3).astype(np.float64),
        old_indices=np.arange(n, dtype=np.int64),
        old_energies=rng.randn(n).astype(np.float64),
        old_forces=rng.randn(n, na, 3).astype(np.float64),
    )

    with pytest.warns(UserWarning, match="recommended limit"):
        RMD17Dataset(
            molecule="benzene", root=str(root),
            n_samples=2000, download=False,
        )


def test_rmd17_synthetic_contract_and_attrs(tmp_path):
    """Full (x, y) contract + attribute shapes via a synthetic npz."""
    import numpy as np
    from ai4mat.datasets.rmd17 import RMD17Dataset
    from tests.conftest import assert_dataset_contract

    root = tmp_path / "rmd17"
    root.mkdir()
    n, na = 2500, 6
    rng = np.random.RandomState(2)
    np.savez(
        root / "rmd17_benzene.npz",
        nuclear_charges=np.array([6, 6, 6, 6, 6, 6], dtype=np.uint8),
        coords=rng.randn(n, na, 3).astype(np.float64),
        energies=rng.randn(n).astype(np.float64),
        forces=rng.randn(n, na, 3).astype(np.float64),
        old_indices=np.arange(n, dtype=np.int64),
        old_energies=rng.randn(n).astype(np.float64),
        old_forces=rng.randn(n, na, 3).astype(np.float64),
    )

    ds = RMD17Dataset(
        molecule="benzene", root=str(root), n_samples=1000,
        split="train", seed=0, download=False,
    )
    assert_dataset_contract(
        ds,
        expected_x_shape=[4 * 6],  # Z (6) + flattened coords (18)
        expected_y_dtype=torch.float32,
        min_len=1000,
    )
    assert len(ds) == 1000
    assert ds.Z.shape == (6,)
    assert ds.Z.dtype == torch.long
    assert ds.coords.shape == (1000, 6, 3)
    assert ds.forces.shape == (1000, 6, 3)
    assert ds.energies.shape == (1000,)
    assert ds.n_atoms == 6


def test_rmd17_train_test_disjoint(tmp_path):
    import numpy as np
    from ai4mat.datasets.rmd17 import RMD17Dataset

    root = tmp_path / "rmd17"
    root.mkdir()
    n, na = 2500, 6
    rng = np.random.RandomState(3)
    np.savez(
        root / "rmd17_benzene.npz",
        nuclear_charges=np.array([6, 6, 6, 6, 6, 6], dtype=np.uint8),
        coords=rng.randn(n, na, 3).astype(np.float64),
        energies=rng.randn(n).astype(np.float64),
        forces=rng.randn(n, na, 3).astype(np.float64),
        old_indices=np.arange(n, dtype=np.int64),
        old_energies=rng.randn(n).astype(np.float64),
        old_forces=rng.randn(n, na, 3).astype(np.float64),
    )
    tr = RMD17Dataset("benzene", root=str(root), n_samples=500,
                      split="train", seed=0, download=False)
    te = RMD17Dataset("benzene", root=str(root), n_samples=500,
                      split="test", seed=0, download=False)
    assert set(tr.indices.tolist()).isdisjoint(set(te.indices.tolist()))


@skip_if_no_data
def test_rmd17_real_data_contract():
    from ai4mat.datasets.rmd17 import RMD17Dataset
    from tests.conftest import assert_dataset_contract

    ds = RMD17Dataset(molecule=_MOL, root=DATA_DIR, n_samples=1000,
                      download=False)
    assert_dataset_contract(
        ds,
        expected_x_shape=[4 * ds.n_atoms],
        expected_y_dtype=torch.float32,
        min_len=1000,
    )


@pytest.mark.slow
def test_rmd17_real_download(tmp_path):
    """Live figshare download of the smallest molecule + contract."""
    from ai4mat.datasets.rmd17 import RMD17Dataset
    from tests.conftest import assert_dataset_contract

    ds = RMD17Dataset(
        molecule="ethanol", root=str(tmp_path / "rmd17"),
        n_samples=1000, download=True,
    )
    assert_dataset_contract(
        ds,
        expected_x_shape=[4 * ds.n_atoms],
        expected_y_dtype=torch.float32,
        min_len=1000,
    )
    assert (tmp_path / "rmd17" / "README.md").is_file()
