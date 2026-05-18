import os

import numpy as np
import pytest
import torch

DATA_DIR = "data/qm9"
CSV = os.path.join(DATA_DIR, "qm9.csv")


def _data_present() -> bool:
    return os.path.isfile(CSV)


skip_if_no_data = pytest.mark.skipif(
    not _data_present(), reason="QM9 CSV not present at data/qm9/qm9.csv"
)


def test_qm9_importable():
    from ai4mat.datasets.qm9 import QM9Dataset  # noqa: F401


def test_qm9_smiles_featurizer_known_molecules():
    """Fast, offline check of the hand-rolled SMILES feature parser.

    Feature layout: [n_C, n_H, n_N, n_O, n_F, n_heavy, n_atoms_total].
    Implicit-H uses the crude flat valence model documented in qm9.py:
    each C donates 3, N 2, O 1, F 0 hydrogens (valence-1, no bond accounting).
    """
    from ai4mat.datasets.qm9 import _featurize_smiles, _tokenize_smiles_atoms

    # Methane "C": 1 heavy C, implicit H = 1*(4-1) = 3.
    f = _featurize_smiles("C")
    assert f.tolist() == [1, 3, 0, 0, 0, 1, 4]

    # Methanol "CO": C + O heavy. implicit H = C:3 + O:1 = 4.
    f = _featurize_smiles("CO")
    assert f.tolist() == [1, 4, 0, 1, 0, 2, 6]

    # Hydrogen cyanide "C#N": tokenizer ignores the triple bond '#'.
    assert _tokenize_smiles_atoms("C#N") == ["C", "N"]
    f = _featurize_smiles("C#N")
    # n_C=1, n_N=1, n_heavy=2, implicit H = C:3 + N:2 = 5.
    assert f.tolist() == [1, 5, 1, 0, 0, 2, 7]

    # Bracket atom + aromatic lowercase: pyridine-like "[nH]1cccc1".
    toks = _tokenize_smiles_atoms("c1cc[nH]c1")
    assert toks == ["C", "C", "C", "N", "C"]

    # Fluoromethane "CF": F contributes count + heavy, 0 implicit H
    # (max(1-1,0)=0). H = C:3 only; total atoms = heavy(2) + H(3) = 5.
    f = _featurize_smiles("CF")
    assert f.tolist() == [1, 3, 0, 0, 1, 2, 5]


def test_qm9_download_false_errors_on_empty(tmp_path):
    from ai4mat.datasets.qm9 import QM9Dataset

    with pytest.raises(FileNotFoundError):
        QM9Dataset(root=str(tmp_path / "empty"), download=False)


def test_qm9_invalid_target(tmp_path):
    from ai4mat.datasets.qm9 import QM9Dataset

    with pytest.raises(ValueError, match="target must be one of"):
        QM9Dataset(root=str(tmp_path), target="not_a_property", download=False)


def _write_synthetic_qm9(path):
    """Write a tiny CSV with the exact verified QM9 schema."""
    cols = (
        "mol_id,smiles,A,B,C,mu,alpha,homo,lumo,gap,r2,zpve,"
        "u0,u298,h298,g298,cv,u0_atom,u298_atom,h298_atom,g298_atom"
    )
    rows = [
        "gdb_1,C,157.7,157.7,157.7,0,13.2,-0.38,0.11,0.50,35.3,0.04,"
        "-40.4,-40.4,-40.4,-40.4,6.4,-395.9,-398.6,-401.0,-372.4",
        "gdb_2,N,293.6,293.5,191.3,1.6,9.4,-0.25,0.08,0.33,26.1,0.03,"
        "-56.5,-56.5,-56.5,-56.5,6.3,-276.8,-278.6,-280.3,-259.3",
        "gdb_3,O,799.5,437.9,282.9,1.8,6.3,-0.29,0.06,0.36,19.0,0.02,"
        "-76.4,-76.4,-76.4,-76.4,6.0,-213.0,-213.9,-215.1,-201.4",
        "gdb_4,CO,1.0,1.0,1.0,1.7,9.0,-0.27,0.05,0.40,30.0,0.05,"
        "-115.0,-115.0,-115.0,-115.0,7.0,-300.0,-301.0,-302.0,-290.0",
    ] * 4  # 16 rows so contract min_len passes
    path.write_text(cols + "\n" + "\n".join(rows) + "\n")


def test_qm9_offline_synthetic_contract(tmp_path):
    """Exercise CSV-parse + feature + contract path with a synthetic CSV."""
    from ai4mat.datasets.qm9 import QM9Dataset
    from tests.conftest import assert_dataset_contract

    root = tmp_path / "qm9"
    root.mkdir()
    _write_synthetic_qm9(root / "qm9.csv")

    ds = QM9Dataset(root=str(root), target="gap", download=False)
    assert_dataset_contract(
        ds, expected_x_shape=[7], expected_y_dtype=torch.float32, min_len=10
    )
    assert ds.target == "gap"
    assert len(ds.smiles) == len(ds)
    assert ds.df.shape[0] == len(ds)
    assert "gap" in ds.property_names
    assert ds.X.shape == (len(ds), 7)

    # Different target selects a different column.
    ds2 = QM9Dataset(root=str(root), target="mu", download=False)
    assert not torch.allclose(ds.y, ds2.y)


@skip_if_no_data
@pytest.mark.slow
def test_qm9_real_download_contract():
    from ai4mat.datasets.qm9 import QM9Dataset
    from tests.conftest import assert_dataset_contract

    ds = QM9Dataset(target="gap", download=False)
    assert_dataset_contract(
        ds, expected_x_shape=[7], expected_y_dtype=torch.float32,
        min_len=100000,
    )
    assert len(ds) > 130000
    assert np.isfinite(ds.X.numpy()).all()
