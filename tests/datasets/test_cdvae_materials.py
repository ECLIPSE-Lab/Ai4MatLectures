import os

import pytest
import torch

ROOT = "data/cdvae"


def _csv_present(subset: str, split: str) -> bool:
    return os.path.isfile(os.path.join(ROOT, subset, f"{split}.csv"))


skip_if_no_data = pytest.mark.skipif(
    not _csv_present("perov_5", "val"),
    reason="CDVAE perov_5/val.csv not present at data/cdvae/perov_5/val.csv",
)


def test_cdvae_importable():
    from ai4mat.datasets.cdvae_materials import CDVAEMaterialsDataset  # noqa: F401


def test_cdvae_invalid_subset(tmp_path):
    from ai4mat.datasets.cdvae_materials import CDVAEMaterialsDataset

    with pytest.raises(ValueError, match="subset must be one of"):
        CDVAEMaterialsDataset(subset="bogus", root=str(tmp_path), download=False)


def test_cdvae_invalid_split(tmp_path):
    from ai4mat.datasets.cdvae_materials import CDVAEMaterialsDataset

    with pytest.raises(ValueError, match="split must be one of"):
        CDVAEMaterialsDataset(split="bogus", root=str(tmp_path), download=False)


def test_cdvae_download_false_errors_when_missing(tmp_path):
    from ai4mat.datasets.cdvae_materials import CDVAEMaterialsDataset

    with pytest.raises(FileNotFoundError):
        CDVAEMaterialsDataset(root=str(tmp_path / "empty"), download=False)


def _write_synthetic_perov5(path):
    """Tiny CSV mirroring the real perov_5 schema discovered upstream:
    columns: ['', material_id, cif, formula, heat_all, heat_ref,
              dir_gap, ind_gap].
    """
    import pandas as pd

    cif_tmpl = (
        "# generated using pymatgen\n"
        "data_{f}\n"
        "_symmetry_space_group_name_H-M   'P 1'\n"
        "_chemical_formula_structural   {f}\n"
        "_chemical_formula_sum   '{fsum}'\n"
        "_cell_length_a   3.9\n"
    )
    rows = []
    for i, (f, fsum, ha, hr) in enumerate([
        ("CuCdN3", "Cu1 Cd1 N3", 2.76, 2.709),
        ("BaTiO3", "Ba1 Ti1 O3", 1.10, 0.950),
        ("SrZrO3", "Sr1 Zr1 O3", 0.42, 0.333),
        ("KNbO3", "K1 Nb1 O3", 1.88, 1.501),
        ("LaAlO3", "La1 Al1 O3", 0.05, 0.044),
        ("NaTaO3", "Na1 Ta1 O3", 1.23, 1.100),
        ("CaSnO3", "Ca1 Sn1 O3", 2.01, 1.870),
        ("MgSiO3", "Mg1 Si1 O3", 3.40, 3.200),
        ("ZnGeO3", "Zn1 Ge1 O3", 2.55, 2.400),
        ("FeTiO3", "Fe1 Ti1 O3", 0.30, 0.270),
        ("CoMnO3", "Co1 Mn1 O3", 0.61, 0.550),
        ("NiCrO3", "Ni1 Cr1 O3", 0.77, 0.700),
    ]):
        rows.append({
            "": i,
            "material_id": 10000 + i,
            "cif": cif_tmpl.format(f=f, fsum=fsum),
            "formula": f,
            "heat_all": ha,
            "heat_ref": hr,
            "dir_gap": 0.0,
            "ind_gap": 0.0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def synthetic_root(tmp_path):
    d = tmp_path / "cdvae" / "perov_5"
    d.mkdir(parents=True)
    _write_synthetic_perov5(d / "val.csv")
    return str(tmp_path / "cdvae")


def test_cdvae_synthetic_contract(synthetic_root):
    from ai4mat.datasets.cdvae_materials import CDVAEMaterialsDataset
    from tests.conftest import assert_dataset_contract

    ds = CDVAEMaterialsDataset(
        subset="perov_5", split="val", root=synthetic_root,
        target="heat_ref", download=False,
    )
    assert_dataset_contract(
        ds, expected_x_shape=[118], expected_y_dtype=torch.float32, min_len=12
    )


def test_cdvae_synthetic_attributes(synthetic_root):
    from ai4mat.datasets.cdvae_materials import CDVAEMaterialsDataset

    ds = CDVAEMaterialsDataset(
        subset="perov_5", split="val", root=synthetic_root, download=False,
    )
    assert len(ds.cif) == len(ds) == len(ds.ids)
    assert len(ds.feature_names) == 118
    # Global default not present in perov_5 -> falls back to heat_ref.
    assert ds.target == "heat_ref"
    # Element fractions: each row sums to ~1.
    sums = ds.X.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # CuCdN3 -> Cu, Cd, N fractions 0.2/0.2/0.6.
    from ai4mat.datasets.cdvae_materials import _ELEM_IDX

    row0 = ds.X[0]
    assert abs(row0[_ELEM_IDX["Cu"]].item() - 0.2) < 1e-5
    assert abs(row0[_ELEM_IDX["N"]].item() - 0.6) < 1e-5


def test_cdvae_invalid_target_lists_available(synthetic_root):
    from ai4mat.datasets.cdvae_materials import CDVAEMaterialsDataset

    with pytest.raises(ValueError, match="Available numeric targets"):
        CDVAEMaterialsDataset(
            subset="perov_5", split="val", root=synthetic_root,
            target="not_a_column", download=False,
        )


def test_cdvae_cif_from_synthetic_when_no_formula_col(tmp_path):
    """carbon_24 has no formula column -> formula parsed from CIF text."""
    import pandas as pd

    from ai4mat.datasets.cdvae_materials import CDVAEMaterialsDataset

    d = tmp_path / "cdvae" / "carbon_24"
    d.mkdir(parents=True)
    cif = (
        "# generated using pymatgen\n"
        "data_C\n"
        "_chemical_formula_sum   'C4'\n"
        "_cell_length_a   2.5\n"
    )
    rows = [
        {"": i, "material_id": f"C-{i}", "cif": cif,
         "energy_per_atom": -154.2 - i * 0.01}
        for i in range(12)
    ]
    pd.DataFrame(rows).to_csv(d / "val.csv", index=False)

    ds = CDVAEMaterialsDataset(
        subset="carbon_24", split="val",
        root=str(tmp_path / "cdvae"), download=False,
    )
    assert ds.target == "energy_per_atom"
    from ai4mat.datasets.cdvae_materials import _ELEM_IDX

    assert abs(ds.X[0][_ELEM_IDX["C"]].item() - 1.0) < 1e-5


@pytest.mark.slow
def test_cdvae_real_download(tmp_path):
    """Real network download of the small perov_5/val split."""
    from ai4mat.datasets.cdvae_materials import CDVAEMaterialsDataset
    from tests.conftest import assert_dataset_contract

    ds = CDVAEMaterialsDataset(
        subset="perov_5", split="val", root=str(tmp_path / "cdvae"),
        target="heat_ref", download=True,
    )
    assert_dataset_contract(
        ds, expected_x_shape=[118], expected_y_dtype=torch.float32,
        min_len=1000,
    )
    assert os.path.isfile(str(tmp_path / "cdvae" / "README.md"))
