import gzip
import json
import os

import pytest
import torch

DATA_DIR = "data/matbench"


def _task_present(task: str) -> bool:
    return os.path.isfile(os.path.join(DATA_DIR, f"{task}.json.gz"))


skip_if_no_steels = pytest.mark.skipif(
    not _task_present("matbench_steels"),
    reason="matbench_steels not present at data/matbench/matbench_steels.json.gz",
)


def _write_synthetic(root, task, columns, rows):
    """Write a tiny matminer-format (pandas orient='split') gz JSON.

    Bypasses SHA-256 by registering the synthetic file's real digest in
    the in-memory _TASKS registry for the duration of a test.
    """
    import hashlib

    from ai4mat.datasets import matbench as mb

    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{task}.json.gz")
    payload = {
        "index": list(range(len(rows))),
        "columns": columns,
        "data": rows,
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    mb._TASKS[task]["sha256"] = digest
    return path


def test_matbench_importable():
    from ai4mat.datasets import MatBenchDataset  # noqa: F401


def test_supported_tasks_listed():
    from ai4mat.datasets.matbench import supported_tasks

    tasks = supported_tasks()
    for t in (
        "matbench_steels",
        "matbench_jdft2d",
        "matbench_phonons",
        "matbench_perovskites",
        "matbench_dielectric",
        "matbench_log_gvrh",
        "matbench_log_kvrh",
    ):
        assert t in tasks


def test_unknown_task_errors():
    from ai4mat.datasets import MatBenchDataset

    with pytest.raises(ValueError, match="Unknown Matbench task"):
        MatBenchDataset(task="matbench_not_a_task", download=False)


def test_download_false_errors_on_empty(tmp_path):
    from ai4mat.datasets import MatBenchDataset

    with pytest.raises(FileNotFoundError):
        MatBenchDataset(
            task="matbench_steels", root=str(tmp_path / "empty"), download=False
        )


def test_synthetic_composition_task(tmp_path):
    """Exercise the composition parsing path offline."""
    from ai4mat.datasets import MatBenchDataset
    from tests.conftest import assert_dataset_contract

    rows = [
        ["Fe0.8Ni0.2", 500.0],
        ["Al2O3", 300.0],
        ["Fe", 250.0],
        ["TiNi", 420.0],
        ["Fe0.5Cr0.5", 380.0],
        ["SiC", 290.0],
        ["NbTi", 410.0],
        ["W", 600.0],
        ["MgO", 270.0],
        ["CoCr", 350.0],
        ["FeC0.1", 480.0],
        ["NiCrMo", 360.0],
    ]
    root = tmp_path / "mb"
    _write_synthetic(str(root), "matbench_steels", ["composition", "yield strength"], rows)

    ds = MatBenchDataset(task="matbench_steels", root=str(root), download=False)
    assert_dataset_contract(
        ds, expected_x_shape=[118], expected_y_dtype=torch.float32, min_len=12
    )
    assert ds.feature_kind == "composition"
    assert ds.target_name == "yield strength"
    assert ds.formulas is not None and len(ds.formulas) == len(ds)
    assert ds.structures is None
    # Fe0.8Ni0.2 fractions
    x0, _ = ds[0]
    fe = ds.element_names.index("Fe")
    ni = ds.element_names.index("Ni")
    assert pytest.approx(float(x0[fe]), abs=1e-5) == 0.8
    assert pytest.approx(float(x0[ni]), abs=1e-5) == 0.2
    assert pytest.approx(float(x0.sum()), abs=1e-5) == 1.0


def test_synthetic_structure_task(tmp_path):
    """Exercise the structure-dict parsing path offline (no pymatgen)."""
    from ai4mat.datasets import MatBenchDataset
    from tests.conftest import assert_dataset_contract

    def struct(species_per_site):
        return {
            "@module": "pymatgen.core.structure",
            "@class": "Structure",
            "charge": 0,
            "lattice": {"matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]]},
            "sites": [
                {"species": [{"element": el, "occu": 1}], "abc": [0, 0, 0]}
                for el in species_per_site
            ],
        }

    rows = []
    for i in range(12):
        rows.append([struct(["Sr", "Ti", "O", "O", "O"]), float(i) * 0.1 - 0.5])
    root = tmp_path / "mbs"
    _write_synthetic(str(root), "matbench_perovskites", ["structure", "e_form"], rows)

    ds = MatBenchDataset(task="matbench_perovskites", root=str(root), download=False)
    assert_dataset_contract(
        ds, expected_x_shape=[118], expected_y_dtype=torch.float32, min_len=12
    )
    assert ds.feature_kind == "structure"
    assert ds.structures is not None and len(ds.structures) == len(ds)
    assert ds.formulas is None
    x0, _ = ds[0]
    o = ds.element_names.index("O")
    sr = ds.element_names.index("Sr")
    assert pytest.approx(float(x0[o]), abs=1e-5) == 3.0 / 5.0
    assert pytest.approx(float(x0[sr]), abs=1e-5) == 1.0 / 5.0
    assert pytest.approx(float(x0.sum()), abs=1e-5) == 1.0


def test_folds_structure(tmp_path):
    from ai4mat.datasets import MatBenchDataset

    rows = [["Fe", float(i)] for i in range(25)]
    root = tmp_path / "mbf"
    _write_synthetic(str(root), "matbench_steels", ["composition", "yield strength"], rows)
    ds = MatBenchDataset(task="matbench_steels", root=str(root), download=False)

    assert ds.official_folds is False
    assert len(ds.folds) == 5
    all_test = []
    for train, test in ds.folds:
        assert set(train).isdisjoint(set(test))
        assert len(train) + len(test) == len(ds)
        all_test.extend(test.tolist())
    # Every sample appears in exactly one test fold.
    assert sorted(all_test) == list(range(len(ds)))


@pytest.mark.slow
@skip_if_no_steels
def test_matbench_steels_real_contract():
    from ai4mat.datasets import MatBenchDataset
    from tests.conftest import assert_dataset_contract

    ds = MatBenchDataset(task="matbench_steels", download=False)
    assert_dataset_contract(
        ds, expected_x_shape=[118], expected_y_dtype=torch.float32, min_len=300
    )
    assert len(ds) == 312


@pytest.mark.slow
def test_matbench_steels_download(tmp_path):
    """Real network download of the smallest task into a temp root."""
    from ai4mat.datasets import MatBenchDataset
    from tests.conftest import assert_dataset_contract

    ds = MatBenchDataset(
        task="matbench_steels", root=str(tmp_path / "mb"), download=True
    )
    assert_dataset_contract(
        ds, expected_x_shape=[118], expected_y_dtype=torch.float32, min_len=300
    )
