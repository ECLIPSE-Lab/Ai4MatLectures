import pytest
import torch
from tests.conftest import assert_dataset_contract


@pytest.mark.slow
def test_cahn_hilliard_contract():
    """Use simulation_number=0 for fast loading (single simulation ~1000 images)."""
    from ai4mat.datasets.cahn_hilliard import CahnHilliardDataset
    ds = CahnHilliardDataset(simulation_number=0)
    assert_dataset_contract(ds, expected_x_shape=[1, 64, 64], expected_y_dtype=torch.float32, min_len=50)


@pytest.mark.slow
def test_cahn_hilliard_normalised():
    from ai4mat.datasets.cahn_hilliard import CahnHilliardDataset
    ds = CahnHilliardDataset(simulation_number=0)
    x, _ = ds[0]
    assert x.min() >= 0.0 and x.max() <= 1.0


@pytest.mark.slow
def test_cahn_hilliard_energy_positive():
    from ai4mat.datasets.cahn_hilliard import CahnHilliardDataset
    ds = CahnHilliardDataset(simulation_number=0)
    for i in range(min(10, len(ds))):
        _, y = ds[i]
        assert y.item() > 0, "Cahn-Hilliard energy should be positive"
