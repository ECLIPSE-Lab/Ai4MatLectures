import pytest
import torch
from tests.conftest import assert_dataset_contract


def test_ising_light_contract():
    from ai4mat.datasets.ising import IsingDataset
    ds = IsingDataset(size='light')
    assert_dataset_contract(ds, expected_x_shape=[1, 16, 16], expected_y_dtype=torch.long, min_len=100)


def test_ising_full_contract():
    from ai4mat.datasets.ising import IsingDataset
    ds = IsingDataset(size='full')
    assert_dataset_contract(ds, expected_x_shape=[1, 64, 64], expected_y_dtype=torch.long, min_len=1000)


def test_ising_default_is_light():
    from ai4mat.datasets.ising import IsingDataset
    ds = IsingDataset()
    x, _ = ds[0]
    assert x.shape == torch.Size([1, 16, 16])


def test_ising_normalised_to_0_1():
    from ai4mat.datasets.ising import IsingDataset
    ds = IsingDataset(size='light')
    for i in range(min(10, len(ds))):
        x, _ = ds[i]
        assert x.min() >= 0.0 and x.max() <= 1.0


def test_ising_invalid_size():
    from ai4mat.datasets.ising import IsingDataset
    with pytest.raises(ValueError):
        IsingDataset(size='medium')


def test_ising_labels_binary():
    from ai4mat.datasets.ising import IsingDataset
    ds = IsingDataset(size='light')
    for i in range(min(20, len(ds))):
        _, y = ds[i]
        assert y.item() in {0, 1}
