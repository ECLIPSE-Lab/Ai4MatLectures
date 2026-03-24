import pytest
import torch
from tests.conftest import assert_dataset_contract


@pytest.mark.parametrize("temperature", [0, 400, 600])
def test_tensile_dataset_contract(temperature):
    from ai4mat.datasets.tensile_test import TensileTestDataset
    ds = TensileTestDataset(temperature=temperature)
    assert_dataset_contract(ds, expected_x_shape=[1], expected_y_dtype=torch.float32, min_len=100)


def test_tensile_invalid_temperature():
    from ai4mat.datasets.tensile_test import TensileTestDataset
    with pytest.raises((AssertionError, ValueError)):
        TensileTestDataset(temperature=999)


def test_tensile_default_temperature_is_600():
    from ai4mat.datasets.tensile_test import TensileTestDataset
    ds_default = TensileTestDataset()
    ds_600 = TensileTestDataset(temperature=600)
    x0, y0 = ds_default[0]
    x1, y1 = ds_600[0]
    assert torch.allclose(x0, x1) and torch.allclose(y0, y1)
