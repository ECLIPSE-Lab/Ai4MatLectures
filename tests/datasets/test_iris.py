import torch
from tests.conftest import assert_dataset_contract


def test_iris_dataset_contract():
    from ai4mat.datasets.iris import IrisDataset
    ds = IrisDataset()
    assert_dataset_contract(ds, expected_x_shape=[4], expected_y_dtype=torch.long, min_len=150)


def test_iris_dataset_length():
    from ai4mat.datasets.iris import IrisDataset
    assert len(IrisDataset()) == 150


def test_iris_transform_applied():
    from ai4mat.datasets.iris import IrisDataset
    transform = lambda x: x * 2
    ds = IrisDataset(transform=transform)
    x_raw, _ = IrisDataset()[0]
    x_transformed, _ = ds[0]
    assert torch.allclose(x_transformed, x_raw * 2)


def test_iris_labels_in_range():
    from ai4mat.datasets.iris import IrisDataset
    ds = IrisDataset()
    for i in range(len(ds)):
        _, y = ds[i]
        assert y.item() in {0, 1, 2}
