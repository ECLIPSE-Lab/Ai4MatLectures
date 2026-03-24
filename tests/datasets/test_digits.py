import torch
from tests.conftest import assert_dataset_contract


def test_digits_dataset_contract():
    from ai4mat.datasets.digits import DigitsDataset
    ds = DigitsDataset()
    # X is flat (64,) -- deliberately not (1,8,8), see spec note
    assert_dataset_contract(ds, expected_x_shape=[64], expected_y_dtype=torch.long, min_len=100)


def test_digits_x_range():
    """Feature values should be normalised to [0, 1]."""
    from ai4mat.datasets.digits import DigitsDataset
    ds = DigitsDataset()
    x, _ = ds[0]
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_digits_labels_in_range():
    from ai4mat.datasets.digits import DigitsDataset
    ds = DigitsDataset()
    for i in range(min(50, len(ds))):
        _, y = ds[i]
        assert 0 <= y.item() <= 9
