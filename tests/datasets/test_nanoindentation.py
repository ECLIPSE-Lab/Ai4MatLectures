import torch
from tests.conftest import assert_dataset_contract


def test_nanoindentation_contract():
    from ai4mat.datasets.nanoindentation import NanoindentationDataset
    ds = NanoindentationDataset()
    assert_dataset_contract(ds, expected_x_shape=[2], expected_y_dtype=torch.long, min_len=100)


def test_nanoindentation_labels_valid():
    from ai4mat.datasets.nanoindentation import NanoindentationDataset
    ds = NanoindentationDataset()
    unique_labels = set(ds[i][1].item() for i in range(min(50, len(ds))))
    assert unique_labels.issubset({0, 1, 2, 3})  # 4 Cu/Cr composite classes
