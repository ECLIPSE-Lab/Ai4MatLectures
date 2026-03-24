import torch
from tests.conftest import assert_dataset_contract


def test_elements_contract():
    from ai4mat.datasets.chemical_elements import ChemicalElementsDataset
    ds = ChemicalElementsDataset()
    # n_features depends on MDS4 columns; contract checks shape consistency
    x, y = ds[0]
    assert x.ndim == 1 and x.dtype == torch.float32
    assert y.ndim == 0 and y.dtype == torch.float32
    assert len(ds) > 10


def test_elements_no_nan():
    from ai4mat.datasets.chemical_elements import ChemicalElementsDataset
    ds = ChemicalElementsDataset()
    for i in range(len(ds)):
        x, y = ds[i]
        assert not torch.isnan(x).any(), f"NaN in X at index {i}"
        assert not torch.isnan(y), f"NaN in y at index {i}"
