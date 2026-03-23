"""Shared pytest fixtures for ai4mat dataset tests.

All tests use small subsets (size parameter or simulation_number) to keep
test suite fast — mdsdata loads from disk so we cap at ~50 samples per test.
"""
import pytest
import torch


def assert_dataset_contract(ds, expected_x_shape, expected_y_dtype, min_len=10):
    """Reusable contract check for any ai4mat Dataset."""
    from torch.utils.data import DataLoader

    assert len(ds) >= min_len, f"Expected at least {min_len} samples, got {len(ds)}"

    x, y = ds[0]
    assert isinstance(x, torch.Tensor), "x must be a torch.Tensor"
    assert isinstance(y, torch.Tensor), "y must be a torch.Tensor"
    assert x.shape == torch.Size(expected_x_shape), (
        f"Expected x shape {expected_x_shape}, got {x.shape}"
    )
    assert y.dtype == expected_y_dtype, (
        f"Expected y dtype {expected_y_dtype}, got {y.dtype}"
    )
    assert y.ndim == 0, "y must be a scalar (0-dim) tensor"

    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[1:] == torch.Size(expected_x_shape)
    assert batch_y.shape[0] <= 4 and len(batch_y) > 0
