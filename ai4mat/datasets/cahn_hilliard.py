import torch
from torch.utils.data import Dataset


class CahnHilliardDataset(Dataset):
    """Cahn-Hilliard phase-field microstructure regression dataset.

    17,866 images (64x64) from 18 simulations. Target is free energy (float).
    Pixel values in [0, 255], normalised to [0, 1].

    Args:
        simulation_number: int or list of ints in [0, 17], or -1 for all 18.
            Default: -1 (all simulations, ~17k images -- slow to load).
            Use simulation_number=0 for fast single-simulation loading in tests/demos.

    X shape: (1, 64, 64)  dtype: float32  range: [0, 1]
    y shape: ()  dtype: float32  (free energy, positive real)
    """

    def __init__(self, simulation_number=-1, transform=None, target_transform=None):
        from mdsdata import MDS3
        images, energies = MDS3.load_data(
            simulation_number=simulation_number, return_X_y=True
        )
        # images: (N, 64, 64) int array; pixel range [0, 255]
        self.X = torch.tensor(images, dtype=torch.float32).reshape(-1, 1, 64, 64) / 255.0
        self.y = torch.tensor(energies, dtype=torch.float32)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
