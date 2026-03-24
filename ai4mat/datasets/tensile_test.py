import torch
from torch.utils.data import Dataset


class TensileTestDataset(Dataset):
    """Tensile test stress-strain dataset as a PyTorch Dataset.

    350 samples per temperature. X = strain (scalar), y = stress (scalar).
    X shape: (1,)  dtype: float32
    y shape: ()    dtype: float32

    Args:
        temperature: one of {0, 400, 600} degrees Celsius. Default 600
            (most nonlinear curve -- pedagogically richest for regression).
    """

    def __init__(self, temperature=600, transform=None, target_transform=None):
        if temperature not in {0, 400, 600}:
            raise ValueError(f"temperature must be one of {{0, 400, 600}}, got {temperature}")
        from mdsdata import load_tensile_test
        strain, stress = load_tensile_test(temperature=temperature)
        # strain is (350,) float64; unsqueeze to (350, 1) so [idx] gives (1,)
        self.X = torch.tensor(strain, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(stress, dtype=torch.float32)
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
