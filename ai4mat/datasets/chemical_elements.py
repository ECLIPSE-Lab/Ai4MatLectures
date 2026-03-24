import torch
from torch.utils.data import Dataset


class ChemicalElementsDataset(Dataset):
    """Periodic table element properties dataset as a PyTorch Dataset.

    38 elements (22 metals, 16 non-metals). 4 features: atomic radius,
    electron affinity, ionization energy, electronegativity.
    Task: binary classification (metallic vs non-metallic) treated as regression.

    X shape: (4,)  dtype: float32
    y shape: ()    dtype: float32  (0.0 = non-metallic, 1.0 = metallic)
    """

    def __init__(self, transform=None, target_transform=None):
        from mdsdata import MDS4
        import numpy as np
        X, y = MDS4.load_data(return_X_y=True)
        self.X = torch.tensor(np.array(X, dtype=float), dtype=torch.float32)
        self.y = torch.tensor(np.array(y, dtype=float).ravel(), dtype=torch.float32)
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
