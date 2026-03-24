import torch
from torch.utils.data import Dataset


class NanoindentationDataset(Dataset):
    """Nanoindentation dataset of Cu/Cr composites as a PyTorch Dataset.

    938 samples (outliers included by default load), 2 features
    (Young's modulus E, hardness H in GPa), 4 classes.
    Task: classify Cu/Cr composite from indentation measurements.

    X shape: (2,)  dtype: float32
    y shape: ()    dtype: long  (4 Cu/Cr composite classes: 0%/25%/60%/100% Cr)
    """

    def __init__(self, transform=None, target_transform=None):
        from mdsdata import MDS5
        import numpy as np
        X, y = MDS5.load_data(return_X_y=True)
        self.X = torch.tensor(np.array(X, dtype=float), dtype=torch.float32)
        self.y = torch.tensor(np.array(y, dtype=int).ravel(), dtype=torch.long)
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
