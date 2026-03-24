import torch
from torch.utils.data import Dataset


class DigitsDataset(Dataset):
    """Alpaydin optical digits dataset as a PyTorch Dataset.

    8x8 digit images stored flat as (64,) float32 tensors, normalised to [0, 1].
    Intentionally flat (not (1,8,8)) to pair with fully-connected models.
    For CNN use, reshape in the notebook: x.reshape(-1, 1, 8, 8).

    y: long scalar, digit class 0-9.
    """

    def __init__(self, transform=None, target_transform=None):
        from mdsdata import load_Alpaydin_digits
        X, y = load_Alpaydin_digits()
        # X is (N, 8, 8) float64; pixel range 0-255; normalise to [0, 1]
        self.X = torch.tensor(X, dtype=torch.float32).reshape(-1, 64) / 255.0
        self.y = torch.tensor(y, dtype=torch.long)
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
