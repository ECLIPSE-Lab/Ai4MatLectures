import torch
from torch.utils.data import Dataset


class IrisDataset(Dataset):
    """Iris flower classification dataset as a PyTorch Dataset.

    150 samples, 4 features (sepal/petal length & width), 3 classes.
    X: float32 tensor of shape (4,)
    y: long tensor scalar (0=Iris-setosa, 1=Iris-versicolor, 2=Iris-virginica)
    """

    def __init__(self, transform=None, target_transform=None):
        from mdsdata import DS1
        X, y = DS1.load_data(return_X_y=True)
        self.X = torch.tensor(X, dtype=torch.float32)
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
