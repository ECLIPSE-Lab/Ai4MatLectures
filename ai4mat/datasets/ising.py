import torch
from torch.utils.data import Dataset


class IsingDataset(Dataset):
    """Ising model microstructure classification dataset as a PyTorch Dataset.

    Images are grayscale with pixel values in [0, 255], normalised to [0, 1].
    Binary classification: 0 = below Curie temperature, 1 = above.

    Args:
        size: 'light' -> 16x16 images (5000 samples, fast for demos)
              'full'  -> 64x64 images (5000 samples, for CNN training)
              Default: 'light'

    X shape: (1, 16, 16) or (1, 64, 64)  dtype: float32  range: [0, 1]
    y shape: ()  dtype: long  values: {0, 1}
    """

    def __init__(self, size='light', transform=None, target_transform=None):
        if size == 'light':
            from mdsdata import load_Ising_light
            images, labels, _ = load_Ising_light()
            H, W = 16, 16
        elif size == 'full':
            from mdsdata import load_Ising
            images, labels, _ = load_Ising()
            H, W = 64, 64
        else:
            raise ValueError(f"size must be 'light' or 'full', got '{size}'")

        # images: (N, H, W) int64 array; pixel range [0, 255]
        self.X = torch.tensor(images, dtype=torch.float32).reshape(-1, 1, H, W) / 255.0
        self.y = torch.tensor(labels, dtype=torch.long)
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
