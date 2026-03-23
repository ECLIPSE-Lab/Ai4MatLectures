# Ai4MatLectures

PyTorch dataset wrappers and teaching notebooks for the ECLIPSE Lab lecture triad:
- **MFML** — Mathematical Foundations of AI & ML
- **MLPC** — Machine Learning in Materials Processing & Characterization
- **MG** — Materials Genomics

## Install

```bash
pip install git+https://github.com/ECLIPSE-Lab/Ai4MatLectures.git "mdsdata>=0.1.5"
```

## Usage

```python
from ai4mat.datasets import IsingDataset
from torch.utils.data import DataLoader

ds = IsingDataset(size='light')
loader = DataLoader(ds, batch_size=32, shuffle=True)
x, y = next(iter(loader))  # x: (32, 1, 16, 16), y: (32,)
```

## Notebooks

See [eclipse-lab.github.io/Ai4MatLectures](https://eclipse-lab.github.io/Ai4MatLectures) for rendered notebooks with Colab links.
