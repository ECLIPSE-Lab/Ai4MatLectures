from .iris import IrisDataset
from .digits import DigitsDataset
from .tensile_test import TensileTestDataset
from .ising import IsingDataset
from .cahn_hilliard import CahnHilliardDataset
from .chemical_elements import ChemicalElementsDataset
from .nanoindentation import NanoindentationDataset
from .neu_det import NEUDETDataset
from .crystal_graphs import CrystalGraphsDataset
from .estm import ESTMDataset

__all__ = [
    "IrisDataset",
    "DigitsDataset",
    "TensileTestDataset",
    "IsingDataset",
    "CahnHilliardDataset",
    "ChemicalElementsDataset",
    "NanoindentationDataset",
    "NEUDETDataset",
    "CrystalGraphsDataset",
    "ESTMDataset",
]
