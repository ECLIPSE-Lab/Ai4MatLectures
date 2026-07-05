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
from .matbench import MatBenchDataset
from .rmd17 import RMD17Dataset
from .cdvae_materials import CDVAEMaterialsDataset
from .qm9 import QM9Dataset
from .metaldam import MetalDAMDataset

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
    "MatBenchDataset",
    "RMD17Dataset",
    "CDVAEMaterialsDataset",
    "QM9Dataset",
    "MetalDAMDataset",
]
