import math

import numpy as np
import torch
from torch.utils.data import Dataset


# Pauling electronegativities for the 38 elements used by the dataset.
# Source: standard periodic-table reference values (single-bonded atom Pauling
# scale).  We hard-code them here so the dataset has no external dependency.
_ELECTRONEGATIVITY = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
    26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01,
    33: 2.18, 34: 2.55, 35: 2.96, 37: 0.82, 38: 0.95, 47: 1.93, 50: 1.96,
    53: 2.66, 56: 0.89,
}

# Covalent radii (Å) for the same set, used as a stand-in for ionic radii in
# the toy formation-energy model.
_RADIUS = {
    1: 0.31, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39,
    26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20,
    33: 1.19, 34: 1.20, 35: 1.20, 37: 2.20, 38: 1.95, 47: 1.45, 50: 1.39,
    53: 1.39, 56: 2.15,
}

# Element groups we draw cations and anions from for each prototype.
_CATIONS = [3, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 37, 38, 47, 50, 56]
_ANIONS = [7, 8, 9, 15, 16, 17, 33, 34, 35, 53]

# Five tiny crystal "prototypes" defined as fixed graph templates over a
# 4-site unit cell + first-shell neighbour images.  Each entry is
# (n_nodes, edge_list, prototype_bias).  Edges are undirected (we'll
# duplicate them when building the data tensors).
_PROTOTYPES = {
    "rocksalt":   (8,  [(0,1),(1,2),(2,3),(3,0),(0,4),(1,5),(2,6),(3,7),
                        (4,5),(5,6),(6,7),(7,4)],                     -2.30),
    "zincblende": (8,  [(0,4),(0,5),(1,4),(1,5),(2,5),(2,6),(3,6),(3,7),
                        (0,6),(1,7),(2,7),(3,4)],                     -1.85),
    "wurtzite":   (8,  [(0,4),(1,4),(2,4),(3,4),(0,5),(1,5),(2,6),(3,7),
                        (4,5),(5,6),(6,7)],                           -1.65),
    "fluorite":   (12, [(0,4),(0,5),(0,6),(0,7),(1,4),(1,5),(1,8),(1,9),
                        (2,5),(2,6),(2,9),(2,10),(3,6),(3,7),(3,10),(3,11),
                        (0,1),(1,2),(2,3),(3,0)],                     -2.55),
    "perovskite": (10, [(0,5),(0,6),(0,7),(0,8),(0,9),
                        (1,5),(1,6),(2,7),(2,8),(3,5),(3,9),(4,7),(4,9),
                        (1,2),(2,3),(3,4),(4,1)],                     -2.10),
}

# How many nodes are cations vs anions in each prototype.  For each prototype
# we list the cation node indices; the rest are anions.
_CATION_INDICES = {
    "rocksalt":   [0, 1, 2, 3],
    "zincblende": [0, 1, 2, 3],
    "wurtzite":   [0, 1, 2, 3],
    "fluorite":   [0, 1, 2, 3],
    "perovskite": [0, 1, 2, 3, 4],
}


def _toy_formation_energy(species, cation_idx, prototype_bias, edges, rng):
    """Synthetic formation energy in eV/atom.

    Three additive contributions, all chosen so the resulting numbers behave
    like *real* DFT formation energies for teaching purposes:

    1. Prototype baseline (more stable prototypes are more negative).
    2. Mean cation-anion electronegativity difference (more ionic = more bound).
    3. Mean radius mismatch penalty (wrong cation/anion size = strain).
    4. Tiny Gaussian noise so optimizers see something to regress against.
    """
    chi = np.array([_ELECTRONEGATIVITY[int(z)] for z in species])
    r = np.array([_RADIUS[int(z)] for z in species])

    is_cation = np.zeros(len(species), dtype=bool)
    is_cation[cation_idx] = True

    chi_diffs = []
    radius_mismatch = []
    for i, j in edges:
        if is_cation[i] != is_cation[j]:
            chi_diffs.append(abs(chi[i] - chi[j]))
            r_target = (r[i] + r[j]) / 2.0
            radius_mismatch.append(abs(r[i] - r[j]) / max(r_target, 0.1))

    chi_term = -0.55 * (float(np.mean(chi_diffs)) if chi_diffs else 0.0)
    radius_term = +1.20 * (float(np.mean(radius_mismatch)) if radius_mismatch else 0.0)
    noise = float(rng.normal(0.0, 0.05))

    return prototype_bias + chi_term + radius_term + noise


def _build_dataset(n_total=200, seed=0):
    """Procedurally generate the 200-crystal dataset; returns a dict of arrays.

    Each crystal i is represented by:
      - species[i]: int64 array (n_i,)         atomic numbers
      - edge_index[i]: int64 array (2, m_i)    undirected edges (both directions)
      - edge_distance[i]: float32 array (m_i,) toy bond lengths
      - prototype[i]: int64 in {0..4}
      - y[i]: float32 scalar                   toy formation energy (eV/atom)
    """
    rng = np.random.default_rng(seed)
    proto_names = list(_PROTOTYPES.keys())
    n_proto = len(proto_names)
    per_proto = n_total // n_proto                # 40 each at n_total=200

    species_list, edge_index_list, edge_dist_list = [], [], []
    proto_idx_list, y_list = [], []

    for p_idx, proto in enumerate(proto_names):
        n_nodes, edges, baseline = _PROTOTYPES[proto]
        cat_pos = _CATION_INDICES[proto]

        for _ in range(per_proto):
            # Pick one cation species (used for all cation sites) and one
            # anion species (all anion sites).  Real crystals are messier;
            # this keeps the toy model interpretable.
            z_cat = int(rng.choice(_CATIONS))
            z_an = int(rng.choice(_ANIONS))
            species = np.full(n_nodes, z_an, dtype=np.int64)
            species[cat_pos] = z_cat

            # Edge distances: ideal sum of covalent radii plus a small
            # perturbation so different crystals see different geometry.
            ideal = _RADIUS[z_cat] + _RADIUS[z_an]
            distortion = rng.uniform(0.92, 1.08, size=len(edges))
            d_undir = ideal * distortion

            # Build directed edges (i->j and j->i) so message passing is
            # symmetric; mirror the distances.
            ei = np.array([[i, j] for i, j in edges] +
                          [[j, i] for i, j in edges]).T
            ed = np.concatenate([d_undir, d_undir]).astype(np.float32)

            y = _toy_formation_energy(species, cat_pos, baseline, edges, rng)

            species_list.append(species)
            edge_index_list.append(ei.astype(np.int64))
            edge_dist_list.append(ed)
            proto_idx_list.append(p_idx)
            y_list.append(np.float32(y))

    return {
        "species": species_list,
        "edge_index": edge_index_list,
        "edge_distance": edge_dist_list,
        "prototype": np.array(proto_idx_list, dtype=np.int64),
        "y": np.array(y_list, dtype=np.float32),
        "prototype_names": proto_names,
    }


class CrystalGraphsDataset(Dataset):
    """Tiny synthetic crystal-graphs dataset for hand-rolled GNN training.

    200 crystals across 5 prototype templates (rocksalt, zincblende,
    wurtzite, fluorite, perovskite), each populated with a randomly
    chosen cation and anion species.  Targets are *toy* formation
    energies built from a hand-coded model that nonetheless behaves like
    real DFT formation energies (electronegativity-difference + radius-
    mismatch + prototype-baseline + Gaussian noise).

    Each sample is a graph:
      - species:        int64 tensor (n_i,)        atomic numbers
      - edge_index:     int64 tensor (2, m_i)      directed edges (both ways)
      - edge_distance:  float32 tensor (m_i,)      toy bond lengths in Angstrom
      - prototype:      int64 scalar in {0..4}
      - y:              float32 scalar            formation energy (eV/atom)

    The dataset is *deterministic* (fixed seed=0) and pure-Python: no
    pymatgen, no PyTorch Geometric, no network calls. Designed for the
    Week 6 hand-rolled GNN exercise.
    """

    def __init__(self, n_total=200, seed=0):
        data = _build_dataset(n_total=n_total, seed=seed)
        self.species = [torch.from_numpy(z) for z in data["species"]]
        self.edge_index = [torch.from_numpy(e) for e in data["edge_index"]]
        self.edge_distance = [torch.from_numpy(d) for d in data["edge_distance"]]
        self.prototype = torch.from_numpy(data["prototype"])
        self.y = torch.from_numpy(data["y"])
        self.prototype_names = data["prototype_names"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "species": self.species[idx],
            "edge_index": self.edge_index[idx],
            "edge_distance": self.edge_distance[idx],
            "prototype": int(self.prototype[idx].item()),
            "y": self.y[idx],
        }
