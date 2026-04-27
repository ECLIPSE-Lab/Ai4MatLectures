"""Build data/week3_mystery.npz and data/week3_mystery_solutions.txt.

Three deliberately-leaky train/test splits derived from TensileTestDataset
across temperatures {0, 400, 600} degC. The mapping from split-name (A/B/C)
to leakage flavour (preprocessing / group / temporal) is randomised with
a fixed seed so the artefact is reproducible bit-for-bit.

Run from repo root:
    python scripts/build_week3_mystery.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ai4mat.datasets import TensileTestDataset

SEED = 20260427
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_NPZ = REPO_ROOT / "data" / "week3_mystery.npz"
OUT_TXT = REPO_ROOT / "data" / "week3_mystery_solutions.txt"


def load_combined() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, temperature_group, time_index_within_temp)."""
    Xs, ys, groups, t_idx = [], [], [], []
    for T in (0, 400, 600):
        ds = TensileTestDataset(temperature=T)
        # ds.X is (350, 1) torch.float32; ds.y is (350,) torch.float32.
        strain = ds.X.numpy().astype(np.float64).reshape(-1, 1)
        stress = ds.y.numpy().astype(np.float64)
        n = strain.shape[0]
        # Feature matrix has [strain, temperature]: T is a real input feature.
        X_T = np.hstack([strain, np.full((n, 1), float(T))])
        Xs.append(X_T)
        ys.append(stress)
        groups.append(np.full(n, T, dtype=np.int64))
        t_idx.append(np.arange(n, dtype=np.int64))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    grp = np.concatenate(groups)
    tix = np.concatenate(t_idx)
    return X, y, grp, tix


def make_split_preprocessing_leak(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Standardise X using stats computed over the FULL (train+test) data, then split.

    The leak: test-set means/stds bleed into train via the shared scaling.
    Visible symptom: test-R^2 is suspiciously close to train-R^2 even with
    a model that should not generalise across temperatures.
    """
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    n = X_scaled.shape[0]
    perm = rng.permutation(n)
    n_train = int(0.7 * n)
    tr, te = perm[:n_train], perm[n_train:]
    return X_scaled[tr], X_scaled[te], y[tr], y[te]


def make_split_group_leak(X: np.ndarray, y: np.ndarray, grp: np.ndarray, rng: np.random.Generator):
    """Random row-split that mixes ALL three temperatures into both train and test.

    The leak: every test row has near-neighbour train rows from the same temperature,
    so the model can memorise temperature-specific structure. An honest split would
    hold out an entire temperature.
    """
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train = int(0.7 * n)
    tr, te = perm[:n_train], perm[n_train:]
    return X[tr], X[te], y[tr], y[te]


def make_split_temporal_leak(X: np.ndarray, y: np.ndarray, grp: np.ndarray, t_idx: np.ndarray, rng: np.random.Generator):
    """Train on LATER time-indices (high strain), test on EARLIER ones (low strain).

    The leak: the model learned the post-yield plateau, then is asked to interpolate
    backwards to elastic regime — but stress at low strain is simpler than at high
    strain, so the test 'looks easy' compared to honest random splits within the
    elastic-only regime.
    Mechanism: per temperature, take the late-half rows for training, early-half for test.
    """
    tr_mask = np.zeros(X.shape[0], dtype=bool)
    te_mask = np.zeros(X.shape[0], dtype=bool)
    for T in np.unique(grp):
        in_T = np.where(grp == T)[0]
        # Sort by time-index within this temperature.
        sorted_in_T = in_T[np.argsort(t_idx[in_T])]
        cut = int(0.5 * sorted_in_T.size)
        te_mask[sorted_in_T[:cut]] = True   # earliest 50% -> test
        tr_mask[sorted_in_T[cut:]] = True   # latest 50%   -> train
    return X[tr_mask], X[te_mask], y[tr_mask], y[te_mask]


def main() -> None:
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    X, y, grp, t_idx = load_combined()

    builders = {
        "preprocessing": lambda: make_split_preprocessing_leak(X, y, np.random.default_rng(SEED + 1)),
        "group":         lambda: make_split_group_leak(X, y, grp, np.random.default_rng(SEED + 2)),
        "temporal":      lambda: make_split_temporal_leak(X, y, grp, t_idx, np.random.default_rng(SEED + 3)),
    }

    # Randomise the A/B/C -> flavour mapping deterministically.
    flavours = ["preprocessing", "group", "temporal"]
    rng.shuffle(flavours)
    mapping = dict(zip(["A", "B", "C"], flavours))

    npz_payload: dict[str, np.ndarray] = {}
    for letter, flavour in mapping.items():
        X_tr, X_te, y_tr, y_te = builders[flavour]()
        npz_payload[f"split_{letter}_X_train"] = X_tr.astype(np.float64)
        npz_payload[f"split_{letter}_X_test"]  = X_te.astype(np.float64)
        npz_payload[f"split_{letter}_y_train"] = y_tr.astype(np.float64)
        npz_payload[f"split_{letter}_y_test"]  = y_te.astype(np.float64)

    np.savez(OUT_NPZ, **npz_payload)

    OUT_TXT.write_text(
        "# week3_mystery_solutions.txt\n"
        "# DO NOT OPEN until you have written down your guess for each split.\n"
        f"split_A: {mapping['A']}\n"
        f"split_B: {mapping['B']}\n"
        f"split_C: {mapping['C']}\n"
    )
    print(f"wrote {OUT_NPZ.relative_to(REPO_ROOT)} ({OUT_NPZ.stat().st_size} bytes)")
    print(f"wrote {OUT_TXT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
