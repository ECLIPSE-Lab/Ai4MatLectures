"""Tests for scripts/build_week3_mystery.py — reproducibility + npz structure."""
from pathlib import Path
import subprocess
import sys
import hashlib

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_week3_mystery.py"
NPZ = REPO_ROOT / "data" / "week3_mystery.npz"
SOLUTIONS = REPO_ROOT / "data" / "week3_mystery_solutions.txt"

EXPECTED_KEYS = {
    f"split_{name}_{piece}"
    for name in ("A", "B", "C")
    for piece in ("X_train", "X_test", "y_train", "y_test")
}

EXPECTED_FLAVOURS = {"preprocessing", "group", "temporal"}


def _run_builder():
    subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=REPO_ROOT)


def test_builder_produces_both_files():
    if NPZ.exists():
        NPZ.unlink()
    if SOLUTIONS.exists():
        SOLUTIONS.unlink()
    _run_builder()
    assert NPZ.exists(), "build script did not produce data/week3_mystery.npz"
    assert SOLUTIONS.exists(), "build script did not produce data/week3_mystery_solutions.txt"


def test_npz_has_expected_keys_and_no_answer_key():
    _run_builder()
    with np.load(NPZ) as z:
        keys = set(z.files)
    assert keys == EXPECTED_KEYS, f"npz key mismatch: extra={keys - EXPECTED_KEYS}, missing={EXPECTED_KEYS - keys}"


def test_solutions_file_lists_three_flavours_one_per_split():
    _run_builder()
    lines = [
        ln.strip()
        for ln in SOLUTIONS.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert len(lines) == 3, f"expected 3 non-comment lines, got {len(lines)}: {lines}"
    flavours = set()
    for ln in lines:
        # Lines look like:  split_A: preprocessing
        name, _, flavour = ln.partition(":")
        assert name.strip() in {"split_A", "split_B", "split_C"}, ln
        flavours.add(flavour.strip())
    assert flavours == EXPECTED_FLAVOURS, f"flavour set mismatch: {flavours}"


def test_builder_is_deterministic():
    """Running the builder twice produces byte-identical npz and identical solutions text."""
    _run_builder()
    h1 = hashlib.sha256(NPZ.read_bytes()).hexdigest()
    s1 = SOLUTIONS.read_text()
    NPZ.unlink()
    SOLUTIONS.unlink()
    _run_builder()
    h2 = hashlib.sha256(NPZ.read_bytes()).hexdigest()
    s2 = SOLUTIONS.read_text()
    assert h1 == h2, "npz is not bit-for-bit reproducible across runs"
    assert s1 == s2, "solutions text is not reproducible across runs"


def test_split_shapes_are_sane():
    _run_builder()
    with np.load(NPZ) as z:
        for name in ("A", "B", "C"):
            X_tr = z[f"split_{name}_X_train"]
            X_te = z[f"split_{name}_X_test"]
            y_tr = z[f"split_{name}_y_train"]
            y_te = z[f"split_{name}_y_test"]
            assert X_tr.ndim == 2 and X_te.ndim == 2, f"split_{name} X arrays should be 2-D"
            assert X_tr.shape[1] == X_te.shape[1], f"split_{name} feature dim mismatch"
            assert X_tr.shape[0] == y_tr.shape[0], f"split_{name} train rows mismatch"
            assert X_te.shape[0] == y_te.shape[0], f"split_{name} test rows mismatch"
            assert X_tr.shape[0] >= 100 and X_te.shape[0] >= 100, f"split_{name} too small"
