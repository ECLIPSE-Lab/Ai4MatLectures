# HRTEMDataset — multi-mode PyTorch dataset for HRTEM micrographs

**Date:** 2026-05-13
**Author:** Philipp Pelz (with Claude)
**Sibling specs:** `2026-05-11-neu-det-clustering-mlpc-unit5-design.md`, `2026-05-11-estm-clustering-mlpc-unit5-design.md`

## Goal

Wrap the 25 HRTEM TIFFs currently sitting at `data/hrtem/` as a proper PyTorch
`Dataset` so they are reusable across notebooks. A single class
`HRTEMDataset` exposes three behaviours via a `mode` constructor flag:

- `mode="image"` — whole-image inputs, intended for MLPC W4 receptive-field
  visualisation demos (show what a deep activation "sees" in an atomic-
  resolution micrograph).
- `mode="patch"` — deterministic tiled patches, intended as training inputs
  for small CNNs that don't fit 2048×2048 images.
- `mode="ae"` — `(noisy, clean)` patch pairs for autoencoder training
  (MLPC W11 anomaly / autoencoder section).

The class is the minimal infrastructure needed to *make these images usable*
in lecture notebooks; it does not commit to a specific downstream notebook.

## Scope

In scope:

- A new `HRTEMDataset(torch.utils.data.Dataset)` class in
  `ai4mat/datasets/hrtem.py`, re-exported from `ai4mat.datasets.__init__`.
- Eager load of all 25 TIFFs from `data/hrtem/` (≈ 400 MB float32 in
  memory) with per-image 1st–99th percentile normalisation into `[0, 1]`.
- Smoke / contract tests in `tests/datasets/test_hrtem.py`.
- A `.gitignore` line for `data/hrtem/`, matching the precedent set by
  `data/NEU-DET/` and `data/estm/`.

Out of scope (YAGNI):

- A `download_if_missing` helper. The TIFFs are sitting in the repo from a
  prior session and there is no public URL to fetch them from. If
  `data/hrtem/` is missing or empty, the class raises `FileNotFoundError`
  with a one-line hint.
- Random crops / on-the-fly augmentation. Patches use a deterministic
  tiled grid — reproducible across runs is more valuable for lecture
  demos and TDD than stochastic sampling. A future `mode="patch_random"`
  is easy to add later if a notebook needs it.
- Multiple noise models in `mode="ae"`. Only additive Gaussian. Poisson /
  Anscombe-transformed Gaussian are easy follow-ups if needed.
- Multi-image-size support. All 25 TIFFs are 2048×2048 (verified at spec
  time); the loader asserts this and raises a clear error otherwise.
- A lecture notebook. The dataset is the prerequisite; downstream
  notebooks are separate specs.

## Data layer — `ai4mat/datasets/hrtem.py`

A new `HRTEMDataset(torch.utils.data.Dataset)`. Like `NEUDETDataset` and
`ESTMDataset`, this dataset owns its on-disk layout rather than going
through `mdsdata`.

**Expected on-disk layout (already present, not downloaded):**

```
data/hrtem/
├── image_000.tiff   (2048 × 2048, int16, ~8 MB each)
├── image_001.tiff
├── ...
└── image_024.tiff   (25 files total)
```

Image content: int16 detector counts, observed range ≈ −321 … 1666, mean
≈ 120 in the sample inspected at spec time. The negative values are
real (after detector dark-current subtraction).

**Constructor:**

```python
class HRTEMDataset(Dataset):
    def __init__(
        self,
        root: str = "data/hrtem",
        mode: str = "image",          # "image" | "patch" | "ae"
        patch_size: int = 256,
        noise_std: float = 0.05,      # Gaussian noise std (only for mode="ae")
        normalize: bool = True,       # per-image 1st–99th percentile → [0, 1]
        transform=None,
        target_transform=None,
        seed: int = 0,                # base RNG seed for "ae" noise
    ):
```

**Validation:**

- `mode` must be one of `{"image", "patch", "ae"}` — else `ValueError`.
- `patch_size` must be a positive divisor of 2048 in patch / ae modes —
  else `ValueError(f"patch_size {patch_size} must divide 2048 evenly")`.
- `noise_std` must be ≥ 0 in ae mode — else `ValueError`.
- `root` directory must exist and contain ≥ 1 file matching `image_*.tiff` —
  else `FileNotFoundError(f"No HRTEM TIFFs found in {root}. Place the files manually.")`.
- Each loaded TIFF must be 2048×2048 — else `ValueError(f"{path}: expected 2048×2048, got {shape}")`.

**Eager load + normalisation:**

For each `image_NNN.tiff`, sorted by filename:

1. Read with `imageio.v3.imread` → `numpy.int16 (2048, 2048)`.
2. If `normalize=True`:
   - `lo, hi = np.percentile(img, [1, 99])`
   - `img = np.clip(img, lo, hi)`
   - `img = (img - lo) / max(hi - lo, 1e-8)` (guard against constant images)
   - Result is float32 in `[0, 1]`.
3. If `normalize=False`: cast to float32, leave as raw counts.
4. Stack into `self.images: torch.float32 (N, 1, 2048, 2048)`.

Memory budget: `25 × 1 × 2048² × 4 B = 400 MB`. Acceptable for a lecture
machine; matches the `NEUDETDataset` eager-load idiom.

**Public attributes:**

- `self.images: torch.float32 (N, 1, 2048, 2048)` — always populated.
  Available even in patch / ae modes so notebooks can ask "show me the
  source image for patch idx j".
- `self.image_paths: list[str]` — sorted absolute paths.
- `self.mode`, `self.patch_size`, `self.noise_std`, `self.normalize`,
  `self.seed` — frozen copies of the constructor args.
- `self.tiles_per_image: int` — set to `(2048 // patch_size) ** 2` in
  patch / ae modes, set to `1` in image mode.

### Mode behaviour

**`mode="image"`:**

- `len(ds) = N` (= 25 with the current TIFFs).
- `ds[i]` returns `(image: float32 (1, 2048, 2048), idx: long ())`.
- `idx` is the integer image index as a 0-D `torch.long` tensor.
- The companion "label" is the index — useful for receptive-field demos
  that need to know which source image a CNN activation came from.

**`mode="patch"`:**

- `T = 2048 // patch_size`, `tiles_per_image = T²` (= 64 at P=256).
- `len(ds) = N × T²`.
- `ds[j]` decomposes `j` as `image_idx = j // T²`, then
  `tile_idx = j % T²`, then `ty, tx = divmod(tile_idx, T)`.
- Returns `(patch: float32 (1, P, P), image_idx: long ())` where the
  patch is `self.images[image_idx, :, ty*P:(ty+1)*P, tx*P:(tx+1)*P]`.

**`mode="ae"`:**

- Same tiling and length as `mode="patch"`.
- `ds[j]` returns `(noisy: float32 (1, P, P), clean: float32 (1, P, P))`.
- `clean` is the same patch as in `mode="patch"`.
- `noisy = clean + ε` where `ε ~ N(0, noise_std)` sampled with
  `torch.Generator().manual_seed(seed * 1_000_003 + j)` so the noise
  pattern is reproducible per-index across runs and across `DataLoader`
  worker processes.
- Noisy values are *not* clipped back to `[0, 1]` — this preserves
  noise statistics for downstream training; if a notebook needs clipping
  it can apply a transform.

### `__getitem__` interface

For all three modes, after building the raw `(x, y)` tuple, the standard
transform / target_transform pipeline applies:

```python
def __getitem__(self, idx):
    x, y = self._get(idx)   # mode-specific
    if self.transform:
        x = self.transform(x)
    if self.target_transform:
        y = self.target_transform(y)
    return x, y
```

(Matches the convention in `ai4mat/datasets/iris.py`,
`nanoindentation.py`, etc.)

## Re-export

Append `"HRTEMDataset"` to `__all__` in `ai4mat/datasets/__init__.py`
and add `from .hrtem import HRTEMDataset` alongside the other imports.

## `.gitignore`

Append `data/hrtem/` so the 25 TIFFs (≈ 200 MB on disk) stay out of git,
matching the existing pattern:

```diff
 data/NEU-DET/
 data/estm/
+data/hrtem/
```

## Tests — `tests/datasets/test_hrtem.py`

Mirror the style of `tests/datasets/test_neu_det.py`. CI-safe tests run
without the data; data-gated tests skip when `data/hrtem/` is missing.

```python
import os
import pytest
import torch

DATA_DIR = "data/hrtem"


def _data_present() -> bool:
    return os.path.isdir(DATA_DIR) and any(
        f.startswith("image_") and f.endswith(".tiff")
        for f in os.listdir(DATA_DIR)
    )


skip_if_no_data = pytest.mark.skipif(
    not _data_present(), reason="HRTEM TIFFs not present in data/hrtem"
)


def test_hrtem_importable():
    from ai4mat.datasets import HRTEMDataset  # noqa: F401


def test_hrtem_mode_validation():
    from ai4mat.datasets import HRTEMDataset
    with pytest.raises(ValueError):
        HRTEMDataset(mode="nonsense")


def test_hrtem_missing_data_raises(tmp_path):
    from ai4mat.datasets import HRTEMDataset
    with pytest.raises(FileNotFoundError):
        HRTEMDataset(root=str(tmp_path / "empty"))


@skip_if_no_data
def test_hrtem_image_mode():
    from ai4mat.datasets import HRTEMDataset
    ds = HRTEMDataset(mode="image")
    assert len(ds) == 25
    x, y = ds[0]
    assert x.shape == (1, 2048, 2048)
    assert x.dtype == torch.float32
    assert 0.0 <= x.min().item() and x.max().item() <= 1.0
    assert y.dtype == torch.long
    assert y.item() == 0


@skip_if_no_data
def test_hrtem_patch_mode():
    from ai4mat.datasets import HRTEMDataset
    ds = HRTEMDataset(mode="patch", patch_size=256)
    assert len(ds) == 25 * 64
    x, y = ds[0]
    assert x.shape == (1, 256, 256)
    assert x.dtype == torch.float32
    assert y.item() == 0
    # Index 64 is the first tile of image 1.
    _, y1 = ds[64]
    assert y1.item() == 1


@skip_if_no_data
def test_hrtem_ae_mode():
    from ai4mat.datasets import HRTEMDataset
    ds = HRTEMDataset(mode="ae", patch_size=256, noise_std=0.05, seed=0)
    noisy, clean = ds[0]
    assert noisy.shape == (1, 256, 256)
    assert clean.shape == (1, 256, 256)
    assert not torch.allclose(noisy, clean)
    diff_std = (noisy - clean).std().item()
    assert 0.04 < diff_std < 0.06   # noise_std ± 20 %


@skip_if_no_data
def test_hrtem_ae_noise_reproducible():
    from ai4mat.datasets import HRTEMDataset
    ds1 = HRTEMDataset(mode="ae", seed=42)
    ds2 = HRTEMDataset(mode="ae", seed=42)
    n1, _ = ds1[10]
    n2, _ = ds2[10]
    assert torch.allclose(n1, n2)


@skip_if_no_data
def test_hrtem_ae_clean_matches_patch_mode():
    """The clean target in ae mode is exactly the patch from patch mode."""
    from ai4mat.datasets import HRTEMDataset
    ds_patch = HRTEMDataset(mode="patch", patch_size=256)
    ds_ae = HRTEMDataset(mode="ae", patch_size=256, noise_std=0.05, seed=0)
    for j in [0, 64, 500, len(ds_patch) - 1]:
        x_patch, _ = ds_patch[j]
        _, clean = ds_ae[j]
        assert torch.allclose(x_patch, clean)


@skip_if_no_data
def test_hrtem_patch_size_must_divide():
    from ai4mat.datasets import HRTEMDataset
    with pytest.raises(ValueError):
        HRTEMDataset(mode="patch", patch_size=300)  # 2048 % 300 != 0
```

## Risks / open questions

- **Eager memory footprint.** 400 MB float32 in RAM is acceptable for
  development but could be the largest dataset in the package. If a
  future student's machine struggles, a `lazy=True` flag is a
  one-screen addition that mmaps the TIFFs and decodes per `__getitem__`.
  Not building it now.
- **Negative-value preservation.** Per-image 1st–99th percentile
  clipping discards the most-extreme dark and bright pixels (≈ 4 %
  of the area each). For receptive-field demos this is fine — the
  features students care about live well inside the percentile band.
  For denoising work that requires faithful intensity reconstruction,
  set `normalize=False` and let the notebook handle it.
- **Noise reproducibility under multi-worker DataLoader.** The
  per-index seed `seed * 1_000_003 + j` is computed from the index
  alone (not from worker state), so noise is identical regardless of
  worker count. This is the safe default; if a notebook explicitly
  wants noise to differ between workers, it can override `transform`.
- **TIFF dtype assumption.** All inspected files are int16. The loader
  uses `imageio.v3.imread` which returns whatever dtype the file
  declares; the per-image percentile normalisation handles uint16,
  int32, etc. transparently. If `normalize=False` the caller gets
  raw counts in whatever dtype the file used, cast to float32.

## Definition of done

- `ai4mat/datasets/hrtem.py` exists; `HRTEMDataset` re-exported.
  [`python -c "from ai4mat.datasets import HRTEMDataset; print(HRTEMDataset.__doc__[:80])"`]
- All three modes load 25 images.
  [`python -c "from ai4mat.datasets import HRTEMDataset as D; print(len(D(mode='image')), len(D(mode='patch')), len(D(mode='ae')))"`
   prints `25 1600 1600`]
- All tests in `tests/datasets/test_hrtem.py` pass when the data is
  present; CI-safe subset passes when it is not.
  [`pytest tests/datasets/test_hrtem.py -v` — 8 passed locally with
  data; with `data/hrtem/` removed the 3 CI-safe tests pass and 5
  data-gated tests skip.]
- `.gitignore` has `data/hrtem/`.
  [`grep -F 'data/hrtem/' .gitignore`]
- Committed on `main` (matching the precedent set by NEU-DET and
  ESTM).
