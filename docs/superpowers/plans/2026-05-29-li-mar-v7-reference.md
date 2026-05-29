# LI-MAR v7 Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parameter-free linear-interpolation MAR (LI-MAR) on the v7.0.0 fan-beam dataset that produces a CHO-ready `slice_0129.dcm` per realization — a positive control and reproducible ΔAUC anchor (non-normative).

**Architecture:** A single self-contained script `algorithms/v7/reference_li_mar_v7.py` with small pure helpers, importing the validated fan-beam `forward_project_slice` / `fbp_reconstruct_slice` from `generator_v7_0_0` (the locked physics is reused, never modified). Metal is detected by thresholding the existing noMAR reconstruction, its sinogram trace is filled by linear interpolation along the detector axis, and the result is reconstructed with fan-beam FBP and written via `mar_ils_core.dicom_utils.write_dicom_slice`.

**Tech Stack:** Python 3.10, NumPy, h5py, pydicom, tqdm; reuses `generator_v7_0_0` and `mar_ils_core`. Tests: pytest.

---

## File Structure

- **Create** `algorithms/v7/reference_li_mar_v7.py` — the algorithm + CLI (all functions live here).
- **Create** `tests/test_li_mar_v7.py` — unit + integration tests.
- **Create** `algorithms/v7/README.md` — purpose, usage, ΔAUC anchor placeholder.
- **Modify** `algorithms/README.md` — add a status-table row marking this the one fan-beam-compatible reference.

**Reused interfaces (do not modify):**
- `generator_v7_0_0.forward_project_slice(mu: ndarray[512,512]) -> ndarray[720,512]`
- `generator_v7_0_0.fbp_reconstruct_slice(sino: ndarray[720,512], dc_offset_cm: float=0.0) -> ndarray[512,512]` (HU)
- `mar_ils_core.dicom_utils.write_dicom_slice(hu, z, *, output_dir, realization_idx, condition_label, study_uid, series_uid, metal_mask, ...)` — writes `slice_{z+1:04d}.dcm`, hard-sets `metal_mask` pixels to 3000 HU.
- `mar_ils_core.constants`: `X_DIM`, `Y_DIM`, `LESION_SLICE_INDEX` (128), `METAL_HU`.
- `mar_ils_core.phantom.build_attenuation_map(place_lesion, jitter_deg)`, `build_metal_mask(yy, xx)` — test fixtures only.

**Note on imports:** the module runs from `algorithms/v7/`, so it prepends the repo root to `sys.path` (three levels up) before importing `generator_v7_0_0` / `mar_ils_core`. Tests prepend `algorithms/v7/` to `sys.path` to import the module by name.

---

### Task 1: Module skeleton + `linear_interp_metal`

**Files:**
- Create: `algorithms/v7/reference_li_mar_v7.py`
- Test: `tests/test_li_mar_v7.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_li_mar_v7.py
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "algorithms" / "v7"))
import reference_li_mar_v7 as li  # noqa: E402


class TestLinearInterpMetal:
    def test_fills_metal_columns_linearly(self):
        # one view (row), detector bins 0..9; bins 4,5 are metal (W<0.5)
        sino = np.array([[0.0, 1.0, 2.0, 3.0, 99.0, 99.0, 6.0, 7.0, 8.0, 9.0]])
        W = np.ones_like(sino)
        W[0, 4] = 0.0
        W[0, 5] = 0.0
        out = li.linear_interp_metal(sino, W)
        # bins 4,5 linearly interpolated between clean neighbours 3.0 and 6.0
        assert out[0, 4] == pytest.approx(4.0)
        assert out[0, 5] == pytest.approx(5.0)

    def test_metal_free_view_unchanged(self):
        sino = np.array([[0.0, 1.0, 2.0, 3.0]])
        W = np.ones_like(sino)  # all clean
        out = li.linear_interp_metal(sino, W)
        assert np.array_equal(out, sino)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_li_mar_v7.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'reference_li_mar_v7'`.

- [ ] **Step 3: Write minimal implementation**

```python
# algorithms/v7/reference_li_mar_v7.py
"""
LI-MAR v7 reference — parameter-free linear-interpolation Metal Artifact
Reduction on the v7.0.0 fan-beam dataset.

Non-normative. Serves as (a) a positive control for the CHO pipeline and
(b) a reproducible delta-AUC anchor. The ASTM type test measures the *lab's*
algorithm; this reference is for validation and calibration only.

Pipeline (per realization, slice 128 / slice_0129.dcm only):
  1. metal_mask  = noMAR_HU > METAL_HU_THRESH       (textbook detection)
  2. metal_trace = forward_project_slice(metal_mask) -> clean-ray weights W
  3. sino_li     = linear interpolation across the metal trace, per view
  4. hu_corr     = fbp_reconstruct_slice(sino_li, dc_offset_cm)
  5. write slice_0129.dcm (metal hard-set to 3000 HU, as in noMAR)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import pydicom
from pydicom.uid import generate_uid
from tqdm import tqdm

# Run from algorithms/v7/ -> prepend repo root so the shared modules import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from generator_v7_0_0 import forward_project_slice, fbp_reconstruct_slice  # noqa: E402
from mar_ils_core.dicom_utils import write_dicom_slice  # noqa: E402
from mar_ils_core.constants import LESION_SLICE_INDEX, X_DIM, Y_DIM  # noqa: E402

# Metal reconstructs to 3000 HU (hard-set) in the noMAR series; 2000 HU is a
# robust detection floor.
METAL_HU_THRESH = 2000.0
# Clean-ray fraction (matches the v6 references' threshold).
CLEAN_RAY_FRAC = 0.05


def linear_interp_metal(sino: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Linear interpolation along the detector axis to fill metal-traced rays.

    W convention: >= 0.5 is a clean ray, < 0.5 is a metal-traced ray.
    Ported from algorithms/reference_nmar.py (geometry-agnostic).
    """
    sino_out = sino.copy()
    for a in range(sino.shape[0]):
        metal = np.where(W[a] < 0.5)[0]
        if metal.size == 0:
            continue
        clean = np.where(W[a] >= 0.5)[0]
        if clean.size < 2:
            continue
        sino_out[a, metal] = np.interp(metal, clean, sino[a, clean])
    return sino_out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_li_mar_v7.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algorithms/v7/reference_li_mar_v7.py tests/test_li_mar_v7.py
git commit -m "feat: LI-MAR v7 module skeleton + linear_interp_metal"
```

---

### Task 2: `detect_metal_mask`

**Files:**
- Modify: `algorithms/v7/reference_li_mar_v7.py`
- Test: `tests/test_li_mar_v7.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_li_mar_v7.py
class TestDetectMetalMask:
    def test_finds_high_hu_disc(self):
        hu = np.full((512, 512), 40.0)
        yy, xx = np.mgrid[0:512, 0:512]
        disc = (xx - 256) ** 2 + (yy - 256) ** 2 <= 10 ** 2
        hu[disc] = 3000.0
        mask = li.detect_metal_mask(hu)
        assert np.array_equal(mask, disc)

    def test_threshold_is_configurable(self):
        hu = np.full((4, 4), 1500.0)
        assert li.detect_metal_mask(hu, thresh=1000.0).all()
        assert not li.detect_metal_mask(hu, thresh=2000.0).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_li_mar_v7.py::TestDetectMetalMask -q`
Expected: FAIL — `AttributeError: module 'reference_li_mar_v7' has no attribute 'detect_metal_mask'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to algorithms/v7/reference_li_mar_v7.py (after linear_interp_metal)
def detect_metal_mask(hu: np.ndarray, thresh: float = METAL_HU_THRESH) -> np.ndarray:
    """Boolean metal mask from a noMAR HU image (textbook threshold detection)."""
    return hu > thresh
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_li_mar_v7.py::TestDetectMetalMask -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algorithms/v7/reference_li_mar_v7.py tests/test_li_mar_v7.py
git commit -m "feat: detect_metal_mask (threshold-based metal detection)"
```

---

### Task 3: `metal_trace_weights`

**Files:**
- Modify: `algorithms/v7/reference_li_mar_v7.py`
- Test: `tests/test_li_mar_v7.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_li_mar_v7.py
from mar_ils_core.phantom import build_metal_mask  # noqa: E402


class TestMetalTraceWeights:
    def test_centred_rod_flags_central_detector_bins(self):
        yy, xx = np.mgrid[0:512, 0:512]
        metal_mask = build_metal_mask(yy, xx)  # centred 10-vox iron rod
        W = li.metal_trace_weights(metal_mask)
        assert W.shape == (720, 512)            # (N_ANGLES, N_DET)
        assert ((W == 0.0) | (W == 1.0)).all()  # binary
        # A centred rod projects to the central detector region for every view,
        # leaving the detector edges clean.
        assert (W < 0.5).any()                   # some metal-traced rays
        assert W[0, 0] == 1.0 and W[0, -1] == 1.0  # edges clean
        # Metal-traced bins cluster near detector centre.
        metal_cols = np.where(W[0] < 0.5)[0]
        assert abs(metal_cols.mean() - 256) < 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_li_mar_v7.py::TestMetalTraceWeights -q`
Expected: FAIL — `AttributeError: ... has no attribute 'metal_trace_weights'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to algorithms/v7/reference_li_mar_v7.py (after detect_metal_mask)
def metal_trace_weights(
    metal_mask: np.ndarray, clean_frac: float = CLEAN_RAY_FRAC
) -> np.ndarray:
    """Forward-project the metal mask through fan-beam geometry, then mark each
    ray clean (1.0) where its metal path length is below ``clean_frac`` of the
    per-view peak, else metal-traced (0.0)."""
    trace = forward_project_slice(metal_mask.astype(np.float64))
    peak = trace.max(axis=1, keepdims=True)
    return (trace < clean_frac * np.maximum(peak, 1e-9)).astype(np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_li_mar_v7.py::TestMetalTraceWeights -q`
Expected: PASS (1 passed). First run triggers numba JIT compile (~5 s).

- [ ] **Step 5: Commit**

```bash
git add algorithms/v7/reference_li_mar_v7.py tests/test_li_mar_v7.py
git commit -m "feat: metal_trace_weights (fan-beam metal trace -> clean-ray mask)"
```

---

### Task 4: `li_mar_slice` (per-slice orchestration)

**Files:**
- Modify: `algorithms/v7/reference_li_mar_v7.py`
- Test: `tests/test_li_mar_v7.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_li_mar_v7.py
from mar_ils_core.phantom import build_attenuation_map  # noqa: E402


class TestLiMarSlice:
    @pytest.fixture(scope="class")
    def slice_inputs(self):
        from generator_v7_0_0 import forward_project_slice, fbp_reconstruct_slice
        mu = build_attenuation_map(place_lesion=True, jitter_deg=0.0)
        sino = forward_project_slice(mu).astype(np.float64)
        nomar_hu = fbp_reconstruct_slice(sino, dc_offset_cm=0.0)
        return sino, nomar_hu

    def test_returns_finite_corrected_slice_and_mask(self, slice_inputs):
        sino, nomar_hu = slice_inputs
        hu_corr, metal_mask = li.li_mar_slice(sino, nomar_hu, dc_offset_cm=0.0)
        assert hu_corr.shape == (512, 512)
        assert np.isfinite(hu_corr).all()
        assert metal_mask.dtype == bool
        assert metal_mask.sum() > 0          # rod detected
        assert metal_mask.sum() < 2000       # but not the whole image

    def test_metal_trace_is_replaced(self, slice_inputs):
        # After LI, the metal region reconstructs to ~tissue, far below the
        # ~10000 HU raw metal in the noMAR image. Confirms the trace was filled.
        sino, nomar_hu = slice_inputs
        hu_corr, metal_mask = li.li_mar_slice(sino, nomar_hu, dc_offset_cm=0.0)
        assert hu_corr[metal_mask].mean() < nomar_hu[metal_mask].mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_li_mar_v7.py::TestLiMarSlice -q`
Expected: FAIL — `AttributeError: ... has no attribute 'li_mar_slice'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to algorithms/v7/reference_li_mar_v7.py (after metal_trace_weights)
def li_mar_slice(
    sino: np.ndarray,
    nomar_hu: np.ndarray,
    dc_offset_cm: float,
    thresh: float = METAL_HU_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """Run LI-MAR on one fan-beam slice.

    Returns (hu_corrected, metal_mask). Metal is NOT hard-set here; the writer
    applies the 3000 HU hard-set via metal_mask.
    """
    metal_mask = detect_metal_mask(nomar_hu, thresh)
    W = metal_trace_weights(metal_mask)
    sino_li = linear_interp_metal(np.asarray(sino, dtype=np.float64), W)
    hu_corr = fbp_reconstruct_slice(sino_li, dc_offset_cm=dc_offset_cm)
    return hu_corr, metal_mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_li_mar_v7.py::TestLiMarSlice -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algorithms/v7/reference_li_mar_v7.py tests/test_li_mar_v7.py
git commit -m "feat: li_mar_slice end-to-end per-slice correction"
```

---

### Task 5: `load_dc_offset`

**Files:**
- Modify: `algorithms/v7/reference_li_mar_v7.py`
- Test: `tests/test_li_mar_v7.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_li_mar_v7.py
import json as _json


class TestLoadDcOffset:
    def test_reads_value(self, tmp_path):
        (tmp_path / "generator_provenance.json").write_text(
            _json.dumps({"dc_offset_cm": -0.029})
        )
        assert li.load_dc_offset(tmp_path) == pytest.approx(-0.029)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            li.load_dc_offset(tmp_path)

    def test_missing_key_raises(self, tmp_path):
        (tmp_path / "generator_provenance.json").write_text(_json.dumps({"x": 1}))
        with pytest.raises(KeyError):
            li.load_dc_offset(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_li_mar_v7.py::TestLoadDcOffset -q`
Expected: FAIL — `AttributeError: ... has no attribute 'load_dc_offset'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to algorithms/v7/reference_li_mar_v7.py (after li_mar_slice)
def load_dc_offset(dataset_dir) -> float:
    """Read dc_offset_cm from <dataset_dir>/generator_provenance.json.

    Raises rather than silently defaulting to 0.0 (a wrong offset mis-calibrates
    HU by ~141 HU). Callers may bypass this with an explicit --dc-offset-cm.
    """
    prov = Path(dataset_dir) / "generator_provenance.json"
    if not prov.exists():
        raise FileNotFoundError(
            f"{prov} not found; pass --dc-offset-cm to supply it explicitly"
        )
    data = json.loads(prov.read_text())
    if "dc_offset_cm" not in data:
        raise KeyError(
            "dc_offset_cm missing from provenance; pass --dc-offset-cm explicitly"
        )
    return float(data["dc_offset_cm"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_li_mar_v7.py::TestLoadDcOffset -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add algorithms/v7/reference_li_mar_v7.py tests/test_li_mar_v7.py
git commit -m "feat: load_dc_offset from generator provenance"
```

---

### Task 6: `discover_realizations`

**Files:**
- Modify: `algorithms/v7/reference_li_mar_v7.py`
- Test: `tests/test_li_mar_v7.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_li_mar_v7.py
class TestDiscoverRealizations:
    def test_counts_h5_files(self, tmp_path):
        d = tmp_path / "sinograms" / "LP"
        d.mkdir(parents=True)
        for n in (1, 2, 3):
            (d / f"realization_{n:03d}.h5").touch()
        assert li.discover_realizations(tmp_path, "LP") == 3

    def test_zero_when_absent(self, tmp_path):
        assert li.discover_realizations(tmp_path, "LP") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_li_mar_v7.py::TestDiscoverRealizations -q`
Expected: FAIL — `AttributeError: ... has no attribute 'discover_realizations'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to algorithms/v7/reference_li_mar_v7.py (after load_dc_offset)
def discover_realizations(dataset_dir, cond: str = "LP") -> int:
    """Count realization_*.h5 files under <dataset_dir>/sinograms/<cond>/."""
    d = Path(dataset_dir) / "sinograms" / cond
    return len(list(d.glob("realization_*.h5")))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_li_mar_v7.py::TestDiscoverRealizations -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algorithms/v7/reference_li_mar_v7.py tests/test_li_mar_v7.py
git commit -m "feat: discover_realizations helper"
```

---

### Task 7: `process_realization` (I/O integration)

**Files:**
- Modify: `algorithms/v7/reference_li_mar_v7.py`
- Test: `tests/test_li_mar_v7.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_li_mar_v7.py
from mar_ils_core.dicom_utils import write_dicom_slice  # noqa: E402


def _build_tiny_dataset(root: Path):
    """One-slice (slice_index 0) LP realization: sinogram H5 + noMAR DICOM."""
    from generator_v7_0_0 import forward_project_slice, fbp_reconstruct_slice
    import h5py
    import numpy as np
    mu = build_attenuation_map(place_lesion=True, jitter_deg=0.0)
    sino = forward_project_slice(mu).astype(np.float32)        # (720, 512)
    nomar_hu = fbp_reconstruct_slice(sino, dc_offset_cm=0.0)   # (512, 512)

    sdir = root / "sinograms" / "LP"
    sdir.mkdir(parents=True)
    with h5py.File(str(sdir / "realization_001.h5"), "w") as f:
        f.create_dataset("line_integrals", data=sino[np.newaxis, :, :])  # (1,720,512)

    ndir = root / "noMAR_recon" / "LP" / "realization_001"
    yy, xx = np.mgrid[0:512, 0:512]
    from mar_ils_core.phantom import build_metal_mask
    write_dicom_slice(
        nomar_hu, 0, output_dir=ndir, realization_idx=0, condition_label="LP",
        study_uid=generate_uid(), series_uid=generate_uid(),
        metal_mask=build_metal_mask(yy, xx),
    )
    (root / "generator_provenance.json").write_text(_json.dumps({"dc_offset_cm": 0.0}))


class TestProcessRealization:
    def test_writes_readable_corrected_dicom(self, tmp_path):
        _build_tiny_dataset(tmp_path)
        out = tmp_path / "li_mar_recon"
        li.process_realization(
            tmp_path, out, "LP", 0, dc_offset_cm=0.0, slice_index=0,
        )
        dcm = out / "LP" / "realization_001" / "slice_0001.dcm"
        assert dcm.exists()
        ds = pydicom.dcmread(str(dcm))
        hu = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        assert hu.shape == (512, 512)
        # metal hard-set to 3000 HU by the writer. Check the rod *core* (always
        # detected); the geometric boundary can blur below the 2000 HU threshold.
        yy, xx = np.mgrid[0:512, 0:512]
        core = (xx - 256) ** 2 + (yy - 256) ** 2 <= 5 ** 2
        assert hu[256, 256] == pytest.approx(3000.0)
        assert np.allclose(hu[core], 3000.0)

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            li.process_realization(tmp_path, tmp_path / "o", "LP", 0, 0.0, slice_index=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_li_mar_v7.py::TestProcessRealization -q`
Expected: FAIL — `AttributeError: ... has no attribute 'process_realization'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to algorithms/v7/reference_li_mar_v7.py (after discover_realizations)
def process_realization(
    dataset_dir,
    output_dir,
    cond: str,
    idx: int,
    dc_offset_cm: float,
    thresh: float = METAL_HU_THRESH,
    *,
    slice_index: int = LESION_SLICE_INDEX,
) -> None:
    """Read one realization's sinogram + noMAR slice, apply LI-MAR, write DICOM."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    tag = f"realization_{idx + 1:03d}"
    h5_path = dataset_dir / "sinograms" / cond / f"{tag}.h5"
    dcm_path = dataset_dir / "noMAR_recon" / cond / tag / f"slice_{slice_index + 1:04d}.dcm"
    for p in (h5_path, dcm_path):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")

    with h5py.File(str(h5_path), "r") as f:
        sino = f["line_integrals"][slice_index].astype(np.float64)  # (720, 512)

    ds = pydicom.dcmread(str(dcm_path))
    nomar_hu = (
        ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope)
        + float(ds.RescaleIntercept)
    )

    hu_corr, metal_mask = li_mar_slice(sino, nomar_hu, dc_offset_cm, thresh)

    write_dicom_slice(
        hu_corr, slice_index,
        output_dir=output_dir / cond / tag,
        realization_idx=idx,
        condition_label=cond,
        study_uid=generate_uid(),
        series_uid=generate_uid(),
        metal_mask=metal_mask,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_li_mar_v7.py::TestProcessRealization -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algorithms/v7/reference_li_mar_v7.py tests/test_li_mar_v7.py
git commit -m "feat: process_realization I/O glue (read H5 + noMAR, write LI-MAR DICOM)"
```

---

### Task 8: `main()` CLI + docs

**Files:**
- Modify: `algorithms/v7/reference_li_mar_v7.py`
- Create: `algorithms/v7/README.md`
- Modify: `algorithms/README.md`
- Test: `tests/test_li_mar_v7.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_li_mar_v7.py
class TestMain:
    def test_runs_over_tiny_dataset(self, tmp_path):
        _build_tiny_dataset(tmp_path)          # LP only
        out = tmp_path / "li_mar_recon"
        li.main(
            ["--dataset-dir", str(tmp_path), "--output-dir", str(out),
             "--realizations", "1"],
            slice_index=0,
        )
        assert (out / "LP" / "realization_001" / "slice_0001.dcm").exists()

    def test_dc_offset_override_skips_provenance(self, tmp_path):
        _build_tiny_dataset(tmp_path)
        (tmp_path / "generator_provenance.json").unlink()  # force reliance on override
        out = tmp_path / "li_mar_recon"
        li.main(
            ["--dataset-dir", str(tmp_path), "--output-dir", str(out),
             "--realizations", "1", "--dc-offset-cm", "0.0"],
            slice_index=0,
        )
        assert (out / "LP" / "realization_001" / "slice_0001.dcm").exists()
```

Note: `_build_tiny_dataset` creates only an `LP` condition. `main` iterates `("LP", "LA")`; with `--realizations 1` the `LA` loop calls `process_realization` for a missing file. To keep `main` simple and the test honest, `main` skips a condition whose realization count is 0 **and** logs when an explicit `--realizations` is given but inputs are absent. The implementation below skips conditions with no discoverable sinogram directory; the test passes `--realizations 1` and only `LP` exists, so `LA` is skipped because its `sinograms/LA` folder is absent.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_li_mar_v7.py::TestMain -q`
Expected: FAIL — `AttributeError: ... has no attribute 'main'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to algorithms/v7/reference_li_mar_v7.py (after process_realization)
def main(argv=None, *, slice_index: int = LESION_SLICE_INDEX) -> None:
    ap = argparse.ArgumentParser(
        description="LI-MAR v7 reference (non-normative): linear-interpolation "
                    "MAR on the v7.0.0 fan-beam dataset; writes slice_0129.dcm."
    )
    ap.add_argument("--dataset-dir", required=True,
                    help="generator_v7_0_0.py output directory")
    ap.add_argument("--output-dir", default="./li_mar_recon",
                    help="destination for {LP,LA}/realization_NNN/slice_0129.dcm")
    ap.add_argument("--realizations", type=int, default=None,
                    help="count per condition (default: auto-detect)")
    ap.add_argument("--metal-hu-thresh", type=float, default=METAL_HU_THRESH)
    ap.add_argument("--dc-offset-cm", type=float, default=None,
                    help="override; default reads generator_provenance.json")
    args = ap.parse_args(argv)

    dc = (args.dc_offset_cm if args.dc_offset_cm is not None
          else load_dc_offset(args.dataset_dir))

    for cond in ("LP", "LA"):
        sino_dir = Path(args.dataset_dir) / "sinograms" / cond
        if not sino_dir.is_dir():
            continue  # condition not present in this dataset
        n = (args.realizations if args.realizations is not None
             else discover_realizations(args.dataset_dir, cond))
        for i in tqdm(range(n), desc=f"LI-MAR {cond}"):
            process_realization(
                args.dataset_dir, args.output_dir, cond, i, dc,
                args.metal_hu_thresh, slice_index=slice_index,
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_li_mar_v7.py::TestMain -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full new test module**

Run: `python -m pytest tests/test_li_mar_v7.py -q`
Expected: PASS (all tests across Tasks 1-8).

- [ ] **Step 6: Write `algorithms/v7/README.md`**

```markdown
# algorithms/v7/ — fan-beam reference MAR

Reference MAR implementations compatible with the **v7.0.0 fan-beam** dataset
(ASTM Rev 04). Unlike the parallel-beam references in `algorithms/`, these
operate on the (256, 720, 512) fan-beam sinograms and reuse the validated
`forward_project_slice` / `fbp_reconstruct_slice` from `generator_v7_0_0.py`.

## reference_li_mar_v7.py — Linear-Interpolation MAR

Parameter-free LI-MAR. **Non-normative**: the ASTM type test measures the
*lab's* algorithm. This reference exists as a CHO-pipeline positive control and
a reproducible delta-AUC anchor (LI-MAR is the universal baseline every
commercial MAR is expected to beat).

It detects metal by thresholding the existing noMAR reconstruction, fills the
metal trace in the sinogram by linear interpolation along the detector axis,
reconstructs with fan-beam FBP, and writes `slice_0129.dcm` only (what the CHO
reads).

### Usage

```bash
python algorithms/v7/reference_li_mar_v7.py \
    --dataset-dir ./astm_reference_dataset \
    --output-dir  ./li_mar_recon
# then score it:
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./li_mar_recon \
    --internal-noise-sigma 15
```

Options: `--realizations N` (default auto-detect), `--metal-hu-thresh 2000`,
`--dc-offset-cm <float>` (default read from `generator_provenance.json`).

### delta-AUC anchor

Run the two commands above on the locked reference dataset (N=40, sigma=15) and
record the result here:

> delta-AUC (LI-MAR vs noMAR) = _TBD — fill in after the manual anchor run_

This is a manual step (a full dataset is ~35 GB / ~10 min), not part of CI.
```

- [ ] **Step 7: Update `algorithms/README.md` status table**

Add this row to the status table in `algorithms/README.md` (the table whose
header is `| Algorithm | File | Geometry | Compatible with v7.0.0? |`):

```markdown
| LI-MAR | `v7/reference_li_mar_v7.py` | **Fan-beam (720 angles)** | **Yes** |
```

- [ ] **Step 8: Commit**

```bash
git add algorithms/v7/reference_li_mar_v7.py algorithms/v7/README.md \
        algorithms/README.md tests/test_li_mar_v7.py
git commit -m "feat: LI-MAR v7 CLI + docs; mark fan-beam-compatible reference"
```

---

## Final Verification

- [ ] **Run the full suite** (confirm no regressions in the existing 80 tests):

Run: `python -m pytest -q`
Expected: PASS (existing 80 + the new LI-MAR tests).

- [ ] **Lint the new module** (it is under `algorithms/`, outside the current CI lint scope, but keep it clean):

Run: `ruff check algorithms/v7/reference_li_mar_v7.py tests/test_li_mar_v7.py`
Expected: All checks passed.

---

## Self-Review Notes (filled by plan author)

**Spec coverage:** every spec section maps to a task — purpose/decisions (docs in Task 8), data flow (Tasks 2→3→4→7), components table (Tasks 1–8 one function each), CLI (Task 8), error handling (Task 5 dc_offset raise, Task 7 missing-input raise, `linear_interp_metal` clean-ray guard in Task 1), testing split (unit Tasks 1–6, integration Task 7, main smoke Task 8, manual anchor documented in Task 8 README), deliverables (all four files), non-goals (single slice via `slice_index`, no shared layer, no parallelism — honored).

**Anchor not in CI:** intentional, per spec §7 — the delta-AUC anchor needs the full 35 GB dataset; Task 8's README captures it as a manual step with a placeholder.

**Geometry untouched:** no task modifies `generator_v7_0_0.py`, `mar_ils_core/`, or any locked constant — they are imported only.
