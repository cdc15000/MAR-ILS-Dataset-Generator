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


def detect_metal_mask(hu: np.ndarray, thresh: float = METAL_HU_THRESH) -> np.ndarray:
    """Boolean metal mask from a noMAR HU image (textbook threshold detection)."""
    return hu > thresh


def metal_trace_weights(
    metal_mask: np.ndarray, clean_frac: float = CLEAN_RAY_FRAC
) -> np.ndarray:
    """Forward-project the metal mask through fan-beam geometry, then mark each
    ray clean (1.0) where its metal path length is below ``clean_frac`` of the
    per-view peak, else metal-traced (0.0)."""
    trace = forward_project_slice(metal_mask.astype(np.float64))
    peak = trace.max(axis=1, keepdims=True)
    return (trace < clean_frac * np.maximum(peak, 1e-9)).astype(np.float64)


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


def discover_realizations(dataset_dir, cond: str = "LP") -> int:
    """Count realization_*.h5 files under <dataset_dir>/sinograms/<cond>/."""
    d = Path(dataset_dir) / "sinograms" / cond
    return len(list(d.glob("realization_*.h5")))
