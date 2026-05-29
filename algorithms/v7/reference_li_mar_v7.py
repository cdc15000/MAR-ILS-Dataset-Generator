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
