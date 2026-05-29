#!/usr/bin/env python3
"""
apply_mar_sir.py
Linear Interpolation MAR (LI-MAR) with sinogram-domain metal masking.

Algorithm
---------
1. Metal ray detection  : Forward-project the noMAR metal mask → weight matrix W
                          (W=1 clean ray, W=0 metal-contaminated ray)
2. Linear interpolation : Fill metal trace in sinogram along detector axis
3. FBP                  : scikit-image iradon with Ram-Lak filter
4. HU calibration       : Match soft-tissue mean/std to the noMAR reference image
5. Metal restore        : Set metal voxels to 3000 HU
6. Output               : DICOM slice_0129.dcm per realization per condition

Note: POCS+TV iterative refinement was removed because scatter saturation
(5% scatter floor → -log(0.05)≈3 nepers) makes the sinogram scale ambiguous,
causing the POCS loop to amplify rather than reduce streak artifacts.

Outputs
-------
  <input_dir>/sir_mar_recon/LP/realization_NNN/slice_0129.dcm  (N=1..40)
  <input_dir>/sir_mar_recon/LA/realization_NNN/slice_0129.dcm  (N=1..40)
"""

import argparse
import copy
import os
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from skimage.transform import iradon, radon

# ---------------------------------------------------------------------------
# Normative constants (ASTM WKXXXXX Rev 03)
# ---------------------------------------------------------------------------
LESION_SLICE_INDEX = 128          # 0-indexed; DICOM slice_0129.dcm
N_ANGLES           = 360
N_DETECTORS        = 512
THETA_DEG          = np.linspace(0, 180, N_ANGLES, endpoint=False)
METAL_HU_THRESH    = 1500         # HU threshold to identify metal in noMAR image
METAL_HU           = 3000         # restore metal to this HU (§A1.3(d,f))
N_REALIZATIONS     = 40


# ---------------------------------------------------------------------------
# Core WLS-TV SIR-MAR (single 2D slice)
# ---------------------------------------------------------------------------

def _linear_interp_metal(sino: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Linear interpolation along detector axis to fill metal-traced rays."""
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


def _fbp(sino: np.ndarray) -> np.ndarray:
    """FBP via Ram-Lak filter. sino (360,512) → img (512,512)."""
    return iradon(sino.T, theta=THETA_DEG, filter_name="ramp", circle=True)


def _fwd(img: np.ndarray) -> np.ndarray:
    """Radon forward projection. img (512,512) → sino (360,512)."""
    return radon(img, theta=THETA_DEG, circle=True).T


def sir_mar_slice(sino_meas: np.ndarray, ref_hu: np.ndarray) -> tuple:
    """
    Parameters
    ----------
    sino_meas : (360, 512) float64 – measured line integrals (neper)
    ref_hu    : (512, 512) float64 – noMAR FBP image (DICOM HU)

    Returns
    -------
    img_li  : (512, 512) float64  LI-MAR corrected image in HU
    costs   : list of float  (empty — no iterative solver)
    """
    # ── 1. Metal mask from noMAR image ───────────────────────────────────
    metal_mask = ref_hu > METAL_HU_THRESH

    # ── 2. Build sinogram weight matrix via forward projection of metal ──
    metal_sino = _fwd(metal_mask.astype(float))
    # Threshold: any ray carrying ≥ 5% of the per-angle metal projection peak
    per_angle_peak = metal_sino.max(axis=1, keepdims=True)
    W = (metal_sino < 0.05 * np.maximum(per_angle_peak, 1e-6)).astype(float)

    # ── 3. Linear interpolation to fill metal trace ───────────────────────
    sino_LI = _linear_interp_metal(sino_meas, W)

    # ── 4. FBP reconstruction ─────────────────────────────────────────────
    x = _fbp(sino_LI)

    # ── 5. HU calibration: mean-shift only (no std rescaling) ────────────
    # Std-rescaling would amplify noise when MAR removes streak variance.
    body_mask = (ref_hu > -800) & (~metal_mask)
    ref_vals = ref_hu[body_mask]
    x_cal = x + (np.mean(ref_vals) - np.mean(x[body_mask]))

    # ── 6. Restore metal voxels ───────────────────────────────────────────
    x_cal[metal_mask] = float(METAL_HU)

    return x_cal, []


# ---------------------------------------------------------------------------
# DICOM I/O helpers
# ---------------------------------------------------------------------------

def _load_noMAR_dicom(dcm_path: Path) -> tuple[np.ndarray, object]:
    dcm = pydicom.dcmread(str(dcm_path))
    hu = dcm.pixel_array.astype(float) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
    return hu, dcm


def _save_sir_dicom(img_hu: np.ndarray, template_dcm, out_path: Path) -> None:
    """Save SIR-MAR image as DICOM by cloning the noMAR template."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dcm = copy.deepcopy(template_dcm)

    # Pixel array as int16 (RescaleSlope=1, RescaleIntercept=0 assumed from template)
    slope  = float(dcm.RescaleSlope)
    intercept = float(dcm.RescaleIntercept)
    pixel_hu = (img_hu - intercept) / slope
    pixel_hu = np.clip(np.round(pixel_hu), -32768, 32767).astype(np.int16)
    dcm.PixelData = pixel_hu.tobytes()

    # Update UIDs so PACS treats this as a new series
    dcm.SeriesInstanceUID = generate_uid()
    dcm.SOPInstanceUID    = generate_uid()
    dcm.SeriesDescription = "SIR-MAR WLS-TV"

    dcm.save_as(str(out_path))


# ---------------------------------------------------------------------------
# Worker (one realization × one condition)
# ---------------------------------------------------------------------------

def _process_one(args):
    """Worker function executed in subprocess."""
    cond, real_idx, dataset_dir, output_dir = args
    dataset_dir = Path(dataset_dir)
    output_dir  = Path(output_dir)

    tag     = f"realization_{real_idx:03d}"
    h5_path = dataset_dir / "sinograms" / cond / f"{tag}.h5"
    dcm_src = dataset_dir / "noMAR_recon" / cond / tag / "slice_0129.dcm"
    dcm_dst = output_dir  / cond / tag / "slice_0129.dcm"

    if dcm_dst.exists():
        return cond, real_idx, None, True   # already done → skip

    # Load sinogram slice 128
    with h5py.File(str(h5_path), "r") as f:
        sino = f["line_integrals"][LESION_SLICE_INDEX].astype(np.float64)

    # Load noMAR reference image (HU + DICOM template)
    ref_hu, template_dcm = _load_noMAR_dicom(dcm_src)

    # Run WLS-TV SIR-MAR
    img_sir, costs = sir_mar_slice(sino, ref_hu)

    # Write DICOM
    _save_sir_dicom(img_sir, template_dcm, dcm_dst)

    return cond, real_idx, costs, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="LI-MAR benchmark for ASTM WKXXXXX Rev 03 dataset")
    ap.add_argument("--input-dir",  default="./astm_reference_dataset",
                    help="Root of the reference dataset  (default: ./astm_reference_dataset)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers  (0 = os.cpu_count())")
    args = ap.parse_args()

    dataset_dir = Path(args.input_dir).resolve()
    output_dir  = dataset_dir / "sir_mar_recon"
    n_workers   = args.workers if args.workers > 0 else os.cpu_count()

    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {output_dir}")
    print(f"Workers : {n_workers}  |  Algorithm: LI-MAR (linear interpolation + FBP)")

    # Build task list: 40 LP + 40 LA
    tasks = [
        (cond, r, str(dataset_dir), str(output_dir))
        for cond in ("LP", "LA")
        for r in range(1, N_REALIZATIONS + 1)
    ]

    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_process_one, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                cond, ridx, _costs, skipped = fut.result()
                done += 1
                status = "skip" if skipped else "done"
                print(f"  [{done:3d}/80] {cond}/realization_{ridx:03d} {status}", flush=True)
            except Exception as exc:
                t = futs[fut]
                print(f"  ERROR {t[0]}/realization_{t[1]:03d}: {exc}")


    print(f"\nSIR-MAR output → {output_dir}")
    print("Next step:")
    print("  python run_cho_analysis_v5_3.py \\")
    print(f"      --dataset-dir {dataset_dir} \\")
    print(f"      --mar-output-dir {output_dir} \\")
    print("      --internal-noise-sigma 15 \\")
    print("      --results-file mar_benchmark_results.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
