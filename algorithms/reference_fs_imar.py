#!/usr/bin/env python3
"""
reference_fs_imar.py
====================
Frequency-Split iMAR (FS-iMAR) — detail-preserving metal artifact reduction.
ASTM WKXXXXX v1.0.0 — Tier-aware.

Motivation
----------
Standard iMAR (reference_imar.py) replaces metal-trace sinogram angles with
an analytic seamless prior, then reconstructs with FBP.  The prior is a
Gaussian-blurred (σ=1.0) analytic body ellipse — intentionally smooth.  For
large lesions (T1_AB: 6×4 vox) the lesion is well above the smoothing scale
and iMAR achieves NON-INFERIOR ΔAUC.  For tight-geometry tiers with small
lesions (T2_SB: 4×3 vox, gap=3 vox; T3_HEAD: 3×2 vox), the same smoothing
attenuates the lesion signal → INDETERMINATE.

Algorithm
---------
  1. Correction layer  x_low   = iMAR(sino_meas, ref_hu, tier)
     Standard seamless-prior reconstruction.  Streak-free but over-smooth
     in the vicinity of the lesion.

  2. Detail layer      hf      = noMAR_masked − LP(noMAR_masked, σ_hp)
     High-pass residual of the noMAR image.  Captures the sharp lesion disc
     (a 3–6 vox feature → high-frequency) without capturing the broad HU
     depression/elevation streaks (low-frequency), which the iMAR base already
     corrects.  The noMAR image is metal-masked before the HP filter to prevent
     the 3000 HU step from creating a ring artifact in the detail layer.

  3. Recombination     x_fs    = x_low + hf
     Streak-free base with original fine spatial detail restored.

  4. Mean-shift        x_fs   += BACKGROUND_HU − mean(x_fs[body & ~metal])
     Anchors tissue background exactly at 40 HU (§10.1.1).

  5. Metal restore     x_fs[metal_mask] = 3000 HU

Metal masking before HF extraction
------------------------------------
  ref_masked = ref_hu.copy()
  ref_masked[metal_mask] = BACKGROUND_HU      # replace 3000 HU → 40 HU
  lp = gaussian_filter(ref_masked, σ_hp)      # smooth the masked image
  hf = ref_masked − lp                        # high-frequency residual ≈ 0 at metal

  Setting metal pixels to BACKGROUND_HU eliminates the sharp 3000 HU step
  from the HP filter input.  The Gaussian blur then produces a smooth transition
  at the metal boundary, so hf ≈ 0 inside and just outside the metal mask.
  The metal voxels are restored to 3000 HU in step 5 regardless.

Outputs
-------
  <input_dir>/fs_imar_recon/LP/realization_NNN/slice_0129.dcm  (N=1..40)
  <input_dir>/fs_imar_recon/LA/realization_NNN/slice_0129.dcm  (N=1..40)

Tunable parameter
-----------------
  --gaussian-sigma  (default 1.5 vox)
  Larger σ → more low-frequency content removed from noMAR before adding back
  as detail, i.e. coarser detail layer.  Smaller σ → detail layer contains
  more medium-frequency content (larger streak features may contaminate).
  σ = 1.5 vox is tuned so that the 12 HU 4×3 vox lesion disc (characteristic
  spatial scale ≈ 4 vox) falls entirely in the detail layer while streak bands
  (spatial scale ≫ 10 vox) remain in the corrected iMAR base.
"""

import argparse
import copy
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import gaussian_filter

from reference_imar import imar_slice
from tier_config import (
    BACKGROUND_HU,
    TIER_REGISTRY,
    TierConfig,
)

# ---------------------------------------------------------------------------
# Normative constants (tier-independent)
# ---------------------------------------------------------------------------
LESION_SLICE_INDEX = 128
N_REALIZATIONS     = 40
METAL_HU_THRESH    = 1500     # threshold for metal identification in noMAR HU image
METAL_HU           = 3000.0   # restored HU for metal voxels (§A1.3(d,f))

DEFAULT_GAUSSIAN_SIGMA = 1.5  # HP filter sigma (vox); tuned for 4×3 vox lesion


# ---------------------------------------------------------------------------
# FS-iMAR core (single 2D slice)
# ---------------------------------------------------------------------------

def fs_imar_slice(
    sino_meas:      np.ndarray,
    ref_hu:         np.ndarray,
    tier:           TierConfig,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> np.ndarray:
    """
    Parameters
    ----------
    sino_meas      : (360, 512) float64  measured line integrals (neper)
    ref_hu         : (512, 512) float64  noMAR FBP image (DICOM HU)
    tier           : TierConfig          tier geometry for body ellipse and prior
    gaussian_sigma : float               HP filter Gaussian σ (vox)

    Returns
    -------
    img_fs : (512, 512) float64  FS-iMAR-corrected image in HU
    """
    metal_mask = ref_hu > METAL_HU_THRESH

    # ── 1. Correction layer: standard iMAR (streak-free, smooth) ──────────
    x_low = imar_slice(sino_meas, ref_hu, tier)

    # ── 2. Detail layer: high-pass residual of noMAR ───────────────────────
    #    Replace metal voxels with BACKGROUND_HU to suppress the 3000 HU
    #    step function before Gaussian blur, preventing a ring artifact
    #    in hf at the metal boundary.
    ref_masked = ref_hu.astype(np.float64).copy()
    ref_masked[metal_mask] = BACKGROUND_HU

    lp_noMAR = gaussian_filter(ref_masked, sigma=gaussian_sigma)
    hf_detail = ref_masked - lp_noMAR   # ≈ lesion disc + tissue microstructure

    # ── 3. Recombination ───────────────────────────────────────────────────
    combined = x_low + hf_detail

    # ── 4. Mean-shift: anchor body background at BACKGROUND_HU ────────────
    _rows, _cols = np.mgrid[0:512, 0:512]
    body_ellipse = (
        (_cols - 256.0) ** 2 / float(tier.body_semi_x_vox ** 2) +
        (_rows - 256.0) ** 2 / float(tier.body_semi_y_vox ** 2)
    ) < 1.0
    body_mask = body_ellipse & ~metal_mask
    body_mean = float(np.mean(combined[body_mask]))
    combined  = combined + (BACKGROUND_HU - body_mean)

    # ── 5. Restore metal ───────────────────────────────────────────────────
    combined[metal_mask] = METAL_HU

    return combined


# ---------------------------------------------------------------------------
# DICOM I/O helpers
# ---------------------------------------------------------------------------

def _load_noMAR_dicom(dcm_path: Path) -> tuple:
    dcm = pydicom.dcmread(str(dcm_path))
    hu  = dcm.pixel_array.astype(float) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
    return hu, dcm


def _save_fs_imar_dicom(img_hu: np.ndarray, template_dcm, out_path: Path) -> None:
    """Save FS-iMAR image as DICOM by cloning the noMAR template."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dcm = copy.deepcopy(template_dcm)

    slope     = float(dcm.RescaleSlope)
    intercept = float(dcm.RescaleIntercept)
    pixel_arr = np.clip(
        np.round((img_hu - intercept) / slope), -32768, 32767
    ).astype(np.int16)
    dcm.PixelData = pixel_arr.tobytes()

    dcm.SeriesInstanceUID = generate_uid()
    dcm.SOPInstanceUID    = generate_uid()
    dcm.SeriesDescription = f"FS-iMAR (σ={DEFAULT_GAUSSIAN_SIGMA:.1f} vox)"

    dcm.save_as(str(out_path))


# ---------------------------------------------------------------------------
# Worker (one realization × one condition)
# ---------------------------------------------------------------------------

def _process_one(args):
    """Worker function executed in subprocess."""
    cond, real_idx, dataset_dir, output_dir, tier_id, gaussian_sigma = args
    dataset_dir = Path(dataset_dir)
    output_dir  = Path(output_dir)
    tier        = TIER_REGISTRY[tier_id]

    tag     = f"realization_{real_idx:03d}"
    h5_path = dataset_dir / "sinograms" / cond / f"{tag}.h5"
    dcm_src = dataset_dir / "noMAR_recon" / cond / tag / "slice_0129.dcm"
    dcm_dst = output_dir  / cond / tag / "slice_0129.dcm"

    if dcm_dst.exists():
        return cond, real_idx, True

    with h5py.File(str(h5_path), "r") as f:
        sino = f["line_integrals"][LESION_SLICE_INDEX].astype(np.float64)

    ref_hu, template_dcm = _load_noMAR_dicom(dcm_src)
    img_fs = fs_imar_slice(sino, ref_hu, tier, gaussian_sigma=gaussian_sigma)
    _save_fs_imar_dicom(img_fs, template_dcm, dcm_dst)

    return cond, real_idx, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    warnings.warn(
        "This algorithm uses parallel-beam geometry (Rev 03). "
        "It is NOT compatible with v7.0.0 fan-beam datasets.",
        DeprecationWarning, stacklevel=2,
    )
    ap = argparse.ArgumentParser(
        description="FS-iMAR benchmark for ASTM WKXXXXX v1.0.0 — tier-aware")
    ap.add_argument("--input-dir", default="./astm_reference_dataset",
                    help="Root of the tier dataset (default: ./astm_reference_dataset)")
    ap.add_argument("--tier", default="T1_AB",
                    choices=list(TIER_REGISTRY.keys()),
                    help="Imaging scenario tier (default: T1_AB)")
    ap.add_argument("--gaussian-sigma", type=float, default=DEFAULT_GAUSSIAN_SIGMA,
                    help=f"HP filter Gaussian σ in voxels (default: {DEFAULT_GAUSSIAN_SIGMA})")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0 = os.cpu_count())")
    args = ap.parse_args()

    tier        = TIER_REGISTRY[args.tier]
    dataset_dir = Path(args.input_dir).resolve()
    output_dir  = dataset_dir / "fs_imar_recon"
    n_workers   = args.workers if args.workers > 0 else os.cpu_count()

    print(f"Tier       : {tier.tier_id}  —  {tier.description}")
    print(f"Body       : {tier.body_semi_x_vox}×{tier.body_semi_y_vox} vox  "
          f"({'circle' if tier.is_circular_body else 'ellipse'})")
    print(f"Metal      : {tier.metal_material}  μ={tier.metal_mu_cm} cm⁻¹  "
          f"r={tier.metal_radius_vox} vox  blockage={tier.blockage_frac*100:.1f}%")
    print(f"Lesion     : {tier.lesion_semi_major_vox}×{tier.lesion_semi_minor_vox} vox  "
          f"gap={tier.gap_vox} vox ({tier.gap_mm:.1f} mm)")
    print(f"HP σ       : {args.gaussian_sigma:.1f} vox  "
          f"(lesion scale ≈ {tier.lesion_semi_major_vox} vox)")
    print(f"Dataset    : {dataset_dir}")
    print(f"Output     : {output_dir}")
    print(f"Workers    : {n_workers}  |  Algorithm: FS-iMAR (iMAR base + HF detail)")

    tasks = [
        (cond, r, str(dataset_dir), str(output_dir), args.tier, args.gaussian_sigma)
        for cond in ("LP", "LA")
        for r in range(1, N_REALIZATIONS + 1)
    ]

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_process_one, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                cond, ridx, skipped = fut.result()
                done += 1
                status = "skip" if skipped else "done"
                print(f"  [{done:3d}/80] {cond}/realization_{ridx:03d} {status}",
                      flush=True)
            except Exception as exc:
                t = futs[fut]
                print(f"  ERROR {t[0]}/realization_{t[1]:03d}: {exc}")

    print(f"\nFS-iMAR output → {output_dir}")
    print("Next step:")
    print(f"  python run_cho_analysis_v6_0.py \\")
    print(f"      --dataset-dir {dataset_dir} \\")
    print(f"      --mar-output-dir {output_dir} \\")
    print(f"      --tier {args.tier} \\")
    print(f"      --internal-noise-sigma 15 \\")
    print(f"      --noise-sigma-sweep \\")
    print(f"      --results-file fs_imar_{args.tier.lower()}_results.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
