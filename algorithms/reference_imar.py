#!/usr/bin/env python3
"""
reference_imar.py
Iterative Metal Artifact Reduction (iMAR) — seamless tissue prior with
sinogram synthesis and Savitzky–Golay continuity enforcement.
ASTM WKXXXXX v1.0.0 — Tier-aware version.

Algorithm
---------
1. Metal mask + weight matrix  : Forward-project noMAR metal mask → W
2. Seamless tissue prior       : Analytic body ellipse (dimensions from
                                  TierConfig) at BACKGROUND_HU = 40 HU.
                                  Metal voxels are set to 40 HU (NOT 3000 HU)
                                  — eliminates the metal signature from the
                                  prior so sino_prior reflects tissue-only
                                  attenuation in the metal trace.
3. Boundary stabilization      : Gaussian blur (σ=1.0 voxel) on the prior
                                  to smooth the body/air boundary before
                                  forward projection.
4. Forward project + scatter   : sino_prior = A·μ_prior; apply 5% scatter
                                  transform: sino_prior_scatter =
                                  -log(exp(-sino_prior) + 0.05)
5. Synthetic sinogram          : Clean rays (W≥0.5) → sino_meas (unchanged)
                                  Metal trace (W<0.5) → sino_prior_scatter
6. SG continuity enforcement   : 1-D Savitzky–Golay filter (window=9,
                                  order=3) applied at each metal-trace
                                  boundary to suppress discontinuity-driven
                                  FBP streak artifacts.
7. Final FBP                   : x = FBP(sino_smooth)
8. HU calibration              : x_hu = (x / (μ_water·voxel_cm) − 1)×1000;
                                  global mean-shift to anchor tissue at 40 HU
9. Restore metal               : x_hu[metal_mask] = 3000 HU

Tier-aware changes (v1.0.0)
---------------------------
The seamless prior body geometry is now pulled from TierConfig:
  T1_AB   → 170×120-vox ellipse  (adult body,   CoCr rod)
  T2_SB   →  85×60-vox  ellipse  (pediatric,    SS316L rod)
  T3_HEAD → 200×200-vox circle   (head phantom, Ti6Al4V rod)

All other algorithm parameters (SG filter, scatter fraction, HU constants,
metal threshold) are tier-independent normative constants.

Outputs
-------
  <input_dir>/imar_recon/LP/realization_NNN/slice_0129.dcm  (N=1..40)
  <input_dir>/imar_recon/LA/realization_NNN/slice_0129.dcm  (N=1..40)
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
from scipy.signal import savgol_filter
from skimage.transform import iradon, radon

from tier_config import (
    BACKGROUND_HU,
    TIER_REGISTRY,
    TierConfig,
    validate_tier_registry,
)

# ---------------------------------------------------------------------------
# Normative constants (ASTM WKXXXXX v1.0.0 — tier-independent)
# ---------------------------------------------------------------------------
LESION_SLICE_INDEX = 128          # 0-indexed; DICOM slice_0129.dcm
N_ANGLES           = 360
N_DETECTORS        = 512
THETA_DEG          = np.linspace(0, 180, N_ANGLES, endpoint=False)
METAL_HU_THRESH    = 1500         # HU threshold to identify metal in noMAR image
METAL_HU           = 3000         # restore metal to this HU (§A1.3(d,f))
N_REALIZATIONS     = 40

# Physical constants
MU_WATER   = 0.2059   # cm⁻¹ at 60 keV (soft tissue attenuation)
VOXEL_CM   = 0.05     # 0.5 mm isotropic voxel → 0.05 cm
SCATTER_FRAC = 0.05   # 5% scatter floor (matches generator scatter_frac)

# Savitzky–Golay filter parameters for sinogram continuity enforcement
SG_WINDOW = 9    # detector-bin window width (must be odd)
SG_ORDER  = 3    # polynomial order


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def _fbp(sino: np.ndarray) -> np.ndarray:
    """FBP via Ram-Lak filter.  sino (360,512) → img (512,512)."""
    return iradon(sino.T, theta=THETA_DEG, filter_name="ramp", circle=True)


def _fwd(img: np.ndarray) -> np.ndarray:
    """Radon forward projection.  img (512,512) → sino (360,512)."""
    return radon(img, theta=THETA_DEG, circle=True).T


# ---------------------------------------------------------------------------
# iMAR core (single 2D slice)
# ---------------------------------------------------------------------------

def imar_slice(sino_meas: np.ndarray, ref_hu: np.ndarray,
               tier: TierConfig) -> np.ndarray:
    """
    Parameters
    ----------
    sino_meas : (360, 512) float64  measured line integrals (neper)
    ref_hu    : (512, 512) float64  noMAR FBP image (DICOM HU)
    tier      : TierConfig          tier geometry for seamless prior

    Returns
    -------
    img_imar : (512, 512) float64  iMAR-corrected image in HU
    """
    # ── 1. Metal mask and weight matrix ──────────────────────────────────
    metal_mask     = ref_hu > METAL_HU_THRESH
    metal_sino     = _fwd(metal_mask.astype(float))
    per_angle_peak = metal_sino.max(axis=1, keepdims=True)
    W = (metal_sino < 0.05 * np.maximum(per_angle_peak, 1e-6)).astype(float)

    # ── 2. Seamless tissue prior (tier-specific body geometry) ────────────
    #    Analytic body shape set uniformly to BACKGROUND_HU = 40 HU.
    #    Metal voxels are included at 40 HU (seamless), NOT 3000 HU.
    #    Body semi-axes come from TierConfig:
    #      T1_AB  : (170, 120) vox ellipse
    #      T2_SB  : ( 85,  60) vox ellipse
    #      T3_HEAD: (200, 200) vox circle (degenerate ellipse A=B)
    _rows, _cols = np.mgrid[0:512, 0:512]
    body_ellipse = (
        (_cols - 256.0) ** 2 / float(tier.body_semi_x_vox ** 2) +
        (_rows - 256.0) ** 2 / float(tier.body_semi_y_vox ** 2)
    ) < 1.0
    prior_hu = np.where(body_ellipse, BACKGROUND_HU, -1000.0).astype(np.float64)

    # ── 3. Boundary stabilization (Gaussian blur σ = 1.0 voxel) ──────────
    prior_hu_smooth = gaussian_filter(prior_hu, sigma=1.0)

    # ── 4. Forward-project prior → sino_prior, apply scatter transform ───
    mu_prior           = np.maximum((prior_hu_smooth / 1000.0 + 1.0) * MU_WATER, 0.0)
    sino_prior         = _fwd(mu_prior) * VOXEL_CM          # nepers
    sino_prior_scatter = -np.log(np.exp(-sino_prior) + SCATTER_FRAC)

    # ── 5. Synthetic sinogram construction ───────────────────────────────
    sino_synth              = sino_meas.copy()
    metal_trace             = W < 0.5
    sino_synth[metal_trace] = sino_prior_scatter[metal_trace]

    # ── 6. Savitzky–Golay continuity enforcement ─────────────────────────
    sino_smooth = sino_synth.copy()
    hw = SG_WINDOW // 2 + 2
    for a in range(sino_smooth.shape[0]):
        metal_cols = np.where(W[a] < 0.5)[0]
        if metal_cols.size == 0:
            continue
        for boundary in (int(metal_cols.min()), int(metal_cols.max())):
            lo  = max(0, boundary - hw)
            hi  = min(sino_smooth.shape[1], boundary + hw + 1)
            seg = sino_smooth[a, lo:hi].copy()
            n   = len(seg)
            if n < SG_WINDOW:
                continue
            wl = SG_WINDOW if SG_WINDOW <= n else (n if n % 2 == 1 else n - 1)
            if wl > SG_ORDER:
                sino_smooth[a, lo:hi] = savgol_filter(seg, wl, SG_ORDER)

    # ── 7. Final FBP ──────────────────────────────────────────────────────
    x = _fbp(sino_smooth)

    # ── 8. HU calibration ─────────────────────────────────────────────────
    x_hu      = (x / (MU_WATER * VOXEL_CM) - 1.0) * 1000.0
    body_mask = body_ellipse & ~metal_mask
    body_mean = float(np.mean(x_hu[body_mask]))
    x_hu      = x_hu + (BACKGROUND_HU - body_mean)

    # ── 9. Restore metal voxels ───────────────────────────────────────────
    x_hu[metal_mask] = float(METAL_HU)

    return x_hu


# ---------------------------------------------------------------------------
# DICOM I/O helpers
# ---------------------------------------------------------------------------

def _load_noMAR_dicom(dcm_path: Path) -> tuple:
    dcm = pydicom.dcmread(str(dcm_path))
    hu  = dcm.pixel_array.astype(float) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
    return hu, dcm


def _save_imar_dicom(img_hu: np.ndarray, template_dcm, out_path: Path) -> None:
    """Save iMAR image as DICOM by cloning the noMAR template."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dcm = copy.deepcopy(template_dcm)

    slope     = float(dcm.RescaleSlope)
    intercept = float(dcm.RescaleIntercept)
    pixel_arr = np.clip(np.round((img_hu - intercept) / slope), -32768, 32767).astype(np.int16)
    dcm.PixelData = pixel_arr.tobytes()

    dcm.SeriesInstanceUID = generate_uid()
    dcm.SOPInstanceUID    = generate_uid()
    dcm.SeriesDescription = "iMAR (seamless tissue prior)"

    dcm.save_as(str(out_path))


# ---------------------------------------------------------------------------
# Worker (one realization × one condition)
# ---------------------------------------------------------------------------

def _process_one(args):
    """Worker function executed in subprocess."""
    cond, real_idx, dataset_dir, output_dir, tier_id = args
    dataset_dir = Path(dataset_dir)
    output_dir  = Path(output_dir)
    tier        = TIER_REGISTRY[tier_id]

    tag     = f"realization_{real_idx:03d}"
    h5_path = dataset_dir / "sinograms" / cond / f"{tag}.h5"
    dcm_src = dataset_dir / "noMAR_recon" / cond / tag / "slice_0129.dcm"
    dcm_dst = output_dir  / cond / tag / "slice_0129.dcm"

    if dcm_dst.exists():
        return cond, real_idx, True   # already done → skip

    with h5py.File(str(h5_path), "r") as f:
        sino = f["line_integrals"][LESION_SLICE_INDEX].astype(np.float64)

    ref_hu, template_dcm = _load_noMAR_dicom(dcm_src)
    img_imar = imar_slice(sino, ref_hu, tier)
    _save_imar_dicom(img_imar, template_dcm, dcm_dst)

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
        description="iMAR benchmark for ASTM WKXXXXX v1.0.0 — tier-aware")
    ap.add_argument("--input-dir", default="./astm_reference_dataset",
                    help="Root of the tier dataset (default: ./astm_reference_dataset)")
    ap.add_argument("--tier", default="T1_AB",
                    choices=list(TIER_REGISTRY.keys()),
                    help="Imaging scenario tier (default: T1_AB)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0 = os.cpu_count())")
    args = ap.parse_args()

    tier        = TIER_REGISTRY[args.tier]
    dataset_dir = Path(args.input_dir).resolve()
    output_dir  = dataset_dir / "imar_recon"
    n_workers   = args.workers if args.workers > 0 else os.cpu_count()

    print(f"Tier    : {tier.tier_id}  —  {tier.description}")
    print(f"Body    : {tier.body_semi_x_vox}×{tier.body_semi_y_vox} vox  "
          f"({'circle' if tier.is_circular_body else 'ellipse'})")
    print(f"Metal   : {tier.metal_material}  μ={tier.metal_mu_cm} cm⁻¹  "
          f"r={tier.metal_radius_vox} vox")
    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {output_dir}")
    print(f"Workers : {n_workers}  |  Algorithm: iMAR (seamless tissue prior + SG continuity)")

    tasks = [
        (cond, r, str(dataset_dir), str(output_dir), args.tier)
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

    print(f"\niMAR output → {output_dir}")
    print("Next step:")
    print(f"  python run_cho_analysis_v6_0.py \\")
    print(f"      --dataset-dir {dataset_dir} \\")
    print(f"      --mar-output-dir {output_dir} \\")
    print(f"      --tier {args.tier} \\")
    print(f"      --internal-noise-sigma 15 \\")
    print(f"      --results-file imar_{args.tier.lower()}_results.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
