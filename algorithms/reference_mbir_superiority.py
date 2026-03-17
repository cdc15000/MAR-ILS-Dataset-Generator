#!/usr/bin/env python3
"""
reference_mbir_superiority.py
=============================
Prior-Free, Binary-Weighted MBIR for MAR Superiority.
ASTM WKXXXXX v1.0.0 — Tier-aware.

Motivation
----------
The previous MBIR (reference_mbir_imar.py) achieved ΔAUC = −0.006 on De Man-
calibrated T2_SB (23% blockage) by using statistical weights W=exp(−y) and a
Stage 1 sinogram fusion with the iMAR tissue prior.  Two sources of residual
signal dilution prevented crossing the ΔAUC > 0 barrier:

  1. Stage 1 fusion mixed 16% of a lesion-free prior into clean rays
     (W_clean = 0.843 ≠ 1.0).
  2. The warm start was the fused image, which already had attenuated
     lesion contrast.

This solver eliminates both sources:

  (a)  Warm start = noMAR FBP (100% of original lesion signal).
  (b)  Binary weights: W = 1.0 for non-metal rays, W = 0.05 for metal rays.
       Clean rays are NEVER diluted.  Metal rays contribute negligible gradient.

Algorithm
---------
  1. Metal ray identification via forward-projected metal mask.
  2. Scatter-correct measured sinogram: y_corr = −log(max(exp(−y) − S, ε)).
  3. Warm start x₀ = noMAR FBP (converted to μ units).
  4. Fix x₀[metal] = μ_metal (correct metal attenuation).
  5. PWLS-TV gradient descent with BB adaptive step sizing:
       argmin_x  (1/2) ||W · (Ax·h − y_corr)||² + λ·TV_Huber(x)
     Metal pixels are pinned after each update.
  6. HU calibration + metal restore.

The optimizer barely touches the lesion ROI (clean-ray residuals are already
small from the FBP warm start) while reducing metal-streak energy in the
regions affected by the 23% corrupted projections.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import gaussian_filter
from skimage.transform import radon as _sk_radon, iradon as _sk_iradon

# Add parent dir for tier_config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tier_config import (
    BACKGROUND_HU,
    TIER_REGISTRY,
    TierConfig,
)

# ---------------------------------------------------------------------------
# Normative constants
# ---------------------------------------------------------------------------
LESION_SLICE_INDEX = 128
N_ANGLES           = 360
N_DET              = 512
IMG_SIZE           = 512
THETA_DEG          = np.linspace(0, 180, N_ANGLES, endpoint=False)
METAL_HU_THRESH    = 1500
METAL_HU           = 3000
N_REALIZATIONS     = 40

MU_WATER     = 0.2059
VOXEL_CM     = 0.05
SCATTER_FRAC = 0.05

# Solver defaults
N_ITERATIONS    = 40
TV_LAMBDA       = 0.00002
TV_HUBER_DELTA  = 0.01

# Metal-ray weight (binary: clean=1.0, metal=W_METAL)
W_METAL = 0.05

# Circle mask
_cx = IMG_SIZE / 2.0
_yy, _xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
_CIRCLE_MASK = ((_xx - _cx) ** 2 + (_yy - _cx) ** 2) <= _cx ** 2


# ---------------------------------------------------------------------------
# Projectors (matrix-free, skimage)
# ---------------------------------------------------------------------------

def _fwd(img: np.ndarray) -> np.ndarray:
    """Forward projection. img (512,512) → sino (360,512) in voxel units."""
    return _sk_radon(img, theta=THETA_DEG, circle=True).T


def _fbp(sino: np.ndarray) -> np.ndarray:
    """FBP with Ram-Lak. sino (360,512) → img (512,512)."""
    return _sk_iradon(sino.T, theta=THETA_DEG, filter_name="ramp", circle=True)


def _back_proj(sino: np.ndarray) -> np.ndarray:
    """Unfiltered backprojection. sino (360,512) → img (512,512)."""
    return _sk_iradon(sino.T, theta=THETA_DEG, filter_name=None, circle=True)


# ---------------------------------------------------------------------------
# TV gradient (Huber norm)
# ---------------------------------------------------------------------------

def _tv_gradient(img: np.ndarray, delta: float = TV_HUBER_DELTA) -> np.ndarray:
    """Gradient of Huber-norm isotropic TV."""
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, :-1] = img[:, 1:] - img[:, :-1]
    dy[:-1, :] = img[1:, :] - img[:-1, :]

    grad_mag = np.sqrt(dx ** 2 + dy ** 2 + 1e-12)
    weight = np.where(grad_mag <= delta, 1.0 / delta, 1.0 / grad_mag)

    px = weight * dx
    py = weight * dy

    div = np.zeros_like(img)
    div[:, 1:]  -= px[:, :-1]
    div[:, :-1] += px[:, :-1]
    div[1:, :]  -= py[:-1, :]
    div[:-1, :] += py[:-1, :]

    return -div


# ---------------------------------------------------------------------------
# Core solver: Prior-Free Binary-Weighted PWLS-TV
# ---------------------------------------------------------------------------

def mbir_superiority_slice(
    sino_meas: np.ndarray,
    ref_hu: np.ndarray,
    tier: TierConfig,
    n_iter: int = N_ITERATIONS,
    lam_tv: float = TV_LAMBDA,
    verbose: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """
    Prior-free, binary-weighted MBIR.

    Parameters
    ----------
    sino_meas : (360, 512) float64  measured line integrals (neper)
    ref_hu    : (512, 512) float64  noMAR FBP image (DICOM HU)
    tier      : TierConfig          tier geometry
    n_iter    : int                  PWLS-TV iterations
    lam_tv    : float                TV regularisation weight

    Returns
    -------
    img_hu : (512, 512) float64  corrected image in HU
    losses : list[float]         PWLS cost per iteration
    """
    h = VOXEL_CM

    # ── 1. Metal mask and binary weight matrix ─────────────────────────
    metal_mask = ref_hu > METAL_HU_THRESH

    # Forward-project metal mask to identify metal-trace rays
    metal_sino = _fwd(metal_mask.astype(float))
    per_angle_peak = metal_sino.max(axis=1, keepdims=True)
    is_metal_ray = metal_sino > 0.05 * np.maximum(per_angle_peak, 1e-6)

    # Binary weights: clean=1.0, metal=W_METAL
    W = np.ones_like(sino_meas)
    W[is_metal_ray] = W_METAL
    W2 = W * W

    n_metal_rays = is_metal_ray.sum()
    n_total_rays = is_metal_ray.size
    metal_frac = n_metal_rays / n_total_rays

    if verbose:
        print(f"  Binary weights: {metal_frac*100:.1f}% metal rays (W={W_METAL}), "
              f"{(1-metal_frac)*100:.1f}% clean rays (W=1.0)")

    # ── 2. Scatter-correct measured sinogram ───────────────────────────
    sino_corr = -np.log(np.maximum(np.exp(-sino_meas) - SCATTER_FRAC, 1e-8))

    # ── 3. Warm start: noMAR FBP → attenuation (μ in cm⁻¹) ───────────
    #    100% of the original lesion signal is in ref_hu.
    x = np.maximum((ref_hu / 1000.0 + 1.0) * MU_WATER, 0.0)

    # ── 4. Pin metal to true attenuation ───────────────────────────────
    x[metal_mask] = tier.metal_mu_cm
    x[~_CIRCLE_MASK] = 0.0

    if verbose:
        xs, xe = tier.roi_x_bounds()
        ys, ye = tier.roi_y_bounds()
        roi_init = ref_hu[ys:ye, xs:xe]
        print(f"  Warm start (noMAR) ROI: mean={roi_init.mean():.1f}, "
              f"std={roi_init.std():.1f}")

    # ── 5. PWLS-TV gradient descent with BB step sizing ────────────────
    losses = []
    alpha = 1e-4  # initial conservative step

    x_prev = None
    g_prev = None

    for it in range(n_iter):
        t0 = time.time()

        # Forward project current image
        Ax = _fwd(x)  # voxel units

        # Weighted residual
        residual = h * Ax - sino_corr
        cost = 0.5 * float(np.sum(W2 * residual ** 2))
        losses.append(cost)

        # Gradient: data fidelity + TV
        g_data = h * _back_proj(W2 * residual)
        g_tv = lam_tv * _tv_gradient(x) if lam_tv > 0 else 0.0
        g = g_data + g_tv

        # BB adaptive step sizing (after first iteration)
        if x_prev is not None:
            dx = x - x_prev
            dg = g - g_prev
            dxdg = float(np.sum(dx * dg))
            if abs(dxdg) > 1e-12:
                alpha = abs(float(np.sum(dx * dx)) / dxdg)
            alpha = np.clip(alpha, 1e-8, 0.5)

        x_prev = x.copy()
        g_prev = g.copy()

        # Gradient step
        x = x - alpha * g

        # Enforce constraints
        x = np.maximum(x, 0.0)       # non-negative attenuation
        x[~_CIRCLE_MASK] = 0.0       # zero outside circle
        x[metal_mask] = tier.metal_mu_cm  # pin metal

        elapsed = time.time() - t0
        if verbose:
            print(f"  Iter {it+1:3d}/{n_iter}  cost={cost:.1f}  "
                  f"α={alpha:.2e}  ({elapsed:.1f}s)")

    # ── 6. HU calibration ──────────────────────────────────────────────
    #    x is in cm⁻¹ (NOT μ·VOXEL_CM like FBP output).
    #    Correct formula: x_hu = (x / MU_WATER - 1) × 1000.
    body_ellipse = (
        (_xx - 256.0) ** 2 / float(tier.body_semi_x_vox ** 2) +
        (_yy - 256.0) ** 2 / float(tier.body_semi_y_vox ** 2)
    ) < 1.0

    x_hu = (x / MU_WATER - 1.0) * 1000.0
    body_mask = body_ellipse & ~metal_mask
    if body_mask.any():
        body_mean = float(np.mean(x_hu[body_mask]))
        x_hu += (BACKGROUND_HU - body_mean)

    x_hu[metal_mask] = float(METAL_HU)

    if verbose:
        roi_final = x_hu[ys:ye, xs:xe]
        print(f"  Final ROI: mean={roi_final.mean():.1f}, "
              f"std={roi_final.std():.1f}")
        print(f"  Cost reduction: {losses[0]:.0f} → {losses[-1]:.0f} "
              f"({losses[-1]/losses[0]*100:.1f}%)")

    return x_hu, losses


# ---------------------------------------------------------------------------
# DICOM I/O
# ---------------------------------------------------------------------------

def _load_noMAR_dicom(dcm_path: Path) -> tuple:
    dcm = pydicom.dcmread(str(dcm_path))
    hu = dcm.pixel_array.astype(float) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
    return hu, dcm


def _save_dicom(img_hu: np.ndarray, template_dcm, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dcm = copy.deepcopy(template_dcm)
    slope = float(dcm.RescaleSlope)
    intercept = float(dcm.RescaleIntercept)
    pixel_arr = np.clip(
        np.round((img_hu - intercept) / slope), -32768, 32767
    ).astype(np.int16)
    dcm.PixelData = pixel_arr.tobytes()
    dcm.SeriesInstanceUID = generate_uid()
    dcm.SOPInstanceUID = generate_uid()
    dcm.SeriesDescription = "MBIR-Superiority (prior-free, binary W)"
    dcm.save_as(str(out_path))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _process_one(args_tuple):
    cond, real_idx, dataset_dir, output_dir, tier_id, n_iter, lam_tv = args_tuple
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    tier = TIER_REGISTRY[tier_id]

    tag = f"realization_{real_idx:03d}"
    h5_path = dataset_dir / "sinograms" / cond / f"{tag}.h5"
    dcm_src = dataset_dir / "noMAR_recon" / cond / tag / "slice_0129.dcm"
    dcm_dst = output_dir / cond / tag / "slice_0129.dcm"

    if dcm_dst.exists():
        return cond, real_idx, True, []

    with h5py.File(str(h5_path), "r") as f:
        sino = f["line_integrals"][LESION_SLICE_INDEX].astype(np.float64)

    ref_hu, template_dcm = _load_noMAR_dicom(dcm_src)

    v = (cond == "LP" and real_idx == 1)
    img_hu, losses = mbir_superiority_slice(
        sino, ref_hu, tier, n_iter=n_iter, lam_tv=lam_tv, verbose=v)
    _save_dicom(img_hu, template_dcm, dcm_dst)

    return cond, real_idx, False, losses


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def generate_comparison_plot(dataset_dir: Path, output_dir: Path,
                             tier: TierConfig, losses: list[float]) -> None:
    """Side-by-side: noMAR / MBIR-Superiority + convergence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, xe = tier.roi_x_bounds()
    ys, ye = tier.roi_y_bounds()

    dcm_nomar = dataset_dir / "noMAR_recon/LP/realization_001/slice_0129.dcm"
    dcm_mbir  = output_dir / "LP/realization_001/slice_0129.dcm"

    if not dcm_nomar.exists() or not dcm_mbir.exists():
        print("  Warning: cannot generate plot (DICOMs not found)")
        return

    nomar_hu, _ = _load_noMAR_dicom(dcm_nomar)
    mbir_hu, _  = _load_noMAR_dicom(dcm_mbir)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    vmin, vmax = -200, 400
    for ax, img, title in zip(
        axes[0, :2],
        [nomar_hu, mbir_hu],
        ["noMAR (FBP)", "MBIR-Superiority"],
    ):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        rect = plt.Rectangle((xs, ys), xe - xs, ye - ys,
                              linewidth=1.5, edgecolor="cyan", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Difference image
    diff = mbir_hu - nomar_hu
    axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-50, vmax=50)
    axes[0, 2].set_title("MBIR − noMAR (Δ HU)", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # Zoomed ROIs
    vmin_z, vmax_z = -20, 120
    for ax, img, title in zip(
        axes[1, :2],
        [nomar_hu, mbir_hu],
        ["noMAR — lesion ROI", "MBIR-Superiority — lesion ROI"],
    ):
        roi = img[ys:ye, xs:xe]
        ax.imshow(roi, cmap="gray", vmin=vmin_z, vmax=vmax_z)
        ax.set_title(f"{title}\nμ={roi.mean():.1f}  σ={roi.std():.1f}",
                     fontsize=10)
        ax.axis("off")

    # Convergence
    ax_conv = axes[1, 2]
    if losses:
        ax_conv.semilogy(range(1, len(losses) + 1), losses, "o-",
                         color="steelblue", linewidth=1.5, markersize=3)
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("PWLS cost")
    ax_conv.set_title("Convergence", fontsize=11, fontweight="bold")
    ax_conv.grid(True, alpha=0.3)

    plt.suptitle(
        f"MBIR-Superiority — {tier.tier_id}  (Prior-Free, Binary W)\n"
        f"Blockage={tier.blockage_frac*100:.1f}%  "
        f"n_iter={len(losses)}  TV={TV_LAMBDA:.0e}",
        fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plot_path = Path(__file__).resolve().parent.parent / "superiority_achieved_roi.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="MBIR-Superiority (Prior-Free, Binary W) — ASTM WKXXXXX")
    ap.add_argument("--input-dir", default="./deman_t2_sb")
    ap.add_argument("--tier", default="T2_SB",
                    choices=list(TIER_REGISTRY.keys()))
    ap.add_argument("--n-iter", type=int, default=N_ITERATIONS)
    ap.add_argument("--tv-lambda", type=float, default=TV_LAMBDA)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    tier = TIER_REGISTRY[args.tier]
    dataset_dir = Path(args.input_dir).resolve()
    output_dir = dataset_dir / "mbir_superiority_recon"

    print(f"{'='*60}")
    print(f"MBIR-Superiority  —  Prior-Free, Binary W")
    print(f"{'='*60}")
    print(f"Tier      : {tier.tier_id} — {tier.description}")
    print(f"Blockage  : {tier.blockage_frac*100:.1f}%")
    print(f"Dataset   : {dataset_dir}")
    print(f"Output    : {output_dir}")
    print(f"Warm start: noMAR FBP (prior-free — 100% lesion signal)")
    print(f"Weights   : binary (clean=1.0, metal={W_METAL})")
    print(f"Iterations: {args.n_iter}")
    print(f"TV λ      : {args.tv_lambda}")
    print()

    tasks = [
        (cond, r, str(dataset_dir), str(output_dir), args.tier,
         args.n_iter, args.tv_lambda)
        for cond in ("LP", "LA")
        for r in range(1, N_REALIZATIONS + 1)
    ]

    first_losses = []
    done = 0
    total = len(tasks)
    t_start = time.time()

    for task in tasks:
        cond = task[0]
        ridx = task[1]
        try:
            cond_r, real_idx, skipped, losses = _process_one(task)
            done += 1
            status = "skip" if skipped else "done"
            elapsed = time.time() - t_start
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"  [{done:3d}/{total}] {cond_r}/realization_{real_idx:03d} "
                  f"{status}  (ETA {eta/60:.1f}min)")
            if cond == "LP" and ridx == 1 and losses:
                first_losses = losses
        except Exception as exc:
            done += 1
            print(f"  ERROR {cond}/realization_{ridx:03d}: {exc}")

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total/60:.1f} min")
    print(f"Output → {output_dir}")

    # Generate comparison plot
    if not args.skip_plot:
        try:
            generate_comparison_plot(dataset_dir, output_dir, tier, first_losses)
        except Exception as exc:
            print(f"  Warning: plot failed: {exc}")

    print("\nNext step:")
    print(f"  python run_cho_analysis_v6_0.py \\")
    print(f"      --dataset-dir {dataset_dir} \\")
    print(f"      --mar-output-dir {output_dir} \\")
    print(f"      --tier {args.tier} \\")
    print(f"      --internal-noise-sigma 15 \\")
    print(f"      --results-file mbir_superiority_{args.tier.lower()}_results.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
