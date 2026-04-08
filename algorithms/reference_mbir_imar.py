#!/usr/bin/env python3
"""
reference_mbir_imar.py
======================
Model-Based Iterative Reconstruction with Metal Artifact Reduction
(MBIR-iMAR) — ASTM WKXXXXX v1.0.0, Tier-aware.

Hypothesis
----------
Sinogram inpainting approaches (iMAR, FS-iMAR, ASD-iMAR, DLSC-iMAR) all
converge to ΔAUC ≈ −0.055 on T2_SB because they *discard* the metal-shadow
rays.  MBIR keeps every measured ray in the loop via a statistical weighting
matrix W that down-weights (but never zeroes) high-attenuation rays.

Algorithm (Two-Stage PWLS-TV)
-----------------------------
Stage 1 — Soft-Weighted Sinogram Fusion:
    sino_fused = W · sino_meas + (1−W) · sino_prior
    where W_i = exp(−y_i) ∈ (0,1]: clean rays → W≈1 (trust measurement),
    metal rays → W≈0.05 (mostly trust prior, but retain measured info).
    The prior is iMAR's seamless tissue sinogram.
    → FBP of fused sinogram gives Stage-1 image.

Stage 2 — PWLS-TV Iterative Refinement (image domain):
    argmin_x  (1/2) ||√W · (Ax·h − y)||² + λ·TV_Huber(x)
    Gradient descent with Barzilai-Borwein adaptive step sizing.
    Warm start from Stage-1 FBP result.
    20 iterations → refines the soft-blended image toward the measured data.

Key: the metal-shadow rays are NEVER discarded — they contribute with
statistically appropriate weight throughout both stages.

Forward/back projectors are matrix-free (skimage radon/iradon).
No system matrix is ever stored.  Peak memory < 4 GB.

Outputs
-------
  <input_dir>/mbir_imar_recon/LP/realization_NNN/slice_0129.dcm
  <input_dir>/mbir_imar_recon/LA/realization_NNN/slice_0129.dcm
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
from scipy.signal import savgol_filter
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

MU_WATER   = 0.2059
VOXEL_CM   = 0.05
SCATTER_FRAC = 0.05

# MBIR parameters
N_ITERATIONS    = 20         # PWLS-TV iterations (Stage 2)
TV_LAMBDA       = 0.01       # TV regularisation strength
TV_HUBER_DELTA  = 0.01       # Huber norm transition (avoids staircasing)

# Savitzky-Golay (for Stage 1 fusion boundary smoothing)
SG_WINDOW = 9
SG_ORDER  = 3

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
# Statistical weights
# ---------------------------------------------------------------------------

def compute_weights(sino_meas: np.ndarray) -> np.ndarray:
    """
    w_i = exp(-y_i) with floor 1e-3 and cap 1.0.

    Metal rays (y≈2-5 neper) → w≈0.007-0.14 (low trust).
    Clean rays (y≈0.5-1.5 neper) → w≈0.22-0.61 (high trust).
    """
    w = np.exp(-sino_meas)
    return np.clip(w, 1e-3, 1.0)


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
# Stage 1: Soft-Weighted Sinogram Fusion
# ---------------------------------------------------------------------------

def stage1_fusion(
    sino_meas: np.ndarray,
    ref_hu: np.ndarray,
    tier: TierConfig,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse measured sinogram with iMAR tissue prior using statistical weights.

    Returns (img_fused_hu, metal_mask, imar_hu) — all in HU.
    """
    # ── Metal mask ───────────────────────────────────────────────────
    metal_mask = ref_hu > METAL_HU_THRESH

    # ── Seamless tissue prior (same as iMAR) ─────────────────────────
    body_ellipse = (
        (_xx - 256.0) ** 2 / float(tier.body_semi_x_vox ** 2) +
        (_yy - 256.0) ** 2 / float(tier.body_semi_y_vox ** 2)
    ) < 1.0
    prior_hu = np.where(body_ellipse, BACKGROUND_HU, -1000.0).astype(np.float64)
    prior_hu_smooth = gaussian_filter(prior_hu, sigma=1.0)
    mu_prior = np.maximum((prior_hu_smooth / 1000.0 + 1.0) * MU_WATER, 0.0)
    sino_prior = _fwd(mu_prior) * VOXEL_CM             # neper
    sino_prior_scatter = -np.log(np.exp(-sino_prior) + SCATTER_FRAC)

    # ── Metal trace identification ───────────────────────────────────
    metal_sino = _fwd(metal_mask.astype(float))
    per_angle_peak = metal_sino.max(axis=1, keepdims=True)
    clean_mask = (metal_sino < 0.05 * np.maximum(per_angle_peak, 1e-6))

    # ── Statistical weights ──────────────────────────────────────────
    W = compute_weights(sino_meas)

    # ── Soft fusion ──────────────────────────────────────────────────
    # Clean rays: mostly measured data (W high)
    # Metal rays: mostly prior (W low), but measured data retained
    sino_fused = W * sino_meas + (1.0 - W) * sino_prior_scatter

    if verbose:
        metal_trace = ~clean_mask
        w_metal_mean = W[metal_trace].mean() if metal_trace.any() else 0
        w_clean_mean = W[clean_mask].mean() if clean_mask.any() else 0
        print(f"  Fusion weights: clean_mean={w_clean_mean:.3f}, "
              f"metal_mean={w_metal_mean:.3f}")

    # ── Savitzky-Golay boundary smoothing ────────────────────────────
    sino_smooth = sino_fused.copy()
    hw = SG_WINDOW // 2 + 2
    for a in range(sino_smooth.shape[0]):
        metal_cols = np.where(~clean_mask[a])[0]
        if metal_cols.size == 0:
            continue
        for boundary in (int(metal_cols.min()), int(metal_cols.max())):
            lo = max(0, boundary - hw)
            hi = min(sino_smooth.shape[1], boundary + hw + 1)
            seg = sino_smooth[a, lo:hi].copy()
            n = len(seg)
            if n < SG_WINDOW:
                continue
            wl = SG_WINDOW if SG_WINDOW <= n else (n if n % 2 == 1 else n - 1)
            if wl > SG_ORDER:
                sino_smooth[a, lo:hi] = savgol_filter(seg, wl, SG_ORDER)

    # ── FBP of fused sinogram ────────────────────────────────────────
    x = _fbp(sino_smooth)

    # ── HU calibration ───────────────────────────────────────────────
    x_hu = (x / (MU_WATER * VOXEL_CM) - 1.0) * 1000.0
    body_mask = body_ellipse & ~metal_mask
    if body_mask.any():
        body_mean = float(np.mean(x_hu[body_mask]))
        x_hu += (BACKGROUND_HU - body_mean)
    x_hu[metal_mask] = float(METAL_HU)

    # Also compute plain iMAR for comparison
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reference_imar import imar_slice
    imar_hu = imar_slice(sino_meas, ref_hu, tier)

    return x_hu, metal_mask, imar_hu


# ---------------------------------------------------------------------------
# Stage 2: PWLS-TV Iterative Refinement
# ---------------------------------------------------------------------------

def stage2_pwls_tv(
    sino_meas: np.ndarray,
    x_init_hu: np.ndarray,
    metal_mask: np.ndarray,
    tier: TierConfig,
    n_iter: int = N_ITERATIONS,
    lam_tv: float = TV_LAMBDA,
    verbose: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """
    Refine Stage-1 image using PWLS gradient descent.

    Parameters
    ----------
    sino_meas : (360, 512)  measured line integrals (neper)
    x_init_hu : (512, 512)  Stage-1 image (HU)
    metal_mask: (512, 512)  bool — metal pixel locations
    tier      : TierConfig
    n_iter    : GD iterations
    lam_tv    : TV weight

    Returns
    -------
    x_hu      : refined image (HU)
    losses    : cost per iteration
    """
    h = VOXEL_CM

    # Scatter-correct the sinogram for forward-model consistency
    I_meas = np.exp(-sino_meas)
    sino_corr = -np.log(np.maximum(I_meas - SCATTER_FRAC, 1e-8))

    # Convert to attenuation
    x = np.maximum((x_init_hu / 1000.0 + 1.0) * MU_WATER, 0.0)
    x[metal_mask] = tier.metal_mu_cm

    # Weights
    W = compute_weights(sino_corr)
    W2 = W * W

    def _cost(x_cur):
        Ax = _fwd(x_cur)  # voxel units
        residual = W * (h * Ax - sino_corr)
        return 0.5 * np.sum(residual ** 2)

    def _gradient(x_cur):
        Ax = _fwd(x_cur)
        residual = W2 * (h * Ax - sino_corr)
        g_data = h * _back_proj(residual)
        g_tv = lam_tv * _tv_gradient(x_cur) if lam_tv > 0 else 0.0
        return g_data + g_tv

    losses = []
    alpha = 1e-4  # initial step size

    for it in range(n_iter):
        t0 = time.time()
        g = _gradient(x)

        if it > 0:
            dx = x - x_prev
            dg = g - g_prev
            dxdg = np.sum(dx * dg)
            if abs(dxdg) > 1e-12:
                alpha = abs(np.sum(dx * dx) / dxdg)
            alpha = np.clip(alpha, 1e-8, 0.5)

        x_prev = x.copy()
        g_prev = g.copy()
        x = x - alpha * g
        x = np.maximum(x, 0.0)
        x[~_CIRCLE_MASK] = 0.0

        cost = _cost(x)
        losses.append(cost)

        elapsed = time.time() - t0
        if verbose:
            print(f"  S2 Iter {it+1:3d}/{n_iter}  cost={cost:.1f}  "
                  f"α={alpha:.2e}  ({elapsed:.1f}s)")

    # Convert to HU — x is in cm⁻¹ (not μ·VOXEL_CM like FBP output)
    body_ellipse = (
        (_xx - 256.0) ** 2 / float(tier.body_semi_x_vox ** 2) +
        (_yy - 256.0) ** 2 / float(tier.body_semi_y_vox ** 2)
    ) < 1.0
    x_hu = (x / MU_WATER - 1.0) * 1000.0
    body_mask = body_ellipse & ~metal_mask
    if body_mask.any():
        x_hu += (BACKGROUND_HU - float(np.mean(x_hu[body_mask])))
    x_hu[metal_mask] = float(METAL_HU)

    return x_hu, losses


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def mbir_imar_slice(
    sino_meas: np.ndarray,
    ref_hu: np.ndarray,
    tier: TierConfig,
    n_iter: int = N_ITERATIONS,
    lam_tv: float = TV_LAMBDA,
    verbose: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """
    Full MBIR-iMAR pipeline for a single 2D slice.

    Returns (img_hu, losses).
    """
    # Stage 1: Soft-weighted sinogram fusion + FBP
    s1_hu, metal_mask, imar_hu = stage1_fusion(
        sino_meas, ref_hu, tier, verbose=verbose)

    if verbose:
        xs, xe = tier.roi_x_bounds()
        ys, ye = tier.roi_y_bounds()
        roi_s1 = s1_hu[ys:ye, xs:xe]
        roi_im = imar_hu[ys:ye, xs:xe]
        print(f"  Stage 1 ROI: mean={roi_s1.mean():.1f}, std={roi_s1.std():.1f}")
        print(f"  iMAR   ROI: mean={roi_im.mean():.1f}, std={roi_im.std():.1f}")

    # Stage 2: PWLS-TV refinement (optional — only if n_iter > 0)
    losses = []
    if n_iter > 0:
        s2_hu, losses = stage2_pwls_tv(
            sino_meas, s1_hu, metal_mask, tier,
            n_iter=n_iter, lam_tv=lam_tv, verbose=verbose)
        return s2_hu, losses

    return s1_hu, losses


# ---------------------------------------------------------------------------
# DICOM I/O
# ---------------------------------------------------------------------------

def _load_noMAR_dicom(dcm_path: Path) -> tuple:
    dcm = pydicom.dcmread(str(dcm_path))
    hu = dcm.pixel_array.astype(float) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
    return hu, dcm


def _save_mbir_dicom(img_hu: np.ndarray, template_dcm, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dcm = copy.deepcopy(template_dcm)
    slope = float(dcm.RescaleSlope)
    intercept = float(dcm.RescaleIntercept)
    pixel_arr = np.clip(np.round((img_hu - intercept) / slope), -32768, 32767).astype(np.int16)
    dcm.PixelData = pixel_arr.tobytes()
    dcm.SeriesInstanceUID = generate_uid()
    dcm.SOPInstanceUID = generate_uid()
    dcm.SeriesDescription = "MBIR-iMAR (PWLS-TV)"
    dcm.save_as(str(out_path))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def generate_comparison_plot(
    ref_hu: np.ndarray,
    imar_hu: np.ndarray,
    mbir_hu: np.ndarray,
    tier: TierConfig,
    losses: list[float],
    output_path: Path,
) -> None:
    """Side-by-side comparison: noMAR / iMAR / MBIR + convergence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, xe = tier.roi_x_bounds()
    ys, ye = tier.roi_y_bounds()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    vmin, vmax = -200, 400
    for ax, img, title in zip(
        axes[0],
        [ref_hu, imar_hu, mbir_hu],
        ["noMAR (FBP)", "iMAR", "MBIR-iMAR"],
    ):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        rect = plt.Rectangle((xs, ys), xe - xs, ye - ys,
                              linewidth=1.5, edgecolor="cyan", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    vmin_z, vmax_z = -50, 150
    for ax, img, title in zip(
        axes[1, :2],
        [imar_hu, mbir_hu],
        ["iMAR — lesion ROI", "MBIR-iMAR — lesion ROI"],
    ):
        roi = img[ys:ye, xs:xe]
        ax.imshow(roi, cmap="gray", vmin=vmin_z, vmax=vmax_z)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    ax_conv = axes[1, 2]
    if losses:
        ax_conv.semilogy(range(1, len(losses) + 1), losses, "o-",
                         color="steelblue", linewidth=1.5, markersize=4)
    ax_conv.set_xlabel("Iteration", fontsize=10)
    ax_conv.set_ylabel("PWLS cost", fontsize=10)
    ax_conv.set_title("Stage 2 Convergence", fontsize=11, fontweight="bold")
    ax_conv.grid(True, alpha=0.3)

    plt.suptitle(f"MBIR-iMAR vs FBP — {tier.tier_id} ({tier.description})",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot → {output_path}")


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
    img_mbir, losses = mbir_imar_slice(sino, ref_hu, tier,
                                        n_iter=n_iter, lam_tv=lam_tv, verbose=v)
    _save_mbir_dicom(img_mbir, template_dcm, dcm_dst)

    return cond, real_idx, False, losses


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
        description="MBIR-iMAR benchmark for ASTM WKXXXXX v1.0.0")
    ap.add_argument("--input-dir", default="./bench_t2_sb")
    ap.add_argument("--tier", default="T2_SB",
                    choices=list(TIER_REGISTRY.keys()))
    ap.add_argument("--n-iter", type=int, default=N_ITERATIONS)
    ap.add_argument("--tv-lambda", type=float, default=TV_LAMBDA)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    tier = TIER_REGISTRY[args.tier]
    dataset_dir = Path(args.input_dir).resolve()
    output_dir = dataset_dir / "mbir_imar_recon"

    print(f"{'='*60}")
    print(f"MBIR-iMAR  —  ASTM WKXXXXX v1.0.0")
    print(f"{'='*60}")
    print(f"Tier      : {tier.tier_id} — {tier.description}")
    print(f"Blockage  : {tier.blockage_frac*100:.1f}%")
    print(f"Dataset   : {dataset_dir}")
    print(f"Output    : {output_dir}")
    print(f"Iterations: {args.n_iter} (Stage 2 PWLS-TV)")
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
    print(f"MBIR-iMAR output → {output_dir}")

    # Comparison plot
    if not args.skip_plot:
        try:
            dcm_src = dataset_dir / "noMAR_recon/LP/realization_001/slice_0129.dcm"
            dcm_mbir = output_dir / "LP/realization_001/slice_0129.dcm"
            if dcm_src.exists() and dcm_mbir.exists():
                ref_hu, _ = _load_noMAR_dicom(dcm_src)
                mbir_hu, _ = _load_noMAR_dicom(dcm_mbir)
                imar_path = dataset_dir / "imar_recon/LP/realization_001/slice_0129.dcm"
                if imar_path.exists():
                    imar_hu, _ = _load_noMAR_dicom(imar_path)
                else:
                    imar_hu = ref_hu
                plot_path = (Path(__file__).resolve().parent.parent
                             / "mbir_vs_fbp_comparison.png")
                generate_comparison_plot(ref_hu, imar_hu, mbir_hu, tier,
                                        first_losses, plot_path)
        except Exception as exc:
            print(f"  Warning: plot failed: {exc}")

    print("\nNext step:")
    print(f"  python run_cho_analysis_v6_0.py \\")
    print(f"      --dataset-dir {dataset_dir} \\")
    print(f"      --mar-output-dir {output_dir} \\")
    print(f"      --tier {args.tier} \\")
    print(f"      --internal-noise-sigma 15 \\")
    print(f"      --results-file mbir_imar_{args.tier.lower()}_results.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
