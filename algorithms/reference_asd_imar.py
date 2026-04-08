#!/usr/bin/env python3
"""
reference_asd_imar.py
=====================
Anisotropic Sinogram Diffusion iMAR (ASD-iMAR) — structure-tensor-guided
inpainting of metal-trace sinogram data.
ASTM WKXXXXX v1.0.0 — Tier-aware.

Hypothesis
----------
Standard iMAR replaces metal-trace rays with a smooth analytic prior, which
discards the directional flow of the lesion signal present in adjacent clean
rays.  In tight-geometry tiers (T2_SB: 40% blockage, 1.5 mm gap) this
causes "information isolation" — the lesion signal cannot bridge the metal
trace.  Anisotropic diffusion guided by the structure tensor of the clean
sinogram propagates high-frequency gradients (lesion signal) into the metal
trace along paths of maximum coherence.

Algorithm
---------
  1. Metal mask + weight matrix     : Same as iMAR (forward-project noMAR
                                       metal mask → W).
  2. Seamless tissue prior          : Standard iMAR prior as warm start.
  3. Synthetic sinogram (initial)   : Clean rays → sino_meas; metal trace
                                       → sino_prior (iMAR baseline).
  4. Structure tensor computation   : Compute Jρ of sino_meas in clean-ray
                                       region.  Eigendecomposition → local
                                       orientation θ(a,d) and coherence
                                       c(a,d) for each sinogram pixel.
  5. Anisotropic diffusion          : Perona-Malik PDE with structure-tensor
                                       guidance, N iterations.  Only metal-trace
                                       pixels evolve; clean-ray pixels are
                                       pinned to sino_meas (Dirichlet BC).
                                       Conductance κ is auto-tuned from
                                       tier.metal_mu_cm.
  6. SG continuity enforcement      : Same as iMAR (boundary smoothing).
  7. Final FBP + HU calibration     : Same as iMAR.
  8. Metal restore                  : metal voxels → 3000 HU.

Conductance auto-tuning
-----------------------
  κ = κ_base / tier.metal_mu_cm
  Higher metal μ → stronger streak gradients → lower κ → more selective
  diffusion (only coherent gradients propagate).  κ_base is calibrated so
  that T2_SB (μ=2.8) gets κ ≈ 0.036 and T3_HEAD (μ=1.5) gets κ ≈ 0.067.

Outputs
-------
  <input_dir>/asd_imar_recon/LP/realization_NNN/slice_0129.dcm  (N=1..40)
  <input_dir>/asd_imar_recon/LA/realization_NNN/slice_0129.dcm  (N=1..40)
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
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter
from scipy.signal import savgol_filter
from skimage.transform import iradon, radon

from tier_config import (
    BACKGROUND_HU,
    TIER_REGISTRY,
    TierConfig,
)

# ---------------------------------------------------------------------------
# Normative constants (tier-independent)
# ---------------------------------------------------------------------------
LESION_SLICE_INDEX = 128
N_ANGLES           = 360
N_DETECTORS        = 512
THETA_DEG          = np.linspace(0, 180, N_ANGLES, endpoint=False)
METAL_HU_THRESH    = 1500
METAL_HU           = 3000.0
N_REALIZATIONS     = 40

MU_WATER     = 0.2059
VOXEL_CM     = 0.05
SCATTER_FRAC = 0.05

SG_WINDOW = 9
SG_ORDER  = 3

# ASD-iMAR tunable defaults
DEFAULT_N_ITER      = 20       # diffusion iterations
DEFAULT_KAPPA_BASE  = 0.10     # base conductance (scaled by 1/metal_mu_cm)
DEFAULT_DT          = 0.20     # diffusion timestep (CFL-safe for 2D, dt < 0.25)
DEFAULT_STRUCT_SIGMA = 1.5     # Gaussian σ for structure tensor smoothing


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
# Structure tensor computation
# ---------------------------------------------------------------------------

def compute_structure_tensor(
    sino: np.ndarray,
    mask_clean: np.ndarray,
    sigma_deriv: float = 1.0,
    sigma_struct: float = DEFAULT_STRUCT_SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the structure tensor of the sinogram in the clean-ray region.

    Parameters
    ----------
    sino        : (N_a, N_d)  sinogram
    mask_clean  : (N_a, N_d)  bool — True where data is clean (not metal trace)
    sigma_deriv : float       Gaussian σ for gradient estimation
    sigma_struct: float       Gaussian σ for structure tensor averaging (Jρ)

    Returns
    -------
    orientation : (N_a, N_d)  local orientation angle θ in [0, π)
    coherence   : (N_a, N_d)  coherence measure c ∈ [0, 1]
    diff_tensor : (N_a, N_d, 2, 2)  diffusion tensor D at each pixel
    """
    na, nd = sino.shape

    # Mask the sinogram: set metal-trace to local mean to avoid edge artifacts
    # in gradient computation at the clean/metal boundary.
    sino_masked = sino.copy()
    sino_masked[~mask_clean] = np.nan
    # Fill NaN with column-wise (detector-wise) mean of clean data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_mean = np.nanmean(sino_masked, axis=0)
    # Detectors with no clean data at all: use global mean
    global_mean = np.nanmean(sino_masked)
    col_mean = np.where(np.isnan(col_mean), global_mean, col_mean)
    for d in range(nd):
        sino_masked[~mask_clean[:, d], d] = col_mean[d]

    # Gradients along angle (axis 0) and detector (axis 1)
    g_a = gaussian_filter1d(sino_masked, sigma=sigma_deriv, axis=0, order=1)
    g_d = gaussian_filter1d(sino_masked, sigma=sigma_deriv, axis=1, order=1)

    # Structure tensor components J = [[J_aa, J_ad], [J_ad, J_dd]]
    J_aa = gaussian_filter(g_a * g_a, sigma=sigma_struct)
    J_ad = gaussian_filter(g_a * g_d, sigma=sigma_struct)
    J_dd = gaussian_filter(g_d * g_d, sigma=sigma_struct)

    # Eigendecomposition (analytic 2×2)
    # λ₁ ≥ λ₂ (eigenvalues); v₁ = direction of max gradient
    trace = J_aa + J_dd
    det   = J_aa * J_dd - J_ad * J_ad
    disc  = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))

    lam1 = 0.5 * (trace + disc)   # largest eigenvalue
    lam2 = 0.5 * (trace - disc)   # smallest eigenvalue

    # Coherence: (λ₁ - λ₂) / (λ₁ + λ₂ + ε)
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-10)

    # Orientation: eigenvector corresponding to λ₂ (direction of least change
    # = edge direction; we want to diffuse ALONG this direction)
    # For 2×2 symmetric: v₂ = (-J_ad, J_aa - λ₂) normalized
    v2_a = -J_ad
    v2_d = J_aa - lam2
    norm = np.sqrt(v2_a**2 + v2_d**2 + 1e-10)
    v2_a /= norm
    v2_d /= norm

    orientation = np.arctan2(v2_d, v2_a) % np.pi

    # Build diffusion tensor D:
    # D = c₁ v₁v₁ᵀ + c₂ v₂v₂ᵀ
    # where c₂ (along edge) > c₁ (across edge) for anisotropic diffusion.
    # v₁ is perpendicular to v₂:
    v1_a = v2_d    # 90° rotation
    v1_d = -v2_a

    # Diffusivities: diffuse strongly along edges, weakly across
    # c₂ = 1 (along edge), c₁ = 1 - coherence² (across edge)
    c1 = 1.0 - coherence**2
    c2 = np.ones_like(coherence)

    diff_tensor = np.zeros((na, nd, 2, 2), dtype=np.float64)
    diff_tensor[:, :, 0, 0] = c1 * v1_a * v1_a + c2 * v2_a * v2_a
    diff_tensor[:, :, 0, 1] = c1 * v1_a * v1_d + c2 * v2_a * v2_d
    diff_tensor[:, :, 1, 0] = diff_tensor[:, :, 0, 1]  # symmetric
    diff_tensor[:, :, 1, 1] = c1 * v1_d * v1_d + c2 * v2_d * v2_d

    return orientation, coherence, diff_tensor


# ---------------------------------------------------------------------------
# Anisotropic diffusion engine
# ---------------------------------------------------------------------------

def anisotropic_diffusion_inpaint(
    sino_init: np.ndarray,
    sino_meas: np.ndarray,
    metal_trace: np.ndarray,
    diff_tensor: np.ndarray,
    kappa: float,
    n_iter: int = DEFAULT_N_ITER,
    dt: float = DEFAULT_DT,
) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion with structure-tensor guidance.

    Only metal-trace pixels evolve.  Clean-ray pixels are Dirichlet-pinned
    to sino_meas.

    Parameters
    ----------
    sino_init    : (N_a, N_d)      initial sinogram (iMAR warm start)
    sino_meas    : (N_a, N_d)      measured sinogram (clean-ray anchor)
    metal_trace  : (N_a, N_d) bool True where sinogram is metal-corrupted
    diff_tensor  : (N_a, N_d, 2, 2) structure-tensor diffusion guidance
    kappa        : float           conductance parameter
    n_iter       : int             number of diffusion iterations
    dt           : float           timestep (must be < 0.25 for stability)

    Returns
    -------
    sino_diffused : (N_a, N_d)  diffused sinogram
    """
    u = sino_init.astype(np.float64).copy()

    for _ in range(n_iter):
        # Finite differences (central, with wrap on angle axis for continuity)
        # ∂u/∂a and ∂u/∂d
        du_a = np.zeros_like(u)
        du_d = np.zeros_like(u)

        # Forward differences
        du_a_fwd = np.roll(u, -1, axis=0) - u  # u[a+1,d] - u[a,d]
        du_a_bwd = u - np.roll(u, 1, axis=0)   # u[a,d] - u[a-1,d]
        du_d_fwd = np.zeros_like(u)
        du_d_bwd = np.zeros_like(u)
        du_d_fwd[:, :-1] = u[:, 1:] - u[:, :-1]
        du_d_bwd[:, 1:]  = u[:, 1:] - u[:, :-1]

        # Perona-Malik conductance: g(|∇u|²) = 1 / (1 + |∇u|²/κ²)
        # Use central gradient magnitude for conductance
        grad_a = 0.5 * (du_a_fwd + du_a_bwd)
        grad_d = 0.5 * (du_d_fwd + du_d_bwd)
        grad_mag_sq = grad_a**2 + grad_d**2
        g = 1.0 / (1.0 + grad_mag_sq / (kappa**2))

        # Structure-tensor-guided flux:
        # J_a = D[0,0]*∂u/∂a + D[0,1]*∂u/∂d
        # J_d = D[1,0]*∂u/∂a + D[1,1]*∂u/∂d
        # Apply conductance to modulate the flux
        D00 = diff_tensor[:, :, 0, 0]
        D01 = diff_tensor[:, :, 0, 1]
        D10 = diff_tensor[:, :, 1, 0]
        D11 = diff_tensor[:, :, 1, 1]

        # Flux using forward differences (for divergence via backward diff)
        flux_a_fwd = g * (D00 * du_a_fwd + D01 * du_d_fwd)
        flux_d_fwd = g * (D10 * du_a_fwd + D11 * du_d_fwd)

        # Divergence via backward differences of fluxes
        div_a = flux_a_fwd - np.roll(flux_a_fwd, 1, axis=0)
        div_d = np.zeros_like(u)
        div_d[:, 1:] = flux_d_fwd[:, 1:] - flux_d_fwd[:, :-1]

        divergence = div_a + div_d

        # Update only metal-trace pixels
        u[metal_trace] += dt * divergence[metal_trace]

        # Re-pin clean rays (Dirichlet boundary condition)
        u[~metal_trace] = sino_meas[~metal_trace]

    return u


# ---------------------------------------------------------------------------
# ASD-iMAR core (single 2D slice)
# ---------------------------------------------------------------------------

def asd_imar_slice(
    sino_meas: np.ndarray,
    ref_hu:    np.ndarray,
    tier:      TierConfig,
    n_iter:     int   = DEFAULT_N_ITER,
    kappa_base: float = DEFAULT_KAPPA_BASE,
    dt:         float = DEFAULT_DT,
) -> np.ndarray:
    """
    Parameters
    ----------
    sino_meas  : (360, 512) float64  measured line integrals (neper)
    ref_hu     : (512, 512) float64  noMAR FBP image (DICOM HU)
    tier       : TierConfig          tier geometry for seamless prior
    n_iter     : int                 diffusion iterations
    kappa_base : float               base conductance (auto-tuned by metal_mu_cm)
    dt         : float               diffusion timestep

    Returns
    -------
    img_asd : (512, 512) float64  ASD-iMAR-corrected image in HU
    """
    # ── 1. Metal mask and weight matrix ──────────────────────────────────
    metal_mask     = ref_hu > METAL_HU_THRESH
    metal_sino     = _fwd(metal_mask.astype(float))
    per_angle_peak = metal_sino.max(axis=1, keepdims=True)
    W = (metal_sino < 0.05 * np.maximum(per_angle_peak, 1e-6)).astype(float)
    metal_trace = W < 0.5
    clean_mask  = ~metal_trace

    # ── 2. Seamless tissue prior (iMAR warm start) ───────────────────────
    _rows, _cols = np.mgrid[0:512, 0:512]
    body_ellipse = (
        (_cols - 256.0) ** 2 / float(tier.body_semi_x_vox ** 2) +
        (_rows - 256.0) ** 2 / float(tier.body_semi_y_vox ** 2)
    ) < 1.0
    prior_hu = np.where(body_ellipse, BACKGROUND_HU, -1000.0).astype(np.float64)
    prior_hu_smooth = gaussian_filter(prior_hu, sigma=1.0)
    mu_prior = np.maximum((prior_hu_smooth / 1000.0 + 1.0) * MU_WATER, 0.0)
    sino_prior = _fwd(mu_prior) * VOXEL_CM
    sino_prior_scatter = -np.log(np.exp(-sino_prior) + SCATTER_FRAC)

    # ── 3. Initial synthetic sinogram ────────────────────────────────────
    sino_init = sino_meas.copy()
    sino_init[metal_trace] = sino_prior_scatter[metal_trace]

    # ── 4. Structure tensor of measured sinogram ─────────────────────────
    _, _, diff_tensor = compute_structure_tensor(
        sino_meas, clean_mask, sigma_deriv=1.0, sigma_struct=DEFAULT_STRUCT_SIGMA,
    )

    # ── 5. Anisotropic diffusion ─────────────────────────────────────────
    # Auto-tune κ: higher metal_mu → stronger streaks → lower κ
    kappa = kappa_base / tier.metal_mu_cm
    sino_diffused = anisotropic_diffusion_inpaint(
        sino_init, sino_meas, metal_trace, diff_tensor,
        kappa=kappa, n_iter=n_iter, dt=dt,
    )

    # ── 6. Savitzky-Golay continuity enforcement ─────────────────────────
    sino_smooth = sino_diffused.copy()
    hw = SG_WINDOW // 2 + 2
    for a in range(sino_smooth.shape[0]):
        metal_cols = np.where(metal_trace[a])[0]
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

    # ── 7. Final FBP ────────────────────────────────────────────────────
    x = _fbp(sino_smooth)

    # ── 8. HU calibration ───────────────────────────────────────────────
    x_hu = (x / (MU_WATER * VOXEL_CM) - 1.0) * 1000.0
    body_mask = body_ellipse & ~metal_mask
    body_mean = float(np.mean(x_hu[body_mask]))
    x_hu = x_hu + (BACKGROUND_HU - body_mean)

    # ── 9. Restore metal voxels ─────────────────────────────────────────
    x_hu[metal_mask] = METAL_HU

    return x_hu


# ---------------------------------------------------------------------------
# DICOM I/O helpers
# ---------------------------------------------------------------------------

def _load_noMAR_dicom(dcm_path: Path) -> tuple:
    dcm = pydicom.dcmread(str(dcm_path))
    hu  = dcm.pixel_array.astype(float) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
    return hu, dcm


def _save_asd_imar_dicom(img_hu: np.ndarray, template_dcm, out_path: Path,
                         n_iter: int, kappa: float) -> None:
    """Save ASD-iMAR image as DICOM by cloning the noMAR template."""
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
    dcm.SeriesDescription = f"ASD-iMAR (N={n_iter}, κ={kappa:.4f})"

    dcm.save_as(str(out_path))


# ---------------------------------------------------------------------------
# Worker (one realization x one condition)
# ---------------------------------------------------------------------------

def _process_one(args):
    """Worker function executed in subprocess."""
    cond, real_idx, dataset_dir, output_dir, tier_id, n_iter, kappa_base, dt = args
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
    kappa = kappa_base / tier.metal_mu_cm
    img_asd = asd_imar_slice(
        sino, ref_hu, tier,
        n_iter=n_iter, kappa_base=kappa_base, dt=dt,
    )
    _save_asd_imar_dicom(img_asd, template_dcm, dcm_dst, n_iter, kappa)

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
        description="ASD-iMAR benchmark for ASTM WKXXXXX v1.0.0 — tier-aware")
    ap.add_argument("--input-dir", default="./astm_reference_dataset",
                    help="Root of the tier dataset (default: ./astm_reference_dataset)")
    ap.add_argument("--tier", default="T1_AB",
                    choices=list(TIER_REGISTRY.keys()),
                    help="Imaging scenario tier (default: T1_AB)")
    ap.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER,
                    help=f"Diffusion iterations (default: {DEFAULT_N_ITER})")
    ap.add_argument("--kappa-base", type=float, default=DEFAULT_KAPPA_BASE,
                    help=f"Base conductance κ (default: {DEFAULT_KAPPA_BASE})")
    ap.add_argument("--dt", type=float, default=DEFAULT_DT,
                    help=f"Diffusion timestep (default: {DEFAULT_DT})")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0 = os.cpu_count())")
    args = ap.parse_args()

    tier        = TIER_REGISTRY[args.tier]
    dataset_dir = Path(args.input_dir).resolve()
    output_dir  = dataset_dir / "asd_imar_recon"
    n_workers   = args.workers if args.workers > 0 else os.cpu_count()
    kappa       = args.kappa_base / tier.metal_mu_cm

    print(f"Tier       : {tier.tier_id}  —  {tier.description}")
    print(f"Body       : {tier.body_semi_x_vox}×{tier.body_semi_y_vox} vox  "
          f"({'circle' if tier.is_circular_body else 'ellipse'})")
    print(f"Metal      : {tier.metal_material}  μ={tier.metal_mu_cm} cm⁻¹  "
          f"r={tier.metal_radius_vox} vox  blockage={tier.blockage_frac*100:.1f}%")
    print(f"Lesion     : {tier.lesion_semi_major_vox}×{tier.lesion_semi_minor_vox} vox  "
          f"gap={tier.gap_vox} vox ({tier.gap_mm:.1f} mm)")
    print(f"Diffusion  : N={args.n_iter}  κ_base={args.kappa_base}  "
          f"κ_eff={kappa:.4f}  dt={args.dt}")
    print(f"Dataset    : {dataset_dir}")
    print(f"Output     : {output_dir}")
    print(f"Workers    : {n_workers}  |  Algorithm: ASD-iMAR "
          f"(structure-tensor anisotropic diffusion)")

    tasks = [
        (cond, r, str(dataset_dir), str(output_dir), args.tier,
         args.n_iter, args.kappa_base, args.dt)
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

    print(f"\nASD-iMAR output → {output_dir}")
    print("Next step:")
    print(f"  python run_cho_analysis_v6_0.py \\")
    print(f"      --dataset-dir {dataset_dir} \\")
    print(f"      --mar-output-dir {output_dir} \\")
    print(f"      --tier {args.tier} \\")
    print(f"      --internal-noise-sigma 15 \\")
    print(f"      --results-file asd_imar_{args.tier.lower()}_results.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
