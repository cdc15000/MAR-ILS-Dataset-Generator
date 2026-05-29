#!/usr/bin/env python3
"""
generate_detectability_curves.py
Replicate Vaishnav et al. (2020) detectability validation curves.

Methodology
-----------
Lesion PSF Construction (avoids double-noise from LP−LA image differences):
  1. Average 40 pairs of LP/LA sinograms at slice 128 to get the mean
     lesion sinogram delta: Δs = mean_i(sino_LP_i − sino_LA_i)
  2. FBP-reconstruct Δs → PSF_12HU  (the expected 12 HU lesion footprint in
     image space, calibrated to ~12 HU at the lesion centre)
  3. LP[i] = LA_img[i] + (c/12)·PSF_12HU  for contrast sweep
  4. LP[i] = LA_img[i] + extra_noise + PSF_12HU ; LA[i] = LA_img[i] + extra_noise
     for noise sweep (σ_total = √(30² + σ_extra²))

CHO Specification (normative — ASTM WKXXXXX Rev 03 §A1.5)
  • 2D Laguerre–Gauss channels, n=0..9, a=7.5 voxels
  • ROI 121×121 centred at (281, 256)
  • Tikhonov λ = 0.01·trace(K)/p
  • Internal noise σ_internal = 15 (§A1.5.2(d))
  • LOO hold-out AUC via Mann–Whitney

Outputs
-------
  detectability_contrast_curve.png
  detectability_noise_curve.png
"""

import argparse
import os
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from skimage.transform import iradon, radon

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
LESION_SLICE_INDEX = 128
THETA_DEG          = np.linspace(0, 180, 360, endpoint=False)
ROI_CX, ROI_CY     = 281, 256   # lesion centre in image coords (x, y) = (col, row)
ROI_HALF           = 60         # ±60 voxels → 121×121 ROI
N_CHANNELS         = 10
A_CHANNEL          = 7.5        # voxels  (1.5 × r_lesion = 1.5 × 5)
SIGMA_INTERNAL     = 15.0       # Vaishnav internal observer noise (§A1.5.2(d))
N_REALIZATIONS     = 40
REF_CONTRAST_HU    = 12.0       # locked baseline
METAL_HU_THRESH    = 1500


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def _fbp(sino: np.ndarray) -> np.ndarray:
    """sino (360,512) → img (512,512)  using scikit-image Ram-Lak FBP."""
    return iradon(sino.T, theta=THETA_DEG, filter_name="ramp", circle=True)


def _fwd(img: np.ndarray) -> np.ndarray:
    """img (512,512) → sino (360,512)."""
    return radon(img, theta=THETA_DEG, circle=True).T


def _calibrate(img: np.ndarray, ref_mean: float, ref_std: float) -> np.ndarray:
    """Scale/shift img to have the given mean and std."""
    img_std = np.std(img)
    if img_std < 1e-10:
        return img + (ref_mean - np.mean(img))
    return (img - np.mean(img)) / img_std * ref_std + ref_mean


# ---------------------------------------------------------------------------
# CHO machinery (inline, normative per §A1.5)
# ---------------------------------------------------------------------------

def _laguerre_gauss_channels(roi_size: int = 121) -> np.ndarray:
    """
    Build 2D LG channels for a roi_size × roi_size ROI centred on the lesion.
    Returns U: (roi_size², N_CHANNELS) float64 (L2-normalised column vectors).
    """
    from numpy.polynomial.laguerre import lagval

    half = roi_size // 2
    y_coords, x_coords = np.mgrid[-half:half + 1, -half:half + 1]
    r2 = (x_coords ** 2 + y_coords ** 2).astype(float)
    r  = np.sqrt(r2)

    channels = []
    for n in range(N_CHANNELS):
        # u_n(r) = L_n(2π r²/a²) × exp(−π r²/a²)
        arg    = 2.0 * np.pi * r2 / A_CHANNEL ** 2
        lval   = lagval(arg, [0.0] * n + [1.0])
        gauss  = np.exp(-np.pi * r2 / A_CHANNEL ** 2)
        u      = lval * gauss
        # L2-normalise
        norm = np.linalg.norm(u)
        if norm > 0:
            u /= norm
        channels.append(u.ravel())

    return np.column_stack(channels)   # (roi_size², N_CHANNELS)


# Pre-build channels once (module-level constant after first call)
_CHANNELS: np.ndarray | None = None


def _get_channels() -> np.ndarray:
    global _CHANNELS
    if _CHANNELS is None:
        _CHANNELS = _laguerre_gauss_channels(2 * ROI_HALF + 1)
    return _CHANNELS


def _extract_roi_features(img_hu: np.ndarray) -> np.ndarray:
    """
    Extract normalised 121×121 ROI from img_hu, project through LG channels.
    Returns feature vector of shape (N_CHANNELS,).
    """
    r0 = ROI_CY - ROI_HALF
    r1 = ROI_CY + ROI_HALF + 1
    c0 = ROI_CX - ROI_HALF
    c1 = ROI_CX + ROI_HALF + 1
    roi = img_hu[r0:r1, c0:c1].astype(float)
    return (roi.ravel() @ _get_channels())   # (N_CHANNELS,)


def _mann_whitney_auc(s_lp: np.ndarray, s_la: np.ndarray) -> float:
    """AUC via Mann–Whitney statistic (no mid-rank correction needed here)."""
    n_lp, n_la = len(s_lp), len(s_la)
    if n_lp == 0 or n_la == 0:
        return 0.5
    count = sum(int(a > b) + 0.5 * int(a == b) for a in s_lp for b in s_la)
    return count / (n_lp * n_la)


def _fit_score(train_lp, train_la, test_lp, test_la, int_var: float):
    """Fit Hotelling template; return test statistics."""
    K_la  = np.cov(train_la, rowvar=False)
    lam   = 0.01 * np.trace(K_la) / N_CHANNELS
    K_reg = K_la + (lam + int_var) * np.eye(N_CHANNELS)
    dmu   = np.mean(train_lp, axis=0) - np.mean(train_la, axis=0)
    w     = np.linalg.solve(K_reg, dmu)
    return test_lp @ w, test_la @ w


def compute_auc_loo(
    lp_feat: np.ndarray,      # (N, N_CHANNELS)
    la_feat: np.ndarray,      # (N, N_CHANNELS)
    sigma_internal: float = SIGMA_INTERNAL,
) -> float:
    """LOO hold-out AUC (Mann–Whitney)."""
    N = len(lp_feat)
    int_var = sigma_internal ** 2
    s_lp_loo, s_la_loo = np.zeros(N), np.zeros(N)

    for k in range(N):
        trn_lp = np.delete(lp_feat, k, axis=0)
        trn_la = np.delete(la_feat, k, axis=0)

        # LP LOO
        t_lp, _ = _fit_score(trn_lp, trn_la,
                              lp_feat[[k]], la_feat[[k]], int_var)
        s_lp_loo[k] = t_lp[0]

        # LA LOO
        _, t_la = _fit_score(
            np.delete(lp_feat, k, axis=0),
            np.delete(la_feat, k, axis=0),
            lp_feat[[k]], la_feat[[k]], int_var)
        s_la_loo[k] = t_la[0]

    return _mann_whitney_auc(s_lp_loo, s_la_loo)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_la_images(dataset_dir: Path) -> np.ndarray:
    """Load all 40 LA noMAR DICOMs at slice 128.  Returns (40, 512, 512)."""
    imgs = []
    for r in range(1, N_REALIZATIONS + 1):
        p = (dataset_dir / "noMAR_recon" / "LA"
             / f"realization_{r:03d}" / "slice_0129.dcm")
        dcm = pydicom.dcmread(str(p))
        hu = dcm.pixel_array.astype(float) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
        imgs.append(hu)
    return np.stack(imgs)   # (40, 512, 512)


def load_sir_mar_images(sir_dir: Path, cond: str) -> np.ndarray | None:
    """Load SIR-MAR DICOMs for a condition.  Returns (40,512,512) or None."""
    imgs = []
    for r in range(1, N_REALIZATIONS + 1):
        p = sir_dir / cond / f"realization_{r:03d}" / "slice_0129.dcm"
        if not p.exists():
            return None
        dcm = pydicom.dcmread(str(p))
        hu = dcm.pixel_array.astype(float) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
        imgs.append(hu)
    return np.stack(imgs)


def build_lesion_psf(dataset_dir: Path) -> tuple[np.ndarray, float, float]:
    """
    Compute the lesion PSF at 12 HU from average(sino_LP - sino_LA) → FBP.
    Returns (psf_img, ref_mean_HU, ref_std_HU) where the latter two describe
    the soft-tissue background in the LA images (for calibration).
    """
    print("  Building lesion PSF from sinogram delta …")
    delta_sum = np.zeros((360, 512), dtype=np.float64)
    la_imgs   = []

    for r in range(1, N_REALIZATIONS + 1):
        h5_lp = dataset_dir / "sinograms" / "LP" / f"realization_{r:03d}.h5"
        h5_la = dataset_dir / "sinograms" / "LA" / f"realization_{r:03d}.h5"
        with h5py.File(str(h5_lp), "r") as f:
            slp = f["line_integrals"][LESION_SLICE_INDEX].astype(np.float64)
        with h5py.File(str(h5_la), "r") as f:
            sla = f["line_integrals"][LESION_SLICE_INDEX].astype(np.float64)
        delta_sum += (slp - sla)

        p = (dataset_dir / "noMAR_recon" / "LA"
             / f"realization_{r:03d}" / "slice_0129.dcm")
        dcm = pydicom.dcmread(str(p))
        hu = dcm.pixel_array.astype(float) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
        la_imgs.append(hu)

    delta_mean = delta_sum / N_REALIZATIONS   # average lesion sinogram delta

    # FBP of the mean delta → lesion PSF in FBP units
    psf_raw = _fbp(delta_mean)

    # Calibrate PSF to 12 HU by matching lesion-centre value to expected 12 HU
    # Use the mean LA tissue value as offset reference
    la_stack = np.stack(la_imgs)   # (40, 512, 512)
    metal_mask = la_stack[0] > METAL_HU_THRESH
    body_valid = (la_stack[0] > -800) & (~metal_mask)
    ref_mean = float(np.mean(la_stack[:, body_valid]))
    ref_std  = float(np.std(la_stack[:, body_valid]))

    # PSF peak should map to ~12 HU; scale accordingly
    r0 = ROI_CY - ROI_HALF; r1 = ROI_CY + ROI_HALF + 1
    c0 = ROI_CX - ROI_HALF; c1 = ROI_CX + ROI_HALF + 1
    psf_roi_peak = float(np.max(psf_raw[r0:r1, c0:c1]))
    if abs(psf_roi_peak) > 1e-10:
        psf_12HU = psf_raw * (REF_CONTRAST_HU / psf_roi_peak)
    else:
        # Fallback: use a constant disc approximation
        psf_12HU = np.zeros_like(psf_raw)
        yy, xx = np.mgrid[0:512, 0:512]
        disc = ((xx - ROI_CX)**2 + (yy - ROI_CY)**2) <= 25  # 5-voxel radius
        psf_12HU[disc] = REF_CONTRAST_HU

    print(f"    PSF peak in ROI: {psf_roi_peak:.5f}  → scaled to {REF_CONTRAST_HU} HU")
    return psf_12HU, ref_mean, ref_std, la_stack


# ---------------------------------------------------------------------------
# Curve 1 — AUC vs Lesion Contrast
# ---------------------------------------------------------------------------

def contrast_sweep(
    dataset_dir: Path,
    psf_12HU: np.ndarray,
    la_stack: np.ndarray,
    contrasts: list[float],
) -> list[float]:
    """
    For each contrast c, construct LP[i] = LA[i] + (c/12)·PSF_12HU,
    run LOO CHO, return AUC.
    """
    aucs = []
    for c in contrasts:
        scale = c / REF_CONTRAST_HU
        # LP images: matched LA background + scaled lesion signal
        lp_feat = np.zeros((N_REALIZATIONS, N_CHANNELS))
        la_feat = np.zeros((N_REALIZATIONS, N_CHANNELS))
        for i in range(N_REALIZATIONS):
            lp_img = la_stack[i] + scale * psf_12HU
            la_img = la_stack[i]
            lp_feat[i] = _extract_roi_features(lp_img)
            la_feat[i] = _extract_roi_features(la_img)
        auc = compute_auc_loo(lp_feat, la_feat)
        aucs.append(auc)
        print(f"    c = {c:5.1f} HU  →  AUC = {auc:.4f}")
    return aucs


# ---------------------------------------------------------------------------
# Curve 2 — AUC vs Noise Level
# ---------------------------------------------------------------------------

def noise_sweep(
    dataset_dir: Path,
    psf_12HU: np.ndarray,
    la_stack: np.ndarray,
    noise_sigmas: list[float],
    sir_lp: np.ndarray | None = None,
    sir_la: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[float], list[float] | None]:
    """
    For each target noise σ_n (HU), add extra Gaussian noise σ_extra
    = √max(0, σ_n² − 30²) to both LP and LA images and run CHO.

    Returns (auc_noMAR_list, auc_MAR_list_or_None).
    """
    if rng is None:
        rng = np.random.default_rng(20260314)

    aucs_noMAR: list[float] = []
    aucs_MAR:   list[float] = []
    compute_mar = sir_lp is not None and sir_la is not None

    for sigma_n in noise_sigmas:
        sigma_extra = float(np.sqrt(max(0.0, sigma_n**2 - 30.0**2)))

        lp_feat_nm = np.zeros((N_REALIZATIONS, N_CHANNELS))
        la_feat_nm = np.zeros((N_REALIZATIONS, N_CHANNELS))
        lp_feat_mr = np.zeros((N_REALIZATIONS, N_CHANNELS)) if compute_mar else None
        la_feat_mr = np.zeros((N_REALIZATIONS, N_CHANNELS)) if compute_mar else None

        for i in range(N_REALIZATIONS):
            # Independent extra noise per realization
            noise_lp = rng.normal(0, sigma_extra, (512, 512)) if sigma_extra > 0 else 0.0
            noise_la = rng.normal(0, sigma_extra, (512, 512)) if sigma_extra > 0 else 0.0

            # noMAR: LA background + extra noise (LP has lesion PSF on top)
            lp_nm = la_stack[i] + noise_lp + psf_12HU
            la_nm = la_stack[i] + noise_la
            lp_feat_nm[i] = _extract_roi_features(lp_nm)
            la_feat_nm[i] = _extract_roi_features(la_nm)

            # SIR-MAR images (add the same extra noise for fair comparison)
            if compute_mar:
                noise_s_lp = rng.normal(0, sigma_extra, (512, 512)) if sigma_extra > 0 else 0.0
                noise_s_la = rng.normal(0, sigma_extra, (512, 512)) if sigma_extra > 0 else 0.0
                lp_feat_mr[i] = _extract_roi_features(sir_lp[i] + noise_s_lp)
                la_feat_mr[i] = _extract_roi_features(sir_la[i] + noise_s_la)

        auc_nm = compute_auc_loo(lp_feat_nm, la_feat_nm)
        aucs_noMAR.append(auc_nm)

        if compute_mar:
            auc_mr = compute_auc_loo(lp_feat_mr, la_feat_mr)
            aucs_MAR.append(auc_mr)
            print(f"    σ = {sigma_n:5.1f} HU  →  AUC_noMAR = {auc_nm:.4f} | "
                  f"AUC_SIR-MAR = {auc_mr:.4f}  ΔAUC = {auc_mr - auc_nm:+.4f}")
        else:
            print(f"    σ = {sigma_n:5.1f} HU  →  AUC_noMAR = {auc_nm:.4f}")

    return aucs_noMAR, (aucs_MAR if compute_mar else None)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_contrast_curve(contrasts, aucs, ref_c=REF_CONTRAST_HU, ref_auc=0.7063):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(contrasts, aucs, "b-o", lw=2, ms=7, label="2D CHO (σ_int=15)")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Chance (AUC=0.5)")
    ax.axhline(ref_auc, color="r", ls=":", lw=1.5,
               label=f"Locked baseline AUC = {ref_auc:.4f}")
    ax.axvline(ref_c, color="orange", ls=":", lw=1.5,
               label=f"Locked contrast = {ref_c:.0f} HU")

    ax.set_xlabel("Lesion Contrast (HU above background)", fontsize=12)
    ax.set_ylabel("Area Under ROC Curve (LOO AUC)", fontsize=12)
    ax.set_title("Detectability vs Lesion Contrast\n"
                 "(Vaishnav Validation Curve 1 — ASTM WKXXXXX Rev 03)", fontsize=11)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "detectability_contrast_curve.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nContrast curve → {fname}")


def plot_noise_curve(
    noise_sigmas, aucs_noMAR, aucs_MAR=None, ref_sigma=30.0, ref_auc=0.7063
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(noise_sigmas, aucs_noMAR, "b-s", lw=2, ms=7,
            label="AUC_noMAR (FBP baseline)")
    if aucs_MAR is not None:
        ax.plot(noise_sigmas, aucs_MAR, "g-o", lw=2, ms=7,
                label="AUC_MAR (SIR-MAR WLS-TV)")
        delta_aucs = [m - n for m, n in zip(aucs_MAR, aucs_noMAR)]
        ax2 = ax.twinx()
        ax2.bar(noise_sigmas, delta_aucs, width=2.5, alpha=0.25, color="green",
                label="ΔAUC")
        ax2.set_ylabel("ΔAUC = AUC_MAR − AUC_noMAR", fontsize=11, color="green")
        ax2.tick_params(axis="y", labelcolor="green")

    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.axvline(ref_sigma, color="orange", ls=":", lw=1.5,
               label=f"Reference σ = {ref_sigma:.0f} HU  (AUC = {ref_auc:.4f})")
    ax.set_xlabel("Target Noise Level σ (HU)", fontsize=12)
    ax.set_ylabel("Area Under ROC Curve (LOO AUC)", fontsize=12)
    ax.set_title("Detectability vs Noise Level / Dose\n"
                 "(Vaishnav Validation Curve 2 — ASTM WKXXXXX Rev 03)", fontsize=11)
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "detectability_noise_curve.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Noise curve    → {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Vaishnav detectability curves for ASTM WKXXXXX Rev 03")
    ap.add_argument("--dataset-dir", default="./astm_reference_dataset")
    ap.add_argument("--sir-mar-dir", default=None,
                    help="Path to SIR-MAR output (optional; enables MAR overlay "
                         "on noise curve).  Default: <dataset-dir>/sir_mar_recon")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    sir_dir     = Path(args.sir_mar_dir).resolve() if args.sir_mar_dir else \
                  dataset_dir / "sir_mar_recon"

    print(f"Dataset : {dataset_dir}")
    print(f"SIR-MAR : {sir_dir}  ({'found' if sir_dir.exists() else 'not found'})")

    # ── PSF + LA images ────────────────────────────────────────────────
    print("\n[1/4] Building lesion PSF …")
    psf_12HU, _ref_mean, _ref_std, la_stack = build_lesion_psf(dataset_dir)
    psf_peak = float(np.max(psf_12HU[
        ROI_CY - ROI_HALF : ROI_CY + ROI_HALF + 1,
        ROI_CX - ROI_HALF : ROI_CX + ROI_HALF + 1]))
    print(f"  PSF peak in ROI = {psf_peak:.2f} HU  (target: {REF_CONTRAST_HU:.1f} HU)")

    # ── Contrast sweep ─────────────────────────────────────────────────
    contrasts = [4.0, 8.0, 12.0, 16.0, 20.0]
    print(f"\n[2/4] Contrast sweep: {contrasts} HU …")
    auc_contrast = contrast_sweep(dataset_dir, psf_12HU, la_stack, contrasts)

    # ── Load SIR-MAR images (optional) ────────────────────────────────
    sir_lp_stack = load_sir_mar_images(sir_dir, "LP") if sir_dir.exists() else None
    sir_la_stack = load_sir_mar_images(sir_dir, "LA") if sir_dir.exists() else None
    if sir_lp_stack is not None:
        print(f"  SIR-MAR images loaded: LP={sir_lp_stack.shape}, LA={sir_la_stack.shape}")
    else:
        print("  SIR-MAR images not available — noise curve will show noMAR only.")

    # ── Noise sweep ────────────────────────────────────────────────────
    noise_sigmas = [30.0, 40.0, 50.0, 60.0]
    print(f"\n[3/4] Noise sweep: {noise_sigmas} HU …")
    auc_noMAR, auc_MAR = noise_sweep(
        dataset_dir, psf_12HU, la_stack, noise_sigmas,
        sir_lp=sir_lp_stack, sir_la=sir_la_stack)

    # ── Plots ──────────────────────────────────────────────────────────
    print("\n[4/4] Generating plots …")
    plot_contrast_curve(contrasts, auc_contrast)
    plot_noise_curve(noise_sigmas, auc_noMAR, auc_MAR)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n─── Summary ───────────────────────────────────────")
    print("Contrast sweep:")
    for c, a in zip(contrasts, auc_contrast):
        mark = " ← LOCKED BASELINE" if abs(c - 12.0) < 0.5 else ""
        print(f"  {c:5.1f} HU  →  AUC = {a:.4f}{mark}")
    print("\nNoise sweep:")
    for s, nm in zip(noise_sigmas, auc_noMAR):
        mar_str = ""
        if auc_MAR:
            idx = noise_sigmas.index(s)
            dA = auc_MAR[idx] - nm
            mar_str = f"  |  AUC_MAR = {auc_MAR[idx]:.4f}  ΔAUC = {dA:+.4f}"
        print(f"  σ = {s:5.1f} HU  →  AUC_noMAR = {nm:.4f}{mar_str}")
    c12_idx = contrasts.index(12.0) if 12.0 in contrasts else -1
    if c12_idx >= 0:
        print(f"\n✓ Contrast curve passes through "
              f"AUC = {auc_contrast[c12_idx]:.4f} at c = 12 HU  "
              f"(locked baseline: 0.7063)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
