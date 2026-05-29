#!/usr/bin/env python3
"""
plot_t3_head_comparison.py
==========================
Generate one LP and one LA realization for T3_HEAD, forward-project to
sinograms, and produce the LP-vs-LA comparison plot with quantitative
annotations.  Also prints raw numerical summary for technical reporting.

No pre-generated dataset required — everything runs in-memory.
"""

import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── T3_HEAD geometry (from tier_config.py) ────────────────────────────────────
VOXEL_MM  = 0.5
VOXEL_CM  = VOXEL_MM / 10.0
N_ANGLES  = 360
N_DET     = 512
X_DIM     = 512
Y_DIM     = 512

PHANTOM_CENTER_X = 256
PHANTOM_CENTER_Y = 256

ANGLES_DEG = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)

# T3_HEAD: circular body r=100 mm = 200 vox
BODY_RADIUS_VOX   = round(100.0 / VOXEL_MM)  # 200

# Ti-6Al-4V rod: r=8 mm = 16 vox, μ=1.5 cm⁻¹ at 60 keV
METAL_RADIUS_VOX  = round(8.0 / VOXEL_MM)    # 16
MU_METAL_CM       = 1.5

# Lesion: 3×2 vox ellipse at x=320
LESION_CENTER_X       = 320
LESION_SEMI_MAJOR_VOX = round(1.5 / VOXEL_MM)  # 3
LESION_SEMI_MINOR_VOX = round(1.0 / VOXEL_MM)  # 2

MU_AIR_CM        = 0.000196
MU_TISSUE_CM     = 0.2059
LESION_DELTA_HU  = 12.0
BACKGROUND_HU    = 40.0
METAL_HU_RESTORE = 3000.0

SCATTER_FRAC     = 0.05
SIGMA_E_COUNTS   = 5.0
BASE_SEED        = 20260314

# Blockage fraction: (2/pi)*arcsin(R_metal / l_nominal)
L_NOMINAL_VOX    = LESION_CENTER_X - PHANTOM_CENTER_X  # 64
import math
BLOCKAGE_FRAC    = (2.0 / math.pi) * math.asin(METAL_RADIUS_VOX / L_NOMINAL_VOX)

# ── Masks ─────────────────────────────────────────────────────────────────────
yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]

body_mask = (
    (xx - PHANTOM_CENTER_X) ** 2 + (yy - PHANTOM_CENTER_Y) ** 2
    <= BODY_RADIUS_VOX ** 2
)

metal_mask = (
    (xx - PHANTOM_CENTER_X) ** 2 + (yy - PHANTOM_CENTER_Y) ** 2
    <= METAL_RADIUS_VOX ** 2
)

lesion_mask = (
    ((xx - LESION_CENTER_X) / max(LESION_SEMI_MAJOR_VOX, 1)) ** 2
    + ((yy - PHANTOM_CENTER_Y) / max(LESION_SEMI_MINOR_VOX, 1)) ** 2
    <= 1.0
)


def build_mu(place_lesion):
    mu = np.full((Y_DIM, X_DIM), MU_AIR_CM, dtype=np.float64)
    mu[body_mask] = MU_TISSUE_CM
    mu[metal_mask] = MU_METAL_CM
    if place_lesion:
        mu_lesion = MU_TISSUE_CM * (1.0 + LESION_DELTA_HU / 1000.0)
        mu[lesion_mask] = mu_lesion
    return mu


def forward_project(mu):
    sino = np.zeros((N_ANGLES, N_DET), dtype=np.float64)
    mu64 = mu.astype(np.float64)
    for i, ang in enumerate(ANGLES_DEG):
        rot = scipy.ndimage.rotate(
            mu64, -ang, reshape=False, order=1,
            mode="constant", cval=MU_AIR_CM,
        )
        sino[i] = rot.sum(axis=0) * VOXEL_CM
    return sino


def apply_noise(sino_clean, I0, rng):
    S = SCATTER_FRAC * I0
    I_expected = I0 * np.exp(-sino_clean) + S
    I_measured = rng.poisson(I_expected).astype(np.float64)
    I_measured += rng.normal(0.0, SIGMA_E_COUNTS, size=I_measured.shape)
    I_measured = np.maximum(I_measured, 0.1)
    return (-np.log(I_measured / I0)).astype(np.float32)


def main():
    out_path = Path("results_archive/plots/t3_head_lp_vs_la_sinogram.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Quick I0 calibration ──────────────────────────────────────────────────
    print("Calibrating I0 for T3_HEAD...")
    mu_body = np.full((Y_DIM, X_DIM), MU_AIR_CM, dtype=np.float64)
    mu_body[body_mask] = MU_TISSUE_CM
    sino_body_nf = forward_project(mu_body)

    # Raw FBP helper
    def fbp_raw(sino):
        n_proj, n_det = sino.shape
        freq = np.fft.rfftfreq(n_det)
        ramp = np.abs(freq)
        filtered = np.fft.irfft(
            np.fft.rfft(sino.astype(np.float64), axis=1) * ramp, n=n_det, axis=1
        )
        recon = np.zeros((Y_DIM, X_DIM), dtype=np.float64)
        for i, ang in enumerate(ANGLES_DEG):
            recon += scipy.ndimage.rotate(
                np.tile(filtered[i], (Y_DIM, 1)),
                ang, reshape=False, order=1, mode="constant", cval=0.0,
            )
        return recon * np.pi / n_proj / VOXEL_CM

    # Cal ROI (upper body region)
    by = BODY_RADIUS_VOX
    y0 = max(0, PHANTOM_CENTER_Y - round(0.90 * by))
    y1 = min(Y_DIM, PHANTOM_CENTER_Y - round(0.40 * by))
    x0 = max(0, PHANTOM_CENTER_X - round(0.20 * by))
    x1 = min(X_DIM, PHANTOM_CENTER_X + round(0.20 * by))

    mu_body_recon = fbp_raw(sino_body_nf)
    dc_cal = float(mu_body_recon[y0:y1, x0:x1].mean()) - MU_TISSUE_CM

    I0_ref = 1e5
    rng_cal = np.random.default_rng(42)
    sigma_samples = []
    for _ in range(5):
        sino_mc = apply_noise(sino_body_nf, I0_ref, rng_cal)
        mu_mc = fbp_raw(sino_mc)
        hu_mc = (mu_mc - MU_TISSUE_CM - dc_cal) / MU_TISSUE_CM * 1000
        sigma_samples.append(float(hu_mc[y0:y1, x0:x1].std()))
    sigma_ref = np.mean(sigma_samples)
    I0_cal = I0_ref * (sigma_ref / 30.0) ** 2
    print(f"  I0 = {I0_cal:.0f}")

    # ── Build LP and LA sinograms ─────────────────────────────────────────────
    print("Forward projecting LP (lesion present)...")
    mu_lp = build_mu(place_lesion=True)
    sino_lp_clean = forward_project(mu_lp)

    print("Forward projecting LA (lesion absent)...")
    mu_la = build_mu(place_lesion=False)
    sino_la_clean = forward_project(mu_la)

    # Add noise (realization 1)
    rng_lp = np.random.default_rng(BASE_SEED + 1)
    rng_la = np.random.default_rng(BASE_SEED + 1)
    sino_lp = apply_noise(sino_lp_clean, I0_cal, rng_lp)
    sino_la = apply_noise(sino_la_clean, I0_cal, rng_la)

    diff = sino_lp.astype(np.float64) - sino_la.astype(np.float64)

    # ── Quantitative analysis ─────────────────────────────────────────────────
    # Metal shadow region: detector bins near centre
    metal_lo = N_DET // 2 - METAL_RADIUS_VOX - 2
    metal_hi = N_DET // 2 + METAL_RADIUS_VOX + 2
    # Lesion shadow region: approximate detector offset
    lesion_det_center = N_DET // 2 + L_NOMINAL_VOX
    lesion_lo = lesion_det_center - LESION_SEMI_MAJOR_VOX - 2
    lesion_hi = lesion_det_center + LESION_SEMI_MAJOR_VOX + 2

    # Metal-to-tissue contrast ratio
    metal_tissue_ratio = MU_METAL_CM / MU_TISSUE_CM

    # Peak line integral through metal
    peak_lp = float(sino_lp.max())
    peak_la = float(sino_la.max())
    mean_tissue_lp = float(np.median(sino_lp[:, metal_hi + 20:metal_hi + 60]))

    # Difference statistics
    diff_metal_region = diff[:, metal_lo:metal_hi]
    diff_lesion_region = diff[:, lesion_lo:lesion_hi]
    diff_outside = np.concatenate([diff[:, :metal_lo - 30], diff[:, metal_hi + 30:]], axis=1)

    # Photon starvation metric: fraction of rays where I_measured < 100
    I_through_metal_lp = I0_cal * np.exp(-sino_lp_clean[:, metal_lo:metal_hi])
    photon_starved_frac = float((I_through_metal_lp < 100).sum()) / I_through_metal_lp.size

    print("\n" + "=" * 72)
    print("T3_HEAD SINOGRAM COMPARISON — QUANTITATIVE SUMMARY")
    print("=" * 72)
    print(f"\n  Phantom geometry:")
    print(f"    Body:    circular, r = {BODY_RADIUS_VOX * VOXEL_MM:.0f} mm ({BODY_RADIUS_VOX} vox)")
    print(f"    Metal:   Ti-6Al-4V, r = {METAL_RADIUS_VOX * VOXEL_MM:.0f} mm ({METAL_RADIUS_VOX} vox)")
    print(f"    Lesion:  {LESION_SEMI_MAJOR_VOX}x{LESION_SEMI_MINOR_VOX} vox ellipse at x={LESION_CENTER_X}")
    print(f"    l_nominal = {L_NOMINAL_VOX} vox ({L_NOMINAL_VOX * VOXEL_MM:.1f} mm)")
    print(f"    Blockage fraction = {BLOCKAGE_FRAC * 100:.2f}%")

    print(f"\n  Attenuation coefficients (60 keV):")
    print(f"    mu_tissue  = {MU_TISSUE_CM:.4f} cm^-1")
    print(f"    mu_Ti6Al4V = {MU_METAL_CM:.4f} cm^-1")
    print(f"    mu ratio   = {metal_tissue_ratio:.2f}x")
    mu_cocr = 4.2
    mu_ss316l = 2.8
    print(f"    (cf. CoCr = {mu_cocr:.1f} cm^-1 [{mu_cocr/MU_TISSUE_CM:.1f}x], "
          f"SS316L = {mu_ss316l:.1f} cm^-1 [{mu_ss316l/MU_TISSUE_CM:.1f}x])")

    print(f"\n  Sinogram statistics:")
    print(f"    Peak line integral (LP):  {peak_lp:.3f} neper")
    print(f"    Peak line integral (LA):  {peak_la:.3f} neper")
    print(f"    Median tissue region:     {mean_tissue_lp:.3f} neper")
    print(f"    Metal/tissue peak ratio:  {peak_la / mean_tissue_lp:.2f}x")

    print(f"\n  LP - LA difference:")
    print(f"    Metal region:  mean = {diff_metal_region.mean():.6f}, "
          f"std = {diff_metal_region.std():.6f} neper")
    print(f"    Lesion region: mean = {diff_lesion_region.mean():.6f}, "
          f"std = {diff_lesion_region.std():.6f} neper")
    print(f"    Background:    mean = {diff_outside.mean():.6f}, "
          f"std = {diff_outside.std():.6f} neper")
    print(f"    Lesion SNR (diff_mean/diff_std): "
          f"{abs(diff_lesion_region.mean()) / max(diff_lesion_region.std(), 1e-12):.3f}")

    print(f"\n  Photon starvation:")
    print(f"    Fraction of metal-shadow rays with I < 100 photons: "
          f"{photon_starved_frac * 100:.2f}%")
    # Compare with CoCr
    I_through_cocr = I0_cal * np.exp(-sino_la_clean[:, metal_lo:metal_hi] * (mu_cocr / MU_METAL_CM))
    cocr_starved = float((I_through_cocr < 100).sum()) / I_through_cocr.size
    print(f"    (cf. CoCr equivalent:  {cocr_starved * 100:.2f}%)")
    I_through_ss = I0_cal * np.exp(-sino_la_clean[:, metal_lo:metal_hi] * (mu_ss316l / MU_METAL_CM))
    ss_starved = float((I_through_ss < 100).sum()) / I_through_ss.size
    print(f"    (cf. SS316L equivalent: {ss_starved * 100:.2f}%)")

    # Dynamic range
    dr_ti = peak_la / mean_tissue_lp
    dr_cocr_approx = peak_la * (mu_cocr / MU_METAL_CM) / mean_tissue_lp
    dr_ss_approx = peak_la * (mu_ss316l / MU_METAL_CM) / mean_tissue_lp
    print(f"\n  Dynamic range (peak_metal / median_tissue):")
    print(f"    Ti-6Al-4V: {dr_ti:.2f}x")
    print(f"    SS316L:    ~{dr_ss_approx:.2f}x  (scaled estimate)")
    print(f"    CoCr:      ~{dr_cocr_approx:.2f}x  (scaled estimate)")

    print("=" * 72)

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nRendering comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # LP sinogram
    ax = axes[0, 0]
    im = ax.imshow(sino_lp, aspect="auto", cmap="inferno",
                   extent=[0, N_DET, 180, 0])
    ax.set_title("LP (Lesion Present)", fontsize=11)
    ax.set_ylabel("Projection angle (deg)")
    ax.set_xlabel("Detector bin")
    plt.colorbar(im, ax=ax, label="neper", shrink=0.85)

    # LA sinogram
    ax = axes[0, 1]
    im = ax.imshow(sino_la, aspect="auto", cmap="inferno",
                   extent=[0, N_DET, 180, 0])
    ax.set_title("LA (Lesion Absent)", fontsize=11)
    ax.set_ylabel("Projection angle (deg)")
    ax.set_xlabel("Detector bin")
    plt.colorbar(im, ax=ax, label="neper", shrink=0.85)

    # Difference map
    ax = axes[1, 0]
    lim = max(np.abs(diff).max() * 0.5, 1e-6)  # tighten for visibility
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r",
                   extent=[0, N_DET, 180, 0],
                   vmin=-lim, vmax=lim)
    ax.set_title("LP \u2212 LA  (lesion signal in sinogram domain)", fontsize=11)
    ax.set_ylabel("Projection angle (deg)")
    ax.set_xlabel("Detector bin")
    plt.colorbar(im, ax=ax, label="\u0394 neper", shrink=0.85)
    # Mark lesion shadow location
    ax.axvline(lesion_det_center, color="lime", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(lesion_det_center + 5, 10, "lesion", color="lime", fontsize=8, va="top")

    # Detector profiles at 0 deg
    ax = axes[1, 1]
    ax.plot(sino_lp[0], color="#d62728", linewidth=1.0, label="LP", alpha=0.8)
    ax.plot(sino_la[0], color="#1f77b4", linewidth=1.0, label="LA", alpha=0.8)
    ax.set_title("Detector profile at 0\u00b0 projection", fontsize=11)
    ax.set_xlabel("Detector bin")
    ax.set_ylabel("Line integral (neper)")
    ax.legend(fontsize=9)
    ax.axvspan(metal_lo, metal_hi, alpha=0.10, color="gray")
    ax.annotate(
        f"Ti-6Al-4V shadow\n\u03bc = {MU_METAL_CM:.1f} cm\u207b\u00b9\n"
        f"({metal_tissue_ratio:.1f}\u00d7 tissue)",
        xy=(N_DET // 2, sino_la[0, N_DET // 2]),
        xytext=(N_DET // 2 - 110, peak_la * 0.85),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray"),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
    )
    # Mark lesion location
    ax.axvspan(lesion_lo, lesion_hi, alpha=0.15, color="green")
    ax.annotate(
        "Lesion\n(+12 HU)",
        xy=(lesion_det_center, sino_lp[0, lesion_det_center]),
        xytext=(lesion_det_center + 30, sino_lp[0, lesion_det_center] + 0.3),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="green"),
        bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "T3_HEAD LP vs LA Sinogram Comparison\n"
        f"Ti-6Al-4V rod (\u03bc={MU_METAL_CM} cm\u207b\u00b9, r={METAL_RADIUS_VOX * VOXEL_MM:.0f} mm)  |  "
        f"Blockage = {BLOCKAGE_FRAC * 100:.1f}%  |  "
        f"l_nominal = {L_NOMINAL_VOX * VOXEL_MM:.0f} mm  |  "
        f"Lesion = +{LESION_DELTA_HU:.0f} HU",
        fontsize=11.5, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
