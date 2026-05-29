#!/usr/bin/env python3
"""
plot_spectral_transparency.py
=============================
Generate a single-realization T2_SB sinogram at 60 keV and 140 keV,
then plot the pair side-by-side with a difference map to visualise the
spectral transparency jump through the SS-316L rod.

No dataset generation required — everything is computed in-memory.

Usage:
    python plot_spectral_transparency.py
    python plot_spectral_transparency.py --output spectral_comparison.png
    python plot_spectral_transparency.py --slice 128  # default
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Shared acquisition geometry (from tier_config) ────────────────────────────
VOXEL_MM  = 0.5
VOXEL_CM  = VOXEL_MM / 10.0
N_ANGLES  = 360
N_DET     = 512
X_DIM     = 512
Y_DIM     = 512

PHANTOM_CENTER_X = 256
PHANTOM_CENTER_Y = 256

ANGLES_DEG = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)

# ── T2_SB geometry ────────────────────────────────────────────────────────────
BODY_SEMI_X_VOX = round(42.5 / VOXEL_MM)   # 85 vox
BODY_SEMI_Y_VOX = round(30.0 / VOXEL_MM)   # 60 vox
METAL_RADIUS_VOX = round(5.0 / VOXEL_MM)   # 10 vox
LESION_CENTER_X  = 273
LESION_SEMI_MAJOR_VOX = round(2.0 / VOXEL_MM)  # 4 vox
LESION_SEMI_MINOR_VOX = round(1.5 / VOXEL_MM)  # 3 vox

# ── Attenuation coefficients (NIST XCOM) ─────────────────────────────────────
#                       60 keV      140 keV
MU_AIR_CM          = 0.000196     # essentially the same at both
MU_TISSUE_60       = 0.2059       # cm⁻¹  (soft tissue, 60 keV)
MU_TISSUE_140      = 0.1492       # cm⁻¹  (soft tissue, 140 keV)
MU_SS316L_60       = 2.80         # cm⁻¹  (316L stainless steel, 60 keV)
MU_SS316L_140      = 0.58         # cm⁻¹  (316L stainless steel, 140 keV)

LESION_DELTA_HU    = 12.0         # normative +12 HU contrast

# Metal-to-tissue ratios for annotation
RATIO_60  = MU_SS316L_60  / MU_TISSUE_60    # ~13.6×
RATIO_140 = MU_SS316L_140 / MU_TISSUE_140   # ~3.9×


def _build_body_mask(yy, xx):
    return (
        ((xx - PHANTOM_CENTER_X) / BODY_SEMI_X_VOX) ** 2
        + ((yy - PHANTOM_CENTER_Y) / BODY_SEMI_Y_VOX) ** 2
        <= 1.0
    )


def _build_metal_mask(yy, xx):
    return (
        (xx - PHANTOM_CENTER_X) ** 2 + (yy - PHANTOM_CENTER_Y) ** 2
        <= METAL_RADIUS_VOX ** 2
    )


def _build_lesion_mask(yy, xx):
    lma = max(LESION_SEMI_MAJOR_VOX, 1)
    lmi = max(LESION_SEMI_MINOR_VOX, 1)
    return (
        ((xx - LESION_CENTER_X) / lma) ** 2
        + ((yy - PHANTOM_CENTER_Y) / lmi) ** 2
        <= 1.0
    )


def build_mu_map(mu_tissue, mu_metal, place_lesion=True):
    """Build 2D attenuation map for one energy."""
    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    mu = np.full((Y_DIM, X_DIM), MU_AIR_CM, dtype=np.float64)
    mu[_build_body_mask(yy, xx)] = mu_tissue
    mu[_build_metal_mask(yy, xx)] = mu_metal
    if place_lesion:
        mu_lesion = mu_tissue * (1.0 + LESION_DELTA_HU / 1000.0)
        mu[_build_lesion_mask(yy, xx)] = mu_lesion
    return mu


def forward_project(mu):
    """Parallel-beam forward projection → sinogram (N_ANGLES, N_DET) in neper."""
    sino = np.zeros((N_ANGLES, N_DET), dtype=np.float64)
    mu64 = mu.astype(np.float64)
    for i, ang in enumerate(ANGLES_DEG):
        rot = scipy.ndimage.rotate(
            mu64, -ang, reshape=False, order=1,
            mode="constant", cval=MU_AIR_CM,
        )
        sino[i] = rot.sum(axis=0) * VOXEL_CM
    return sino


def main():
    parser = argparse.ArgumentParser(
        description="Plot 60 keV vs 140 keV sinogram transparency for T2_SB"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="results_archive/plots/spectral_transparency_60v140.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Build attenuation maps ────────────────────────────────────────────────
    print("Building T2_SB attenuation maps...")
    mu_60  = build_mu_map(MU_TISSUE_60,  MU_SS316L_60,  place_lesion=True)
    mu_140 = build_mu_map(MU_TISSUE_140, MU_SS316L_140, place_lesion=True)

    # ── Forward project (noise-free) ──────────────────────────────────────────
    print("Forward projecting at 60 keV...")
    sino_60 = forward_project(mu_60)
    print("Forward projecting at 140 keV...")
    sino_140 = forward_project(mu_140)

    diff = sino_60 - sino_140

    # ── Peak line-integral through metal centre ───────────────────────────────
    peak_60  = sino_60.max()
    peak_140 = sino_140.max()

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("Rendering figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: 60 keV sinogram
    ax = axes[0, 0]
    im0 = ax.imshow(
        sino_60, aspect="auto", cmap="inferno",
        extent=[0, N_DET, 180, 0],
    )
    ax.set_title(f"60 keV sinogram  (peak = {peak_60:.2f} neper)", fontsize=11)
    ax.set_ylabel("Projection angle (deg)")
    ax.set_xlabel("Detector bin")
    plt.colorbar(im0, ax=ax, label="Line integral (neper)", shrink=0.85)

    # Top-right: 140 keV sinogram
    ax = axes[0, 1]
    im1 = ax.imshow(
        sino_140, aspect="auto", cmap="inferno",
        extent=[0, N_DET, 180, 0],
        vmin=0, vmax=sino_60.max(),  # shared scale to show contrast drop
    )
    ax.set_title(f"140 keV sinogram  (peak = {peak_140:.2f} neper)", fontsize=11)
    ax.set_ylabel("Projection angle (deg)")
    ax.set_xlabel("Detector bin")
    plt.colorbar(im1, ax=ax, label="Line integral (neper)", shrink=0.85)

    # Bottom-left: difference map
    ax = axes[1, 0]
    lim = np.abs(diff).max()
    im2 = ax.imshow(
        diff, aspect="auto", cmap="RdBu_r",
        extent=[0, N_DET, 180, 0],
        vmin=-lim, vmax=lim,
    )
    ax.set_title("60 keV \u2212 140 keV  (transparency jump)", fontsize=11)
    ax.set_ylabel("Projection angle (deg)")
    ax.set_xlabel("Detector bin")
    plt.colorbar(im2, ax=ax, label="\u0394 Line integral (neper)", shrink=0.85)

    # Bottom-right: detector profile at 0° projection
    ax = axes[1, 1]
    ax.plot(sino_60[0], color="#d62728", linewidth=1.2, label="60 keV")
    ax.plot(sino_140[0], color="#1f77b4", linewidth=1.2, label="140 keV")
    ax.set_title("Detector profile at 0\u00b0 projection", fontsize=11)
    ax.set_xlabel("Detector bin")
    ax.set_ylabel("Line integral (neper)")
    ax.legend(loc="upper right", fontsize=9)
    ax.axvspan(
        N_DET / 2 - METAL_RADIUS_VOX, N_DET / 2 + METAL_RADIUS_VOX,
        alpha=0.12, color="gray", label="_metal shadow",
    )
    ax.annotate(
        f"Metal shadow\n\u03bc ratio: {RATIO_60:.1f}\u00d7 \u2192 {RATIO_140:.1f}\u00d7",
        xy=(N_DET / 2, max(sino_60[0, N_DET // 2], sino_140[0, N_DET // 2])),
        xytext=(N_DET / 2 + 60, peak_60 * 0.8),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="gray"),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "T2_SB Spectral Transparency Jump  \u2014  SS-316L at 60 keV vs 140 keV\n"
        f"\u03bc_metal: {MU_SS316L_60:.2f} \u2192 {MU_SS316L_140:.2f} cm\u207b\u00b9  "
        f"({MU_SS316L_140/MU_SS316L_60*100:.0f}% reduction)    "
        f"\u03bc_tissue: {MU_TISSUE_60:.4f} \u2192 {MU_TISSUE_140:.4f} cm\u207b\u00b9  "
        f"({MU_TISSUE_140/MU_TISSUE_60*100:.0f}% reduction)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    print(f"\n--- Spectral summary ---")
    print(f"  SS-316L \u03bc:  {MU_SS316L_60:.3f} \u2192 {MU_SS316L_140:.3f} cm\u207b\u00b9  "
          f"({(1 - MU_SS316L_140/MU_SS316L_60)*100:.0f}% drop)")
    print(f"  Tissue \u03bc:   {MU_TISSUE_60:.4f} \u2192 {MU_TISSUE_140:.4f} cm\u207b\u00b9  "
          f"({(1 - MU_TISSUE_140/MU_TISSUE_60)*100:.0f}% drop)")
    print(f"  Metal/tissue ratio: {RATIO_60:.1f}\u00d7 \u2192 {RATIO_140:.1f}\u00d7")
    print(f"  Peak sinogram:  {peak_60:.2f} \u2192 {peak_140:.2f} neper  "
          f"({(1 - peak_140/peak_60)*100:.0f}% drop)")


if __name__ == "__main__":
    main()
