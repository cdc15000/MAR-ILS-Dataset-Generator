#!/usr/bin/env python3
"""
reference_spectral_mar.py — Dual-Energy Spectral Metal Artifact Reduction
=========================================================================
ASTM WKXXXXX v1.0.0 — Tier-aware reference implementation.

Simulates dual-energy CT acquisition to exploit the energy-dependent opacity
of metal implants.  At high keV (140 keV), the metal rod transmits ~55% of
photons (vs ~19% at 60 keV), so metal-trace rays carry usable lesion signal.

Algorithm
---------
1.  Load the 60 keV sinogram from the ILS dataset (existing HDF5).
2.  Synthesize a 140 keV sinogram from the same phantom geometry + noise.
3.  Basis Material Decomposition (water + metal) in the sinogram domain.
4.  Synthesize a Virtual Monochromatic Image (VMI) sinogram at 70 keV.
5.  FBP reconstruct the VMI sinogram.
6.  Metal-weighted spectral blend: for metal-trace rays, weight the 140 keV
    data more heavily (cleaner, higher transmission); for clean rays, use
    standard VMI synthesis.

The key advantage over sinogram-inpainting MAR (iMAR, NMAR, FS-iMAR): the
high-energy metal-trace rays CONTAIN the lesion signal.  No prior-based
replacement is needed; the lesion information is physically present in the
140 keV measurement.

Usage
-----
    PYTHONPATH=. python algorithms/reference_spectral_mar.py \\
        --input-dir ./spectral_t2_sb \\
        --tier T2_SB \\
        --workers 0

Author  : ASTM F04 Subcommittee Working Draft
Standard: ASTM WKXXXXX v1.0.0
Date    : 2026-03-17
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage
import pydicom
from pydicom.uid import generate_uid

from tier_config import (
    TierConfig, TIER_REGISTRY, validate_tier_registry,
    VOXEL_MM, VOXEL_CM, N_ANGLES, N_DET,
    PHANTOM_CENTER_X, PHANTOM_CENTER_Y, LESION_SLICE_INDEX,
    MU_AIR_CM, MU_TISSUE_CM, BACKGROUND_HU, METAL_HU_RESTORE,
    SCATTER_FRAC, SIGMA_E_COUNTS, LESION_DELTA_HU,
    MU_LESION_SCALE, NUM_REALIZATIONS,
)

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# Volume geometry
# ═════════════════════════════════════════════════════════════════════════════
X_DIM = Y_DIM = 512
ANGLES_DEG = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)

# ═════════════════════════════════════════════════════════════════════════════
# Energy-dependent linear attenuation coefficients (cm⁻¹)
# Monochromatic values from NIST XCOM (parameterized for metals)
# ═════════════════════════════════════════════════════════════════════════════
ENERGY_LOW  = 60   # keV — matches existing ILS dataset
ENERGY_HIGH = 140  # keV — high-energy acquisition
VMI_ENERGY  = 70   # keV — optimal VMI synthesis energy

MU_WATER = {
    60:  0.2059,   # cm⁻¹  (normative baseline)
    70:  0.1929,   # NIST XCOM
    140: 0.1505,   # NIST XCOM
}

MU_METAL = {
    "SS316L":  {60: 2.800, 70: 2.260, 140: 1.000},
    "CoCr":    {60: 4.200, 70: 3.360, 140: 1.470},
    "Ti6Al4V": {60: 1.500, 70: 1.170, 140: 0.467},
}

# Lesion: same fractional excess at all energies (Compton-dominated water)
MU_LESION_FRAC = MU_LESION_SCALE   # 1.012 → +12 HU at any energy

# ═════════════════════════════════════════════════════════════════════════════
# Forward projection & FBP  (self-contained, matches generator_v6_0_0.py)
# ═════════════════════════════════════════════════════════════════════════════

def _forward_project(mu: np.ndarray) -> np.ndarray:
    """Parallel-beam forward projection.  mu (512,512) cm⁻¹ → sino (360,512) neper."""
    sino = np.zeros((N_ANGLES, N_DET), dtype=np.float64)
    mu64 = mu.astype(np.float64)
    for i, ang in enumerate(ANGLES_DEG):
        rot = scipy.ndimage.rotate(
            mu64, -ang, reshape=False, order=1,
            mode='constant', cval=MU_AIR_CM,
        )
        sino[i] = rot.sum(axis=0) * VOXEL_CM
    return sino


def _apply_noise(sino_clean: np.ndarray, I0: float, rng: np.random.Generator) -> np.ndarray:
    """Physics-based CT noise: Poisson + electronic.  Returns noisy sinogram (neper)."""
    S = SCATTER_FRAC * I0
    I_expected = I0 * np.exp(-sino_clean) + S
    I_measured = rng.poisson(I_expected).astype(np.float64)
    I_measured += rng.normal(0.0, SIGMA_E_COUNTS, size=I_measured.shape)
    I_measured = np.maximum(I_measured, 0.1)
    return (-np.log(I_measured / I0)).astype(np.float32)


def _fbp_raw(sino: np.ndarray) -> np.ndarray:
    """Ram-Lak FBP → μ map (cm⁻¹).  No HU conversion."""
    n_proj, n_det = sino.shape
    freq = np.fft.rfftfreq(n_det)
    ramp = np.abs(freq)
    filtered = np.fft.irfft(
        np.fft.rfft(sino.astype(np.float64), axis=1) * ramp, n=n_det, axis=1
    )
    recon = np.zeros((Y_DIM, X_DIM), dtype=np.float64)
    for i, ang in enumerate(ANGLES_DEG):
        recon += scipy.ndimage.rotate(
            np.tile(filtered[i], (Y_DIM, 1)), ang,
            reshape=False, order=1, mode='constant', cval=0.0,
        )
    return (recon * np.pi / n_proj / VOXEL_CM).astype(np.float32)


def _mu_to_hu(mu_recon: np.ndarray, mu_water: float, dc_offset: float = 0.0) -> np.ndarray:
    """Convert reconstructed μ (cm⁻¹) to HU at the specified water reference energy."""
    return ((mu_recon - mu_water - dc_offset) / mu_water * 1000.0
            + BACKGROUND_HU).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Phantom construction at arbitrary energy
# ═════════════════════════════════════════════════════════════════════════════

def _build_body_mask(yy, xx, tier):
    if tier.is_circular_body:
        return ((xx - PHANTOM_CENTER_X)**2 + (yy - PHANTOM_CENTER_Y)**2
                <= tier.body_semi_x_vox**2)
    return (
        ((xx - PHANTOM_CENTER_X) / tier.body_semi_x_vox)**2 +
        ((yy - PHANTOM_CENTER_Y) / tier.body_semi_y_vox)**2
        <= 1.0
    )


def _build_metal_mask(yy, xx, tier):
    return ((xx - PHANTOM_CENTER_X)**2 + (yy - PHANTOM_CENTER_Y)**2
            <= tier.metal_radius_vox**2)


def _build_lesion_mask(yy, xx, tier):
    lma = max(tier.lesion_semi_major_vox, 1)
    lmi = max(tier.lesion_semi_minor_vox, 1)
    return (
        ((xx - tier.lesion_center_x) / lma)**2 +
        ((yy - PHANTOM_CENTER_Y)     / lmi)**2
        <= 1.0
    )


def build_phantom_at_energy(
    energy_keV: int,
    tier: TierConfig,
    place_lesion: bool,
    jitter_deg: float = 0.0,
) -> np.ndarray:
    """
    Build 512×512 attenuation map at the specified energy.

    Uses the energy-dependent MU_WATER and MU_METAL tables.
    Lesion fractional excess (MU_LESION_FRAC = 1.012) is energy-independent.
    """
    mu_w = MU_WATER[energy_keV]
    mu_m = MU_METAL[tier.metal_material][energy_keV]

    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    body_mask   = _build_body_mask(yy, xx, tier)
    metal_mask  = _build_metal_mask(yy, xx, tier)

    mu = np.full((Y_DIM, X_DIM), MU_AIR_CM, dtype=np.float64)
    mu[body_mask]  = mu_w
    mu[metal_mask] = mu_m

    if place_lesion:
        lesion_mask = _build_lesion_mask(yy, xx, tier)
        mu[lesion_mask] = mu_w * MU_LESION_FRAC

    if abs(jitter_deg) > 1e-6:
        mu = scipy.ndimage.rotate(
            mu, -jitter_deg, reshape=False, order=1,
            mode='constant', cval=MU_AIR_CM,
        )
        body_rot = scipy.ndimage.rotate(
            body_mask.astype(np.float32), -jitter_deg,
            reshape=False, order=1, mode='constant', cval=0.0,
        ) > 0.5
        mu[~body_rot] = MU_AIR_CM

    return mu.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Metal-trace identification via forward-projected metal mask
# ═════════════════════════════════════════════════════════════════════════════

def _metal_trace_mask(tier: TierConfig, jitter_deg: float = 0.0,
                      threshold: float = 0.01) -> np.ndarray:
    """
    Binary sinogram mask: True where the ray passes through the metal rod.

    Threshold = 0.01 neper (~1% of metal contribution) ensures we catch
    all rays with any metal in the beam path.
    """
    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    metal_only = np.zeros((Y_DIM, X_DIM), dtype=np.float64)
    metal_mask = _build_metal_mask(yy, xx, tier)
    metal_only[metal_mask] = tier.metal_mu_cm

    if abs(jitter_deg) > 1e-6:
        metal_only = scipy.ndimage.rotate(
            metal_only, -jitter_deg, reshape=False, order=1,
            mode='constant', cval=0.0,
        )

    sino_metal = _forward_project(metal_only)
    return sino_metal > threshold


# ═════════════════════════════════════════════════════════════════════════════
# Basis Material Decomposition + VMI Synthesis
# ═════════════════════════════════════════════════════════════════════════════

def basis_material_decomposition(
    sino_low: np.ndarray,
    sino_high: np.ndarray,
    metal_material: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-material (water + metal) sinogram-domain decomposition.

    Given sinograms at E_low=60 keV and E_high=140 keV, solve per-ray:
        [p_low ]   [μ_w(60)  μ_m(60) ] [A_w]
        [p_high] = [μ_w(140) μ_m(140)] [A_m]

    Returns (A_water, A_metal) — basis material path-length sinograms.
    """
    mu_w_lo = MU_WATER[ENERGY_LOW]
    mu_w_hi = MU_WATER[ENERGY_HIGH]
    mu_m_lo = MU_METAL[metal_material][ENERGY_LOW]
    mu_m_hi = MU_METAL[metal_material][ENERGY_HIGH]

    det = mu_w_lo * mu_m_hi - mu_w_hi * mu_m_lo
    A_water = (mu_m_hi * sino_low  - mu_m_lo * sino_high) / det
    A_metal = (mu_w_lo * sino_high - mu_w_hi * sino_low)  / det

    return A_water, A_metal


def synthesize_vmi(
    A_water: np.ndarray,
    A_metal: np.ndarray,
    metal_material: str,
    target_keV: int = VMI_ENERGY,
) -> np.ndarray:
    """
    Synthesize a Virtual Monochromatic Image (VMI) sinogram at target_keV.

    sino_vmi = μ_w(target) × A_water + μ_m(target) × A_metal
    """
    mu_w = MU_WATER[target_keV]
    mu_m = MU_METAL[metal_material][target_keV]
    return mu_w * A_water + mu_m * A_metal


# ═════════════════════════════════════════════════════════════════════════════
# Spectral MAR — core algorithm
# ═════════════════════════════════════════════════════════════════════════════

def spectral_mar_slice(
    sino_60: np.ndarray,
    tier: TierConfig,
    I0: float,
    jitter_deg: float,
    place_lesion: bool,
    seed_60: int,
    dc_offset_60: float = 0.0,
    metal_blend_weight: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Dual-energy spectral MAR for one 2D slice.

    Steps
    -----
    1. Build phantom at 140 keV (same geometry + jitter as 60 keV).
    2. Forward project at 140 keV and apply independent noise.
    3. Basis material decomposition (water + metal) in sinogram domain.
    4. Synthesize VMI sinogram at 70 keV.
    5. Metal-weighted spectral blend:
       - For clean rays: standard VMI (from both energies)
       - For metal-trace rays: weight the 140 keV data more heavily
    6. FBP reconstruct → HU image.

    Returns
    -------
    hu_spectral : (512, 512) float32 — MAR-corrected HU image
    diagnostics : dict — intermediate statistics for debugging
    """
    # ── Step 1: Generate 140 keV sinogram ─────────────────────────────
    mu_140 = build_phantom_at_energy(ENERGY_HIGH, tier, place_lesion, jitter_deg)
    sino_140_clean = _forward_project(mu_140)

    # Independent noise: offset seed to ensure uncorrelated noise
    rng_140 = np.random.default_rng(seed_60 + 2_000_000)
    sino_140 = _apply_noise(sino_140_clean, I0, rng_140)

    # ── Step 2: Metal-trace mask ──────────────────────────────────────
    metal_trace = _metal_trace_mask(tier, jitter_deg)

    # ── Step 3: Basis Material Decomposition ──────────────────────────
    sino_60_f64 = sino_60.astype(np.float64)
    sino_140_f64 = sino_140.astype(np.float64)
    A_water, A_metal = basis_material_decomposition(
        sino_60_f64, sino_140_f64, tier.metal_material
    )

    # ── Step 4: VMI sinogram at 70 keV ────────────────────────────────
    sino_vmi_70 = synthesize_vmi(A_water, A_metal, tier.metal_material, VMI_ENERGY)

    # ── Step 5: Metal-weighted spectral blend ─────────────────────────
    #
    # For metal-trace rays, the 60 keV data suffers from photon starvation
    # noise. The 140 keV data is cleaner because the metal transmits more
    # photons.  We blend toward the high-energy contribution:
    #
    #   sino_blend = (1-w) × sino_vmi_70 + w × sino_140_scaled
    #
    # where w = 0 for clean rays, w = metal_blend_weight for metal traces,
    # and sino_140_scaled is the 140 keV sinogram rescaled to 70 keV
    # tissue contrast: sino_140 × (μ_w(70) / μ_w(140)).
    #
    # The scaling preserves the tissue HU calibration while using the
    # cleaner high-energy metal-trace data.
    #
    METAL_BLEND_WEIGHT = metal_blend_weight

    mu_w70  = MU_WATER[VMI_ENERGY]
    mu_w140 = MU_WATER[ENERGY_HIGH]
    sino_140_at_70 = sino_140_f64 * (mu_w70 / mu_w140)

    sino_blend = sino_vmi_70.copy()
    sino_blend[metal_trace] = (
        (1.0 - METAL_BLEND_WEIGHT) * sino_vmi_70[metal_trace] +
        METAL_BLEND_WEIGHT * sino_140_at_70[metal_trace]
    )

    # ── Step 6: FBP reconstruction ────────────────────────────────────
    mu_recon = _fbp_raw(sino_blend.astype(np.float32))

    # DC offset: scale from 60 keV calibration
    dc_offset_vmi = dc_offset_60 * (mu_w70 / MU_WATER[ENERGY_LOW])
    hu_spectral = _mu_to_hu(mu_recon, mu_w70, dc_offset_vmi)

    # Metal restoration
    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    img_metal_mask = _build_metal_mask(yy, xx, tier)
    if abs(jitter_deg) > 1e-6:
        img_metal_mask = scipy.ndimage.rotate(
            img_metal_mask.astype(np.float32), -jitter_deg,
            reshape=False, order=1, mode='constant', cval=0.0,
        ) > 0.5
    hu_spectral[img_metal_mask] = METAL_HU_RESTORE

    # ── Diagnostics ───────────────────────────────────────────────────
    xs, xe = tier.roi_x_bounds()
    ys, ye = tier.roi_y_bounds()
    roi_vmi = hu_spectral[ys:ye, xs:xe]

    # Transmission stats for metal rays
    trans_60 = np.exp(-sino_60_f64[metal_trace]).mean() if metal_trace.any() else 0
    trans_140 = np.exp(-sino_140_f64[metal_trace]).mean() if metal_trace.any() else 0

    diag = {
        "roi_mean": float(roi_vmi.mean()),
        "roi_std": float(roi_vmi.std()),
        "metal_trace_frac": float(metal_trace.mean()),
        "trans_60_mean": float(trans_60),
        "trans_140_mean": float(trans_140),
        "A_water_range": [float(A_water.min()), float(A_water.max())],
        "A_metal_range": [float(A_metal.min()), float(A_metal.max())],
    }

    return hu_spectral, diag


# ═════════════════════════════════════════════════════════════════════════════
# DICOM I/O
# ═════════════════════════════════════════════════════════════════════════════

def _load_noMAR_dicom(dcm_path: Path) -> tuple[np.ndarray, object]:
    dcm = pydicom.dcmread(str(dcm_path))
    hu = (dcm.pixel_array.astype(float) * float(dcm.RescaleSlope)
          + float(dcm.RescaleIntercept))
    return hu, dcm


def _save_mar_dicom(img_hu: np.ndarray, template_dcm, out_path: Path) -> None:
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
    dcm.SeriesDescription = "Spectral MAR (70 keV VMI)"
    dcm.save_as(str(out_path))


# ═════════════════════════════════════════════════════════════════════════════
# Worker and main
# ═════════════════════════════════════════════════════════════════════════════

def _read_dc_offset(dataset_dir: Path) -> float:
    """Read DC offset from generator_provenance.json if available."""
    prov_path = dataset_dir / "generator_provenance.json"
    if prov_path.exists():
        import json
        with open(prov_path) as f:
            prov = json.load(f)
        return prov.get("dc_offset_cm", 0.0)
    return 0.0


def _process_one(args) -> tuple:
    """Process one realization."""
    cond, real_idx, dataset_dir, output_dir, tier_id, blend_weight = args
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    tier = TIER_REGISTRY[tier_id]

    tag = f"realization_{real_idx:03d}"
    h5_path = dataset_dir / "sinograms" / cond / f"{tag}.h5"
    dcm_src = dataset_dir / "noMAR_recon" / cond / tag / "slice_0129.dcm"
    dcm_dst = output_dir / cond / tag / "slice_0129.dcm"

    if dcm_dst.exists():
        return cond, real_idx, True, {}

    # Read 60 keV sinogram + metadata
    with h5py.File(str(h5_path), "r") as f:
        sino_60 = f["line_integrals"][LESION_SLICE_INDEX].astype(np.float64)
        I0 = float(f["noise_params"].attrs["I0"])
        seed = int(f["noise_params"].attrs["seed"])
        jitter_deg = float(f["noise_params"].attrs["jitter_deg"])
        place_lesion = bool(f["noise_params"].attrs["place_lesion"])

    dc_offset = _read_dc_offset(dataset_dir)

    # Run spectral MAR
    hu_spectral, diag = spectral_mar_slice(
        sino_60, tier, I0, jitter_deg, place_lesion, seed, dc_offset,
        metal_blend_weight=blend_weight,
    )

    # Load template DICOM and save
    _, template_dcm = _load_noMAR_dicom(dcm_src)
    _save_mar_dicom(hu_spectral, template_dcm, dcm_dst)

    return cond, real_idx, False, diag


def generate_sinogram_comparison(
    tier: TierConfig,
    I0: float,
    output_path: str = "spectral_vs_single_energy_sinogram.png",
) -> None:
    """Generate the spectral comparison visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n  Generating spectral comparison plot...", flush=True)

    # Build phantoms at both energies (with lesion, no jitter for visualization)
    mu_60  = build_phantom_at_energy(ENERGY_LOW, tier, True, 0.0)
    mu_140 = build_phantom_at_energy(ENERGY_HIGH, tier, True, 0.0)

    # Forward project
    sino_60_clean  = _forward_project(mu_60)
    sino_140_clean = _forward_project(mu_140)

    # Add noise (fixed seed for reproducibility)
    rng_60  = np.random.default_rng(42)
    rng_140 = np.random.default_rng(43)
    sino_60  = _apply_noise(sino_60_clean, I0, rng_60)
    sino_140 = _apply_noise(sino_140_clean, I0, rng_140)

    # BMD + VMI
    A_w, A_m = basis_material_decomposition(
        sino_60.astype(np.float64), sino_140.astype(np.float64), tier.metal_material
    )
    sino_vmi = synthesize_vmi(A_w, A_m, tier.metal_material, VMI_ENERGY)

    # Metal-weighted blend
    metal_trace = _metal_trace_mask(tier, 0.0)
    mu_w70  = MU_WATER[VMI_ENERGY]
    mu_w140 = MU_WATER[ENERGY_HIGH]
    sino_140_at_70 = sino_140.astype(np.float64) * (mu_w70 / mu_w140)
    sino_blend = sino_vmi.copy()
    sino_blend[metal_trace] = (
        0.5 * sino_vmi[metal_trace] + 0.5 * sino_140_at_70[metal_trace]
    )

    # FBP reconstructions
    mu_w60 = MU_WATER[ENERGY_LOW]
    hu_60  = _mu_to_hu(_fbp_raw(sino_60), mu_w60)
    hu_140 = _mu_to_hu(_fbp_raw(sino_140), mu_w140)
    hu_vmi = _mu_to_hu(_fbp_raw(sino_blend.astype(np.float32)), mu_w70)

    # Metal mask for image display
    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    metal_img = _build_metal_mask(yy, xx, tier)
    hu_60[metal_img]  = METAL_HU_RESTORE
    hu_140[metal_img] = METAL_HU_RESTORE
    hu_vmi[metal_img] = METAL_HU_RESTORE

    # ROI
    xs, xe = tier.roi_x_bounds()
    ys, ye = tier.roi_y_bounds()

    # Transmission statistics
    trans_60  = float(np.exp(-sino_60.astype(np.float64)[metal_trace]).mean())
    trans_140 = float(np.exp(-sino_140.astype(np.float64)[metal_trace]).mean())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Sinograms
    vmin_s, vmax_s = 0, 5
    ax = axes[0, 0]
    im0 = ax.imshow(sino_60, aspect='auto', cmap='inferno', vmin=vmin_s, vmax=vmax_s)
    ax.set_title(f"60 keV Sinogram\nMetal μ={tier.metal_mu_cm:.1f} cm⁻¹  "
                 f"Trans={trans_60:.1%}", fontsize=10, fontweight='bold')
    ax.set_ylabel("Projection angle")
    ax.set_xlabel("Detector bin")
    plt.colorbar(im0, ax=ax, fraction=0.046, label="neper")

    ax = axes[0, 1]
    mu_m_hi = MU_METAL[tier.metal_material][ENERGY_HIGH]
    im1 = ax.imshow(sino_140, aspect='auto', cmap='inferno', vmin=vmin_s, vmax=vmax_s)
    ax.set_title(f"140 keV Sinogram\nMetal μ={mu_m_hi:.1f} cm⁻¹  "
                 f"Trans={trans_140:.1%}  ← \"transparent\" rod", fontsize=10, fontweight='bold')
    ax.set_xlabel("Detector bin")
    plt.colorbar(im1, ax=ax, fraction=0.046, label="neper")

    ax = axes[0, 2]
    sino_diff = sino_60.astype(np.float64) - sino_140.astype(np.float64)
    im2 = ax.imshow(sino_diff, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title("Δ Sinogram (60 − 140 keV)\nMetal contrast difference",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Detector bin")
    plt.colorbar(im2, ax=ax, fraction=0.046, label="Δ neper")

    # Row 2: FBP reconstructions
    vmin_h, vmax_h = -200, 400
    ax = axes[1, 0]
    ax.imshow(hu_60, cmap='gray', vmin=vmin_h, vmax=vmax_h)
    rect = plt.Rectangle((xs, ys), xe-xs, ye-ys,
                          linewidth=1.5, edgecolor='cyan', facecolor='none')
    ax.add_patch(rect)
    roi_60 = hu_60[ys:ye, xs:xe]
    ax.set_title(f"60 keV FBP (noMAR)\nROI: μ={roi_60.mean():.0f} σ={roi_60.std():.0f} HU",
                 fontsize=10, fontweight='bold')
    ax.axis('off')

    ax = axes[1, 1]
    ax.imshow(hu_140, cmap='gray', vmin=vmin_h, vmax=vmax_h)
    rect = plt.Rectangle((xs, ys), xe-xs, ye-ys,
                          linewidth=1.5, edgecolor='cyan', facecolor='none')
    ax.add_patch(rect)
    roi_140 = hu_140[ys:ye, xs:xe]
    ax.set_title(f"140 keV FBP\nROI: μ={roi_140.mean():.0f} σ={roi_140.std():.0f} HU",
                 fontsize=10, fontweight='bold')
    ax.axis('off')

    ax = axes[1, 2]
    ax.imshow(hu_vmi, cmap='gray', vmin=vmin_h, vmax=vmax_h)
    rect = plt.Rectangle((xs, ys), xe-xs, ye-ys,
                          linewidth=1.5, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    roi_vmi = hu_vmi[ys:ye, xs:xe]
    ax.set_title(f"70 keV VMI — Spectral MAR\nROI: μ={roi_vmi.mean():.0f} "
                 f"σ={roi_vmi.std():.0f} HU", fontsize=10, fontweight='bold')
    ax.axis('off')

    fig.suptitle(
        f"Spectral MAR — {tier.tier_id} ({tier.description})\n"
        f"Dual-energy (60 + 140 keV) → 70 keV VMI  |  "
        f"Blockage = {tier.blockage_frac*100:.0f}%  |  "
        f"Metal trans: {trans_60:.1%} → {trans_140:.1%}",
        fontsize=13, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Dual-Energy Spectral MAR — Reference Implementation"
    )
    ap.add_argument("--input-dir", default="./spectral_t2_sb",
                    help="Path to tier dataset (with sinograms/ and noMAR_recon/)")
    ap.add_argument("--tier", default="T2_SB",
                    choices=list(TIER_REGISTRY.keys()))
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0 = all CPUs)")
    ap.add_argument("--blend-weight", type=float, default=0.7,
                    help="Metal-trace blend weight (0=pure VMI, 1=pure 140keV)")
    ap.add_argument("--output-dir", default=None,
                    help="Custom output directory (default: <input>/spectral_mar_recon)")
    ap.add_argument("--skip-plot", action="store_true",
                    help="Skip sinogram comparison visualization")
    args = ap.parse_args()

    validate_tier_registry()
    tier = TIER_REGISTRY[args.tier]
    dataset_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "spectral_mar_recon"
    n_workers = args.workers if args.workers > 0 else os.cpu_count()

    print(f"\n{'═'*70}")
    print(f"  Spectral MAR — Dual-Energy (60 + 140 keV) → 70 keV VMI")
    print(f"{'═'*70}")
    print(f"  Tier       : {tier.tier_id}  —  {tier.description}")
    print(f"  Blockage   : {tier.blockage_frac*100:.1f}%")
    print(f"  Metal      : {tier.metal_material}  μ(60)={tier.metal_mu_cm:.1f}  "
          f"μ(140)={MU_METAL[tier.metal_material][ENERGY_HIGH]:.2f} cm⁻¹")
    print(f"  VMI energy : {VMI_ENERGY} keV")
    print(f"  Dataset    : {dataset_dir}")
    print(f"  Output     : {output_dir}")
    print(f"  Workers    : {n_workers}")
    print(f"{'═'*70}\n")

    # Metal transmission comparison
    rod_diam_cm = tier.metal_radius_mm * 2 / 10.0
    trans_60  = np.exp(-tier.metal_mu_cm * rod_diam_cm)
    trans_140 = np.exp(-MU_METAL[tier.metal_material][ENERGY_HIGH] * rod_diam_cm)
    print(f"  Metal transmission through {rod_diam_cm*10:.0f} mm {tier.metal_material} rod:")
    print(f"    60 keV:  {trans_60:.1%}  (μ = {tier.metal_mu_cm:.1f} cm⁻¹)")
    print(f"    140 keV: {trans_140:.1%}  (μ = {MU_METAL[tier.metal_material][ENERGY_HIGH]:.2f} cm⁻¹)")
    print(f"    Improvement: {trans_140/trans_60:.1f}× more photons at high energy")
    print()

    # Build task list
    tasks = [
        (cond, r, str(dataset_dir), str(output_dir), args.tier, args.blend_weight)
        for cond in ("LP", "LA")
        for r in range(1, NUM_REALIZATIONS + 1)
    ]

    t0 = time.time()
    done, skipped = 0, 0
    diag_samples = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_process_one, t): t for t in tasks}
        for fut in as_completed(futs):
            cond, ridx, was_skipped, diag = fut.result()
            if was_skipped:
                skipped += 1
            else:
                done += 1
                if len(diag_samples) < 5:
                    diag_samples.append((f"{cond}/{ridx:03d}", diag))
            total = done + skipped
            print(f"\r  [{total}/{len(tasks)}]  {cond}/realization_{ridx:03d}  "
                  f"{'skip' if was_skipped else 'done'}",
                  end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n\n  Completed {done} realizations in {elapsed:.0f}s "
          f"({skipped} skipped)\n")

    # Print sample diagnostics
    if diag_samples:
        print("  Sample diagnostics:")
        for tag, d in diag_samples[:3]:
            print(f"    {tag}: ROI μ={d['roi_mean']:.1f} σ={d['roi_std']:.1f}  "
                  f"metal_frac={d['metal_trace_frac']:.1%}  "
                  f"trans_60={d['trans_60_mean']:.1%} trans_140={d['trans_140_mean']:.1%}")
        print()

    # Visualization
    if not args.skip_plot:
        # Read I0 from first HDF5
        h5_path = dataset_dir / "sinograms" / "LP" / "realization_001.h5"
        with h5py.File(str(h5_path), "r") as f:
            I0 = float(f["noise_params"].attrs["I0"])
        generate_sinogram_comparison(tier, I0)

    # Print CHO command
    print(f"  Next step — CHO analysis:")
    print(f"    PYTHONPATH=. python run_cho_analysis_v6_0.py \\")
    print(f"        --dataset-dir {dataset_dir} \\")
    print(f"        --mar-output-dir {output_dir} \\")
    print(f"        --tier {args.tier} \\")
    print(f"        --internal-noise-sigma 15")
    print()


if __name__ == "__main__":
    main()
