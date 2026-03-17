#!/usr/bin/env python3
"""
generator_v6_0_0.py
====================
MAR ILS Dataset Generator v6.0.0 — ASTM WKXXXXX v1.0.0 Tiered Framework

Changes from v5.3.0
-------------------
[T1]  Tier-parameterized geometry via TierConfig (tier_config.py).
      Three tiers: T1_AB (mandatory anchor), T2_SB, T3_HEAD.
      Phantom, metal, and lesion geometry are all tier-specific.

[T2]  Elliptical lesion disc (semi-major × semi-minor), not circular.
      Lesion semi-axes, center, and gap are all specified per tier.

[T3]  Tier-specific metal attenuation coefficient (CoCr / SS316L / Ti6Al4V).
      Metal radius is tier-specific (larger for T1_AB high-blockage scenario).

[T4]  --contrast-factor CLI flag: scales μ_lesion (sensitivity sweep).
      contrast_factor = 1.0 gives the normative +12 HU sinogram contrast.
      Value stored in HDF5 /tier_config.attrs['contrast_factor'].

[T5]  --sweep-mode flag: writes full HDF5 but ONLY slice_0129.dcm per
      realization (256× DICOM storage reduction for sensitivity sweeps).
      The noise RNG is advanced identically so slice-128 noise matches a
      full-mode run with the same seed.

[T6]  Tier-aware calibration: body geometry is tier-specific in all three
      FBP calibration passes (DC offset, clean tissue ROI, MC noise draws).

[T7]  HDF5 /tier_config group: all TierConfig fields stored as attrs.

Usage
-----
    python generator_v6_0_0.py --tier T1_AB --output-dir ./t1_ab_dataset
    python generator_v6_0_0.py --tier T2_SB --contrast-factor 0.5
    python generator_v6_0_0.py --tier T1_AB --sweep-mode --contrast-factor 2.0
    python generator_v6_0_0.py --dry-run --tier T3_HEAD

Author  : ASTM F04 Subcommittee Working Draft
Standard: ASTM WKXXXXX v1.0.0
Date    : 2026-03-15
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import scipy.ndimage
import h5py
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from tqdm import tqdm

from tier_config import (
    TierConfig, TIER_REGISTRY, validate_tier_registry,
    VOXEL_MM, VOXEL_CM, N_ANGLES, N_DET,
    PHANTOM_CENTER_X, PHANTOM_CENTER_Y, LESION_SLICE_INDEX,
    MU_AIR_CM, MU_TISSUE_CM, BACKGROUND_HU, METAL_HU_RESTORE,
    SCATTER_FRAC, SIGMA_E_COUNTS, NOISE_SIGMA_TARGET_HU,
    LESION_DELTA_HU, MU_LESION_SCALE, NUM_REALIZATIONS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Volume and acquisition geometry (normative — shared with tier_config)
# ═══════════════════════════════════════════════════════════════════════════════
X_DIM: int = 512
Y_DIM: int = 512
Z_DIM: int = 256

ANGLES_DEG: np.ndarray = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)
ANGLES_RAD: np.ndarray = np.deg2rad(ANGLES_DEG)
DET_COORDS_CM: np.ndarray = (np.arange(N_DET) - N_DET / 2 + 0.5) * VOXEL_CM

# ═══════════════════════════════════════════════════════════════════════════════
# Study parameters
# ═══════════════════════════════════════════════════════════════════════════════
JITTER_MAX_DEG: float = 15.0
BASE_SEED:       int   = 20260314   # YYYYMMDD (v5.3.0 baseline date)
DATASET_VERSION: str   = "v6.0.0"
STANDARD_REF:    str   = "ASTM-WKXXXXX-v1.0.0"
METAL_HU:        float = METAL_HU_RESTORE

# DejaVu Sans paths; fallback to Helvetica if absent
_DEJAVU_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_BOLD    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PDF_FONT        = "DejaVuSans"
PDF_FONT_BOLD   = "DejaVuSans-Bold"


# ═══════════════════════════════════════════════════════════════════════════════
# Tier-specific mask construction
# ═══════════════════════════════════════════════════════════════════════════════

def _build_body_mask(yy: np.ndarray, xx: np.ndarray, tier: TierConfig) -> np.ndarray:
    if tier.is_circular_body:
        return ((xx - PHANTOM_CENTER_X)**2 + (yy - PHANTOM_CENTER_Y)**2
                <= tier.body_semi_x_vox**2)
    return (
        ((xx - PHANTOM_CENTER_X) / tier.body_semi_x_vox)**2 +
        ((yy - PHANTOM_CENTER_Y) / tier.body_semi_y_vox)**2
        <= 1.0
    )


def _build_metal_mask(yy: np.ndarray, xx: np.ndarray, tier: TierConfig) -> np.ndarray:
    return ((xx - PHANTOM_CENTER_X)**2 + (yy - PHANTOM_CENTER_Y)**2
            <= tier.metal_radius_vox**2)


def _build_lesion_mask(yy: np.ndarray, xx: np.ndarray, tier: TierConfig) -> np.ndarray:
    """Elliptical lesion disc (semi-major along x, semi-minor along y)."""
    lma = max(tier.lesion_semi_major_vox, 1)
    lmi = max(tier.lesion_semi_minor_vox, 1)
    return (
        ((xx - tier.lesion_center_x) / lma)**2 +
        ((yy - PHANTOM_CENTER_Y)     / lmi)**2
        <= 1.0
    )


def _cal_roi_bounds(tier: TierConfig) -> tuple[int, int, int, int]:
    """
    Calibration ROI: upper portion of body, well clear of metal and lesion.

    Position: 40–90% of body_semi_y above centre, ±20% of body_semi_x.
    Verified to lie inside the body ellipse for T1_AB, T2_SB, and T3_HEAD.
    """
    by = tier.body_semi_y_vox
    bx = tier.body_semi_x_vox
    y0 = PHANTOM_CENTER_Y - round(0.90 * by)
    y1 = PHANTOM_CENTER_Y - round(0.40 * by)
    x0 = PHANTOM_CENTER_X - round(0.20 * bx)
    x1 = PHANTOM_CENTER_X + round(0.20 * bx)
    return max(0, y0), min(Y_DIM, y1), max(0, x0), min(X_DIM, x1)


# ═══════════════════════════════════════════════════════════════════════════════
# Phantom attenuation map
# ═══════════════════════════════════════════════════════════════════════════════

def build_attenuation_map_v6(
    place_lesion: bool,
    tier: TierConfig,
    jitter_deg: float = 0.0,
    contrast_factor: float = 1.0,
) -> np.ndarray:
    """
    Construct a 2D attenuation map (cm⁻¹) for one axial slice.

    Parameters
    ----------
    place_lesion     : True → LP (lesion present in elliptical disc)
    tier             : TierConfig specifying all phantom geometry
    jitter_deg       : Phantom rotation for this realization (§A1.8)
    contrast_factor  : Multiplier on lesion Δμ (1.0 = normative +12 HU)
    """
    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    body_mask   = _build_body_mask(yy, xx, tier)
    metal_mask  = _build_metal_mask(yy, xx, tier)
    lesion_mask = _build_lesion_mask(yy, xx, tier)

    mu = np.full((Y_DIM, X_DIM), MU_AIR_CM, dtype=np.float64)
    mu[body_mask]  = MU_TISSUE_CM
    mu[metal_mask] = tier.metal_mu_cm

    if place_lesion:
        mu_lesion = MU_TISSUE_CM * (1.0 + contrast_factor * LESION_DELTA_HU / 1000.0)
        mu[lesion_mask] = mu_lesion

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


# ═══════════════════════════════════════════════════════════════════════════════
# Forward projection and FBP (algorithm unchanged from v5.3.0)
# ═══════════════════════════════════════════════════════════════════════════════

def forward_project_slice(mu: np.ndarray) -> np.ndarray:
    """Parallel-beam forward projection.  mu (Y,X) cm⁻¹ → sino (N_ANGLES, N_DET) neper."""
    sino = np.zeros((N_ANGLES, N_DET), dtype=np.float64)
    mu64 = mu.astype(np.float64)
    for i, ang in enumerate(ANGLES_DEG):
        rot = scipy.ndimage.rotate(
            mu64, -ang, reshape=False, order=1,
            mode='constant', cval=MU_AIR_CM,
        )
        sino[i] = rot.sum(axis=0) * VOXEL_CM
    return sino


def apply_noise(sino_clean: np.ndarray, I0: float, rng: np.random.Generator) -> np.ndarray:
    """
    Apply physics-based CT acquisition noise.

      I_meas = Poisson(I₀·exp(−p) + S) + N(0, σ_e²)
      p_meas = −ln(max(I_meas, 0.1) / I₀)
    """
    S = SCATTER_FRAC * I0
    I_expected = I0 * np.exp(-sino_clean) + S
    I_measured = rng.poisson(I_expected).astype(np.float64)
    I_measured += rng.normal(0.0, SIGMA_E_COUNTS, size=I_measured.shape)
    I_measured = np.maximum(I_measured, 0.1)
    return (-np.log(I_measured / I0)).astype(np.float32)


def fbp_reconstruct_slice(sino: np.ndarray, dc_offset_cm: float = 0.0) -> np.ndarray:
    """Ram-Lak FBP with BACKGROUND_HU offset and DC correction."""
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
    mu_recon = recon * np.pi / n_proj / VOXEL_CM
    hu = ((mu_recon - MU_TISSUE_CM - dc_offset_cm) / MU_TISSUE_CM * 1000.0
          + BACKGROUND_HU)
    return hu.astype(np.float32)


def _fbp_raw(sino: np.ndarray) -> np.ndarray:
    """Internal FBP → raw μ map (cm⁻¹). Used during calibration only."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration (tier-specific geometry)
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_v6(
    tier: TierConfig,
    target_sigma_hu: float = NOISE_SIGMA_TARGET_HU,
) -> tuple[float, float]:
    """
    Tier-specific calibration: 3-FBP analytic approach.

      FBP-1  Noise-free full phantom (with metal) → DC offset.
      FBP-2  Noise-free body-only phantom → clean tissue ROI for noise cal.
      FBP-3  N_MC_CAL noisy FBPs at I₀_ref=1×10⁵ → mean σ_ref.
      Analytic scaling: I₀ = I₀_ref × (σ_ref / target_sigma_hu)².

    Returns
    -------
    (I0_calibrated, dc_offset_cm)
    """
    print(f"Calibrating forward model for tier {tier.tier_id!r}...", flush=True)
    y0, y1, x0, x1 = _cal_roi_bounds(tier)
    print(f"  Calibration ROI: y=[{y0}:{y1}] x=[{x0}:{x1}] "
          f"({(y1-y0)*(x1-x0)} pixels)", flush=True)

    print("  [1/3] Noise-free FBP (full phantom, DC offset)...",
          end=" ", flush=True)
    mu_full = build_attenuation_map_v6(False, tier, jitter_deg=0.0)
    sino_full_nf = forward_project_slice(mu_full)
    mu_full_recon = _fbp_raw(sino_full_nf)
    dc_offset_cm = float(mu_full_recon[y0:y1, x0:x1].mean()) - MU_TISSUE_CM
    print(f"DC offset = {dc_offset_cm/MU_TISSUE_CM*1000:+.1f} HU", flush=True)

    print("  [2/3] Noise-free FBP (body-only calibration phantom)...",
          end=" ", flush=True)
    yy_c, xx_c = np.mgrid[0:Y_DIM, 0:X_DIM]
    mu_cal = np.full((Y_DIM, X_DIM), MU_AIR_CM, dtype=np.float32)
    mu_cal[_build_body_mask(yy_c, xx_c, tier)] = MU_TISSUE_CM
    sino_cal_nf = forward_project_slice(mu_cal)
    mu_cal_recon = _fbp_raw(sino_cal_nf)
    dc_cal = float(mu_cal_recon[y0:y1, x0:x1].mean()) - MU_TISSUE_CM
    hu_cal_nf = (mu_cal_recon - MU_TISSUE_CM - dc_cal) / MU_TISSUE_CM * 1000
    nf_std = float(hu_cal_nf[y0:y1, x0:x1].std())
    print(f"done  (ROI noise-free std = {nf_std:.2f} HU)", flush=True)

    I0_ref = 1e5
    N_MC_CAL = 5
    print(f"  [3/3] {N_MC_CAL} noisy FBPs at I₀_ref={I0_ref:.0f}...", flush=True)
    sigmas_ref = []
    for mc_seed in tqdm(range(N_MC_CAL), desc="    MC draws", leave=False):
        rng_mc = np.random.default_rng(900_000 + mc_seed)
        sino_mc = apply_noise(sino_cal_nf, I0_ref, rng_mc)
        mu_mc   = _fbp_raw(sino_mc)
        hu_mc   = (mu_mc - MU_TISSUE_CM - dc_cal) / MU_TISSUE_CM * 1000
        s = float(hu_mc[y0:y1, x0:x1].std())
        sigmas_ref.append(s)
        print(f"    draw {mc_seed+1}/{N_MC_CAL}: σ = {s:.1f} HU", flush=True)

    sigma_ref = float(np.mean(sigmas_ref))
    print(f"  Mean σ_ref = {sigma_ref:.1f} HU  (std = {float(np.std(sigmas_ref)):.1f} HU)",
          flush=True)

    I0_cal = I0_ref * (sigma_ref / target_sigma_hu) ** 2
    print(f"  Calibrated I₀ = {I0_cal:.0f}  (target σ = {target_sigma_hu} HU)",
          flush=True)

    return float(I0_cal), float(dc_offset_cm)


# ═══════════════════════════════════════════════════════════════════════════════
# Sinogram generation (HDF5)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sinogram_realization_v6(
    output_path: Path,
    *,
    place_lesion: bool,
    seed: int,
    jitter_deg: float,
    I0: float,
    realization_idx: int,
    tier: TierConfig,
    contrast_factor: float = 1.0,
) -> None:
    """
    Generate one realization of a 3D sinogram and write to HDF5.

    HDF5 structure
    --------------
    /line_integrals  float32  (Z_DIM, N_ANGLES, N_DET)
    /geometry        group    attrs: geometry parameters
    /noise_params    group    attrs: acquisition noise parameters
    /tier_config     group    attrs: all TierConfig fields + contrast_factor
    """
    rng = np.random.default_rng(seed)

    mu_no = build_attenuation_map_v6(False, tier, jitter_deg)
    sino_no = forward_project_slice(mu_no)

    if place_lesion:
        mu_with = build_attenuation_map_v6(True, tier, jitter_deg, contrast_factor)
        sino_with = forward_project_slice(mu_with)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(output_path), 'w') as f:
        sino_vol = f.create_dataset(
            'line_integrals',
            shape=(Z_DIM, N_ANGLES, N_DET),
            dtype=np.float32,
            compression='gzip', compression_opts=4,
            chunks=(1, N_ANGLES, N_DET),
        )
        for z in range(Z_DIM):
            if place_lesion and z == LESION_SLICE_INDEX:
                sino_vol[z] = apply_noise(sino_with, I0, rng)
            else:
                sino_vol[z] = apply_noise(sino_no, I0, rng)

        geo = f.create_group('geometry')
        geo.attrs['n_slices']      = Z_DIM
        geo.attrs['n_angles']      = N_ANGLES
        geo.attrs['n_det']         = N_DET
        geo.attrs['voxel_mm']      = VOXEL_MM
        geo.attrs['angles_deg']    = ANGLES_DEG.tolist()
        geo.attrs['det_coords_mm'] = (DET_COORDS_CM * 10).tolist()

        np_grp = f.create_group('noise_params')
        np_grp.attrs['I0']                 = I0
        np_grp.attrs['scatter_frac']       = SCATTER_FRAC
        np_grp.attrs['sigma_e_counts']     = SIGMA_E_COUNTS
        np_grp.attrs['seed']               = seed
        np_grp.attrs['jitter_deg']         = jitter_deg
        np_grp.attrs['place_lesion']       = int(place_lesion)
        np_grp.attrs['lesion_slice_index'] = LESION_SLICE_INDEX
        np_grp.attrs['lesion_z_extent']    = 1 if place_lesion else 0

        tc_grp = f.create_group('tier_config')
        for k, v in tier.to_dict().items():
            tc_grp.attrs[k] = v
        tc_grp.attrs['contrast_factor'] = contrast_factor


# ═══════════════════════════════════════════════════════════════════════════════
# DICOM writing
# ═══════════════════════════════════════════════════════════════════════════════

def _write_dicom_slice_v6(
    hu: np.ndarray,
    z: int,
    *,
    output_dir: Path,
    realization_idx: int,
    condition_label: str,
    study_uid: str,
    series_uid: str,
    metal_mask: np.ndarray,
) -> None:
    """Write one 2D HU array as DICOM.  Metal hard-set to 3000 HU (§A1.3(d,f))."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hu = hu.copy()
    hu[metal_mask] = METAL_HU   # metal restoration (§A1.3(d,f))

    hu_clipped = np.clip(hu, -1024, 32767).astype(np.int16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID          = '1.2.840.10008.1.2.1'

    ds = FileDataset(str(output_dir), {}, file_meta=file_meta, preamble=b'\0' * 128)
    ds.is_implicit_VR   = False
    ds.is_little_endian = True

    now = datetime.now(timezone.utc)
    ds.ContentDate = now.strftime('%Y%m%d')
    ds.ContentTime = now.strftime('%H%M%S.%f')

    ds.Modality              = 'CT'
    ds.Manufacturer          = 'ASTM WKXXXXX ILS'
    ds.StudyDescription      = f'MAR ILS {DATASET_VERSION}'
    ds.SeriesDescription     = condition_label
    ds.ProtocolName          = f'{STANDARD_REF}-{condition_label}'
    ds.ConvolutionKernel     = 'RAM-LAK'
    ds.KVP                   = '60'
    ds.ExposureTime          = '0'
    ds.SliceThickness        = str(VOXEL_MM)

    ds.SOPClassUID           = '1.2.840.10008.5.1.4.1.1.2'
    ds.SOPInstanceUID        = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID      = study_uid
    ds.SeriesInstanceUID     = series_uid

    ds.Rows                  = Y_DIM
    ds.Columns               = X_DIM
    ds.PixelSpacing          = [VOXEL_MM, VOXEL_MM]
    ds.ImagePositionPatient  = [0.0, 0.0, float(z * VOXEL_MM)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SliceLocation         = float(z * VOXEL_MM)
    ds.InstanceNumber        = z + 1

    ds.SamplesPerPixel           = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated             = 16
    ds.BitsStored                = 16
    ds.HighBit                   = 15
    ds.PixelRepresentation       = 1   # signed
    ds.RescaleIntercept          = 0
    ds.RescaleSlope              = 1
    ds.PixelData                 = hu_clipped.tobytes()

    ds.save_as(str(output_dir / f'slice_{z+1:04d}.dcm'), write_like_original=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Reconstruction (noMAR FBP)
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_realization_v6(
    sino_path: Path,
    output_dir: Path,
    *,
    dc_offset_cm: float,
    realization_idx: int,
    condition_label: str,
    place_lesion: bool,
    study_uid: str,
    series_uid: str,
    metal_mask: np.ndarray,
    sweep_mode: bool = False,
) -> None:
    """
    FBP-reconstruct sinogram HDF5 → DICOM slices.

    sweep_mode=True writes ONLY slice_0129.dcm (LESION_SLICE_INDEX + 1).
    sweep_mode=False writes all Z_DIM DICOM slices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(sino_path), 'r') as f:
        if sweep_mode:
            sino_128 = f['line_integrals'][LESION_SLICE_INDEX].astype(np.float64)
        else:
            line_integrals = f['line_integrals'][:]   # (Z_DIM, N_ANGLES, N_DET)

    if sweep_mode:
        hu = fbp_reconstruct_slice(sino_128, dc_offset_cm=dc_offset_cm)
        _write_dicom_slice_v6(
            hu, LESION_SLICE_INDEX,
            output_dir=output_dir,
            realization_idx=realization_idx,
            condition_label=condition_label,
            study_uid=study_uid,
            series_uid=series_uid,
            metal_mask=metal_mask,
        )
    else:
        for z in range(Z_DIM):
            hu = fbp_reconstruct_slice(
                line_integrals[z].astype(np.float64), dc_offset_cm=dc_offset_cm
            )
            _write_dicom_slice_v6(
                hu, z,
                output_dir=output_dir,
                realization_idx=realization_idx,
                condition_label=condition_label,
                study_uid=study_uid,
                series_uid=series_uid,
                metal_mask=metal_mask,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel worker
# ═══════════════════════════════════════════════════════════════════════════════

def _realization_worker_v6(
    series: str,
    place_lesion: bool,
    realization_idx: int,
    output_dir_str: str,
    I0: float,
    jitter_deg: float,
    dc_offset_cm: float,
    tier_id: str,
    contrast_factor: float,
    sweep_mode: bool,
) -> str:
    """ProcessPoolExecutor worker: sinogram + FBP for one realization."""
    tier = TIER_REGISTRY[tier_id]
    output_dir = Path(output_dir_str)
    seed = BASE_SEED + realization_idx
    tag  = f"realization_{realization_idx+1:03d}"

    h5_path   = output_dir / 'sinograms'    / series / f'{tag}.h5'
    recon_dir = output_dir / 'noMAR_recon'  / series / tag
    recon_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute metal mask for this tier (used in DICOM writing)
    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    metal_mask = _build_metal_mask(yy, xx, tier)

    # Skip if already done
    sentinel = recon_dir / f'slice_{LESION_SLICE_INDEX+1:04d}.dcm'
    if sentinel.exists():
        return f'{series}/{tag} (skipped)'

    # Generate UIDs once (shared across all slices of this realization)
    study_uid  = generate_uid()
    series_uid = generate_uid()

    generate_sinogram_realization_v6(
        h5_path,
        place_lesion=place_lesion,
        seed=seed,
        jitter_deg=jitter_deg,
        I0=I0,
        realization_idx=realization_idx,
        tier=tier,
        contrast_factor=contrast_factor,
    )
    reconstruct_realization_v6(
        h5_path, recon_dir,
        dc_offset_cm=dc_offset_cm,
        realization_idx=realization_idx,
        condition_label=f'noMAR_{series}_{tier_id}',
        place_lesion=place_lesion,
        study_uid=study_uid,
        series_uid=series_uid,
        metal_mask=metal_mask,
        sweep_mode=sweep_mode,
    )
    return f'{series}/{tag}'


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata, checksums, provenance
# ═══════════════════════════════════════════════════════════════════════════════

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(output_dir: Path) -> None:
    files = [p for p in sorted(output_dir.rglob('*'))
             if p.is_file() and p.name != 'checksums_sha256.txt']
    manifest = output_dir / 'checksums_sha256.txt'
    print(f"Computing SHA-256 checksums ({len(files)} files)...", flush=True)
    with open(manifest, 'w') as fout:
        fout.write(f'# ASTM WKXXXXX {DATASET_VERSION} — SHA-256 manifest\n')
        for p in tqdm(files, desc='  SHA-256', unit='file'):
            fout.write(f'{_sha256_file(p)}  {p.relative_to(output_dir)}\n')


def write_provenance_json_v6(
    output_dir: Path,
    tier: TierConfig,
    jitters: dict[int, float],
    I0: float,
    dc_offset_cm: float,
    contrast_factor: float,
    sweep_mode: bool,
) -> None:
    prov = {
        'generator_version':    DATASET_VERSION,
        'standard_reference':   STANDARD_REF,
        'generation_timestamp': datetime.now(timezone.utc).isoformat(),
        'tier':                 tier.to_dict(),
        'contrast_factor':      contrast_factor,
        'sweep_mode':           sweep_mode,
        'geometry': {
            'x_dim': X_DIM, 'y_dim': Y_DIM, 'z_dim': Z_DIM,
            'voxel_mm': VOXEL_MM,
            'n_angles': N_ANGLES, 'n_det': N_DET,
            'angle_range_deg': [0.0, 180.0],
        },
        'physics': {
            'energy_keV': 60,
            'spectrum': 'monochromatic',
            'mu_air_cm':    MU_AIR_CM,
            'mu_tissue_cm': MU_TISSUE_CM,
            'mu_metal_cm':  tier.metal_mu_cm,
            'mu_lesion_cm': tier.mu_lesion_cm * contrast_factor,
            'lesion_delta_hu': LESION_DELTA_HU * contrast_factor,
        },
        'noise_model': {
            'type': 'Poisson+Gaussian (Vaishnav 2020)',
            'I0_calibrated':   round(I0, 2),
            'scatter_frac':    SCATTER_FRAC,
            'sigma_e_counts':  SIGMA_E_COUNTS,
            'target_sigma_hu': NOISE_SIGMA_TARGET_HU,
            'dc_offset_cm':    round(dc_offset_cm, 8),
        },
        'realizations': {
            'count':     NUM_REALIZATIONS,
            'base_seed': BASE_SEED,
            'jitters':   {str(i+1): round(jitters[i], 4) for i in range(NUM_REALIZATIONS)},
        },
    }
    with open(output_dir / 'generator_provenance.json', 'w') as f:
        json.dump(prov, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# PDF lab instructions (tier-aware)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pdf_v6(output_dir: Path, tier: TierConfig, contrast_factor: float) -> None:
    """Generate MAR_ILS_Lab_Instructions.pdf with tier-specific parameters."""
    _font_ok = False
    if os.path.exists(_DEJAVU_REGULAR) and os.path.exists(_DEJAVU_BOLD):
        try:
            pdfmetrics.registerFont(TTFont(PDF_FONT,      _DEJAVU_REGULAR))
            pdfmetrics.registerFont(TTFont(PDF_FONT_BOLD, _DEJAVU_BOLD))
            _font_ok = True
        except Exception:
            pass
    if not _font_ok:
        globals()['PDF_FONT']      = "Helvetica"
        globals()['PDF_FONT_BOLD'] = "Helvetica-Bold"
        warnings.warn("DejaVu fonts not found — falling back to Helvetica.",
                      RuntimeWarning, stacklevel=2)

    pdf_path = output_dir / "MAR_ILS_Lab_Instructions.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
        leftMargin=inch, rightMargin=inch,
    )
    styles = getSampleStyleSheet()

    T      = ParagraphStyle("T",   parent=styles["Title"],    fontName=PDF_FONT_BOLD, fontSize=16, spaceAfter=6)
    H1     = ParagraphStyle("H1",  parent=styles["Heading1"], fontName=PDF_FONT_BOLD, fontSize=13, spaceBefore=14, spaceAfter=4)
    B      = ParagraphStyle("B",   parent=styles["Normal"],   fontName=PDF_FONT, fontSize=10, leading=14, spaceAfter=6)
    TC     = ParagraphStyle("TC",  parent=styles["Normal"],   fontName=PDF_FONT, fontSize=9, leading=11)
    TC_HDR = ParagraphStyle("TCH", parent=styles["Normal"],   fontName=PDF_FONT_BOLD, fontSize=9, leading=11, textColor=colors.white)

    def _cell(v, h=False):
        return Paragraph(str(v), TC_HDR if h else TC)

    def _tbl(data, widths):
        wrapped = [[_cell(c, r == 0) for c in row] for r, row in enumerate(data)]
        t = Table(wrapped, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#003366")),
            ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
            ("VALIGN",         (0,0), (-1,-1), "TOP"),
            ("GRID",           (0,0), (-1,-1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF2F7")]),
            ("TOPPADDING",     (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
            ("LEFTPADDING",    (0,0), (-1,-1), 5),
            ("RIGHTPADDING",   (0,0), (-1,-1), 5),
        ]))
        return t

    body_desc = (
        f"circular r={tier.body_semi_x_mm:.0f} mm ({tier.body_semi_x_vox} vox)"
        if tier.is_circular_body
        else f"ellipse {tier.body_semi_x_mm:.0f}×{tier.body_semi_y_mm:.0f} mm "
             f"({tier.body_semi_x_vox}×{tier.body_semi_y_vox} vox)"
    )
    contrast_note = "" if abs(contrast_factor - 1.0) < 1e-6 else f" (×{contrast_factor:.2f})"

    story = []
    story.append(Paragraph(f"MAR ILS Lab Instructions — {tier.tier_id}", T))
    story.append(Paragraph(
        f"Metal Artifact Reduction ILS — {DATASET_VERSION} ({STANDARD_REF}) — {tier.description}",
        styles["Heading2"],
    ))
    story.append(Paragraph(f"Generated {date.today().isoformat()}", styles["Italic"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey, spaceAfter=12))

    story.append(Paragraph("1. Tier Parameters", H1))
    story.append(_tbl([
        ["Parameter",         "Value",                         "Notes"],
        ["Tier ID",            tier.tier_id,                   tier.description],
        ["Body geometry",      body_desc,                      "Outer phantom boundary"],
        ["Metal material",     tier.metal_material,            f"μ = {tier.metal_mu_cm:.3f} cm⁻¹ at 60 keV"],
        ["Metal radius",       f"{tier.metal_radius_mm:.1f} mm ({tier.metal_radius_vox} vox)",
                               f"Co-axial with phantom centre"],
        ["Lesion geometry",    f"{tier.lesion_semi_major_mm:.1f}×{tier.lesion_semi_minor_mm:.1f} mm ellipse",
                               f"Semi-major/minor axes; x={tier.lesion_center_x}"],
        ["L_nominal",          f"{tier.l_nominal_mm:.1f} mm ({tier.l_nominal_vox} vox)",
                               "Metal centre → lesion centre"],
        ["Edge gap",           f"{tier.gap_mm:.1f} mm ({tier.gap_vox} vox)",
                               "Metal edge → lesion edge"],
        ["Blockage fraction",  f"{tier.blockage_frac*100:.2f}%",
                               "(2/π)·arcsin(R_metal/L_nominal)"],
        ["Contrast factor",    f"{contrast_factor:.2f}{contrast_note}",
                               f"≈ {LESION_DELTA_HU*contrast_factor:.1f} HU sinogram-domain lesion contrast"],
        ["Lesion slice index", str(LESION_SLICE_INDEX),        "zero-indexed; slice_0129.dcm"],
        ["Realizations",       f"{NUM_REALIZATIONS} LP + {NUM_REALIZATIONS} LA",
                               "80 total tasks"],
    ], [1.8*inch, 2.2*inch, 2.6*inch]))

    story.append(Paragraph("2. Submission Instructions", H1))
    story.append(Paragraph(
        "Apply your MAR algorithm to the sinograms in sinograms/LP/ and sinograms/LA/ "
        "and submit reconstructed DICOM images in the following structure:", B,
    ))
    story.append(_tbl([
        ["Folder / File",     "Contents"],
        ["mar_recon/LP/realization_001/ ... /realization_040/",
         f"MAR-corrected DICOMs for LP realizations. "
         f"Name slices slice_NNNN.dcm (1-indexed). "
         f"Only slice_{LESION_SLICE_INDEX+1:04d}.dcm is scored."],
        ["mar_recon/LA/realization_001/ ... /realization_040/",
         "MAR-corrected DICOMs for LA realizations."],
    ], [3.0*inch, 3.6*inch]))

    story.append(Paragraph("3. CHO Analysis", H1))
    story.append(Paragraph(
        f"Run the following command to score your submission "
        f"(--internal-noise-sigma 15 is normative for {STANDARD_REF}):", B,
    ))
    story.append(Paragraph(
        f"python run_cho_analysis_v6_0.py "
        f"--dataset-dir &lt;output_dir&gt; "
        f"--mar-output-dir ./mar_recon "
        f"--tier {tier.tier_id} "
        f"--internal-noise-sigma 15 "
        f"--results-file results.json",
        ParagraphStyle("code", parent=styles["Normal"], fontName="Courier",
                       fontSize=9, leading=12, backColor=colors.HexColor("#F5F5F5"),
                       spaceAfter=8),
    ))

    doc.build(story)
    print(f"  PDF written → {pdf_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="MAR ILS Dataset Generator v6.0.0 — ASTM WKXXXXX v1.0.0 Tiered Framework"
    )
    ap.add_argument("--tier", default="T1_AB", choices=list(TIER_REGISTRY),
                    help="Imaging scenario tier (default: T1_AB)")
    ap.add_argument("--output-dir", default=None,
                    help="Root output directory (default: ./astm_mar_ils_{tier})")
    ap.add_argument("--contrast-factor", type=float, default=1.0,
                    help="Multiplier on lesion Δμ (default: 1.0 = ~12 HU)")
    ap.add_argument("--sweep-mode", action="store_true",
                    help="Write only slice_0129.dcm per realization (sensitivity sweep)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0 = os.cpu_count(), default: 0)")
    ap.add_argument("--no-pdf", action="store_true",
                    help="Skip PDF lab instructions")
    ap.add_argument("--no-checksums", action="store_true",
                    help="Skip SHA-256 checksum computation")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate configuration without writing any files")
    args = ap.parse_args()

    tier = TIER_REGISTRY[args.tier]
    validate_tier_registry()
    print()

    output_dir = Path(args.output_dir or f"./astm_mar_ils_{args.tier.lower()}").resolve()
    n_workers  = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    print(f"Generator version : {DATASET_VERSION} ({STANDARD_REF})")
    print(f"Tier              : {tier.tier_id}  —  {tier.description}")
    print(f"Contrast factor   : {args.contrast_factor:.3f}  "
          f"(lesion Δμ ≈ {LESION_DELTA_HU * args.contrast_factor:.1f} HU)")
    print(f"Sweep mode        : {'YES (slice_0129.dcm only)' if args.sweep_mode else 'NO (full DICOM volume)'}")
    print(f"Output directory  : {output_dir}")
    print(f"Workers           : {n_workers}")
    print()
    print(tier)
    print()

    if args.dry_run:
        print("DRY RUN — no files written.")
        return

    # Calibrate
    I0, dc_offset_cm = calibrate_v6(tier)
    print()

    # Jitter angles (same deterministic scheme as v5.3.0)
    rng_jitter = np.random.default_rng(BASE_SEED - 1)
    jitters: dict[int, float] = {
        i: float(rng_jitter.uniform(-JITTER_MAX_DEG, JITTER_MAX_DEG))
        for i in range(NUM_REALIZATIONS)
    }

    # Build task list
    tasks = [
        (cond, place, i, str(output_dir), I0, jitters[i], dc_offset_cm,
         args.tier, args.contrast_factor, args.sweep_mode)
        for cond, place in (("LP", True), ("LA", False))
        for i in range(NUM_REALIZATIONS)
    ]

    done = 0
    n_tasks = len(tasks)
    print(f"Generating {n_tasks} realizations ({NUM_REALIZATIONS} LP + {NUM_REALIZATIONS} LA)...")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_realization_worker_v6, *t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                label = fut.result()
                done += 1
                print(f"  [{done:3d}/{n_tasks}] {label}", flush=True)
            except Exception as exc:
                t = futs[fut]
                print(f"  ERROR {t[0]}/realization_{t[2]+1:03d}: {exc}")

    print()

    # Write metadata
    write_provenance_json_v6(output_dir, tier, jitters, I0, dc_offset_cm,
                              args.contrast_factor, args.sweep_mode)

    if not args.no_checksums:
        write_checksums(output_dir)

    if not args.no_pdf:
        generate_pdf_v6(output_dir, tier, args.contrast_factor)

    print()
    print(f"Dataset complete → {output_dir}")
    print("Next step:")
    print(f"  python run_cho_analysis_v6_0.py \\")
    print(f"      --dataset-dir {output_dir} \\")
    print(f"      --mar-output-dir ./mar_recon \\")
    print(f"      --tier {args.tier} \\")
    print(f"      --internal-noise-sigma 15 \\")
    print(f"      --results-file results_{args.tier.lower()}.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
