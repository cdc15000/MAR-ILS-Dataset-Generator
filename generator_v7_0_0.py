#!/usr/bin/env python3
"""
generator_v7_0_0.py
====================
MAR ILS Dataset Generator v7.0.0 — ASTM WKXXXXX Rev 04 (Fan-Beam)

Changes from v6.0.0 / v5.3.0
-----------------------------
[FB1] Fan-beam acquisition geometry: SID=570 mm, SDD=1040 mm, equi-angular
      curved detector, 720 projection angles over full 360° rotation.
[FB2] Fan-beam forward projection via ray-driven integration (replaces
      parallel-beam rotation-sum).
[FB3] Fan-beam FBP: cosine pre-weighting, ramp filter, distance-weighted
      backprojection with (SID/L)² weighting (replaces parallel-beam FBP).
[SC]  Single canonical configuration — v5.3.0 parameters (iron rod r=5 mm,
      circular lesion r=2.5 mm at (281,256), ROI 121×121, channel width
      a=7.5 vox). No tier framework. §1.5: "No additional configurations
      are permitted under this standard."
[RR]  --realizations flag: 40 (normative default) or 20 (screening mode).
[AC]  Acceptance criteria cross-reference to IEC 60601-2-44 §203.6.7.102
      and FDA guidance (layered approach per §5.5).

Canonical Parameters (ASTM WKXXXXX Rev 04)
-------------------------------------------
  Body        : ellipse 85×60 mm (170×120 vox)
  Metal       : iron, μ=2.408 cm⁻¹, r=5 mm (10 vox), centred at (256,256)
  Lesion      : circular disc r=2.5 mm (5 vox) at (281,256), slice 128 only
  Contrast    : ~12 HU sinogram-domain (no post-FBP hard-set)
  Metal HU    : 3000 (restored last)
  Background  : 40 HU
  Noise σ     : 30 HU in soft tissue
  Acquisition : fan-beam, SID=570 mm, SDD=1040 mm, 720 angles × 512 dets
  ROI         : 121×121 at (281,256), 2D slice 128 only
  Channel a   : 7.5 voxels

Usage
-----
    python generator_v7_0_0.py --output-dir ./astm_reference_dataset
    python generator_v7_0_0.py --workers 8
    python generator_v7_0_0.py --realizations 20         # screening mode
    python generator_v7_0_0.py --dry-run                  # validate only

Author  : ASTM F04 Subcommittee Working Draft
Standard: ASTM WKXXXXX Rev 04
Date    : 2026-04-05
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
    HRFlowable, KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from tqdm import tqdm
import math

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# ═══════════════════════════════════════════════════════════════════════════════
# Canonical constants (ASTM WKXXXXX Rev 04 — single configuration)
# ═══════════════════════════════════════════════════════════════════════════════

# Volume geometry (§A1.1)
X_DIM: int = 512
Y_DIM: int = 512
Z_DIM: int = 256
VOXEL_MM: float = 0.5
VOXEL_CM: float = VOXEL_MM / 10.0  # 0.05 cm
PHANTOM_CENTER_X: int = 256
PHANTOM_CENTER_Y: int = 256

# Fan-beam geometry [Rev 04] (§A1.1(f,g))
SID_MM: float = 570.0       # source-to-isocenter distance
SDD_MM: float = 1040.0      # source-to-detector distance
SID_CM: float = SID_MM / 10.0
SDD_CM: float = SDD_MM / 10.0
SID_VOX: float = SID_MM / VOXEL_MM   # 1140.0 voxels
SDD_VOX: float = SDD_MM / VOXEL_MM   # 2080.0 voxels
N_ANGLES: int = 720          # full 360° rotation
N_DET: int = 512             # equi-angular detector elements

# Detector fan angles
_FOV_HALF_MM: float = X_DIM * VOXEL_MM / 2.0  # 128 mm
GAMMA_MAX_RAD: float = float(np.arcsin(_FOV_HALF_MM / SID_MM))  # ~0.2266 rad
DELTA_GAMMA_RAD: float = 2.0 * GAMMA_MAX_RAD / N_DET  # angular pitch
DET_FAN_ANGLES_RAD: np.ndarray = (
    (np.arange(N_DET) - N_DET / 2.0 + 0.5) * DELTA_GAMMA_RAD
)
COS_DET_FAN: np.ndarray = np.cos(DET_FAN_ANGLES_RAD)

# Source rotation angles
ANGLES_DEG: np.ndarray = np.linspace(0.0, 360.0, N_ANGLES, endpoint=False)
ANGLES_RAD: np.ndarray = np.deg2rad(ANGLES_DEG)

# Ray-tracing parameters (forward projection)
_DIAG_VOX: float = float(np.sqrt(X_DIM**2 + Y_DIM**2)) / 2.0
_RAY_T_START: float = SID_VOX - _DIAG_VOX - 10.0
_RAY_T_END: float = SID_VOX + _DIAG_VOX + 10.0
_RAY_STEP: float = 0.4   # voxel step along each ray (sub-voxel accuracy)
_RAY_T_VALS: np.ndarray = np.arange(_RAY_T_START, _RAY_T_END, _RAY_STEP)
_N_RAY_SAMPLES: int = len(_RAY_T_VALS)

# Physical constants (NIST XCOM, 60 keV monochromatic)
MU_AIR_CM: float = 0.000196
MU_TISSUE_CM: float = 0.2059
MU_IRON_CM: float = 2.408
BACKGROUND_HU: float = 40.0
METAL_HU: float = 3000.0

# Phantom geometry (§A1.2–§A1.4)
BODY_SEMI_X_VOX: int = round(85.0 / VOXEL_MM)   # 170
BODY_SEMI_Y_VOX: int = round(60.0 / VOXEL_MM)   # 120
METAL_RADIUS_VOX: int = round(5.0 / VOXEL_MM)    # 10
LESION_RADIUS_VOX: int = round(2.5 / VOXEL_MM)   # 5
LESION_CENTER_X: int = 281    # §A1.4(c): 256 + 10 + 10 + 5
LESION_SLICE_INDEX: int = 128

# Lesion contrast (§A1.4(e))
LESION_DELTA_HU: float = 12.0
MU_LESION_CM: float = MU_TISSUE_CM * (1.0 + LESION_DELTA_HU / 1000.0)

# Noise model
SCATTER_FRAC: float = 0.05
SIGMA_E_COUNTS: float = 5.0
NOISE_SIGMA_TARGET_HU: float = 30.0
JITTER_MAX_DEG: float = 15.0

# Study parameters
NUM_REALIZATIONS_DEFAULT: int = 40
BASE_SEED: int = 20260314
DATASET_VERSION: str = "v7.0.0"
STANDARD_REF: str = "ASTM-WKXXXXX-Rev04"

# PDF fonts
_DEJAVU_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PDF_FONT = "DejaVuSans"
PDF_FONT_BOLD = "DejaVuSans-Bold"


# ═══════════════════════════════════════════════════════════════════════════════
# Numba-accelerated kernels (optional — falls back to NumPy if unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

if _HAS_NUMBA:

    @numba.njit(parallel=True, cache=True)
    def _forward_project_jit(mu, angles_rad, det_fan_angles, ray_t_vals,
                             sid_vox, cx, cy, mu_air, ray_step_cm):
        """Fan-beam forward projection with hand-rolled bilinear interpolation."""
        n_angles = angles_rad.shape[0]
        n_det = det_fan_angles.shape[0]
        n_samples = ray_t_vals.shape[0]
        ny, nx = mu.shape
        sino = np.zeros((n_angles, n_det), dtype=np.float64)
        nx_lim = float(nx - 1)
        ny_lim = float(ny - 1)

        for i in numba.prange(n_angles):
            beta = angles_rad[i]
            sx = cx + sid_vox * math.cos(beta)
            sy = cy + sid_vox * math.sin(beta)

            for j in range(n_det):
                ray_angle = beta + math.pi + det_fan_angles[j]
                dx = math.cos(ray_angle)
                dy = math.sin(ray_angle)

                line_sum = 0.0
                for k in range(n_samples):
                    x = sx + dx * ray_t_vals[k]
                    y = sy + dy * ray_t_vals[k]

                    if x < 0.0 or x >= nx_lim or y < 0.0 or y >= ny_lim:
                        line_sum += mu_air
                    else:
                        ix = int(x)
                        iy = int(y)
                        fx = x - float(ix)
                        fy = y - float(iy)
                        line_sum += (mu[iy, ix] * (1.0 - fx) * (1.0 - fy)
                                     + mu[iy, ix + 1] * fx * (1.0 - fy)
                                     + mu[iy + 1, ix] * (1.0 - fx) * fy
                                     + mu[iy + 1, ix + 1] * fx * fy)

                sino[i, j] = line_sum * ray_step_cm

        return sino

    @numba.njit(parallel=True, cache=True)
    def _backproject_jit(filtered, angles_rad, sid_vox, delta_gamma,
                         gamma_min, cx, cy, nx, ny):
        """Single-slice fan-beam backprojection (pixel-driven)."""
        n_proj = filtered.shape[0]
        n_det = filtered.shape[1]
        sid_vox_sq = sid_vox * sid_vox

        # Precompute source positions and central-ray angles
        sx_arr = np.empty(n_proj)
        sy_arr = np.empty(n_proj)
        cr_arr = np.empty(n_proj)
        for i in range(n_proj):
            sx_arr[i] = cx + sid_vox * math.cos(angles_rad[i])
            sy_arr[i] = cy + sid_vox * math.sin(angles_rad[i])
            cr_arr[i] = math.atan2(cy - sy_arr[i], cx - sx_arr[i])

        recon = np.zeros((ny, nx), dtype=np.float64)
        det_max = float(n_det - 1)

        for row in numba.prange(ny):
            yf = float(row)
            for col in range(nx):
                xf = float(col)
                val = 0.0
                for i in range(n_proj):
                    px = xf - sx_arr[i]
                    py = yf - sy_arr[i]
                    L_sq = px * px + py * py
                    pa = math.atan2(py, px)
                    g = pa - cr_arr[i]
                    g = math.atan2(math.sin(g), math.cos(g))
                    di = (g - gamma_min) / delta_gamma
                    if di >= 0.0 and di <= det_max:
                        lo = int(di)
                        if lo >= n_det - 1:
                            lo = n_det - 2
                        f = di - float(lo)
                        v = (1.0 - f) * filtered[i, lo] + f * filtered[i, lo + 1]
                        val += v * sid_vox_sq / L_sq
                recon[row, col] = val

        return recon

    @numba.njit(parallel=True, cache=True)
    def _backproject_batch_jit(filtered_t, angles_rad, sid_vox, delta_gamma,
                               gamma_min, cx, cy, nx, ny, n_slices):
        """
        Batch backprojection: geometry computed once per (pixel, angle),
        applied to all n_slices. ~4x fewer arctan2 calls vs per-slice.

        filtered_t : (n_proj, n_det, n_slices) float64 — transposed for
                     cache-friendly z-sequential access
        Returns    : (ny, nx, n_slices) float64 — transposed output for
                     cache-friendly z-sequential writes
        """
        n_proj = filtered_t.shape[0]
        n_det = filtered_t.shape[1]
        sid_vox_sq = sid_vox * sid_vox

        sx_arr = np.empty(n_proj)
        sy_arr = np.empty(n_proj)
        cr_arr = np.empty(n_proj)
        for i in range(n_proj):
            sx_arr[i] = cx + sid_vox * math.cos(angles_rad[i])
            sy_arr[i] = cy + sid_vox * math.sin(angles_rad[i])
            cr_arr[i] = math.atan2(cy - sy_arr[i], cx - sx_arr[i])

        recon = np.zeros((ny, nx, n_slices), dtype=np.float64)
        det_max = float(n_det - 1)

        for row in numba.prange(ny):
            yf = float(row)
            for col in range(nx):
                xf = float(col)
                for i in range(n_proj):
                    px = xf - sx_arr[i]
                    py = yf - sy_arr[i]
                    L_sq = px * px + py * py
                    w = sid_vox_sq / L_sq
                    pa = math.atan2(py, px)
                    g = pa - cr_arr[i]
                    g = math.atan2(math.sin(g), math.cos(g))
                    di = (g - gamma_min) / delta_gamma
                    if di >= 0.0 and di <= det_max:
                        lo = int(di)
                        if lo >= n_det - 1:
                            lo = n_det - 2
                        f = di - float(lo)
                        omf = 1.0 - f
                        for z in range(n_slices):
                            v = omf * filtered_t[i, lo, z] + f * filtered_t[i, lo + 1, z]
                            recon[row, col, z] += v * w

        return recon


# ═══════════════════════════════════════════════════════════════════════════════
# Phantom construction
# ═══════════════════════════════════════════════════════════════════════════════

def _build_body_mask(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    return (
        ((xx - PHANTOM_CENTER_X) / BODY_SEMI_X_VOX) ** 2
        + ((yy - PHANTOM_CENTER_Y) / BODY_SEMI_Y_VOX) ** 2
        <= 1.0
    )


def _build_metal_mask(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    return (
        (xx - PHANTOM_CENTER_X) ** 2 + (yy - PHANTOM_CENTER_Y) ** 2
        <= METAL_RADIUS_VOX ** 2
    )


def _build_lesion_mask(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    return (
        (xx - LESION_CENTER_X) ** 2 + (yy - PHANTOM_CENTER_Y) ** 2
        <= LESION_RADIUS_VOX ** 2
    )


def build_attenuation_map(
    place_lesion: bool,
    jitter_deg: float = 0.0,
) -> np.ndarray:
    """Construct 2D attenuation map (cm⁻¹) for one axial slice."""
    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    body_mask = _build_body_mask(yy, xx)
    metal_mask = _build_metal_mask(yy, xx)
    lesion_mask = _build_lesion_mask(yy, xx)

    mu = np.full((Y_DIM, X_DIM), MU_AIR_CM, dtype=np.float64)
    mu[body_mask] = MU_TISSUE_CM
    mu[metal_mask] = MU_IRON_CM
    if place_lesion:
        mu[lesion_mask] = MU_LESION_CM

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
# Fan-beam forward projection [Rev 04]
# ═══════════════════════════════════════════════════════════════════════════════

def forward_project_slice(mu: np.ndarray) -> np.ndarray:
    """
    Fan-beam forward projection via ray-driven integration.

    For each source angle β and detector element γ_j, traces a ray from the
    source through the volume, sampling μ at sub-voxel intervals and
    integrating to compute the line integral in nepers.

    mu  : (Y_DIM, X_DIM) attenuation map in cm⁻¹
    Returns: (N_ANGLES, N_DET) sinogram in nepers
    """
    if _HAS_NUMBA:
        return _forward_project_jit(
            mu.astype(np.float64), ANGLES_RAD, DET_FAN_ANGLES_RAD,
            _RAY_T_VALS, SID_VOX, X_DIM / 2.0, Y_DIM / 2.0,
            MU_AIR_CM, _RAY_STEP * VOXEL_CM,
        ).astype(np.float32)

    # NumPy fallback (no Numba)
    sino = np.zeros((N_ANGLES, N_DET), dtype=np.float64)
    mu64 = mu.astype(np.float64)
    cx, cy = X_DIM / 2.0, Y_DIM / 2.0

    for i, beta in enumerate(ANGLES_RAD):
        sx = cx + SID_VOX * np.cos(beta)
        sy = cy + SID_VOX * np.sin(beta)
        ray_angles = beta + np.pi + DET_FAN_ANGLES_RAD
        dx = np.cos(ray_angles)
        dy = np.sin(ray_angles)
        x_pts = sx + np.outer(dx, _RAY_T_VALS)
        y_pts = sy + np.outer(dy, _RAY_T_VALS)
        coords = np.array([y_pts.ravel(), x_pts.ravel()])
        vals = scipy.ndimage.map_coordinates(
            mu64, coords, order=1, mode='constant', cval=float(MU_AIR_CM),
        ).reshape(N_DET, _N_RAY_SAMPLES)
        sino[i] = vals.sum(axis=1) * (_RAY_STEP * VOXEL_CM)

    return sino.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Noise model (unchanged from v5.3.0/v6.0.0)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_noise(sino_clean: np.ndarray, I0: float, rng: np.random.Generator) -> np.ndarray:
    """
    Physics-based CT acquisition noise (Vaishnav 2020).

      I_meas = Poisson(I₀·exp(−p) + S) + N(0, σ_e²)
      p_meas = −ln(max(I_meas, 0.1) / I₀)
    """
    S = SCATTER_FRAC * I0
    I_expected = I0 * np.exp(-sino_clean) + S
    I_measured = rng.poisson(I_expected).astype(np.float64)
    I_measured += rng.normal(0.0, SIGMA_E_COUNTS, size=I_measured.shape)
    I_measured = np.maximum(I_measured, 0.1)
    return (-np.log(I_measured / I0)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Fan-beam FBP reconstruction [Rev 04]
# ═══════════════════════════════════════════════════════════════════════════════

def _fbp_fanbeam_core(sino: np.ndarray) -> np.ndarray:
    """
    Fan-beam filtered backprojection → raw μ map (cm⁻¹).

    Algorithm (Kak & Slaney §3.4, equi-angular detector, full 360° scan):
      1. Pre-weight projections by cos(γ)
      2. Apply ramp (Ram-Lak) filter along detector dimension
      3. Distance-weighted backprojection: Σ (SID/L)² × Q(β, γ(x,y))
      4. Scale by π / N_angles / (SID_cm × Δγ)
    """
    n_proj, n_det = sino.shape

    # Steps 1-2: cosine pre-weighting + ramp filter (NumPy — already fast)
    weighted = sino.astype(np.float64) * COS_DET_FAN[np.newaxis, :]
    freq = np.fft.rfftfreq(n_det)
    ramp = np.abs(freq)
    filtered = np.fft.irfft(
        np.fft.rfft(weighted, axis=1) * ramp, n=n_det, axis=1,
    )

    if _HAS_NUMBA:
        # Numba-accelerated backprojection
        recon = _backproject_jit(
            filtered, ANGLES_RAD, SID_VOX, DELTA_GAMMA_RAD,
            float(DET_FAN_ANGLES_RAD[0]), X_DIM / 2.0, Y_DIM / 2.0,
            X_DIM, Y_DIM,
        )
        mu_recon = recon * np.pi / n_proj / (SID_CM * DELTA_GAMMA_RAD)
        return mu_recon.astype(np.float32)

    # Step 3: distance-weighted backprojection (NumPy fallback)
    recon = np.zeros((Y_DIM, X_DIM), dtype=np.float64)
    cx, cy = X_DIM / 2.0, Y_DIM / 2.0
    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    xx_f = xx.astype(np.float64)
    yy_f = yy.astype(np.float64)
    gamma_min = DET_FAN_ANGLES_RAD[0]

    for i, beta in enumerate(ANGLES_RAD):
        sx = cx + SID_VOX * np.cos(beta)
        sy = cy + SID_VOX * np.sin(beta)
        px = xx_f - sx
        py = yy_f - sy
        L_sq = px * px + py * py
        cr_angle = np.arctan2(cy - sy, cx - sx)
        pixel_angle = np.arctan2(py, px)
        gamma = pixel_angle - cr_angle
        gamma = np.arctan2(np.sin(gamma), np.cos(gamma))
        det_idx = (gamma - gamma_min) / DELTA_GAMMA_RAD
        valid = (det_idx >= 0) & (det_idx <= n_det - 1)
        idx_lo = np.clip(np.floor(det_idx).astype(np.intp), 0, n_det - 2)
        frac = det_idx - idx_lo
        proj = np.zeros((Y_DIM, X_DIM), dtype=np.float64)
        proj[valid] = (
            (1.0 - frac[valid]) * filtered[i, idx_lo[valid]]
            + frac[valid] * filtered[i, np.minimum(idx_lo[valid] + 1, n_det - 1)]
        )
        recon += proj * (SID_VOX * SID_VOX) / L_sq

    mu_recon = recon * np.pi / n_proj / (SID_CM * DELTA_GAMMA_RAD)
    return mu_recon.astype(np.float32)


def fbp_reconstruct_slice(sino: np.ndarray, dc_offset_cm: float = 0.0) -> np.ndarray:
    """Fan-beam FBP → HU image with DC calibration and background offset."""
    mu_recon = _fbp_fanbeam_core(sino)
    hu = (
        (mu_recon - MU_TISSUE_CM - dc_offset_cm) / MU_TISSUE_CM * 1000.0
        + BACKGROUND_HU
    )
    return hu.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════════════════

def _cal_roi_bounds() -> tuple[int, int, int, int]:
    """Calibration ROI in the upper portion of the body ellipse."""
    y0 = PHANTOM_CENTER_Y - round(0.90 * BODY_SEMI_Y_VOX)
    y1 = PHANTOM_CENTER_Y - round(0.40 * BODY_SEMI_Y_VOX)
    x0 = PHANTOM_CENTER_X - round(0.20 * BODY_SEMI_X_VOX)
    x1 = PHANTOM_CENTER_X + round(0.20 * BODY_SEMI_X_VOX)
    return max(0, y0), min(Y_DIM, y1), max(0, x0), min(X_DIM, x1)


def calibrate(target_sigma_hu: float = NOISE_SIGMA_TARGET_HU) -> tuple[float, float]:
    """
    3-FBP analytic calibration for fan-beam geometry.

      FBP-1: Noise-free full phantom → DC offset
      FBP-2: Noise-free body-only → clean tissue ROI for noise scaling
      FBP-3: N_MC noisy FBPs at I₀_ref → mean σ_ref → I₀ = I₀_ref×(σ_ref/σ_target)²

    Returns (I0_calibrated, dc_offset_cm).
    """
    print("Calibrating fan-beam forward model...", flush=True)
    y0, y1, x0, x1 = _cal_roi_bounds()
    print(f"  Calibration ROI: y=[{y0}:{y1}] x=[{x0}:{x1}] "
          f"({(y1 - y0) * (x1 - x0)} pixels)", flush=True)

    # FBP-1: DC offset from noise-free full phantom
    print("  [1/3] Noise-free FBP (full phantom, DC offset)...",
          end=" ", flush=True)
    mu_full = build_attenuation_map(False, jitter_deg=0.0)
    sino_full = forward_project_slice(mu_full)
    mu_full_recon = _fbp_fanbeam_core(sino_full)
    dc_offset_cm = float(mu_full_recon[y0:y1, x0:x1].mean()) - MU_TISSUE_CM
    print(f"DC offset = {dc_offset_cm / MU_TISSUE_CM * 1000:+.1f} HU", flush=True)

    # FBP-2: Body-only phantom for noise calibration
    print("  [2/3] Noise-free FBP (body-only)...", end=" ", flush=True)
    yy_c, xx_c = np.mgrid[0:Y_DIM, 0:X_DIM]
    mu_cal = np.full((Y_DIM, X_DIM), MU_AIR_CM, dtype=np.float32)
    mu_cal[_build_body_mask(yy_c, xx_c)] = MU_TISSUE_CM
    sino_cal = forward_project_slice(mu_cal)
    mu_cal_recon = _fbp_fanbeam_core(sino_cal)
    dc_cal = float(mu_cal_recon[y0:y1, x0:x1].mean()) - MU_TISSUE_CM
    hu_cal_nf = (mu_cal_recon - MU_TISSUE_CM - dc_cal) / MU_TISSUE_CM * 1000
    nf_std = float(hu_cal_nf[y0:y1, x0:x1].std())
    print(f"done  (noise-free ROI std = {nf_std:.2f} HU)", flush=True)

    # FBP-3: Monte Carlo noise calibration
    I0_ref = 1e5
    N_MC_CAL = 5
    print(f"  [3/3] {N_MC_CAL} noisy FBPs at I₀_ref={I0_ref:.0f}...", flush=True)
    sigmas_ref: list[float] = []
    for mc_seed in tqdm(range(N_MC_CAL), desc="    MC draws", leave=False):
        rng_mc = np.random.default_rng(900_000 + mc_seed)
        sino_mc = apply_noise(sino_cal, I0_ref, rng_mc)
        mu_mc = _fbp_fanbeam_core(sino_mc)
        hu_mc = (mu_mc - MU_TISSUE_CM - dc_cal) / MU_TISSUE_CM * 1000
        s = float(hu_mc[y0:y1, x0:x1].std())
        sigmas_ref.append(s)
        print(f"    draw {mc_seed + 1}/{N_MC_CAL}: σ = {s:.1f} HU", flush=True)

    sigma_ref = float(np.mean(sigmas_ref))
    print(f"  Mean σ_ref = {sigma_ref:.1f} HU  "
          f"(std = {float(np.std(sigmas_ref)):.1f} HU)", flush=True)

    I0_cal = I0_ref * (sigma_ref / target_sigma_hu) ** 2
    print(f"  Calibrated I₀ = {I0_cal:.0f}  "
          f"(target σ = {target_sigma_hu} HU)", flush=True)
    return float(I0_cal), float(dc_offset_cm)


# ═══════════════════════════════════════════════════════════════════════════════
# Sinogram generation (HDF5)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sinogram_realization(
    output_path: Path,
    *,
    place_lesion: bool,
    seed: int,
    jitter_deg: float,
    I0: float,
    realization_idx: int,
) -> None:
    """Generate one 3D sinogram realization and write to HDF5."""
    rng = np.random.default_rng(seed)

    mu_no = build_attenuation_map(False, jitter_deg)
    sino_no = forward_project_slice(mu_no)

    if place_lesion:
        mu_with = build_attenuation_map(True, jitter_deg)
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
        geo.attrs['type'] = 'fan-beam'
        geo.attrs['n_slices'] = Z_DIM
        geo.attrs['n_angles'] = N_ANGLES
        geo.attrs['n_det'] = N_DET
        geo.attrs['voxel_mm'] = VOXEL_MM
        geo.attrs['SID_mm'] = SID_MM
        geo.attrs['SDD_mm'] = SDD_MM
        geo.attrs['gamma_max_deg'] = float(np.degrees(GAMMA_MAX_RAD))
        geo.attrs['delta_gamma_deg'] = float(np.degrees(DELTA_GAMMA_RAD))
        geo.attrs['angles_deg'] = ANGLES_DEG.tolist()
        geo.attrs['det_fan_angles_deg'] = np.degrees(DET_FAN_ANGLES_RAD).tolist()

        np_grp = f.create_group('noise_params')
        np_grp.attrs['I0'] = I0
        np_grp.attrs['scatter_frac'] = SCATTER_FRAC
        np_grp.attrs['sigma_e_counts'] = SIGMA_E_COUNTS
        np_grp.attrs['seed'] = seed
        np_grp.attrs['jitter_deg'] = jitter_deg
        np_grp.attrs['place_lesion'] = int(place_lesion)
        np_grp.attrs['lesion_slice_index'] = LESION_SLICE_INDEX
        np_grp.attrs['lesion_z_extent'] = 1 if place_lesion else 0


# ═══════════════════════════════════════════════════════════════════════════════
# DICOM writing
# ═══════════════════════════════════════════════════════════════════════════════

def _write_dicom_slice(
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
    """Write one 2D HU array as DICOM. Metal hard-set to 3000 HU (§A1.3(d,f))."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hu = hu.copy()
    hu[metal_mask] = METAL_HU
    hu_clipped = np.clip(hu, -1024, 32767).astype(np.int16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'

    ds = FileDataset(str(output_dir), {}, file_meta=file_meta, preamble=b'\0' * 128)
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    now = datetime.now(timezone.utc)
    ds.ContentDate = now.strftime('%Y%m%d')
    ds.ContentTime = now.strftime('%H%M%S.%f')
    ds.Modality = 'CT'
    ds.Manufacturer = 'ASTM WKXXXXX ILS'
    ds.StudyDescription = f'MAR ILS {DATASET_VERSION}'
    ds.SeriesDescription = condition_label
    ds.ProtocolName = f'{STANDARD_REF}-{condition_label}'
    ds.ConvolutionKernel = 'RAM-LAK'
    ds.KVP = '60'
    ds.ExposureTime = '0'
    ds.SliceThickness = str(VOXEL_MM)
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Rows = Y_DIM
    ds.Columns = X_DIM
    ds.PixelSpacing = [VOXEL_MM, VOXEL_MM]
    ds.ImagePositionPatient = [0.0, 0.0, float(z * VOXEL_MM)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SliceLocation = float(z * VOXEL_MM)
    ds.InstanceNumber = z + 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.PixelData = hu_clipped.tobytes()

    # DICOM 2026b CP-2575: Metal Artifact Reduction Macro (C.8.15.3.15)
    mar_item = Dataset()
    mar_item.add_new(0x00189391, 'CS', 'NO')   # Metal Artifact Reduction Applied
    ds.add_new(0x00189390, 'SQ', [mar_item])    # Metal Artifact Reduction Sequence

    ds.save_as(str(output_dir / f'slice_{z + 1:04d}.dcm'), write_like_original=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Reconstruction (noMAR FBP)
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_realization(
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
) -> None:
    """FBP-reconstruct sinogram HDF5 → DICOM slices."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(sino_path), 'r') as f:
        line_integrals = f['line_integrals'][:]  # (Z_DIM, N_ANGLES, N_DET)

    for z in range(Z_DIM):
        hu = fbp_reconstruct_slice(
            line_integrals[z].astype(np.float64), dc_offset_cm=dc_offset_cm,
        )
        _write_dicom_slice(
            hu, z,
            output_dir=output_dir,
            realization_idx=realization_idx,
            condition_label=condition_label,
            study_uid=study_uid,
            series_uid=series_uid,
            metal_mask=metal_mask,
        )


def reconstruct_realization_batch(
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
) -> None:
    """
    Batch FBP-reconstruct all slices with shared geometry (Numba-accelerated).

    Geometry (source positions, fan angles, distance weights) is computed once
    per (pixel, angle) pair and applied to all 256 slices. This reduces arctan2
    calls by ~256x compared to per-slice reconstruction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(sino_path), 'r') as f:
        sinograms = f['line_integrals'][:]  # (Z_DIM, N_ANGLES, N_DET) float32

    n_slices, n_proj, n_det = sinograms.shape

    # Steps 1-2: cosine pre-weight + ramp filter all slices (vectorized NumPy)
    weighted = sinograms.astype(np.float64) * COS_DET_FAN[np.newaxis, np.newaxis, :]
    del sinograms
    freq = np.fft.rfftfreq(n_det)
    ramp = np.abs(freq)
    filtered = np.fft.irfft(
        np.fft.rfft(weighted, axis=2) * ramp[np.newaxis, np.newaxis, :],
        n=n_det, axis=2,
    )
    del weighted

    # Transpose for cache-friendly z-sequential access in batch kernel
    filtered_t = np.ascontiguousarray(filtered.transpose(1, 2, 0))
    del filtered

    # Step 3: batch backprojection (geometry computed once, applied to all slices)
    scale = np.pi / n_proj / (SID_CM * DELTA_GAMMA_RAD)
    recon_yxz = _backproject_batch_jit(
        filtered_t, ANGLES_RAD, SID_VOX, DELTA_GAMMA_RAD,
        float(DET_FAN_ANGLES_RAD[0]), X_DIM / 2.0, Y_DIM / 2.0,
        X_DIM, Y_DIM, n_slices,
    )
    del filtered_t

    # Step 4: scale → HU → DICOM for each slice
    for z in range(n_slices):
        mu_recon = recon_yxz[:, :, z] * scale
        hu = (
            (mu_recon - MU_TISSUE_CM - dc_offset_cm) / MU_TISSUE_CM * 1000.0
            + BACKGROUND_HU
        ).astype(np.float32)
        _write_dicom_slice(
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

def _realization_worker(
    series: str,
    place_lesion: bool,
    realization_idx: int,
    output_dir_str: str,
    I0: float,
    jitter_deg: float,
    dc_offset_cm: float,
) -> str:
    """ProcessPoolExecutor worker: sinogram + FBP for one realization."""
    # Limit Numba threads per worker — parallelism is across workers
    if _HAS_NUMBA:
        numba.set_num_threads(1)

    output_dir = Path(output_dir_str)
    seed = BASE_SEED + realization_idx
    tag = f"realization_{realization_idx + 1:03d}"

    h5_path = output_dir / 'sinograms' / series / f'{tag}.h5'
    recon_dir = output_dir / 'noMAR_recon' / series / tag
    recon_dir.mkdir(parents=True, exist_ok=True)

    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    metal_mask = _build_metal_mask(yy, xx)

    sentinel = recon_dir / f'slice_{LESION_SLICE_INDEX + 1:04d}.dcm'
    if sentinel.exists():
        return f'{series}/{tag} (skipped)'

    study_uid = generate_uid()
    series_uid = generate_uid()

    generate_sinogram_realization(
        h5_path,
        place_lesion=place_lesion,
        seed=seed,
        jitter_deg=jitter_deg,
        I0=I0,
        realization_idx=realization_idx,
    )

    _reconstruct = reconstruct_realization_batch if _HAS_NUMBA else reconstruct_realization
    _reconstruct(
        h5_path, recon_dir,
        dc_offset_cm=dc_offset_cm,
        realization_idx=realization_idx,
        condition_label=f'noMAR_{series}',
        place_lesion=place_lesion,
        study_uid=study_uid,
        series_uid=series_uid,
        metal_mask=metal_mask,
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
    files = [
        p for p in sorted(output_dir.rglob('*'))
        if p.is_file() and p.name != 'checksums_sha256.txt'
    ]
    manifest = output_dir / 'checksums_sha256.txt'
    print(f"Computing SHA-256 checksums ({len(files)} files)...", flush=True)
    with open(manifest, 'w') as fout:
        fout.write(f'# ASTM WKXXXXX {DATASET_VERSION} — SHA-256 manifest\n')
        for p in tqdm(files, desc='  SHA-256', unit='file'):
            fout.write(f'{_sha256_file(p)}  {p.relative_to(output_dir)}\n')


def write_provenance_json(
    output_dir: Path,
    jitters: dict[int, float],
    I0: float,
    dc_offset_cm: float,
    num_realizations: int,
) -> None:
    prov = {
        'generator_version': DATASET_VERSION,
        'standard_reference': STANDARD_REF,
        'generation_timestamp': datetime.now(timezone.utc).isoformat(),
        'geometry': {
            'type': 'fan-beam',
            'x_dim': X_DIM, 'y_dim': Y_DIM, 'z_dim': Z_DIM,
            'voxel_mm': VOXEL_MM,
            'SID_mm': SID_MM, 'SDD_mm': SDD_MM,
            'n_angles': N_ANGLES, 'n_det': N_DET,
            'angle_range_deg': [0.0, 360.0],
            'gamma_max_deg': float(np.degrees(GAMMA_MAX_RAD)),
            'delta_gamma_deg': float(np.degrees(DELTA_GAMMA_RAD)),
        },
        'phantom': {
            'body_semi_x_mm': 85.0, 'body_semi_y_mm': 60.0,
            'metal_material': 'iron',
            'metal_radius_mm': 5.0, 'metal_mu_cm': MU_IRON_CM,
            'lesion_radius_mm': 2.5, 'lesion_center_x': LESION_CENTER_X,
            'lesion_slice_index': LESION_SLICE_INDEX,
        },
        'physics': {
            'energy_keV': 60, 'spectrum': 'monochromatic',
            'mu_air_cm': MU_AIR_CM,
            'mu_tissue_cm': MU_TISSUE_CM,
            'mu_metal_cm': MU_IRON_CM,
            'mu_lesion_cm': float(MU_LESION_CM),
            'lesion_delta_hu': LESION_DELTA_HU,
        },
        'noise_model': {
            'type': 'Poisson+Gaussian (Vaishnav 2020)',
            'I0_calibrated': round(I0, 2),
            'scatter_frac': SCATTER_FRAC,
            'sigma_e_counts': SIGMA_E_COUNTS,
            'target_sigma_hu': NOISE_SIGMA_TARGET_HU,
            'dc_offset_cm': round(dc_offset_cm, 8),
        },
        'realizations': {
            'count': num_realizations,
            'base_seed': BASE_SEED,
            'jitters': {
                str(i + 1): round(jitters[i], 4)
                for i in range(num_realizations)
            },
        },
        'cho_reference': {
            'roi_size': 121, 'roi_center': [LESION_CENTER_X, PHANTOM_CENTER_Y],
            'channel_width_a': 7.5,
            'internal_noise_sigma': 15,
            'auc_tolerance': 0.005,
        },
    }
    with open(output_dir / 'generator_provenance.json', 'w') as f:
        json.dump(prov, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# PDF lab instructions
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pdf(output_dir: Path, num_realizations: int) -> None:
    """Generate MAR_ILS_Lab_Instructions.pdf."""
    _font_ok = False
    if os.path.exists(_DEJAVU_REGULAR) and os.path.exists(_DEJAVU_BOLD):
        try:
            pdfmetrics.registerFont(TTFont(PDF_FONT, _DEJAVU_REGULAR))
            pdfmetrics.registerFont(TTFont(PDF_FONT_BOLD, _DEJAVU_BOLD))
            _font_ok = True
        except Exception:
            pass
    if not _font_ok:
        globals()['PDF_FONT'] = "Helvetica"
        globals()['PDF_FONT_BOLD'] = "Helvetica-Bold"
        warnings.warn("DejaVu fonts not found — falling back to Helvetica.",
                      RuntimeWarning, stacklevel=2)

    pdf_path = output_dir / "MAR_ILS_Lab_Instructions.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=inch, rightMargin=inch,
    )
    styles = getSampleStyleSheet()
    T = ParagraphStyle("T", parent=styles["Title"],
                       fontName=PDF_FONT_BOLD, fontSize=16, spaceAfter=6)
    H1 = ParagraphStyle("H1", parent=styles["Heading1"],
                        fontName=PDF_FONT_BOLD, fontSize=13,
                        spaceBefore=14, spaceAfter=4)
    B = ParagraphStyle("B", parent=styles["Normal"],
                       fontName=PDF_FONT, fontSize=10, leading=14, spaceAfter=6)
    TC = ParagraphStyle("TC", parent=styles["Normal"],
                        fontName=PDF_FONT, fontSize=9, leading=11)
    TC_HDR = ParagraphStyle("TCH", parent=styles["Normal"],
                            fontName=PDF_FONT_BOLD, fontSize=9, leading=11,
                            textColor=colors.white)

    def _cell(v, h=False):
        return Paragraph(str(v), TC_HDR if h else TC)

    def _tbl(data, widths):
        wrapped = [[_cell(c, r == 0) for c in row] for r, row in enumerate(data)]
        t = Table(wrapped, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#EEF2F7")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    story = [
        Paragraph("MAR ILS Lab Instructions", T),
        Paragraph(
            f"Metal Artifact Reduction ILS — {DATASET_VERSION} ({STANDARD_REF})",
            styles["Heading2"],
        ),
        Paragraph(f"Generated {date.today().isoformat()}", styles["Italic"]),
        HRFlowable(width="100%", thickness=1, color=colors.grey, spaceAfter=12),
        Paragraph("1. Canonical Test Configuration", H1),
        _tbl([
            ["Parameter", "Value", "Reference"],
            ["Acquisition", "Fan-beam, SID=570 mm, SDD=1040 mm", "§A1.1(f)"],
            ["Projections", "720 angles × 512 detectors, equi-angular", "§A1.1(f)"],
            ["Body", "Ellipse 85×60 mm (170×120 vox)", "§A1.2(a,b)"],
            ["Metal", f"Iron rod, r=5 mm (10 vox), μ={MU_IRON_CM} cm⁻¹", "§A1.3"],
            ["Lesion", "Circular disc r=2.5 mm at (281,256), slice 128",
             "§A1.4"],
            ["Contrast", "~12 HU sinogram-domain (no post-FBP hard-set)",
             "§A1.4(e)"],
            ["Noise", "30 HU σ in soft tissue", "§A1.7.3"],
            ["Realizations", f"{num_realizations} LP + {num_realizations} LA",
             "§10.2"],
        ], [2.0 * inch, 2.8 * inch, 1.8 * inch]),
        Paragraph("2. Submission Instructions", H1),
        Paragraph(
            "Apply your MAR algorithm to the sinograms in sinograms/LP/ and "
            "sinograms/LA/ and submit reconstructed DICOMs:", B,
        ),
        _tbl([
            ["Folder", "Contents"],
            [f"mar_recon/LP/realization_001/ ... /{num_realizations:03d}/",
             f"MAR-corrected DICOMs (slice_NNNN.dcm, 1-indexed). "
             f"Only slice_{LESION_SLICE_INDEX + 1:04d}.dcm is scored."],
            [f"mar_recon/LA/realization_001/ ... /{num_realizations:03d}/",
             "MAR-corrected DICOMs for LA realizations."],
        ], [3.0 * inch, 3.6 * inch]),
        Paragraph("3. CHO Analysis", H1),
        Paragraph(
            "Run the following command (--internal-noise-sigma 15 is normative):", B,
        ),
        Paragraph(
            f"python run_cho_analysis_v7_0.py "
            f"--dataset-dir &lt;output_dir&gt; "
            f"--mar-output-dir ./mar_recon "
            f"--internal-noise-sigma 15",
            ParagraphStyle("code", parent=styles["Normal"], fontName="Courier",
                           fontSize=9, leading=12,
                           backColor=colors.HexColor("#F5F5F5"), spaceAfter=8),
        ),
    ]
    doc.build(story)
    print(f"  PDF written → {pdf_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "MAR ILS Dataset Generator v7.0.0 — "
            "ASTM WKXXXXX Rev 04 (Fan-Beam, Single Canonical Configuration)"
        ),
    )
    ap.add_argument(
        "--output-dir", default="./astm_reference_dataset",
        help="Root output directory (default: ./astm_reference_dataset)",
    )
    ap.add_argument(
        "--realizations", type=int, default=NUM_REALIZATIONS_DEFAULT,
        help=(
            f"Realizations per condition "
            f"(default: {NUM_REALIZATIONS_DEFAULT}; "
            f"screening: 20; formal reporting requires ≥40)"
        ),
    )
    ap.add_argument(
        "--workers", type=int, default=0,
        help="Parallel workers (0 = os.cpu_count())",
    )
    ap.add_argument("--no-pdf", action="store_true",
                    help="Skip PDF lab instructions")
    ap.add_argument("--no-checksums", action="store_true",
                    help="Skip SHA-256 checksum computation")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate configuration without writing files")
    args = ap.parse_args()

    num_real = args.realizations
    output_dir = Path(args.output_dir).resolve()
    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    print(f"Generator version : {DATASET_VERSION} ({STANDARD_REF})")
    print(f"Geometry          : fan-beam  SID={SID_MM} mm  SDD={SDD_MM} mm")
    print(f"Projections       : {N_ANGLES} angles × {N_DET} detectors (equi-angular)")
    print(f"Fan angle range   : ±{np.degrees(GAMMA_MAX_RAD):.2f}° "
          f"(Δγ = {np.degrees(DELTA_GAMMA_RAD):.4f}°)")
    print(f"Phantom           : ellipse 85×60 mm, iron rod r=5 mm "
          f"(μ={MU_IRON_CM} cm⁻¹)")
    print(f"Lesion            : circular disc r=2.5 mm at x={LESION_CENTER_X}, "
          f"slice {LESION_SLICE_INDEX}")
    print(f"Contrast          : ~{LESION_DELTA_HU:.0f} HU sinogram-domain")
    print(f"Realizations      : {num_real} LP + {num_real} LA "
          f"({'SCREENING' if num_real < 40 else 'FORMAL'})")
    print(f"Output directory  : {output_dir}")
    print(f"Workers           : {n_workers}")
    if _HAS_NUMBA:
        print(f"Acceleration      : Numba {numba.__version__} "
              f"(JIT + batch geometry sharing)")
    else:
        print(f"Acceleration      : none (install numba for ~4-8x speedup)")
    print()

    if num_real < 40:
        print("WARNING: Screening mode (< 40 realizations). Results are not "
              "reportable under ASTM WKXXXXX §10.2.", flush=True)
        print()

    if args.dry_run:
        print("DRY RUN — configuration validated, no files written.")
        return

    I0, dc_offset_cm = calibrate()

    # Warm up batch backprojector JIT (forward + single-slice already compiled
    # during calibration; batch kernel is only used in workers)
    if _HAS_NUMBA:
        print("  Compiling batch backprojector...", end=" ", flush=True)
        _dummy = _backproject_batch_jit(
            np.zeros((2, 4, 2), dtype=np.float64),
            np.array([0.0, 1.0]),
            SID_VOX, DELTA_GAMMA_RAD, float(DET_FAN_ANGLES_RAD[0]),
            4.0, 4.0, 4, 4, 2,
        )
        del _dummy
        print("done", flush=True)
    print()

    rng_jitter = np.random.default_rng(BASE_SEED - 1)
    jitters: dict[int, float] = {
        i: float(rng_jitter.uniform(-JITTER_MAX_DEG, JITTER_MAX_DEG))
        for i in range(num_real)
    }

    tasks = [
        (cond, place, i, str(output_dir), I0, jitters[i], dc_offset_cm)
        for cond, place in (("LP", True), ("LA", False))
        for i in range(num_real)
    ]

    done = 0
    n_tasks = len(tasks)
    print(f"Generating {n_tasks} realizations "
          f"({num_real} LP + {num_real} LA)...")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_realization_worker, *t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                label = fut.result()
                done += 1
                print(f"  [{done:3d}/{n_tasks}] {label}", flush=True)
            except Exception as exc:
                t = futs[fut]
                print(f"  ERROR {t[0]}/realization_{t[2] + 1:03d}: {exc}")

    print()

    write_provenance_json(output_dir, jitters, I0, dc_offset_cm, num_real)

    if not args.no_checksums:
        write_checksums(output_dir)

    if not args.no_pdf:
        generate_pdf(output_dir, num_real)

    print()
    print(f"Dataset complete → {output_dir}")
    print("Next step:")
    print(f"  python run_cho_analysis_v7_0.py \\")
    print(f"      --dataset-dir {output_dir} \\")
    print(f"      --mar-output-dir ./mar_recon \\")
    print(f"      --internal-noise-sigma 15")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
