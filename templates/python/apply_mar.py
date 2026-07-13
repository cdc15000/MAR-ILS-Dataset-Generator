"""
Lab MAR Implementation — REPLACE THIS FILE

This file contains the function that the harness calls for each realization.
Replace the body of apply_mar() with your reconstruction + MAR pipeline.

The example below performs plain FBP (no MAR) as a starting point.
"""

from __future__ import annotations

import numpy as np

# Import the geometry dataclass from the harness
from run_mar_harness import FanBeamGeometry


def apply_mar(
    sinogram: np.ndarray,
    geo: FanBeamGeometry,
) -> np.ndarray:
    """Apply MAR reconstruction to a sinogram volume.

    Args:
        sinogram: float32 array, shape (n_slices, n_angles, n_det).
                  Line integrals in neper (ready to reconstruct).
        geo:      FanBeamGeometry with SID, SDD, angles, detector fan
                  angles, voxel size, etc.

    Returns:
        hu_volume: float32 array, shape (n_slices, 512, 512).
                   Reconstructed image in Hounsfield Units.
    """
    # ── REPLACE EVERYTHING BELOW WITH YOUR MAR PIPELINE ──

    n_slices = geo.n_slices
    n_angles = geo.n_angles
    n_det = geo.n_det
    nx, ny = 512, 512

    sid_vox = geo.sid_mm / geo.voxel_mm
    sid_cm = geo.sid_mm / 10.0
    angles_rad = np.deg2rad(geo.angles_deg)
    det_fan_rad = np.deg2rad(geo.det_fan_angles_deg)
    delta_gamma = np.deg2rad(geo.delta_gamma_deg)
    cos_fan = np.cos(det_fan_rad)
    gamma_min = det_fan_rad[0]
    cx, cy = nx / 2.0, ny / 2.0

    mu_tissue = 0.2059
    dc_offset = -0.029
    background_hu = 40.0

    hu_volume = np.zeros((n_slices, ny, nx), dtype=np.float32)

    for z in range(n_slices):
        sino = sinogram[z].astype(np.float64)

        # 1. Cosine pre-weight
        weighted = sino * cos_fan[np.newaxis, :]

        # 2. Ram-Lak filter
        freq = np.fft.rfftfreq(n_det)
        ramp = np.abs(freq)
        filtered = np.fft.irfft(
            np.fft.rfft(weighted, axis=1) * ramp[np.newaxis, :],
            n=n_det, axis=1,
        )

        # 3. Distance-weighted backprojection
        recon = np.zeros((ny, nx), dtype=np.float64)
        for a_idx in range(n_angles):
            beta = angles_rad[a_idx]
            sx = cx + sid_vox * np.cos(beta)
            sy = cy + sid_vox * np.sin(beta)

            for iy in range(ny):
                for ix in range(nx):
                    dx = ix - sx
                    dy = iy - sy
                    L_sq = dx * dx + dy * dy
                    pixel_angle = np.arctan2(dy, dx)
                    cr_angle = beta + np.pi
                    gamma = np.arctan2(
                        np.sin(pixel_angle - cr_angle),
                        np.cos(pixel_angle - cr_angle),
                    )
                    di = (gamma - gamma_min) / delta_gamma
                    di_int = int(di)
                    if 0 <= di_int < n_det - 1:
                        frac = di - di_int
                        val = (1 - frac) * filtered[a_idx, di_int] + \
                              frac * filtered[a_idx, di_int + 1]
                        recon[iy, ix] += val * (sid_vox * sid_vox) / L_sq

        # 4. Scale and convert to HU
        scale = np.pi / n_angles / (sid_cm * delta_gamma)
        mu_recon = recon * scale
        hu = (
            (mu_recon - mu_tissue - dc_offset) / mu_tissue * 1000.0
            + background_hu
        ).astype(np.float32)
        hu_volume[z] = hu

    return hu_volume
