"""
Phantom construction for MAR ILS framework.

Builds 2D attenuation maps (cm⁻¹) from normative phantom geometry.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage

from mar_ils_core.constants import (
    X_DIM, Y_DIM,
    PHANTOM_CENTER_X, PHANTOM_CENTER_Y,
    BODY_SEMI_X_VOX, BODY_SEMI_Y_VOX,
    METAL_RADIUS_VOX, LESION_RADIUS_VOX,
    LESION_CENTER_X,
    MU_AIR_CM, MU_TISSUE_CM, MU_IRON_CM, MU_LESION_CM,
)


def build_body_mask(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Elliptical body mask (170×120 voxel semi-axes)."""
    return (
        ((xx - PHANTOM_CENTER_X) / BODY_SEMI_X_VOX) ** 2
        + ((yy - PHANTOM_CENTER_Y) / BODY_SEMI_Y_VOX) ** 2
        <= 1.0
    )


def build_metal_mask(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Circular metal rod mask (10 voxel radius, centred)."""
    return (
        (xx - PHANTOM_CENTER_X) ** 2 + (yy - PHANTOM_CENTER_Y) ** 2
        <= METAL_RADIUS_VOX ** 2
    )


def build_lesion_mask(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Circular lesion disc mask (5 voxel radius at (281,256))."""
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
    body_mask = build_body_mask(yy, xx)
    metal_mask = build_metal_mask(yy, xx)
    lesion_mask = build_lesion_mask(yy, xx)

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
