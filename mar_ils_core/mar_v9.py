"""
Vaishnav et al. 2020 §2.C MAR methods (sinogram inpainting by linear interpolation).

MAR 1: metal segmented in the sinogram by a fixed count threshold (rays whose
       detected counts fall below tau_counts), then interpolated.
MAR 2: metal segmented in the FBP image by a fixed attenuation threshold,
       the mask forward projected to find its trace, then interpolated.

Thresholds are fixed, not adapted to dose or signal, as in the paper; their
values are set once by the pre-registered rule and stored in the provenance.
"""
from __future__ import annotations

import math

import numpy as np

from mar_ils_core import vaishnav_phantom as vp
from mar_ils_core import vaishnav_sim as vs

try:
    import numba
    _HAS_NUMBA = True
except ImportError:          # pragma: no cover
    _HAS_NUMBA = False


def interpolate_trace(sino: np.ndarray, trace: np.ndarray) -> np.ndarray:
    """Replace trace samples in each view by linear interpolation across channels."""
    out = np.array(sino, dtype=np.float64, copy=True)
    ch = np.arange(sino.shape[1])
    for i in range(sino.shape[0]):
        bad = trace[i]
        if bad.any() and not bad.all():
            out[i, bad] = np.interp(ch[bad], ch[~bad], out[i, ~bad])
    return out


def mar1(counts: np.ndarray, i0: float, tau_counts: float) -> np.ndarray:
    """MAR 1 → corrected log sinogram."""
    trace = counts < tau_counts
    return interpolate_trace(vs.log_sinogram(counts, i0), trace)


if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _mask_trace(rows, cols, x0, y0, pix, betas, sid, dgamma, gamma0, n_det):
        """Pixel-driven trace: each mask pixel marks the channels its footprint
        covers in every view (half-width pix/√2 seen from the source)."""
        n_views = betas.size
        out = np.zeros((n_views, n_det), dtype=np.bool_)
        half = pix / math.sqrt(2.0)
        for i in range(n_views):
            cb = math.cos(betas[i])
            sb = math.sin(betas[i])
            for k in range(rows.size):
                x = x0 + cols[k] * pix
                y = y0 - rows[k] * pix
                u = sid - (x * cb + y * sb)
                v = x * sb - y * cb
                g = math.atan2(v, u)
                h = half / math.sqrt(u * u + v * v)
                lo = math.floor((g - h - gamma0) / dgamma)
                hi = math.ceil((g + h - gamma0) / dgamma)
                for j in range(max(lo, 0), min(hi, n_det - 1) + 1):
                    out[i, j] = True
        return out


def mask_trace(mask: np.ndarray) -> np.ndarray:
    """Rays (N_VIEWS, N_DET) that pass through any pixel of the image mask."""
    x, y = vp.pixel_grid()
    rows, cols = np.nonzero(mask)
    return _mask_trace(rows.astype(np.int64), cols.astype(np.int64), float(x[0, 0]),
                       float(y[0, 0]), vp.PIXEL_CM, vs.BETAS, vs.SID_CM, vs.DELTA_GAMMA,
                       float(vs.GAMMAS[0]), vs.N_DET)


def mar2(counts: np.ndarray, i0: float, tau_mu: float,
         nomar_image: np.ndarray | None = None) -> np.ndarray:
    """MAR 2 → corrected log sinogram. nomar_image is the full FBP μ image if already computed."""
    p = vs.log_sinogram(counts, i0)
    img = vs.fbp(p) if nomar_image is None else nomar_image
    return interpolate_trace(p, mask_trace(img > tau_mu))
