"""
Reference reconstruction of ASTM WK101378 draft Rev 06, §A1.1(g) and (h).

Equi-angular fan-beam filtered backprojection: projections weighted by
D cos(gamma), the Ram-Lak kernel sampled in the spatial domain at the channel
spacing, zero-padded linear convolution, and 1/L^2 weighted pixel-driven
backprojection with linear interpolation between adjacent channels.
Coordinates follow §A1.1(i). Lengths are in cm, so mu comes out in cm^-1.

For use with the harness under Condition 2 (the laboratory's own MAR), where
draft standard §14.3 allows the supplied reference reconstruction. Under
Condition 1 of the ILS, laboratories do not consult or reuse this file
(protocol §A7).
"""
from __future__ import annotations

import math

import numpy as np

IMAGE_PX = 512
FOV_CM = 30.0
PIXEL_CM = FOV_CM / IMAGE_PX
CENTER_PX = 256
MU_WATER_CM = 0.2124          # §A1.1(h)

try:
    import numba
    _HAS_NUMBA = True
except ImportError:           # pragma: no cover
    _HAS_NUMBA = False


def pixel_centers_cm() -> tuple[np.ndarray, np.ndarray]:
    """x of each column and y of each row, in cm (§A1.1(i)); row 0 is the largest y."""
    idx = np.arange(IMAGE_PX) - CENTER_PX
    return idx * PIXEL_CM, -idx * PIXEL_CM


def filter_sinogram(p: np.ndarray, geo) -> np.ndarray:
    """Weight by D cos(gamma) and convolve each view with the Ram-Lak kernel, times d_gamma."""
    n_ch = geo.n_channels
    dg = geo.delta_gamma_rad
    sid_cm = geo.sid_mm / 10.0
    n_pad = 1 << math.ceil(math.log2(2 * n_ch))
    n = np.arange(n_pad)
    n = np.where(n < n_pad // 2, n, n - n_pad)
    g = np.zeros(n_pad)
    g[n == 0] = 1.0 / (8.0 * dg ** 2)
    odd = (n % 2) != 0
    g[odd] = -1.0 / (2.0 * (np.pi * np.sin(n[odd] * dg)) ** 2)
    weighted = p * (sid_cm * np.cos(geo.gammas_rad))[None, :]
    q = np.fft.irfft(np.fft.rfft(weighted, n=n_pad, axis=1) * np.fft.rfft(g)[None, :],
                     n=n_pad, axis=1)
    return q[:, :n_ch] * dg


def _backproject_numpy(q, betas, sid, dg, g0, xs, ys):
    n_ch = q.shape[1]
    x, y = np.meshgrid(xs, ys)
    out = np.zeros_like(x)
    for i, b in enumerate(betas):
        cb, sb = math.cos(b), math.sin(b)
        u = sid - (x * cb + y * sb)
        v = x * sb - y * cb
        di = (np.arctan2(v, u) - g0) / dg
        ok = (di >= 0.0) & (di <= n_ch - 1)
        lo = np.minimum(np.floor(np.where(ok, di, 0.0)).astype(np.intp), n_ch - 2)
        f = di - lo
        val = ((1.0 - f) * q[i, lo] + f * q[i, lo + 1]) / (u * u + v * v)
        out += np.where(ok, val, 0.0)
    return out


if _HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _backproject_numba(q, betas, sid, dg, g0, xs, ys):
        n_views, n_ch = q.shape
        out = np.zeros((ys.size, xs.size))
        cb = np.cos(betas)
        sb = np.sin(betas)
        for r in numba.prange(ys.size):
            y = ys[r]
            for c in range(xs.size):
                x = xs[c]
                acc = 0.0
                for i in range(n_views):
                    u = sid - (x * cb[i] + y * sb[i])
                    v = x * sb[i] - y * cb[i]
                    di = (math.atan2(v, u) - g0) / dg
                    if di >= 0.0 and di <= n_ch - 1:
                        lo = min(int(di), n_ch - 2)
                        f = di - lo
                        acc += ((1.0 - f) * q[i, lo] + f * q[i, lo + 1]) / (u * u + v * v)
                out[r, c] = acc
        return out


def fbp(p: np.ndarray, geo) -> np.ndarray:
    """Reconstruct one sinogram (n_views, n_channels) to a 512 x 512 image of mu in cm^-1."""
    q = filter_sinogram(np.asarray(p, dtype=np.float64), geo)
    xs, ys = pixel_centers_cm()
    args = (q, np.asarray(geo.betas_rad, dtype=np.float64), geo.sid_mm / 10.0,
            geo.delta_gamma_rad, geo.gamma0_rad, xs, ys)
    bp = _backproject_numba(*args) if _HAS_NUMBA else _backproject_numpy(*args)
    return bp * (2.0 * np.pi / geo.n_views)


def to_hu(mu: np.ndarray) -> np.ndarray:
    """HU per §A1.1(h)."""
    return 1000.0 * (mu - MU_WATER_CM) / MU_WATER_CM
