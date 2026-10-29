"""
Acquisition and reconstruction for the Vaishnav phantom (v9 design).

Geometry      : Mehranian et al. 2013 (Vaishnav ref. 55): fan beam, SID 541 mm,
                SDD 949 mm, 888 channels at 1 mm pitch on an arc, 984 views/360°.
Spectrum      : 120 kVp, W anode, 10° anode angle, 2.5 mm Al (Mehranian), SpekPy.
Attenuation   : NIST μ/ρ tables shipped with SpekPy; water, ICRU cortical bone, Fe.
Noise         : Poisson counts + constant scatter (SPR 1e-4 of the unattenuated
                flux, De Man 1999) + Gaussian electronic noise with variance
                160 counts² (La Rivière 2005).
Projections   : analytic (disk chords, Gaussian line integrals), as in Vaishnav.
Reconstruction: equi-angular fan-beam FBP with the Ram-Lak kernel (Kak and
                Slaney §3.4.2), zero-padded linear convolution.

No scatter or beam-hardening correction is applied (Vaishnav reconstructs with
FBP only). Lengths in cm, angles in radians.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import erf

from mar_ils_core import vaishnav_phantom as vp

try:
    import numba
    _HAS_NUMBA = True
except ImportError:          # pragma: no cover
    _HAS_NUMBA = False

SID_CM: float = 54.1
SDD_CM: float = 94.9
N_DET: int = 888
DET_PITCH_CM: float = 0.1
N_VIEWS: int = 984
DELTA_GAMMA: float = DET_PITCH_CM / SDD_CM
GAMMAS: np.ndarray = (np.arange(N_DET) - (N_DET - 1) / 2.0) * DELTA_GAMMA
BETAS: np.ndarray = np.arange(N_VIEWS) * (2.0 * np.pi / N_VIEWS)

KVP: float = 120.0
ANODE_ANGLE_DEG: float = 10.0
AL_FILTER_MM: float = 2.5
ENERGY_BIN_KEV: float = 1.0

SCATTER_TO_PRIMARY: float = 1e-4
ELECTRONIC_VARIANCE: float = 160.0
LOG_FLOOR_COUNTS: float = 1.0

MATERIALS = {"water": "Water, Liquid", "bone": "Bone, Cortical (ICRU)", "iron": "Fe"}


# ── Ray geometry ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def ray_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Source position (sx, sy) and unit direction (ux, uy), each (N_VIEWS, N_DET).

    Source at SID·(cos β, sin β); the central ray points at the origin and the
    ray to channel j is rotated by γ_j.
    """
    b = BETAS[:, None]
    theta = b + np.pi + GAMMAS[None, :]
    sx = np.broadcast_to(SID_CM * np.cos(b), theta.shape).copy()
    sy = np.broadcast_to(SID_CM * np.sin(b), theta.shape).copy()
    return sx, sy, np.cos(theta), np.sin(theta)


def _offset_and_foot(cx: float, cy: float):
    """Signed distance d from (cx, cy) to every ray, and the ray parameter t of the foot."""
    sx, sy, ux, uy = ray_geometry()
    dx, dy = cx - sx, cy - sy
    return ux * dy - uy * dx, ux * dx + uy * dy


def disk_chords(cx: float, cy: float, r: float) -> np.ndarray:
    d, _ = _offset_and_foot(cx, cy)
    return 2.0 * np.sqrt(np.clip(r * r - d * d, 0.0, None))


def gaussian_integrals(cx: float, cy: float, w: float) -> np.ndarray:
    """∫ exp(-π|p - c|²/w²) along every ray = |w|·exp(-π d²/w²)."""
    d, _ = _offset_and_foot(cx, cy)
    return abs(w) * np.exp(-np.pi * d * d / (w * w))


@lru_cache(maxsize=1)
def _signal_disk_rays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat indices of rays through the centred signal disk, with their entry and exit t."""
    d0, t0 = _offset_and_foot(0.0, 0.0)
    r = vp.SIGNAL_RADIUS_CM
    idx = np.flatnonzero(np.abs(d0) < r)
    half = np.sqrt(r * r - d0.ravel()[idx] ** 2)
    return idx, t0.ravel()[idx] - half, t0.ravel()[idx] + half


def gaussian_integrals_in_disk(cx: float, cy: float, w: float, r: float = vp.SIGNAL_RADIUS_CM) -> np.ndarray:
    """Part of the Gaussian line integral inside the centred signal disk."""
    assert r == vp.SIGNAL_RADIUS_CM
    sx, sy, ux, uy = ray_geometry()
    idx, ta, tb = _signal_disk_rays()
    dx, dy = cx - sx.ravel()[idx], cy - sy.ravel()[idx]
    uxi, uyi = ux.ravel()[idx], uy.ravel()[idx]
    d = uxi * dy - uyi * dx
    tc = uxi * dx + uyi * dy
    aw = abs(w)
    k = math.sqrt(math.pi) / aw
    out = np.zeros(sx.size)
    out[idx] = np.exp(-np.pi * d * d / (w * w)) * 0.5 * aw * (erf(k * (tb - tc)) - erf(k * (ta - tc)))
    return out.reshape(sx.shape)


@lru_cache(maxsize=2)
def _static_line_integrals(metal: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parts of the phantom that do not rotate: skull, water, inset disk, teeth,
    fillings and cochlear implants. metal False leaves the fillings and cochlear
    implants as the bone they displace (oracle MAR)."""
    zero = lambda: np.zeros((N_VIEWS, N_DET))
    water, bone, iron = zero(), zero(), zero()
    c_inner = disk_chords(0, 0, vp.INNER_RADIUS_CM)
    bone += disk_chords(0, 0, vp.OUTER_RADIUS_CM) - c_inner
    water += c_inner
    water += (vp.INSET_WATER_FACTOR - 1.0) * disk_chords(0, 0, vp.INSET_RADIUS_CM)
    for n, (cx, cy) in enumerate(vp.tooth_centres_cm()):
        c = disk_chords(cx, cy, vp.TOOTH_RADIUS_CM)
        water -= c
        bone += c
        if metal and n in vp.FILLED_TEETH:
            f = disk_chords(cx, cy, vp.FILLING_RADIUS_CM)
            bone -= f
            iron += f
    for cx, cy in vp.COCHLEAR_CENTRES_CM if metal else ():
        c = disk_chords(cx, cy, vp.COCHLEAR_RADIUS_CM)
        bone -= c
        iron += c
    for a in (water, bone, iron):
        a.flags.writeable = False
    return water, bone, iron


def line_integrals(*, angle: float, amplitude: float | None, lumps: vp.Lumps,
                   lump_scale: float = 1.0,
                   metal: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic (water, bone, iron) basis line integrals in cm, each (N_VIEWS, N_DET).

    Mirrors vaishnav_phantom.basis_maps. The two DBS implants displace the
    inset (1.05 water); lumps inside their 0.3 cm disks are neglected.
    metal False replaces every metal object by the material it displaces
    (the metal-free scan used for oracle MAR).
    """
    sw, sb, sf = _static_line_integrals(metal)
    water, bone, iron = sw.copy(), sb.copy(), sf.copy()
    rot = vp.rotate(lumps.centres, angle)
    for (cx, cy), w, a in zip(rot, lumps.widths, lumps.amplitudes):
        water += lump_scale * a * gaussian_integrals(cx, cy, w)
    if amplitude is not None:
        c_sig = disk_chords(0, 0, vp.SIGNAL_RADIUS_CM)
        water += (amplitude * vp.signal_baseline(lump_scale) - vp.INSET_WATER_FACTOR) * c_sig
        for (cx, cy), w, a in zip(rot, lumps.widths, lumps.amplitudes):
            water -= lump_scale * a * gaussian_integrals_in_disk(cx, cy, w)
    for cx, cy in vp.rotate(np.array(vp.DBS_CENTRES_CM), angle) if metal else ():
        c = disk_chords(cx, cy, vp.DBS_RADIUS_CM)
        water -= vp.INSET_WATER_FACTOR * c
        iron += c
    return water, bone, iron


# ── Spectrum and attenuation ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Bin-centre energies (keV) and photon-number weights summing to 1."""
    import spekpy
    s = spekpy.Spek(kvp=KVP, th=ANODE_ANGLE_DEG, dk=ENERGY_BIN_KEV)
    s.filter("Al", AL_FILTER_MM)
    k, phi = s.get_spectrum()
    keep = phi > 0
    k, phi = np.asarray(k)[keep], np.asarray(phi)[keep]
    return k, phi / phi.sum()


@lru_cache(maxsize=1)
def attenuation() -> dict[str, np.ndarray]:
    """Linear attenuation μ(E) in cm⁻¹ on the spectrum energies, per basis material."""
    from spekpy.DataTables import MuData
    k, _ = spectrum()
    mu = MuData("nist_mu.dat")
    return {name: np.asarray(mu.get_mu_composition(comp, k)) for name, comp in MATERIALS.items()}


def effective_mu_water() -> float:
    """Spectrum-weighted μ_water, used only to express images in HU."""
    _k, w = spectrum()
    return float(np.sum(w * attenuation()["water"]))


# ── Forward model ───────────────────────────────────────────────────────────────

if _HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _poly(water, bone, iron, w, mw, mb, mf):
        flat_w, flat_b, flat_f = water.ravel(), bone.ravel(), iron.ravel()
        out = np.empty(flat_w.size)
        for k in numba.prange(flat_w.size):
            acc = 0.0
            for e in range(w.size):
                acc += w[e] * math.exp(-(flat_w[k] * mw[e] + flat_b[k] * mb[e] + flat_f[k] * mf[e]))
            out[k] = acc
        return out.reshape(water.shape)


def expected_primary(water: np.ndarray, bone: np.ndarray, iron: np.ndarray,
                     i0: float) -> np.ndarray:
    """Polychromatic expected primary counts per ray."""
    _, w = spectrum()
    mu = attenuation()
    return i0 * _poly(np.ascontiguousarray(water), np.ascontiguousarray(bone),
                      np.ascontiguousarray(iron), w, mu["water"], mu["bone"], mu["iron"])


def measure(primary: np.ndarray, i0: float, rng: np.random.Generator) -> np.ndarray:
    """Noisy detector reading: Poisson(primary + scatter) + N(0, σ_e²)."""
    scatter = SCATTER_TO_PRIMARY * i0
    return (rng.poisson(primary + scatter).astype(np.float64)
            + rng.normal(0.0, math.sqrt(ELECTRONIC_VARIANCE), primary.shape))


def log_sinogram(counts: np.ndarray, i0: float) -> np.ndarray:
    return -np.log(np.maximum(counts, LOG_FLOOR_COUNTS) / i0)


# ── Reconstruction ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _ramlak_kernel_fft() -> np.ndarray:
    """FFT of the equi-angular Ram-Lak kernel g(nα), zero-padded for linear convolution."""
    n_pad = 1 << math.ceil(math.log2(2 * N_DET))
    n = np.arange(n_pad)
    n = np.where(n < n_pad // 2, n, n - n_pad)
    g = np.zeros(n_pad)
    g[n == 0] = 1.0 / (8.0 * DELTA_GAMMA ** 2)
    odd = (n % 2) != 0
    g[odd] = -1.0 / (2.0 * (np.pi * np.sin(n[odd] * DELTA_GAMMA)) ** 2)
    return np.fft.rfft(g)


def filter_sinogram(sino: np.ndarray) -> np.ndarray:
    """Pre-weight by D·cos γ and convolve with the Ram-Lak kernel (× Δγ)."""
    kf = _ramlak_kernel_fft()
    n_pad = 2 * (kf.size - 1)
    weighted = sino * (SID_CM * np.cos(GAMMAS))[None, :]
    q = np.fft.irfft(np.fft.rfft(weighted, n=n_pad, axis=1) * kf[None, :], n=n_pad, axis=1)
    return q[:, :N_DET] * DELTA_GAMMA


if _HAS_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _backproject(q, betas, sid, dgamma, gamma0, xs, ys):
        """Pixel-driven backprojection. In the frame of view β, a pixel lies
        u = D − (x cos β + y sin β) along the central ray and
        v = x sin β − y cos β across it; γ = atan2(v, u), L² = u² + v²."""
        n_views, n_det = q.shape
        ny, nx = ys.size, xs.size
        cb = np.cos(betas)
        sb = np.sin(betas)
        out = np.zeros((ny, nx))
        for r in numba.prange(ny):
            y = ys[r]
            for c in range(nx):
                x = xs[c]
                acc = 0.0
                for i in range(n_views):
                    u = sid - (x * cb[i] + y * sb[i])
                    v = x * sb[i] - y * cb[i]
                    di = (math.atan2(v, u) - gamma0) / dgamma
                    if di >= 0.0 and di <= n_det - 1:
                        lo = min(int(di), n_det - 2)
                        f = di - lo
                        acc += ((1.0 - f) * q[i, lo] + f * q[i, lo + 1]) / (u * u + v * v)
                out[r, c] = acc
        return out


def fbp(sino: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Fan-beam FBP → μ image in cm⁻¹ on the 512 × 512, 30 cm grid.

    roi (row0, row1, col0, col1) reconstructs only that block of pixels.
    """
    q = filter_sinogram(np.asarray(sino, dtype=np.float64))
    x, y = vp.pixel_grid()
    xs, ys = x[0], y[:, 0]
    if roi is not None:
        r0, r1, c0, c1 = roi
        xs, ys = xs[c0:c1], ys[r0:r1]
    dbeta = 2.0 * np.pi / N_VIEWS
    return _backproject(q, BETAS, SID_CM, DELTA_GAMMA, float(GAMMAS[0]),
                        np.ascontiguousarray(xs), np.ascontiguousarray(ys)) * dbeta


def centre_roi(size: int = vp.ROI_PX) -> tuple[int, int, int, int]:
    h = size // 2
    m = vp.IMAGE_PX // 2
    return m - h, m + h + 1, m - h, m + h + 1


# Reconstructed μ of water at the centre (r < 2 cm) of a 20 cm water cylinder,
# noise-free, I₀ = 1e6 with the standard scatter (Rev 06 §A1.1(h)).
MU_WATER_REF_CM: float = 0.2124


def to_hu(mu_img: np.ndarray) -> np.ndarray:
    return 1000.0 * (mu_img - MU_WATER_REF_CM) / MU_WATER_REF_CM
