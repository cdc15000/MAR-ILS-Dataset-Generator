"""
Vaishnav et al. (Med Phys 47(8), 2020) head phantom, Table II and Fig. 1.

The phantom is described by three basis maps, each a multiple of one
material's density, so a polychromatic forward model needs only three line
integrals per ray:

    mu(E, r) = w(r)·mu_water(E) + b(r)·mu_bone(E) + f(r)·mu_iron(E)

Lengths are in cm, angles in radians; the phantom centre is (0, 0).
Interpretations of the paper are listed in private/vaishnav_phantom_spec.md §7–9.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

OUTER_RADIUS_CM: float = 12.0       # bone
INNER_RADIUS_CM: float = 10.0       # water
INSET_RADIUS_CM: float = 5.9
INSET_WATER_FACTOR: float = 1.05
LUMP_COUNT: int = 20
LUMP_CENTRE_MAX_CM: float = 4.9
SIGNAL_RADIUS_CM: float = 0.5

TOOTH_RADIUS_CM: float = 0.7        # bone (Fig. 1); Table II prints iron
FILLING_RADIUS_CM: float = 0.2      # iron
FILLED_TEETH: tuple[int, ...] = (1, 2, 3, 8, 9, 10)   # Fig. 1
COCHLEAR_CENTRES_CM: tuple[tuple[float, float], ...] = ((11.0, 0.0), (-11.0, 0.0))
COCHLEAR_RADIUS_CM: float = 0.6
DBS_CENTRES_CM: tuple[tuple[float, float], ...] = ((1.0, -5.1), (-1.0, -5.1))
DBS_RADIUS_CM: float = 0.3          # rotate with the inset

FOV_CM: float = 30.0                # Fig. 1 axes span ±15 cm
IMAGE_PX: int = 512
PIXEL_CM: float = FOV_CM / IMAGE_PX
ROI_PX: int = 121

LUMP_SEED: int = 20200801           # one physical inset; published with the dataset


def tooth_centres_cm() -> np.ndarray:
    n = np.arange(12)
    return np.stack([7.0 * np.cos(n * np.pi / 11), 9.0 * np.sin(n * np.pi / 11)], axis=1)


@dataclass(frozen=True)
class Lumps:
    """Gaussian lumps in inset coordinates (before rotation).

    amplitude is in units of mu_water: lump_i(r) = a_i·exp(-pi|r - r_i|²/w_i²).
    """
    centres: np.ndarray   # (N, 2) cm
    widths: np.ndarray    # (N,) cm
    amplitudes: np.ndarray  # (N,) × mu_water


def draw_lumps(seed: int = LUMP_SEED) -> Lumps:
    """Draw the lump set as printed: w = 1.3 + 2R, R ~ N(0,1);
    a = 1.5R, R ~ N(-1,1); centres uniform over the disk r < 4.9 cm."""
    rng = np.random.default_rng(seed)
    radius = LUMP_CENTRE_MAX_CM * np.sqrt(rng.uniform(size=LUMP_COUNT))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=LUMP_COUNT)
    centres = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    widths = 1.3 + 2.0 * rng.standard_normal(LUMP_COUNT)
    amplitudes = 1.5 * rng.normal(-1.0, 1.0, size=LUMP_COUNT)
    return Lumps(centres, widths, amplitudes)


def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


def pixel_grid(n: int = IMAGE_PX, pixel_cm: float = PIXEL_CM) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) pixel-centre coordinates in cm; row 0 is +y (image convention).

    Pixel (n//2, n//2) sits on the phantom centre, so the odd-sized ROI and the
    signal are centred on a pixel.
    """
    c = (np.arange(n) - n // 2) * pixel_cm
    x, y = np.meshgrid(c, -c)
    return x, y


def lump_map(x: np.ndarray, y: np.ndarray, lumps: Lumps, angle: float,
             lump_scale: float = 1.0) -> np.ndarray:
    out = np.zeros_like(x)
    for (cx, cy), w, a in zip(rotate(lumps.centres, angle), lumps.widths, lumps.amplitudes):
        out += lump_scale * a * np.exp(-np.pi * ((x - cx) ** 2 + (y - cy) ** 2) / w ** 2)
    return out


@lru_cache(maxsize=64)
def _lump_mean_in_signal_disk(seed: int) -> float:
    """Mean of the printed-scale lump sum over the signal disk, averaged over
    36 inset angles (it varies by about 1e-4 μ_water with angle)."""
    lumps = draw_lumps(seed)
    g = np.linspace(-SIGNAL_RADIUS_CM, SIGNAL_RADIUS_CM, 101)
    x, y = np.meshgrid(g, g)
    inside = x ** 2 + y ** 2 <= SIGNAL_RADIUS_CM ** 2
    return float(np.mean([lump_map(x[inside], y[inside], lumps, a).mean()
                          for a in np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)]))


def signal_baseline(lump_scale: float, seed: int = LUMP_SEED) -> float:
    """Mean background B̄ over the signal disk, × μ_water.

    Vaishnav §3: at amplitude 1.00 the signal equals the mean background, so
    the disk value is amplitude × B̄ = amplitude × (1.05 + s·⟨lumps⟩_disk).
    """
    return INSET_WATER_FACTOR + lump_scale * _lump_mean_in_signal_disk(seed)


def _disk(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def basis_maps(x: np.ndarray, y: np.ndarray, *, angle: float, amplitude: float | None,
               lumps: Lumps, lump_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel-sampled (water, bone, iron) basis maps.

    amplitude None means signal absent; otherwise the signal disk replaces the
    inset and lumps with amplitude × B̄ (see signal_baseline). lump_scale multiplies
    the printed lump amplitudes (calibrated against Vaishnav Fig. 3).
    """
    water = np.zeros_like(x)
    bone = np.zeros_like(x)
    iron = np.zeros_like(x)
    bone[_disk(x, y, 0, 0, OUTER_RADIUS_CM)] = 1.0
    inner = _disk(x, y, 0, 0, INNER_RADIUS_CM)
    bone[inner] = 0.0
    water[inner] = 1.0
    inset = _disk(x, y, 0, 0, INSET_RADIUS_CM)
    water[inset] = INSET_WATER_FACTOR
    water += lump_map(x, y, lumps, angle, lump_scale)   # untruncated, as in the analytic projector
    if amplitude is not None:
        sig = _disk(x, y, 0, 0, SIGNAL_RADIUS_CM)
        water[sig] = amplitude * signal_baseline(lump_scale)
    for n, (cx, cy) in enumerate(tooth_centres_cm()):
        t = _disk(x, y, cx, cy, TOOTH_RADIUS_CM)
        water[t], bone[t] = 0.0, 1.0
        if n in FILLED_TEETH:
            f = _disk(x, y, cx, cy, FILLING_RADIUS_CM)
            bone[f], iron[f] = 0.0, 1.0
    for cx, cy in COCHLEAR_CENTRES_CM:
        m = _disk(x, y, cx, cy, COCHLEAR_RADIUS_CM)
        bone[m], iron[m] = 0.0, 1.0
    for cx, cy in rotate(np.array(DBS_CENTRES_CM), angle):
        m = _disk(x, y, cx, cy, DBS_RADIUS_CM)
        water[m], iron[m] = 0.0, 1.0
    return water, bone, iron
