#!/usr/bin/env python3
"""
MAR ILS generator v9.0.0: Vaishnav et al. (2020) head phantom, 2D, polychromatic.

This first version produces the 121 × 121 centre ROIs the CHO reads, for
FBP, MAR 1 and MAR 2, as .npz files. It is used for calibration and
validation; the lab dataset writer (sinograms + DICOM) follows.

Each image has its own generator, seeded by
  (BASE_SEED, class, amplitude code, pool, index, dose code)
with class 0 = signal absent, 1 = present; pool 0 = training, 1 = test;
amplitude code = round(1000·amplitude) for present images and 0 for absent
ones, so absent images are shared by all amplitude levels. The inset angle is
drawn from the same generator, uniform on [0, 2π).

Usage:
  python generator_v9_0_0.py --out roi.npz --dose 1e6 --amplitude 1.03 \\
      --lump-scale 0.01 --n-train 1000 --n-test 1000 --methods fbp mar1 mar2 \\
      --tau-counts 925 --tau-mu 1.0 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from mar_ils_core import mar_v9 as mv
from mar_ils_core import vaishnav_phantom as vp
from mar_ils_core import vaishnav_sim as vs

DATASET_VERSION: str = "v9.0.0"
BASE_SEED: int = 9_0_0_2026


def image_rng(cls: int, amplitude: float | None, pool: int, idx: int, dose: float | None) -> np.random.Generator:
    """dose None drops dose from the seed (common random numbers across doses,
    used for calibration so inset angles match between grid points)."""
    amp_code = 0 if (cls == 0 or amplitude is None) else round(1000 * amplitude)
    dose_code = 0 if dose is None else round(dose)
    return np.random.default_rng([BASE_SEED, cls, amp_code, pool, idx, dose_code])


MAR1_SWEEP = (2, 4, 8)   # graded-harm multipliers of tau_counts ("mar1x2", ...)
METHODS = ("fbp", "mar1", "mar2", "oracle") + tuple(f"mar1x{k}" for k in MAR1_SWEEP)


def simulate(cls: int, amplitude: float | None, dose: float, lump_scale: float, pool: int, idx: int,
             methods: tuple[str, ...], tau_counts: float, tau_mu: float,
             crn: bool = False) -> dict[str, np.ndarray]:
    """ROI images (μ, cm⁻¹, float32) for one realization, keyed by method.

    cls 1 with amplitude None gives a second, independent signal-absent class
    (null-signal check). "oracle" is the same realization (inset angle) scanned
    with every metal object replaced by the material it displaces, with its own
    noise draw. "mar1xK" is MAR 1 with the threshold multiplied by K.
    """
    rng = image_rng(cls, amplitude, pool, idx, None if crn else dose)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    sig = amplitude if cls else None
    lumps = vp.draw_lumps()
    lw, lb, lf = vs.line_integrals(angle=angle, amplitude=sig, lumps=lumps, lump_scale=lump_scale)
    counts = vs.measure(vs.expected_primary(lw, lb, lf, dose), dose, rng)
    p = vs.log_sinogram(counts, dose)
    roi = vs.centre_roi()
    out = {}
    if "fbp" in methods:
        out["fbp"] = vs.fbp(p, roi)
    if "mar1" in methods:
        out["mar1"] = vs.fbp(mv.mar1(counts, dose, tau_counts), roi)
    for k in MAR1_SWEEP:
        if f"mar1x{k}" in methods:
            out[f"mar1x{k}"] = vs.fbp(mv.mar1(counts, dose, k * tau_counts), roi)
    if "mar2" in methods:
        out["mar2"] = vs.fbp(mv.mar2(counts, dose, tau_mu, vs.fbp(p)), roi)
    if "oracle" in methods:
        ow, ob, of = vs.line_integrals(angle=angle, amplitude=sig, lumps=lumps,
                                       lump_scale=lump_scale, metal=False)
        oc = vs.measure(vs.expected_primary(ow, ob, of, dose), dose, rng)
        out["oracle"] = vs.fbp(vs.log_sinogram(oc, dose), roi)
    return {k: v.astype(np.float32) for k, v in out.items()}


def _job(args, **kw):
    cls, pool, idx = args
    return args, simulate(cls, pool=pool, idx=idx, **kw)


def _init_worker():
    try:
        import numba
        numba.set_num_threads(1)
    except ImportError:
        pass


def generate_rois(*, dose: float, amplitude: float, lump_scale: float, n_train: int, n_test: int,
                  methods: tuple[str, ...], tau_counts: float, tau_mu: float,
                  workers: int, crn: bool = False, pool_offset: int = 0) -> dict[str, np.ndarray]:
    """pool_offset shifts the seed pools (training 0 + offset, test 1 + offset) for
    runs that must not share images with any other run; arrays keep keys _0 and _1."""
    jobs = [(c, pool + pool_offset, i) for c in (0, 1)
            for pool, n in ((0, n_train), (1, n_test)) for i in range(n)]
    fn = partial(_job, amplitude=amplitude, dose=dose, lump_scale=lump_scale,
                 methods=methods, tau_counts=tau_counts, tau_mu=tau_mu, crn=crn)
    s = vp.ROI_PX
    arrays = {f"{m}_{c}_{p}": np.empty((n, s, s), np.float32)
              for m in methods for c in (0, 1) for p, n in ((0, n_train), (1, n_test))}
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        for (c, pool, i), res in ex.map(fn, jobs, chunksize=4):
            for m, img in res.items():
                arrays[f"{m}_{c}_{pool - pool_offset}"][i] = img
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dose", type=float, default=1e6, help="I0, counts per detector bin")
    ap.add_argument("--amplitude", type=float, default=1.03)
    ap.add_argument("--lump-scale", type=float, required=True)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--methods", nargs="+", default=["fbp"], choices=list(METHODS))
    ap.add_argument("--tau-counts", type=float, default=float("nan"))
    ap.add_argument("--tau-mu", type=float, default=float("nan"))
    ap.add_argument("--crn", action="store_true",
                    help="common random numbers: leave dose out of the seed (calibration only)")
    ap.add_argument("--pool-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    t = time.time()
    arrays = generate_rois(dose=a.dose, amplitude=a.amplitude, lump_scale=a.lump_scale,
                           n_train=a.n_train, n_test=a.n_test, methods=tuple(a.methods),
                           tau_counts=a.tau_counts, tau_mu=a.tau_mu, workers=a.workers, crn=a.crn,
                           pool_offset=a.pool_offset)
    meta = {"generator_version": DATASET_VERSION, "base_seed": BASE_SEED, "lump_seed": vp.LUMP_SEED,
            **{k: v for k, v in vars(a).items() if k not in ("out", "workers")},
            "signal_baseline": vp.signal_baseline(a.lump_scale),
            "seconds": round(time.time() - t, 1)}
    np.savez(a.out, meta=json.dumps(meta), **arrays)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
