#!/usr/bin/env python3
"""
pilot_gate_v8_0.py
==================
Pilot gate for the v8.0.0 dataset (ASTM WKXXXXX Rev 06). Run once by the task
group with the reference implementation before the ILS; not a lab step.

A single N = 40 dataset resolves ΔAUC only to about ±0.1, so the gate uses the
mean over independent noise replicates rather than one dataset. For each
replicate (default 6..15, the confirmatory set; replicate 0 is the reference
dataset; replicates 1..5 were used to choose the operating point and are not
reused for the gate)
it:

  1. generates the scored slice (128) of 40 LP + 40 LA realizations with the
     generator's own code path (clean_sinogram, slice_rng,
     apply_noise_scatter_corrected), so slice 128 is bit-identical to slice 128
     of a full ``generator_v8_0_0.py --replicate R`` run;
  2. runs the reference LI-MAR (algorithms/v7/reference_li_mar_v7.py) and the
     null-signal check (algorithms/v7/reference_null_signal_v7.py);
  3. scores both with run_cho_analysis_v8_0.py (σ = 15 regularization).

Gate (reference MAR condition): the two-sided 95 % t-interval of the mean
LI-MAR ΔAUC over the replicates excludes zero.
Null-signal check: the 95 % t-interval of the mean null-signal AUC_MAR
contains 0.5.

Usage
-----
    python pilot_gate_v8_0.py --work-dir ./pilot_v8 --workers 3
    python pilot_gate_v8_0.py --work-dir ./pilot_v8 --replicates 6 7 8 9 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "algorithms" / "v7"))

import generator_v8_0_0 as gen
from mar_ils_core.constants import (
    BASE_SEED,
    JITTER_MAX_DEG,
    LESION_SLICE_INDEX,
    N_ANGLES,
    N_DET,
    NUM_REALIZATIONS_DEFAULT,
    X_DIM,
    Y_DIM,
    Z_DIM,
)

PILOT_REPLICATES = tuple(range(6, 16))
NULL_AUC_TOLERANCE = 0.02   # |mean null-signal AUC - 0.5| (review item 16)
SD_TARGET = 0.05            # replicate SD of ΔAUC (Rev 05 §17.10(b))


def _set_threads() -> None:
    try:
        import numba
        numba.set_num_threads(1)
    except ImportError:
        pass


def _scored_slice_worker(args) -> None:
    """Scored slice of one realization: sinogram (HDF5) + noMAR DICOM."""
    out, replicate, series, i, I0, jitter_deg, dc_offset_cm = args
    _set_threads()
    import h5py
    from pydicom.uid import generate_uid

    from mar_ils_core.dicom_utils import write_dicom_slice
    from mar_ils_core.noise import apply_noise_scatter_corrected
    from mar_ils_core.phantom import build_metal_mask

    clean = gen.clean_sinogram(series == "LP", jitter_deg)
    noisy = apply_noise_scatter_corrected(
        clean, I0, gen.slice_rng(replicate, series, i, LESION_SLICE_INDEX))
    tag = f"realization_{i + 1:03d}"
    h5_path = Path(out) / "sinograms" / series / f"{tag}.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        d = f.create_dataset("line_integrals", shape=(Z_DIM, N_ANGLES, N_DET),
                             dtype=np.float32, compression="gzip",
                             chunks=(1, N_ANGLES, N_DET))
        d[LESION_SLICE_INDEX] = noisy
    hu = gen.fbp_reconstruct_slice(noisy.astype(np.float64), dc_offset_cm=dc_offset_cm)
    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    write_dicom_slice(hu, LESION_SLICE_INDEX,
                      output_dir=Path(out) / "noMAR_recon" / series / tag,
                      realization_idx=i, condition_label=f"noMAR_{series}",
                      study_uid=generate_uid(), series_uid=generate_uid(),
                      metal_mask=build_metal_mask(yy, xx),
                      dataset_version=gen.DATASET_VERSION,
                      standard_ref=gen.STANDARD_REF,
                      round_hu=True)


def _mar_worker(args) -> None:
    method, ds, out, cond, i, dc_offset_cm = args
    _set_threads()
    if method == "li_mar":
        import reference_li_mar_v7 as m
        m.process_realization(ds, out, cond, i, dc_offset_cm)
    else:
        import reference_null_signal_v7 as m
        radius = (m.NULL_RADIUS_VOX if method == "null_signal"
                  else int(method.removeprefix("trace")))
        m.process_realization(ds, out, cond, i, dc_offset_cm, radius)


def _cho(ds: Path, mar_dir: Path, results: Path, workers: int, n: int,
         n_boot: int | None = None) -> dict:
    cmd = [sys.executable, str(REPO / "run_cho_analysis_v8_0.py"),
           "--dataset-dir", str(ds), "--mar-output-dir", str(mar_dir),
           "--internal-noise-sigma", "15", "--workers", str(workers),
           "--realizations", str(n), "--results-file", str(results)]
    if n_boot is not None:
        cmd += ["--n-boot", str(n_boot)]
    with open(results.with_suffix(".log"), "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    r = json.loads(results.read_text())
    return {"AUC_noMAR": r["noMAR"]["AUC"], "AUC_MAR": r["MAR"]["AUC"],
            "delta_AUC": r["delta_AUC"],
            "resub_minus_holdout_delta": r["MAR"]["Bias"] - r["noMAR"]["Bias"]}


def _t_interval(values: list[float]) -> tuple[float, float, float, float]:
    """Mean, SD, and two-sided 95 % t-interval."""
    from scipy.stats import t
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else float("nan")
    return mean, sd, mean - half, mean + half


def run_replicate(work: Path, replicate: int, I0: float, dc_offset_cm: float,
                  sigma_ref: float, jitters: dict[int, float], n: int,
                  workers: int, keep_sinograms: bool,
                  methods: tuple[str, ...] = ("li_mar", "null_signal"),
                  cho_n_boot: int | None = None) -> dict:
    out = work / f"r{replicate}"
    out.mkdir(parents=True, exist_ok=True)
    gen.write_provenance_json(out, jitters, I0, dc_offset_cm, n,
                              replicate=replicate, sigma_ref_hu=sigma_ref)
    jobs = [(str(out), replicate, s, i, I0, jitters[i], dc_offset_cm)
            for s in ("LP", "LA") for i in range(n)]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_scored_slice_worker, jobs))
    res = {}
    for method in methods:
        mar_dir = out / method
        jobs = [(method, str(out), str(mar_dir), c, i, dc_offset_cm)
                for c in ("LP", "LA") for i in range(n)]
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_mar_worker, jobs))
        res[method] = _cho(out, mar_dir, out / f"cho_{method}.json", workers, n,
                           cho_n_boot)
        print(f"  r{replicate} {method:11s}: AUC_noMAR {res[method]['AUC_noMAR']:.4f}  "
              f"AUC_MAR {res[method]['AUC_MAR']:.4f}  "
              f"ΔAUC {res[method]['delta_AUC']:+.4f}", flush=True)
    if not keep_sinograms:
        shutil.rmtree(out / "sinograms")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Pilot gate for the v8.0.0 dataset")
    ap.add_argument("--work-dir", default="./pilot_v8")
    ap.add_argument("--replicates", type=int, nargs="+", default=list(PILOT_REPLICATES))
    ap.add_argument("--realizations", type=int, default=NUM_REALIZATIONS_DEFAULT)
    ap.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    ap.add_argument("--keep-sinograms", action="store_true")
    ap.add_argument("--target-sigma-hu", type=float, default=gen.NOISE_SIGMA_TARGET_HU_V8,
                    help="noise target for the I0 calibration (operating-point sweeps)")
    ap.add_argument("--skip-null", action="store_true",
                    help="LI-MAR only (operating-point sweeps)")
    ap.add_argument("--cho-n-boot", type=int, default=None,
                    help="bootstrap draws for the CHO (large-N reference runs)")
    ap.add_argument("--trace-radii", type=int, nargs="*", default=[],
                    help="widened-trace (graded degrading) controls, radius in voxels")
    args = ap.parse_args()
    if 0 in args.replicates:
        ap.error("replicate 0 is the reference dataset; pilot replicates start at 1")

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    I0, dc_offset_cm, sigma_ref = gen.calibrate(args.target_sigma_hu)
    methods = (("li_mar",) if args.skip_null else ("li_mar", "null_signal")) + tuple(
        f"trace{R}" for R in args.trace_radii)
    rng_jitter = np.random.default_rng(BASE_SEED - 1)
    jitters = {i: float(rng_jitter.uniform(-JITTER_MAX_DEG, JITTER_MAX_DEG))
               for i in range(args.realizations)}

    per_rep = {}
    for r in args.replicates:
        print(f"Replicate {r}...", flush=True)
        per_rep[r] = run_replicate(work, r, I0, dc_offset_cm, sigma_ref, jitters,
                                   args.realizations, args.workers,
                                   args.keep_sinograms, methods,
                                   args.cho_n_boot)

    d_li = [per_rep[r]["li_mar"]["delta_AUC"] for r in args.replicates]
    a_null = ([per_rep[r]["null_signal"]["AUC_MAR"] for r in args.replicates]
              if not args.skip_null else [0.5, 0.5])
    a_nomar = [per_rep[r]["li_mar"]["AUC_noMAR"] for r in args.replicates]
    m, sd, lo, hi = _t_interval(d_li)
    mn, sdn, lon, hin = _t_interval(a_null)
    m0, sd0, _, _ = _t_interval(a_nomar)
    gate_pass = bool(lo > 0 or hi < 0)
    # Tolerance band (review item 16): a LOO AUC at chance sits slightly below
    # 0.5, so "interval contains 0.5" fails on precise data.
    null_pass = bool(abs(mn - 0.5) <= NULL_AUC_TOLERANCE)
    traces = {}
    for R in args.trace_radii:
        v = [per_rep[r][f"trace{R}"]["delta_AUC"] for r in args.replicates]
        tm, tsd, tlo, thi = _t_interval(v)
        traces[str(R)] = {"mean": tm, "sd": tsd, "ci95": [tlo, thi],
                          "gate": "PASS" if (thi < 0 and tsd <= SD_TARGET) else "FAIL"}

    summary = {
        "generator_version": gen.DATASET_VERSION,
        "standard_reference": gen.STANDARD_REF,
        "I0_calibrated": I0,
        "target_sigma_hu": args.target_sigma_hu,
        "replicates": args.replicates,
        "realizations_per_condition": args.realizations,
        "per_replicate": {str(r): per_rep[r] for r in args.replicates},
        "AUC_noMAR": {"mean": m0, "sd": sd0},
        "li_mar_delta_AUC": {"mean": m, "sd": sd, "ci95": [lo, hi],
                             "gate": "PASS" if gate_pass else "FAIL"},
        "null_signal_AUC_MAR": (None if args.skip_null else
                                {"mean": mn, "sd": sdn, "ci95": [lon, hin],
                                 "check": "PASS" if null_pass else "FAIL"}),
        "trace_delta_AUC": traces,
    }
    (work / "pilot_gate_results.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"AUC_noMAR            : mean {m0:.4f}  SD {sd0:.4f}")
    print(f"LI-MAR ΔAUC          : mean {m:+.4f}  SD {sd:.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  gate {'PASS' if gate_pass else 'FAIL'}")
    if not args.skip_null:
        print(f"Null-signal AUC_MAR  : mean {mn:.4f}  SD {sdn:.4f}  "
              f"95% CI [{lon:.4f}, {hin:.4f}]  |mean-0.5| <= {NULL_AUC_TOLERANCE}: "
              f"{'PASS' if null_pass else 'FAIL'}")
    for R, t_ in traces.items():
        print(f"Trace R={R:>2s} ΔAUC      : mean {t_['mean']:+.4f}  SD {t_['sd']:.4f}  "
              f"95% CI [{t_['ci95'][0]:+.4f}, {t_['ci95'][1]:+.4f}]  "
              f"gate {t_['gate']} (CI < 0 and SD <= {SD_TARGET})")
    print(f"Results → {work / 'pilot_gate_results.json'}")


if __name__ == "__main__":
    main()
