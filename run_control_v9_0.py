#!/usr/bin/env python3
"""
Control runner v9.0: applies the supplied control method and the reference
reconstruction to every sinogram of a dataset and writes one DICOM CT image
per realization (ASTM WK101378 draft Rev 06 §17.7 and §A1.9; ILS protocol
§A8.2, Steps 2 and 3).

The improving control (MAR 2) is the method of the interlaboratory study. The
degrading control (MAR 1) is applied to the control dataset in the pilot.
Thresholds are read from the dataset record and must equal the values built
into this software.

Usage:
  # Step 1: verify the SHA-256 checksums of a downloaded dataset
  python run_control_v9_0.py --verify REP01

  # Step 2: installation check, on the familiarization dataset
  python run_control_v9_0.py --recon-check FAMILIARIZATION/recon_check/recon_check.h5
  python run_control_v9_0.py FAMILIARIZATION C1_FAM
  python run_control_v9_0.py --compare C1_FAM FAMILIARIZATION/expected_C1

  # Step 3: one run directory per dataset
  python run_control_v9_0.py REP01 C1_R1
  python run_control_v9_0.py NULL C1_NULL

  # Pilot only: the degrading control on the control dataset
  python run_control_v9_0.py CONTROL C1_CONTROL --control degrading

A run that is interrupted can be started again with the same command; images
already written are kept. Each run writes <run directory>_run_record.json
beside the run directory, with the software and package versions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from mar_ils_core import control_v9, dicom_v9
from mar_ils_core import vaishnav_sim as vs

RUNNER_VERSION = "v9.0.0"
RECON_CHECK_TOL_HU = 1.0                # Rev 06 §12.2
PACKAGES = ("numpy", "scipy", "numba", "h5py", "pydicom")


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform(),
           "machine": platform.machine()}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def source_commit() -> str:
    try:
        here = Path(__file__).resolve().parent
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=False)
        return rev.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def read_line_integrals(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        p = f["line_integrals"][()]
    if p.shape != (vs.N_VIEWS, vs.N_DET):
        raise ValueError(f"{path}: line_integrals shape {p.shape}, expected ({vs.N_VIEWS}, {vs.N_DET})")
    return p


def _init_worker():
    try:
        import numba
        numba.set_num_threads(1)
    except ImportError:
        pass


def _one(args) -> str:
    h5_path, dcm_path, control, i0, uids = args
    if not dcm_path.exists():
        hu = control_v9.control_hu(read_line_integrals(h5_path), control, i0)
        tmp = dcm_path.with_name(dcm_path.name + ".partial")
        dicom_v9.write_ct(hu, tmp, mar_applied=True, **uids)
        os.replace(tmp, dcm_path)
    return dcm_path.name


def _check_record(dataset_dir: Path) -> dict:
    record = json.loads((dataset_dir / "dataset_record.json").read_text())
    for key, value in (("mar2_tau_mu_cm", control_v9.TAU_MU_CM), ("mar1_tau_counts", control_v9.TAU_COUNTS)):
        if record.get(key) != value:
            raise ValueError(f"{dataset_dir}: threshold {key} in the dataset record is {record.get(key)!r}, "
                             f"but this software uses {value!r}; thresholds shall not be changed "
                             "(draft standard 17.7(c))")
    return record


def _run_state(record_path: Path) -> dict:
    """Study and Series UIDs of the run, kept across restarts."""
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if "dicom" in old:
            return old["dicom"]
    t = time.gmtime()
    return {"study_uid": dicom_v9.fixed_length_uid(), "series_uid": dicom_v9.fixed_length_uid(),
            "content_date": time.strftime("%Y%m%d", t), "content_time": time.strftime("%H%M%S", t)}


def run(dataset_dir: Path, run_dir: Path, control: str = "improving", workers: int | None = None) -> int:
    if control not in control_v9.CONTROLS:
        raise ValueError(f"unknown control {control!r}; choose from {', '.join(control_v9.CONTROLS)}")
    record = _check_record(dataset_dir)
    sinograms = sorted((dataset_dir / "sinograms").glob("*.h5"))
    if not sinograms:
        raise ValueError(f"no sinograms in {dataset_dir / 'sinograms'}")
    i0 = float(record["I0"])
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    record_path = run_dir.parent / f"{run_dir.name}_run_record.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in run_dir.glob("*.partial"):
        stale.unlink()
    state = _run_state(record_path)
    run_record = {"tool": "run_control_v9_0.py", "runner_version": RUNNER_VERSION,
                  "source_commit": source_commit(), "standard": "ASTM WK101378 draft Revision 06",
                  "dataset": record["dataset"], "run_dir": run_dir.name, "control": control,
                  "tau_mu_cm": control_v9.TAU_MU_CM, "tau_counts": control_v9.TAU_COUNTS, "I0": i0,
                  "versions": versions(), "dicom": state, "complete": False}
    record_path.write_text(json.dumps(run_record, indent=1))

    todo = [(p, run_dir / f"{p.stem}.dcm", control, i0, state) for p in sinograms]
    done_before = sum(job[1].exists() for job in todo)
    print(f"{record['dataset']} -> {run_dir} ({control} control): {len(todo)} realizations, "
          f"{done_before} already written, {workers} workers", flush=True)
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker) as ex:
        for n, _name in enumerate(ex.map(_one, todo, chunksize=2), 1):
            if n % 200 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / 60:.1f} min, about {el / n * (len(todo) - n) / 60:.0f} min left",
                      flush=True)

    written = {p.stem for p in run_dir.glob("*.dcm")}
    expected = {p.stem for p in sinograms}
    if written != expected:
        print(f"{run_dir}: {len(expected - written)} images missing, {len(written - expected)} unexpected",
              file=sys.stderr)
        return 1
    run_record.update(complete=True, n_images=len(written), seconds=round(time.time() - t0, 1))
    record_path.write_text(json.dumps(run_record, indent=1))
    print(f"Run record: {record_path}")
    return 0


def verify(dataset_dir: Path) -> int:
    """Protocol Step 1: every file listed in checksums_sha256.txt is present and unchanged."""
    listing = dataset_dir / "checksums_sha256.txt"
    if not listing.exists():
        print(f"{listing} not found", file=sys.stderr)
        return 1
    bad, missing, n = [], [], 0
    for line in listing.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        path = dataset_dir / rel
        n += 1
        if not path.exists():
            missing.append(rel)
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != digest:
            bad.append(rel)
    passed = not (bad or missing)
    print(f"Checksums of {dataset_dir}: {n} files listed, {len(bad)} changed, {len(missing)} missing "
          f"({'PASS' if passed else 'FAIL'})")
    for label, items in (("changed", bad), ("missing", missing)):
        if items:
            print(f"  first {label}: {items[:3]}")
    return 0 if passed else 1


def recon_check(check_h5: Path) -> int:
    """Rev 06 §12.2 with the reference reconstruction: an installation check."""
    p = read_line_integrals(check_h5)
    with h5py.File(check_h5, "r") as f:
        ref = f["reference_image_hu"][()]
    hu = vs.to_hu(vs.fbp(p.astype(np.float64)))
    worst = float(np.max(np.abs(hu - ref)))
    passed = worst <= RECON_CHECK_TOL_HU
    print(f"Reconstruction check: largest difference {worst:.6f} HU "
          f"({'PASS' if passed else 'FAIL'}, tolerance {RECON_CHECK_TOL_HU} HU)")
    return 0 if passed else 1


def compare(run_dir: Path, expected_dir: Path) -> int:
    """Protocol Step 2.3: the images of a run equal the expected images, pixel for pixel."""
    got = {p.stem for p in run_dir.glob("*.dcm")}
    want = {p.stem for p in expected_dir.glob("*.dcm")}
    if not want:
        print(f"{expected_dir}: no .dcm files", file=sys.stderr)
        return 1
    n_diff, worst, worst_name = 0, 0.0, None
    for name in sorted(got & want):
        d = np.abs(dicom_v9.read_hu(run_dir / f"{name}.dcm") - dicom_v9.read_hu(expected_dir / f"{name}.dcm"))
        if d.any():
            n_diff += 1
            if d.max() > worst:
                worst, worst_name = float(d.max()), name
    missing, extra = sorted(want - got), sorted(got - want)
    passed = not (n_diff or missing or extra)
    print(f"Comparison of {run_dir} with {expected_dir}: {len(got & want)} images compared, "
          f"{n_diff} differ, {len(missing)} missing, {len(extra)} unexpected "
          f"({'PASS' if passed else 'FAIL'})")
    if n_diff:
        print(f"  largest difference {worst:.0f} HU, in {worst_name}")
    if missing:
        print(f"  first missing: {missing[:3]}")
    if extra:
        print(f"  first unexpected: {extra[:3]}")
    return 0 if passed else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", nargs="?", type=Path, help="dataset directory, e.g. REP01")
    ap.add_argument("run_dir", nargs="?", type=Path, help="run directory to write, e.g. C1_R1")
    ap.add_argument("--control", choices=control_v9.CONTROLS, default="improving")
    ap.add_argument("--workers", type=int, help="default: one fewer than the number of processors")
    ap.add_argument("--recon-check", type=Path, metavar="RECON_CHECK_H5",
                    help="run the reconstruction check of draft standard 12.2 and exit")
    ap.add_argument("--compare", type=Path, nargs=2, metavar=("RUN_DIR", "EXPECTED_DIR"),
                    help="compare the images of a run with the expected images and exit")
    ap.add_argument("--verify", type=Path, metavar="DATASET_DIR",
                    help="check the SHA-256 checksums of a downloaded dataset and exit")
    ap.add_argument("--versions", action="store_true", help="print the Python and package versions and exit")
    a = ap.parse_args()
    if a.versions:
        print(json.dumps(versions(), indent=1))
        return
    if a.verify:
        sys.exit(verify(a.verify))
    if a.recon_check:
        sys.exit(recon_check(a.recon_check))
    if a.compare:
        sys.exit(compare(*a.compare))
    if a.dataset_dir is None or a.run_dir is None:
        ap.error("give dataset_dir and run_dir, or --verify, --recon-check, --compare or --versions")
    try:
        sys.exit(run(a.dataset_dir, a.run_dir, a.control, a.workers))
    except ValueError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
