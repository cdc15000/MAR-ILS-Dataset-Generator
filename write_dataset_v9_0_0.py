#!/usr/bin/env python3
"""
MAR ILS dataset writer v9.0.0: sinograms (HDF5) and noMAR images (DICOM) for
ASTM WK101378 draft Rev 06 §8.2, in the master (labeled) layout of the
sinogram ingestion specification (version 2.1).

Each realization is produced by generator_v9_0_0.realization(), the same
function the validated ROI runs use, so a realization written here is the one
the CHO results describe. For each realization the writer stores:

  sinograms/<name>.h5   float32 line integrals (984, 888), [view, channel],
                        with the fixed acquisition attributes of spec §4.2
  noMAR/<name>.dcm      reference FBP of the stored float32 sinogram, in HU,
                        Metal Artifact Reduction Applied = NO

and, per dataset, dataset_record.json (parameters, spectrum, and the label of
every realization) and checksums_sha256.txt. Labels appear only in the file
names and the record, never inside a file, so the blinding tool only renames
files and writes a stripped record. FAMILIARIZATION also gets labels.csv and
recon_check/recon_check.h5.

Datasets and seed pools follow review item 37:
  REP01..REP09     pools 50+2(m-1) / 51+2(m-1), I0 1e6, levels 1.020 1.025 1.030
  CONTROL          pools 68 / 69, I0 4e6, level 1.030
  FAMILIARIZATION  pools 70 / 71, I0 1e6, levels 1.020 1.025 1.030, small and labeled

Pools 50 to 71 belong to the study material: writing them needs --production
(review item 37: generated under issue #11). For validation, --custom NAME
--pools TR TE writes any other pools.

Usage:
  python write_dataset_v9_0_0.py --dataset REP01 --out-root /Volumes/ILS --production
  python write_dataset_v9_0_0.py --custom VALID_D1 --pools 40 41 --dose 1e6 \\
      --levels 1.03 --n-train 100 --n-test 100 --out-root ./scratch
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from pydicom.uid import generate_uid

import generator_v9_0_0 as gen
from mar_ils_core import dicom_v9
from mar_ils_core import vaishnav_phantom as vp
from mar_ils_core import vaishnav_sim as vs

WRITER_VERSION = "v9.0.0"
LUMP_SCALE = 0.0588                     # Rev 06 §A1.2(a)
TAU_MU_CM = 1.9529025691374509          # Rev 06 §A1.9(b), private/ruggedness/data/meta.json
TAU_COUNTS = 1124.2100350620863         # MAR 1 (degrading control), same source
LEVELS = (1.020, 1.025, 1.030)
STUDY_POOLS = range(50, 72)
SPLITS = ("train", "test")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    pools: tuple[int, int]               # (training, test)
    dose: float
    levels: tuple[float, ...]
    n_train: int
    n_test: int
    familiarization: bool = False


def standard_specs(n_familiarization: int) -> dict[str, DatasetSpec]:
    specs = {f"REP{m:02d}": DatasetSpec(f"REP{m:02d}", (50 + 2 * (m - 1), 51 + 2 * (m - 1)),
                                        1e6, LEVELS, 1000, 1000) for m in range(1, 10)}
    specs["CONTROL"] = DatasetSpec("CONTROL", (68, 69), 4e6, (1.030,), 1000, 1000)
    specs["FAMILIARIZATION"] = DatasetSpec("FAMILIARIZATION", (70, 71), 1e6, LEVELS,
                                           n_familiarization, n_familiarization, True)
    return specs


@dataclass(frozen=True)
class Job:
    name: str
    cls: int                  # 0 absent, 1 present
    level: float | None       # None for absent
    split: str
    pool: int
    index: int


def realization_name(cls: int, level: float | None, split: str, index: int) -> str:
    head = "absent" if cls == 0 else f"present{round(1000 * level)}"
    return f"{head}_{split}_{index:04d}"


def jobs_for(spec: DatasetSpec) -> list[Job]:
    jobs = []
    for split, pool, n in (("train", spec.pools[0], spec.n_train), ("test", spec.pools[1], spec.n_test)):
        for i in range(n):
            jobs.append(Job(realization_name(0, None, split, i), 0, None, split, pool, i))
            for lev in spec.levels:
                jobs.append(Job(realization_name(1, lev, split, i), 1, lev, split, pool, i))
    return jobs


# ── Files (ingestion spec §4) ────────────────────────────────────────────────

def sinogram_attrs(dose: float) -> dict:
    return {
        "geometry": "fan-beam, equi-angular curved detector",
        "SID_mm": 10.0 * vs.SID_CM,
        "SDD_mm": 10.0 * vs.SDD_CM,
        "n_views": np.int32(vs.N_VIEWS),
        "n_channels": np.int32(vs.N_DET),
        "channel_pitch_mm": 10.0 * vs.DET_PITCH_CM,
        "delta_gamma_rad": float(vs.DELTA_GAMMA),
        "gamma0_rad": float(vs.GAMMAS[0]),
        "betas_rad": vs.BETAS.astype(np.float64),
        "gammas_rad": vs.GAMMAS.astype(np.float64),
        "I0": float(dose),
        "scatter_counts": float(vs.SCATTER_TO_PRIMARY * dose),
        "electronic_variance_counts2": float(vs.ELECTRONIC_VARIANCE),
        "log_floor_counts": float(vs.LOG_FLOOR_COUNTS),
    }


def write_sinogram(path: Path, p32: np.ndarray, dose: float, extra: dict | None = None) -> None:
    """Contiguous, uncompressed float32 (984, 888); track_times off so files are reproducible."""
    with h5py.File(path, "w", libver=("earliest", "v110")) as f:
        d = f.create_dataset("line_integrals", data=p32.astype("<f4"), track_times=False)
        for k, v in sinogram_attrs(dose).items():
            d.attrs[k] = v
        for k, v in (extra or {}).items():
            f.create_dataset(k, data=v, track_times=False)


def _atomic(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def stored_sinogram(job: Job, dose: float) -> tuple[np.ndarray, float]:
    """float32 line integrals as stored, and the inset angle."""
    _rng, angle, _lumps, _counts, p = gen.realization(job.cls, job.level, dose, LUMP_SCALE,
                                                      job.pool, job.index)
    return p.astype(np.float32), float(angle)


def nomar_hu(p32: np.ndarray) -> np.ndarray:
    """Reference FBP of the stored float32 sinogram (what laboratories read), in HU."""
    return vs.to_hu(vs.fbp(p32.astype(np.float64)))


def _write_one(args) -> dict:
    job, root, dose, uids = args
    h5_path = root / "sinograms" / f"{job.name}.h5"
    dcm_path = root / "noMAR" / f"{job.name}.dcm"
    p32, angle = stored_sinogram(job, dose)
    if not (h5_path.exists() and dcm_path.exists()):
        tmp = _atomic(h5_path)
        write_sinogram(tmp, p32, dose)
        os.replace(tmp, h5_path)
        tmp = _atomic(dcm_path)
        dicom_v9.write_ct(nomar_hu(p32), tmp, mar_applied=False, **uids)
        os.replace(tmp, dcm_path)
    return {**asdict(job), "inset_angle_rad": angle}


def _init_worker():
    try:
        import numba
        numba.set_num_threads(1)
    except ImportError:
        pass


# ── Dataset ──────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        here = Path(__file__).resolve().parent
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here,
                               capture_output=True, text=True, check=False).stdout.strip()
        return rev.stdout.strip() + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def load_state(root: Path) -> dict:
    """Study and Series UIDs and the Content Date and Time, fixed for the dataset across resumed runs."""
    state_path = root / ".writer_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    t = time.gmtime()
    state = {"study_uid": generate_uid(), "series_uid": generate_uid(),
             "content_date": time.strftime("%Y%m%d", t), "content_time": "000000"}
    state_path.write_text(json.dumps(state))
    return state


def write_recon_check(root: Path, spec: DatasetSpec) -> dict:
    """§12.2 check: a copy of present1030_test_0000 with its reference image in float64 HU."""
    job = Job(realization_name(1, 1.030, "test", 0), 1, 1.030, "test", spec.pools[1], 0)
    p32, _ = stored_sinogram(job, spec.dose)
    ref = nomar_hu(p32)
    path = root / "recon_check" / "recon_check.h5"
    path.parent.mkdir(exist_ok=True)
    tmp = _atomic(path)
    write_sinogram(tmp, p32, spec.dose, extra={"reference_image_hu": ref.astype(np.float64)})
    os.replace(tmp, path)
    return {"source_realization": job.name, "reference": "vaishnav_sim.fbp of the stored float32 "
            "sinogram, HU per Rev 06 A1.1(h), float64"}


def dataset_record(spec: DatasetSpec, realizations: list[dict], state: dict, extra: dict) -> dict:
    k, w = vs.spectrum()
    full = (spec.n_train == 1000 and spec.n_test == 1000) or spec.familiarization
    return {
        "dataset": spec.name,
        "writer_version": WRITER_VERSION,
        "generator_version": gen.DATASET_VERSION,
        "source_commit": git_commit(),
        "standard": "ASTM WK101378 draft Revision 06",
        "ingestion_spec_version": "2.1",
        "subset": not full,
        "base_seed": gen.BASE_SEED,
        "lump_seed": vp.LUMP_SEED,
        "lump_scale": LUMP_SCALE,
        "signal_baseline_mu_water": vp.signal_baseline(LUMP_SCALE),
        "seed_rule": "numpy default_rng([base_seed, class, round(1000*level) or 0, pool, index, round(I0)])",
        "pools": {"train": spec.pools[0], "test": spec.pools[1]},
        "I0": spec.dose,
        "levels": list(spec.levels),
        "n_train_per_class": spec.n_train,
        "n_test_per_class": spec.n_test,
        "n_realizations": len(realizations),
        "acquisition": {k_: (v.tolist() if isinstance(v, np.ndarray) else
                             (int(v) if isinstance(v, np.integer) else v))
                        for k_, v in sinogram_attrs(spec.dose).items() if k_ not in ("betas_rad", "gammas_rad")},
        "spectrum": {"kVp": vs.KVP, "anode_angle_deg": vs.ANODE_ANGLE_DEG, "al_filter_mm": vs.AL_FILTER_MM,
                     "spekpy_version": importlib.metadata.version("spekpy"),
                     "energy_keV": k.tolist(), "photon_fraction": w.tolist()},
        "image": {"pixels": vp.IMAGE_PX, "fov_mm": 300.0, "center_pixel": [256, 256],
                  "mu_water_cm": vs.MU_WATER_REF_CM, "stored": "HU rounded to integer, int16, slope 1, intercept 0"},
        "mar2_tau_mu_cm": TAU_MU_CM,
        "mar1_tau_counts": TAU_COUNTS,
        "dicom": state,
        **extra,
        "realizations": realizations,
    }


def write_dataset(spec: DatasetSpec, out_root: Path, workers: int) -> Path:
    root = out_root / spec.name
    (root / "sinograms").mkdir(parents=True, exist_ok=True)
    (root / "noMAR").mkdir(exist_ok=True)
    state = load_state(root)
    uids = {k: state[k] for k in ("study_uid", "series_uid", "content_date", "content_time")}
    jobs = jobs_for(spec)
    t0 = time.time()
    print(f"{spec.name}: {len(jobs)} realizations -> {root}", flush=True)
    realizations = []
    # spawn, not fork: forking after numba's parallel threading layer has started in
    # the parent kills the workers on Linux (macOS already defaults to spawn)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker) as ex:
        for n, rec in enumerate(ex.map(_write_one, [(j, root, spec.dose, uids) for j in jobs],
                                       chunksize=2), 1):
            realizations.append(rec)
            if n % 200 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el / 60:.1f} min, about {el / n * (len(jobs) - n) / 60:.0f} min left",
                      flush=True)

    extra = {}
    if spec.familiarization:
        extra["recon_check"] = write_recon_check(root, spec)
        with open(root / "labels.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["code", "class", "level", "pool", "index"])
            for r in realizations:
                wr.writerow([r["name"], "present" if r["cls"] else "absent",
                             "" if r["level"] is None else f"{r['level']:.3f}", r["split"], r["index"]])

    record = dataset_record(spec, realizations, state, {**extra, "seconds": round(time.time() - t0, 1)})
    (root / "dataset_record.json").write_text(json.dumps(record, indent=1))

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in
                   ("checksums_sha256.txt", ".writer_state.json") and not p.name.endswith(".partial"))
    with open(root / "checksums_sha256.txt", "w") as f:
        f.writelines(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files)
    print(f"  done in {(time.time() - t0) / 60:.1f} min; {len(files)} files checksummed", flush=True)
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--dataset", help="REP01..REP09, CONTROL or FAMILIARIZATION")
    which.add_argument("--custom", metavar="NAME", help="a validation dataset with the pools given")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--pools", type=int, nargs=2, metavar=("TRAIN", "TEST"))
    ap.add_argument("--dose", type=float, default=1e6)
    ap.add_argument("--levels", type=float, nargs="+", default=list(LEVELS))
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--n-test", type=int)
    ap.add_argument("--n-familiarization", type=int, default=25,
                    help="training and test realizations per class in FAMILIARIZATION")
    ap.add_argument("--production", action="store_true",
                    help="allow the study pools 50 to 71 (review item 37, issue #11)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    if a.dataset:
        specs = standard_specs(a.n_familiarization)
        if a.dataset not in specs:
            ap.error(f"unknown dataset {a.dataset}; choose from {', '.join(specs)}")
        spec = specs[a.dataset]
        if a.n_train or a.n_test:
            spec = DatasetSpec(spec.name, spec.pools, spec.dose, spec.levels,
                               a.n_train or spec.n_train, a.n_test or spec.n_test, spec.familiarization)
    else:
        if not a.pools:
            ap.error("--custom needs --pools")
        spec = DatasetSpec(a.custom, tuple(a.pools), a.dose, tuple(a.levels),
                           a.n_train or 1000, a.n_test or 1000)

    if any(p in STUDY_POOLS for p in spec.pools) and not a.production:
        sys.exit(f"pools {spec.pools} are study material (50 to 71); generating them needs --production")
    write_dataset(spec, a.out_root, a.workers)


if __name__ == "__main__":
    main()
