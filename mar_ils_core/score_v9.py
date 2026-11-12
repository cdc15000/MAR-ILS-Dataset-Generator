"""
Scoring a run directory of DICOM images (ILS protocol §A8.2, Step 5; Rev 06 §14.5, §14.6, §15).

A run directory holds one image per realization of one replicate dataset,
named by the realization's code (coded copies) or by its label (master and
familiarization datasets). The labels come from the released blinding key,
or, for labeled datasets, from the dataset record. For each signal level the
noMAR series (the dataset's noMAR/ images) and the MAR series (the run
directory) are scored on the same realizations with the antler procedure of
§A1.6, using the same random training draws for both, and

  ΔAUC∞ = AUC∞(MAR) − AUC∞(noMAR).

The error of ΔAUC∞ (§15.2) applies §A1.6(e) to the per-draw differences,
which the shared draws make paired: bias = |mean Δ(resubstitution intercept)
− mean Δ(hold-out intercept)| / 2, variance term = SD of the per-draw ΔAUC∞
divided by √(number of draws), error = √(bias² + variance²).
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mar_ils_core import cho_v9, dicom_v9
from mar_ils_core import vaishnav_phantom as vp
from mar_ils_core.ddog import ddog_channels

ROI_PX = vp.ROI_PX
CENTER = vp.IMAGE_PX // 2                   # pixel (256, 256), counted from zero (§A1.4)
ANTLER_SEED = 0                             # §A1.6(f): stated in the dataset record
NULL_AUC_INF = 0.533                        # §A1.7: the estimator's value with no signal
NULL_TOLERANCE = 0.02                       # §17.6(c)


@dataclass(frozen=True)
class Label:
    cls: str            # "absent" or "present"; the second absent set of the null-check
                        # dataset is "present" with level None
    level: float | None
    split: str          # "train" or "test"
    index: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def roi(hu: np.ndarray) -> np.ndarray:
    h = ROI_PX // 2
    return hu[CENTER - h:CENTER + h + 1, CENTER - h:CENTER + h + 1]


def labels_from_key(key_path: Path, record: dict) -> tuple[str, dict[str, Label], str]:
    """(dataset, code -> label, key SHA-256). The key's hash must match the one in the record."""
    key_hash = sha256_file(key_path)
    stated = record.get("blinding_key_sha256")
    if stated and stated != key_hash:
        raise ValueError(f"blinding key SHA-256 {key_hash} does not match the dataset record ({stated})")
    key = json.loads(key_path.read_text())
    name = record["dataset"]
    if name not in key["datasets"]:
        raise ValueError(f"the key has no entries for dataset {name}")
    labels = {code: Label(e["class"], e["level"], e["split"], e["index"])
              for code, e in key["datasets"][name].items()}
    return name, labels, key_hash


def labels_from_record(dataset_dir: Path, record: dict) -> dict[str, Label]:
    """Labeled datasets: the master record lists every realization; FAMILIARIZATION also has labels.csv."""
    if "realizations" in record:
        return {r["name"]: Label("present" if r["cls"] else "absent", r["level"], r["split"], r["index"])
                for r in record["realizations"]}
    csv_path = dataset_dir / "labels.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            return {row["code"]: Label("absent" if row["class"] == "absent" else "present",
                                       float(row["level"]) if row["level"] else None,
                                       row["pool"], int(row["index"])) for row in csv.DictReader(f)}
    raise ValueError(f"{dataset_dir} is coded; give the blinding key")


def load_rois(directory: Path, names: list[str]) -> np.ndarray:
    return np.stack([roi(dicom_v9.read_hu(directory / f"{n}.dcm")).astype(np.float32) for n in names])


def export_regions(run_dir: Path) -> dict:
    """The 121 × 121 center region of every image of a run directory; needs no key (protocol §B4)."""
    names = sorted(p.stem for p in run_dir.glob("*.dcm"))
    if not names:
        raise ValueError(f"{run_dir}: no .dcm files")
    return {"names": np.array(names), "MAR": load_rois(run_dir, names)}


def _run_images(run: Path, names: list[str]) -> np.ndarray:
    """Regions of a run in the order of names, from a run directory or an exported regions .npz."""
    if run.is_file():
        d = np.load(run)
        at = {str(n): i for i, n in enumerate(d["names"])}
        return d["MAR"][[at[n] for n in names]].astype(np.float32)
    return load_rois(run, names)


def _run_names(run: Path) -> set[str]:
    if run.is_file():
        return {str(n) for n in np.load(run)["names"]}
    return {p.stem for p in run.glob("*.dcm")}


def delta_error(mar: cho_v9.AntlerResult, nomar: cho_v9.AntlerResult) -> dict:
    d_r = mar.int_resub - nomar.int_resub
    d_h = mar.int_holdout - nomar.int_holdout
    per_draw = 0.5 * (d_r + d_h)
    bias = float(abs(d_r.mean() - d_h.mean()) / 2.0)
    variance = float(per_draw.std(ddof=1) / np.sqrt(per_draw.size))
    return {"bias": bias, "variance": variance, "error": float(np.hypot(bias, variance))}


def _summary(r: cho_v9.AntlerResult) -> dict:
    return {"AUC_inf": r.auc_inf, "bias": r.bias, "variance": r.variance, "error": r.error,
            "intercept_resub": float(r.int_resub.mean()), "intercept_holdout": float(r.int_holdout.mean())}


def score_run(dataset_dir: Path, run_dir: Path, key_path: Path | None = None,
              n_reps: int = cho_v9.DEFAULT_REPS, seed: int = ANTLER_SEED) -> tuple[dict, dict]:
    """Returns (results for cho_results.json, scored regions for the .npz)."""
    record = json.loads((dataset_dir / "dataset_record.json").read_text())
    if key_path is not None:
        dataset, labels, key_hash = labels_from_key(key_path, record)
    else:
        dataset, labels, key_hash = record["dataset"], labels_from_record(dataset_dir, record), None

    expected = set(labels)
    found = _run_names(run_dir)
    if found != expected:
        missing, extra = sorted(expected - found), sorted(found - expected)
        raise ValueError(f"{run_dir}: {len(missing)} images missing (first {missing[:3]}), "
                         f"{len(extra)} unexpected (first {extra[:3]})")

    names = sorted(labels)
    nomar = load_rois(dataset_dir / "noMAR", names)
    mar = _run_images(run_dir, names)
    channels = ddog_channels(ROI_PX)
    v_nomar = cho_v9.channel_outputs(nomar.astype(np.float64), channels)
    v_mar = cho_v9.channel_outputs(mar.astype(np.float64), channels)
    pos = {n: i for i, n in enumerate(names)}

    def rows(cls, level, split):
        sel = [n for n in names if labels[n].cls == cls and labels[n].split == split
               and (cls == "absent" or labels[n].level == level)]
        return np.array([pos[n] for n in sorted(sel, key=lambda n: labels[n].index)], dtype=np.intp)

    levels = sorted({lab.level for lab in labels.values() if lab.cls == "present"},
                    key=lambda v: -1.0 if v is None else v)
    refs = record.get("reference_auc_inf_nomar", {})
    out_levels = {}
    for lev in levels:
        idx = {(c, s): rows(c, lev, s) for c in ("absent", "present") for s in ("train", "test")}

        def antler(v, idx=idx):
            return cho_v9.antler(v[idx[("absent", "train")]], v[idx[("present", "train")]],
                                 v[idx[("absent", "test")]], v[idx[("present", "test")]],
                                 n_reps=n_reps, seed=seed)

        r_n, r_m = antler(v_nomar), antler(v_mar)
        entry = {"n_per_class": {f"{c}_{s}": int(i.size) for (c, s), i in idx.items()},
                 "noMAR": _summary(r_n), "MAR": _summary(r_m),
                 "delta_AUC_inf": r_m.auc_inf - r_n.auc_inf, "delta_error": delta_error(r_m, r_n)}
        name = "null" if lev is None else f"{lev:.3f}"
        if lev is None:
            entry["null_check"] = {"expected_AUC_inf": NULL_AUC_INF, "tolerance": NULL_TOLERANCE,
                                   "noMAR_within_tolerance": abs(r_n.auc_inf - NULL_AUC_INF) <= NULL_TOLERANCE,
                                   "MAR_within_tolerance": abs(r_m.auc_inf - NULL_AUC_INF) <= NULL_TOLERANCE}
        ref = refs.get(name)
        if ref is not None:
            entry["reference_check"] = {"reference_AUC_inf_noMAR": ref, "difference": r_n.auc_inf - ref,
                                        "within_0.005": abs(r_n.auc_inf - ref) <= 0.005}
        out_levels[name] = entry

    results = {"tool": "run_cho_analysis_v9_0.py", "standard": "ASTM WK101378 draft Revision 06",
               "dataset": dataset, "run_dir": run_dir.stem if run_dir.is_file() else run_dir.name, "blinding_key_sha256": key_hash,
               "n_realizations": len(names), "roi": {"size": ROI_PX, "center_pixel": [CENTER, CENTER]},
               "antler": {"seed": seed, "n_reps": n_reps, "N_T": list(cho_v9.DEFAULT_NT)},
               "levels": out_levels}
    regions = {"names": np.array(names), "noMAR": nomar, "MAR": mar}
    return results, regions
