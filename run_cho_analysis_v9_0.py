#!/usr/bin/env python3
"""
CHO analysis v9.0: D-DOG channels, pooled covariance, antler-plot AUC∞
(Vaishnav et al. 2020). Reads an ROI .npz from generator_v9_0_0.py and reports
AUC∞ per method and ΔAUC∞ = AUC∞(method) − AUC∞(fbp).

Usage: python run_cho_analysis_v9_0.py roi.npz --out cho.json [--internal-noise-c 2.5]
       python run_cho_analysis_v9_0.py --dataset-dir REP04 --mar-output-dir C1_R1
       python run_cho_analysis_v9_0.py --dataset-dir REP04 --record-reference   (coordinator)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mar_ils_core import cho_v9, score_v9
from mar_ils_core import vaishnav_phantom as vp
from mar_ils_core.ddog import ddog_channels


def analyze(npz_path: str, internal_noise_c: float = 0.0, n_reps: int = cho_v9.DEFAULT_REPS,
            first_index: int = 1, seed: int = 0) -> dict:
    d = np.load(npz_path)
    meta = json.loads(str(d["meta"]))
    u = ddog_channels(vp.ROI_PX, first_index=first_index)
    methods = sorted({k.split("_")[0] for k in d.files if k != "meta"})
    res = {"meta": meta, "internal_noise_c": internal_noise_c, "ddog_first_index": first_index,
           "methods": {}}
    for m in methods:
        v = {k: cho_v9.channel_outputs(d[f"{m}_{k}"], u) for k in ("0_0", "1_0", "0_1", "1_1")}
        r = cho_v9.antler(v["0_0"], v["1_0"], v["0_1"], v["1_1"], n_reps=n_reps, seed=seed,
                          internal_noise_c=internal_noise_c)
        res["methods"][m] = r.as_dict()
    if "fbp" in res["methods"]:
        base = res["methods"]["fbp"]["AUC_inf"]
        res["delta_AUC_inf"] = {m: res["methods"][m]["AUC_inf"] - base for m in methods if m != "fbp"}
    return res


def score_run_dir(dataset_dir: Path, run_dir: Path, key: Path | None, out: Path | None,
                  regions_out: Path | None, reps: int) -> None:
    """Protocol §A8.2, Step 5: score one run directory against its replicate dataset."""
    results, regions = score_v9.score_run(dataset_dir, run_dir, key, n_reps=reps)
    stem = results["run_dir"]
    out = out or run_dir.parent / f"{stem}_cho_results.json"
    regions_out = regions_out or run_dir.parent / f"{stem}_scored_regions.npz"
    out.write_text(json.dumps(results, indent=1))
    np.savez_compressed(regions_out, **regions)
    print(f"{results['dataset']} / {results['run_dir']}: {results['n_realizations']} realizations")
    for lev, r in results["levels"].items():
        print(f"  level {lev}: AUC_inf noMAR {r['noMAR']['AUC_inf']:.4f} ± {r['noMAR']['error']:.4f}, "
              f"MAR {r['MAR']['AUC_inf']:.4f} ± {r['MAR']['error']:.4f}, "
              f"ΔAUC_inf {r['delta_AUC_inf']:+.4f} ± {r['delta_error']['error']:.4f}")
        nul = r.get("null_check")
        if nul:
            verdict = "within" if nul["MAR_within_tolerance"] else "OUTSIDE"
            print(f"  null-signal check: AUC_inf {r['MAR']['AUC_inf']:.4f}, {verdict} "
                  f"{nul['tolerance']} of {nul['expected_AUC_inf']} (draft standard 17.6(c))")
        chk = r.get("reference_check")
        if chk and not chk["within_0.005"]:
            print(f"  WARNING: AUC_inf noMAR differs from the reference {chk['reference_AUC_inf_noMAR']:.4f} "
                  "by more than 0.005; contact the coordinator (protocol Step 4.2)")
    print(f"Results: {out}\nScored regions: {regions_out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="v9 CHO: score an ROI .npz from the generator, or a DICOM run directory (ILS Step 5)")
    ap.add_argument("npz", nargs="?", help="ROI .npz from generator_v9_0_0.py")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--internal-noise-c", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=cho_v9.DEFAULT_REPS)
    ap.add_argument("--ddog-first-index", type=int, default=1)
    run = ap.add_argument_group("run directory scoring (ILS protocol §A8.2, Step 5)")
    run.add_argument("--dataset-dir", type=Path, help="the replicate dataset the run was produced from")
    run.add_argument("--mar-output-dir", type=Path, help="the run directory, e.g. C1_R1")
    run.add_argument("--blinding-key", type=Path, help="released key; not needed for labeled datasets")
    run.add_argument("--regions-out", type=Path, help="default <run_dir>_scored_regions.npz")
    run.add_argument("--record-reference", action="store_true",
                     help="coordinator only: write the reference AUC_inf of the noMAR series of "
                          "--dataset-dir to its dataset record, and exit")
    run.add_argument("--export-regions", action="store_true",
                     help="only write <run_dir>_regions.npz, the 121 x 121 regions of --mar-output-dir; "
                          "needs no dataset or key (pilot, protocol B4)")
    a = ap.parse_args()

    if a.export_regions:
        if not a.mar_output_dir or a.npz:
            ap.error("--export-regions needs --mar-output-dir")
        target = a.regions_out or a.mar_output_dir.parent / f"{a.mar_output_dir.name}_regions.npz"
        try:
            np.savez_compressed(target, **score_v9.export_regions(a.mar_output_dir))
        except ValueError as e:
            raise SystemExit(str(e)) from e
        print(f"Regions: {target}")
        return
    if a.record_reference:
        if not a.dataset_dir or a.mar_output_dir or a.npz:
            ap.error("--record-reference needs --dataset-dir only")
        try:
            refs = score_v9.record_reference(a.dataset_dir, a.reps)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        if not refs:
            raise SystemExit(f"{a.dataset_dir}: too few training realizations per class; nothing recorded")
        for lev, v in refs.items():
            print(f"  level {lev}: reference AUC_inf noMAR {v:.4f}")
        print(f"Recorded in {a.dataset_dir / 'dataset_record.json'}")
        return
    if a.mar_output_dir or a.dataset_dir:
        if not (a.mar_output_dir and a.dataset_dir) or a.npz:
            ap.error("run directory scoring needs --dataset-dir and --mar-output-dir, and no npz")
        try:
            score_run_dir(a.dataset_dir, a.mar_output_dir, a.blinding_key, a.out, a.regions_out, a.reps)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        return
    if not a.npz or not a.out:
        ap.error("give an ROI .npz and --out, or --dataset-dir and --mar-output-dir")
    res = analyze(a.npz, a.internal_noise_c, a.reps, a.ddog_first_index)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for m, r in res["methods"].items():
        print(f"{m:5s} AUC_inf {r['AUC_inf']:.4f} ± {r['error']:.4f}  "
              f"(resub {r['intercept_resub']:.4f}, holdout {r['intercept_holdout']:.4f})")
    for m, dv in res.get("delta_AUC_inf", {}).items():
        print(f"ΔAUC_inf {m}: {dv:+.4f}")


if __name__ == "__main__":
    main()
