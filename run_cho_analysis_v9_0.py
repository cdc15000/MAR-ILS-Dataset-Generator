#!/usr/bin/env python3
"""
CHO analysis v9.0: D-DOG channels, pooled covariance, antler-plot AUC∞
(Vaishnav et al. 2020). Reads an ROI .npz from generator_v9_0_0.py and reports
AUC∞ per method and ΔAUC∞ = AUC∞(method) − AUC∞(fbp).

Usage: python run_cho_analysis_v9_0.py roi.npz --out cho.json [--internal-noise-c 2.5]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from mar_ils_core import cho_v9
from mar_ils_core import vaishnav_phantom as vp
from mar_ils_core.ddog import ddog_channels


def analyse(npz_path: str, internal_noise_c: float = 0.0, n_reps: int = cho_v9.DEFAULT_REPS,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--internal-noise-c", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=cho_v9.DEFAULT_REPS)
    ap.add_argument("--ddog-first-index", type=int, default=1)
    a = ap.parse_args()
    res = analyse(a.npz, a.internal_noise_c, a.reps, a.ddog_first_index)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for m, r in res["methods"].items():
        print(f"{m:5s} AUC_inf {r['AUC_inf']:.4f} ± {r['error']:.4f}  "
              f"(resub {r['intercept_resub']:.4f}, holdout {r['intercept_holdout']:.4f})")
    for m, dv in res.get("delta_AUC_inf", {}).items():
        print(f"ΔAUC_inf {m}: {dv:+.4f}")


if __name__ == "__main__":
    main()
