#!/usr/bin/env python3
"""
run_cho_analysis_v8_0.py
========================
ASTM WKXXXXX Rev 06 — Reference 2D CHO Analysis (Fan-Beam Dataset)

Changes from run_cho_analysis_v7_0.py
--------------------------------------
[BR]  Bootstrap with refit: each bootstrap draw resamples realization indices
      and repeats the leave-one-out fit on the resampled set, holding out every
      copy of the left-out index. v7.0 resampled fixed hold-out scores, which
      leaves template-training variability out of the CI (bootstrap SE was
      1.2–1.5× below the replicate SD of ΔAUC on v8 data, review item 13(b)).
      The same draws give the per-condition and the paired ΔAUC intervals.
      Intervals are normal, estimate ± 1.96 × bootstrap SE. Percentile
      intervals are not used: a resample holds about 63 % distinct
      realizations, so refit AUCs are biased low and percentile intervals
      can exclude the point estimate. The refit SE matched the replicate SD
      of ΔAUC (0.069 vs 0.070 at 20 HU; 0.070 vs 0.084 at 30 HU).
[IN]  σ_internal = 15 kept at its Rev 05 value and described as what it is:
      an absolute covariance regularization term, K + σ²I, applied to the
      template only (no noise is added to the decision variable). It was
      fixed before the v8 dose sweep and was not tuned to the gate; at the
      10 HU operating point it matters (LI-MAR ΔAUC +0.010 at σ = 0,
      +0.045 at σ = 15; review item 14).
[NW]  Index-paired Wilcoxon tests removed. LP_i and LA_i have independent noise
      in the v8.0.0 dataset, so pairing them by index has no meaning.

Unchanged from v7.0: channels, ROI, LA-only covariance, λ, σ = 15, LOO
point estimate, Mann-Whitney AUC, 50/50 bias and noise sweep. The AUC and
ΔAUC point estimates equal those of run_cho_analysis_v7_0.py exactly; only
the intervals differ. The bootstrap SE is reported but is not calibrated
across operating points (review item 14); the task group's replicate SD is
the reference sampling uncertainty.

Usage
-----
    python run_cho_analysis_v8_0.py \\
        --dataset-dir ./astm_reference_dataset_v8 \\
        --mar-output-dir ./mar_recon \\
        --internal-noise-sigma 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from mar_ils_core.constants import (
    AUC_TOLERANCE,
    CHANNEL_WIDTH_A,
    LESION_SLICE_INDEX,
    N_BOOT,
    NUM_CHANNELS,
    NUM_REALIZATIONS_DEFAULT,
    ROI_CENTER_X,
    ROI_CENTER_Y,
    ROI_SIZE,
)
from run_cho_analysis_v7_0 import (
    _fit_and_score,
    calculate_lg_channels,
    compute_50_50_auc,
    compute_noise_sweep,
    load_all_rois_parallel,
    mw_auc,
)

SCRIPT_VERSION = "run_cho_analysis_v8_0"
STANDARD_REF = "ASTM-WKXXXXX-Rev06"
BOOT_SEED = 99999


def loo_auc(
    lp_features: np.ndarray,
    la_features: np.ndarray,
    internal_noise_var: float,
    idx: np.ndarray | None = None,
) -> float:
    """Leave-one-out hold-out AUC on the (optionally resampled) set ``idx``.

    Each distinct index j is held out together with all of its copies: the
    template is trained on the remaining entries of ``idx`` and scores LP_j
    and LA_j once, and each copy of j carries that score. With ``idx`` =
    0..N-1 this is the v7.0 LOO estimate.
    """
    N = len(lp_features)
    if idx is None:
        idx = np.arange(N)
    s_lp = np.empty(len(idx))
    s_la = np.empty(len(idx))
    for j in np.unique(idx):
        train = idx[idx != j]
        sl, sa = _fit_and_score(
            lp_features[train], la_features[train],
            lp_features[j:j + 1], la_features[j:j + 1],
            internal_noise_var=internal_noise_var,
        )
        s_lp[idx == j] = sl[0]
        s_la[idx == j] = sa[0]
    return mw_auc(s_lp, s_la)


def resub_auc(lp_features, la_features, internal_noise_var: float) -> float:
    s_lp, s_la = _fit_and_score(lp_features, la_features, lp_features,
                                la_features, internal_noise_var=internal_noise_var)
    return mw_auc(s_lp, s_la)


def refit_bootstrap(
    feats: dict[str, tuple[np.ndarray, np.ndarray]],
    internal_noise_var: float,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, np.ndarray]:
    """Bootstrap AUC distributions with refit, shared draws across conditions.

    ``feats`` maps condition -> (lp_features, la_features), all indexed by the
    same realization numbers, so ΔAUC = AUC_MAR − AUC_noMAR is paired per draw.
    """
    N = len(next(iter(feats.values()))[0])
    rng = np.random.default_rng(seed)
    out = {c: np.empty(n_boot) for c in feats}
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        for c, (lp, la) in feats.items():
            out[c][b] = loo_auc(lp, la, internal_noise_var, idx)
    return out


def _ci(estimate: float, boot: np.ndarray) -> list[float]:
    """Normal 95 % interval: estimate ± 1.96 × bootstrap SE."""
    se = float(np.std(boot, ddof=1))
    return [estimate - 1.96 * se, estimate + 1.96 * se]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASTM WKXXXXX Rev 06 — Reference 2D CHO Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-dir", required=True,
                        help="Root directory of the v8.0.0 dataset")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mar-output-dir",
                      help="(ILS mode) Lab MAR reconstructions directory")
    mode.add_argument("--self-test", action="store_true",
                      help="(Validation) Score noMAR_recon vs itself (ΔAUC=0)")
    parser.add_argument("--reconstruction-pipeline", default="", metavar="DESC",
                        help="Free-text pipeline description (ILS mode)")
    parser.add_argument("--internal-noise-sigma", type=float, default=15.0,
                        help="Covariance regularization σ: σ² added to the "
                             "covariance diagonal (normative value 15)")
    parser.add_argument("--realizations", type=int,
                        default=NUM_REALIZATIONS_DEFAULT,
                        help=f"Realizations per condition (default: "
                             f"{NUM_REALIZATIONS_DEFAULT}; screening: 20)")
    parser.add_argument("--noise-sigma-sweep", action="store_true",
                        help="Compute noise sensitivity sweep (informative)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"Bootstrap resamples (default: {N_BOOT})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Workers for DICOM loading (0=cpu_count)")
    parser.add_argument("--results-file", default="cho_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    num_real = args.realizations
    int_var = args.internal_noise_sigma ** 2
    print("ASTM WKXXXXX Rev 06 — 2D CHO Analysis")
    print(f"ROI: {ROI_SIZE}×{ROI_SIZE}  centre=({ROI_CENTER_X},{ROI_CENTER_Y})")
    print(f"LG channel width a = {CHANNEL_WIDTH_A:.1f} vox")
    print(f"Covariance regularization σ = {args.internal_noise_sigma}")
    print(f"Realizations: {num_real} per condition "
          f"({'SCREENING' if num_real < 40 else 'FORMAL'})")
    print(f"Bootstrap: {args.n_boot} draws with LOO refit")
    print()
    if num_real < 40:
        print("WARNING: Screening mode (< 40 realizations). "
              "Results are not reportable under §10.2.\n")

    base_dir = Path(args.dataset_dir)
    noMAR_dir = base_dir / "noMAR_recon"
    if not noMAR_dir.exists():
        print(f"noMAR_recon/ not found in {base_dir}", file=sys.stderr)
        sys.exit(1)
    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    U = calculate_lg_channels()

    if args.self_test:
        mar_root = noMAR_dir
        ils_mode = False
        recon_pipeline = "N/A (self-test)"
        print("  *** SELF-TEST MODE: ΔAUC = 0 by definition ***\n")
    else:
        mar_root = Path(args.mar_output_dir)
        if not mar_root.exists():
            print(f"--mar-output-dir not found: {mar_root}", file=sys.stderr)
            sys.exit(1)
        ils_mode = True
        recon_pipeline = args.reconstruction_pipeline or "not declared"

    feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond, root in (("noMAR", noMAR_dir), ("MAR", mar_root)):
        print(f"Loading {cond}: {root}")
        lp_rois, la_rois = load_all_rois_parallel(
            root / "LP", root / "LA", num_real, n_workers)
        feats[cond] = ((U @ lp_rois.T).T, (U @ la_rois.T).T)

    print("Bootstrap with refit...", flush=True)
    boots = refit_bootstrap(feats, int_var, n_boot=args.n_boot)

    results: dict[str, dict] = {}
    for cond, (lp, la) in feats.items():
        auc = loo_auc(lp, la, int_var)
        stats = {
            "AUC": float(auc),
            "Bias": float(resub_auc(lp, la, int_var) - auc),
            "CI_95": _ci(float(auc), boots[cond]),
            "boot_SE": float(np.std(boots[cond], ddof=1)),
            "bias_5050": compute_50_50_auc(lp, la, num_real,
                                           internal_noise_sigma=args.internal_noise_sigma),
        }
        if args.noise_sigma_sweep or cond == "noMAR":
            stats["noise_sweep"] = compute_noise_sweep(lp, la, num_real)
        results[cond] = stats

    delta = results["MAR"]["AUC"] - results["noMAR"]["AUC"]
    d_boot = boots["MAR"] - boots["noMAR"]
    delta_ci = _ci(float(delta), d_boot)

    print()
    print("=" * 66)
    print("ASTM WKXXXXX Rev 06 — CHO Result")
    print("=" * 66)
    for cond in ("noMAR", "MAR"):
        r = results[cond]
        print(f"  AUC_{cond:6s}: {r['AUC']:.4f}  (Bias {r['Bias']:+.4f}; "
              f"CI [{r['CI_95'][0]:.4f}, {r['CI_95'][1]:.4f}])")
    print(f"  ΔAUC       : {delta:+.4f}  "
          f"(95% CI [{delta_ci[0]:+.4f}, {delta_ci[1]:+.4f}]; "
          f"SE {np.std(d_boot, ddof=1):.4f})")
    if num_real < 40:
        print("  *** SCREENING MODE — not reportable under §10.2 ***")
    if not ils_mode:
        print("  *** SELF-TEST: ΔAUC = 0 by definition ***")
    else:
        print(f"  Pipeline: {recon_pipeline}")
    print("=" * 66)

    out = {
        "script_version": SCRIPT_VERSION,
        "standard_reference": STANDARD_REF,
        "acquisition_geometry": "fan-beam (SID=570mm, SDD=1040mm)",
        "observer": "2D CHO, single slice (§A1.5.3)",
        "roi_size": ROI_SIZE,
        "roi_center": [ROI_CENTER_X, ROI_CENTER_Y],
        "channel_width_a": CHANNEL_WIDTH_A,
        "num_channels": NUM_CHANNELS,
        "lesion_slice_index": LESION_SLICE_INDEX,
        "internal_noise_sigma": args.internal_noise_sigma,
        "num_realizations": num_real,
        "n_boot": args.n_boot,
        "bootstrap": "realization-index resampling with LOO refit",
        "auc_tolerance": AUC_TOLERANCE,
        "ils_mode": ils_mode,
        "reconstruction_pipeline": recon_pipeline,
        "noMAR": results["noMAR"],
        "MAR": results["MAR"],
        "delta_AUC": float(delta),
        "delta_AUC_CI_95": delta_ci,
        "delta_AUC_boot_SE": float(np.std(d_boot, ddof=1)),
        "self_test": not ils_mode,
        "screening_mode": num_real < 40,
    }
    with open(args.results_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written → {args.results_file}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
