#!/usr/bin/env python3
"""
run_cho_analysis_v6_0.py
========================
ASTM WKXXXXX v1.0.0 — Enhanced 2D CHO with Tiered Framework

Changes from run_cho_analysis_v5_3.py
--------------------------------------
[T1]  Tier-aware ROI, CHO channels, and ROI centre via TierConfig.
      ROI size and channel width a are read from TIER_REGISTRY[args.tier].

[T2]  50/50 estimation bias (compute_50_50_auc): 2-fold CV provides an
      estimate of the optimistic bias when training on all N realizations.

[T3]  One-tailed paired Wilcoxon signed-rank test (compute_wilcoxon_test):
      H₁: median(d[i]) > 0 where d[i] = s_LP[i] − s_LA[i].
      Requires scipy >= 1.7 (alternative='greater').

[T4]  Noise sensitivity sweep (compute_noise_sweep): varies
      internal_noise_sigma at analysis time across DEFAULT_SIGMA_SWEEP
      (11 points from σ=0 to σ=80).

[T5]  Sigmoid AUC fit (fit_auc_sigmoid): fits
      AUC(σ) = A / (1 + exp(k(σ − σ₀))) + 0.5 via scipy curve_fit.

[T6]  Extended JSON output: includes 50/50 bias, Wilcoxon p-value, noise
      sweep, and sigmoid fit parameters alongside the standard v5.3.0 fields.

All v5.3.0 CHO mathematics (Laguerre-Gauss channels, _fit_and_score, LOO,
bootstrap) are unchanged.  The only CHO parameter change is that ROI_SIZE
and CHANNEL_WIDTH_A are now tier-specific (from TierConfig.roi_size and
TierConfig.channel_width_a).

Usage
-----
    # ILS mode:
    python run_cho_analysis_v6_0.py \\
        --dataset-dir ./astm_mar_ils_t1_ab \\
        --mar-output-dir ./mar_recon \\
        --tier T1_AB \\
        --internal-noise-sigma 15

    # Self-test (ΔAUC = 0 by definition):
    python run_cho_analysis_v6_0.py \\
        --dataset-dir ./astm_mar_ils_t1_ab \\
        --self-test --tier T1_AB

    # Noise sweep only (no MAR, just noMAR sensitivity curve):
    python run_cho_analysis_v6_0.py \\
        --dataset-dir ./astm_mar_ils_t1_ab \\
        --self-test --tier T1_AB --noise-sigma-sweep

Normative Parameters (§A1.5, v1.0.0)
--------------------------------------
  Observer      : 2D CHO — slice LESION_SLICE_INDEX = 128 ONLY
  ROI           : tier.roi_size × tier.roi_size centred at lesion_center_x
  Channels      : 10 Laguerre-Gauss, a = tier.channel_width_a voxels
  Covariance    : Estimated from LA slice 128 only (§A1.5.2(a))
  Regularisation: Tikhonov λ = 0.01 × trace(K) / 10
  Metric        : Mann-Whitney AUC (LOO hold-out)
  Internal noise: σ = 15 (normative default, §A1.5.2, Vaishnav 2020)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
from scipy.special import genlaguerre
from scipy.stats import wilcoxon
from scipy.optimize import curve_fit

from tier_config import (
    TIER_REGISTRY, TierConfig, validate_tier_registry,
    LESION_SLICE_INDEX, NUM_REALIZATIONS,
)

# ─── Constants ────────────────────────────────────────────────────────────────
NUM_CHANNELS = 10    # Laguerre-Gauss channels (normative — do not modify)
N_BOOT       = 1000  # bootstrap resamples

# Internal noise sigma sweep: 11 points for the Vaishnav Sensitivity Matrix
DEFAULT_SIGMA_SWEEP = [0, 5, 10, 15, 20, 25, 30, 40, 50, 65, 80]


# ─── Channel templates (tier-aware) ───────────────────────────────────────────

def calculate_lg_channels(tier: TierConfig) -> np.ndarray:
    """
    Generate 10 2D Laguerre-Gauss channel templates for the given tier.

    u_n(x,y) = L_n(2π r²/a²) · exp(−π r²/a²), L2-normalised over the ROI.

    Parameters
    ----------
    tier : TierConfig
        Determines roi_size (ROI side length) and channel_width_a.

    Returns
    -------
    U : ndarray, shape (NUM_CHANNELS, roi_size²)
    """
    roi = tier.roi_size
    a   = tier.channel_width_a
    half = roi // 2

    print(f"  LG channels: ROI={roi}×{roi}, a={a:.1f} vox  ... ", end="", flush=True)

    x    = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)
    r2   = xx**2 + yy**2
    arg  = 2.0 * np.pi * r2 / a**2
    env  = np.exp(-np.pi * r2 / a**2)

    rows = []
    for n in range(NUM_CHANNELS):
        u2d  = genlaguerre(n, 0)(arg) * env
        norm = np.linalg.norm(u2d)
        rows.append((u2d / norm).flatten())

    print("done.")
    return np.vstack(rows)   # (NUM_CHANNELS, roi_size²)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_slice_roi(folder_path: Path, tier: TierConfig) -> np.ndarray:
    """
    Load the tier ROI from LESION_SLICE_INDEX of a DICOM realization folder.

    Reads only slice_{LESION_SLICE_INDEX+1:04d}.dcm.  3D integration is
    prohibited (§A1.5.3).

    Returns
    -------
    roi : ndarray, shape (roi_size²,), float64
    """
    fname = folder_path / f"slice_{LESION_SLICE_INDEX + 1:04d}.dcm"
    if not fname.exists():
        raise FileNotFoundError(
            f"Expected slice not found: {fname}\n"
            f"  Ensure the dataset was generated with generator_v6_0_0.py and "
            f"filenames are 1-indexed (slice_0001.dcm … slice_0256.dcm)."
        )
    dcm = pydicom.dcmread(str(fname))
    img = dcm.pixel_array.astype(np.float64)
    img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

    ys, ye = tier.roi_y_bounds()
    xs, xe = tier.roi_x_bounds()
    roi = img[ys:ye, xs:xe]   # (roi_size, roi_size)
    return roi.flatten()


def _worker_load(args) -> np.ndarray:
    folder_str, tier_id = args
    return load_slice_roi(Path(folder_str), TIER_REGISTRY[tier_id])


def load_all_rois_parallel(
    lp_root: Path,
    la_root: Path,
    tier: TierConfig,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load tier ROIs from all realizations. Returns (lp_rois, la_rois)."""
    lp_folders = [str(lp_root / f"realization_{i+1:03d}") for i in range(NUM_REALIZATIONS)]
    la_folders = [str(la_root / f"realization_{i+1:03d}") for i in range(NUM_REALIZATIONS)]
    all_args = [(f, tier.tier_id) for f in (lp_folders + la_folders)]

    rois = {}
    if n_workers == 1:
        for i, arg in enumerate(all_args):
            print(f"  Loading {i+1}/{len(all_args)}...", end="\r")
            rois[arg[0]] = _worker_load(arg)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futs = {exe.submit(_worker_load, a): a for a in all_args}
            done = 0
            for fut in as_completed(futs):
                arg = futs[fut]
                rois[arg[0]] = fut.result()
                done += 1
                print(f"  Loaded {done}/{len(all_args)}...", end="\r", flush=True)

    print(f"  Loaded {len(all_args)} realizations.              ")
    lp_rois = np.vstack([rois[f] for f in lp_folders])
    la_rois = np.vstack([rois[f] for f in la_folders])
    return lp_rois, la_rois


# ─── Core CHO statistics (unchanged from v5.3.0) ─────────────────────────────

def mw_auc(lp: np.ndarray, la: np.ndarray) -> float:
    """Exact Mann-Whitney AUC with mid-rank tie correction."""
    n_lp, n_la = len(lp), len(la)
    count = 0.0
    for p in lp:
        count += np.sum(p > la) + 0.5 * np.sum(p == la)
    return count / (n_lp * n_la)


def _fit_and_score(
    train_lp: np.ndarray,
    train_la: np.ndarray,
    test_lp:  np.ndarray,
    test_la:  np.ndarray,
    internal_noise_var: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Hotelling template on training features; score test features."""
    K_la  = np.cov(train_la, rowvar=False)
    lam   = 0.01 * np.trace(K_la) / NUM_CHANNELS
    K_reg = K_la + (lam + internal_noise_var) * np.eye(NUM_CHANNELS)
    w     = np.linalg.solve(K_reg, np.mean(train_lp, axis=0) - np.mean(train_la, axis=0))
    return test_lp @ w, test_la @ w


def compute_cho_performance(
    lp_features: np.ndarray,
    la_features: np.ndarray,
    n_boot: int = N_BOOT,
    boot_seed: int = 12345,
    internal_noise_sigma: float = 0.0,
) -> dict:
    """
    Resubstitution AUC, LOO hold-out AUC, bias, and 1000-resample bootstrap CI.

    Returns dict with keys: AUC, Bias, CI_95, _s_lp_ho, _s_la_ho
    """
    N = NUM_REALIZATIONS
    int_var = internal_noise_sigma ** 2

    # Resubstitution
    s_lp_rs, s_la_rs = _fit_and_score(
        lp_features, la_features, lp_features, la_features,
        internal_noise_var=int_var,
    )
    auc_resub = mw_auc(s_lp_rs, s_la_rs)

    # LOO hold-out
    s_lp_ho = np.zeros(N, dtype=np.float64)
    s_la_ho = np.zeros(N, dtype=np.float64)
    for i in range(N):
        tr_lp = np.delete(lp_features, i, axis=0)
        tr_la = np.delete(la_features, i, axis=0)
        te_lp = lp_features[i:i+1]
        te_la = la_features[i:i+1]
        s_lp_ho[i], s_la_ho[i] = (
            v[0] for v in _fit_and_score(tr_lp, tr_la, te_lp, te_la,
                                         internal_noise_var=int_var)
        )
    auc_ho = mw_auc(s_lp_ho, s_la_ho)
    bias   = auc_resub - auc_ho

    # Bootstrap 95% CI
    rng = np.random.default_rng(boot_seed)
    boot_aucs = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot_aucs[b] = mw_auc(s_lp_ho[idx], s_la_ho[idx])
    ci_lo = float(np.percentile(boot_aucs, 2.5))
    ci_hi = float(np.percentile(boot_aucs, 97.5))

    return {
        "AUC":      float(auc_ho),
        "Bias":     float(bias),
        "CI_95":    (ci_lo, ci_hi),
        "_s_lp_ho": s_lp_ho,
        "_s_la_ho": s_la_ho,
    }


# ─── Enhanced statistics (v6.0.0 additions) ───────────────────────────────────

def compute_50_50_auc(
    lp_features: np.ndarray,
    la_features: np.ndarray,
    internal_noise_sigma: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    2-fold cross-validation estimation bias (50/50 split).

    Partitions N realizations into two equal halves.  Each fold trains on
    the opposite half and tests on its own half.  Combined test scores give
    AUC_5050.  Bias_5050 = AUC_resub − AUC_5050 quantifies optimistic bias.

    Returns
    -------
    {"AUC_resub", "AUC_5050", "Bias_5050"}
    """
    N = NUM_REALIZATIONS
    int_var = internal_noise_sigma ** 2
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    mid = N // 2

    # Fold A: train on second half, test on first half
    tr_lp_a, te_lp_a = lp_features[idx[mid:]], lp_features[idx[:mid]]
    tr_la_a, te_la_a = la_features[idx[mid:]], la_features[idx[:mid]]
    s_lp_a, s_la_a = _fit_and_score(tr_lp_a, tr_la_a, te_lp_a, te_la_a, int_var)

    # Fold B: train on first half, test on second half
    tr_lp_b, te_lp_b = lp_features[idx[:mid]], lp_features[idx[mid:]]
    tr_la_b, te_la_b = la_features[idx[:mid]], la_features[idx[mid:]]
    s_lp_b, s_la_b = _fit_and_score(tr_lp_b, tr_la_b, te_lp_b, te_la_b, int_var)

    # Combined out-of-fold scores
    auc_5050 = mw_auc(
        np.concatenate([s_lp_a, s_lp_b]),
        np.concatenate([s_la_a, s_la_b]),
    )

    # Resubstitution
    s_lp_rs, s_la_rs = _fit_and_score(
        lp_features, la_features, lp_features, la_features, int_var
    )
    auc_resub = mw_auc(s_lp_rs, s_la_rs)

    return {
        "AUC_resub":  float(auc_resub),
        "AUC_5050":   float(auc_5050),
        "Bias_5050":  float(auc_resub - auc_5050),
    }


def compute_wilcoxon_test(s_lp_ho: np.ndarray, s_la_ho: np.ndarray) -> dict:
    """
    One-tailed paired Wilcoxon signed-rank test on d[i] = s_LP[i] − s_LA[i].

    H₀: median(d) = 0  vs  H₁: median(d) > 0.
    A p-value < 0.05 indicates statistically significant lesion detectability
    advantage for LP over LA — i.e., the MAR algorithm preserves the signal.

    Returns
    -------
    {"wilcoxon_stat", "p_one_sided", "significant_p05"}
    """
    d = s_lp_ho - s_la_ho
    try:
        stat, p = wilcoxon(d, alternative='greater', zero_method='wilcox')
        p_one_sided = float(p)
    except TypeError:
        # scipy < 1.7: alternative keyword not available; compute manually
        stat, p_two = wilcoxon(d, zero_method='wilcox')
        # Two-sided p → one-sided p for H1: median > 0
        # Valid when W+ ≥ N(N+1)/4 (i.e., positive direction)
        p_one_sided = float(p_two) / 2.0 if float(np.sum(d > 0)) >= len(d) / 2 else 1.0 - float(p_two) / 2.0
    return {
        "wilcoxon_stat":     float(stat),
        "p_one_sided":       p_one_sided,
        "significant_p05":   bool(p_one_sided < 0.05),
    }


def fit_auc_sigmoid(
    sigmas: list[float],
    aucs: list[float],
) -> dict:
    """
    Fit AUC(σ) = A / (1 + exp(k(σ − σ₀))) + 0.5 via least-squares.

    The sigmoid captures the transition from near-perfect detection (σ → 0)
    to chance-level detection (σ → ∞).  σ₀ is the inflection point (50%
    of max detectability above chance); k is the steepness.

    Returns
    -------
    {"A", "k", "sigma_0", "fit_ok", "r_squared"}
    """
    sigmas_arr = np.asarray(sigmas, dtype=float)
    aucs_arr   = np.asarray(aucs,   dtype=float)

    def _sigmoid(s, A, k, s0):
        return A / (1.0 + np.exp(k * (s - s0))) + 0.5

    A0  = float(np.max(aucs_arr) - 0.5)
    s0_0 = float(sigmas_arr[np.argmin(np.abs(aucs_arr - (0.5 + A0 / 2)))])

    try:
        popt, _ = curve_fit(
            _sigmoid, sigmas_arr, aucs_arr,
            p0=[max(A0, 0.01), -0.05, s0_0],
            bounds=([0, -5, 0], [0.5, -1e-6, 200]),
            maxfev=5000,
        )
        A, k, s0 = popt
        ss_res = float(np.sum((aucs_arr - _sigmoid(sigmas_arr, *popt))**2))
        ss_tot = float(np.sum((aucs_arr - np.mean(aucs_arr))**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        return {"A": float(A), "k": float(k), "sigma_0": float(s0),
                "fit_ok": True, "r_squared": float(r2)}
    except Exception as exc:
        return {"A": None, "k": None, "sigma_0": None,
                "fit_ok": False, "r_squared": None, "error": str(exc)}


def compute_noise_sweep(
    lp_features: np.ndarray,
    la_features: np.ndarray,
    sigma_range: list[float] | None = None,
    n_boot_sweep: int = 200,
) -> dict:
    """
    Vaishnav Sensitivity Matrix: vary internal_noise_sigma, compute AUC at each.

    Parameters
    ----------
    sigma_range   : internal noise σ values (default: DEFAULT_SIGMA_SWEEP)
    n_boot_sweep  : bootstrap resamples per sigma (200 is sufficient for sweep)

    Returns
    -------
    {"sweep_points": [...], "sigmoid_fit": {...}}
    """
    if sigma_range is None:
        sigma_range = DEFAULT_SIGMA_SWEEP

    print(f"  Noise sweep ({len(sigma_range)} σ values)...", flush=True)
    sweep_points = []
    for sigma in sigma_range:
        stats = compute_cho_performance(
            lp_features, la_features,
            n_boot=n_boot_sweep,
            internal_noise_sigma=float(sigma),
        )
        sweep_points.append({"sigma": sigma, "AUC": stats["AUC"],
                              "CI_95": list(stats["CI_95"])})
        print(f"    σ={sigma:4.0f}  AUC={stats['AUC']:.4f}", flush=True)

    sigmas = [p["sigma"] for p in sweep_points]
    aucs   = [p["AUC"]   for p in sweep_points]
    return {
        "sweep_points": sweep_points,
        "sigmoid_fit":  fit_auc_sigmoid(sigmas, aucs),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASTM WKXXXXX v1.0.0 — Enhanced 2D CHO Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-dir", required=True,
                        help="Root directory of the v6.0.0 dataset")
    parser.add_argument("--tier", default="T1_AB", choices=list(TIER_REGISTRY),
                        help="Imaging scenario tier (default: T1_AB)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mar-output-dir",
                      help="(ILS mode) Directory with lab MAR reconstructions "
                           "(LP/ and LA/ subdirs, realization_NNN/ folders)")
    mode.add_argument("--self-test", action="store_true",
                      help="(Validation mode) Score noMAR_recon vs itself.  "
                           "ΔAUC = 0 by definition.")

    parser.add_argument("--reconstruction-pipeline", default="", metavar="DESC",
                        help="Free-text pipeline description (ILS mode)")
    parser.add_argument("--internal-noise-sigma", type=float, default=15.0,
                        help="Vaishnav internal observer noise σ (default: 15.0)")
    parser.add_argument("--noise-sigma-sweep", action="store_true",
                        help="Compute Vaishnav Sensitivity Matrix (noise sweep)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"Bootstrap resamples (default: {N_BOOT})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Worker processes for DICOM loading (0=cpu_count, default: 1)")
    parser.add_argument("--results-file", default="cho_results_v6.json",
                        help="Output JSON path (default: cho_results_v6.json)")
    args = parser.parse_args()

    validate_tier_registry()
    tier = TIER_REGISTRY[args.tier]
    print()
    print(f"Tier: {tier.tier_id}  —  {tier.description}")
    print(f"ROI: {tier.roi_size}×{tier.roi_size}  centre=({tier.roi_x_center},{tier.roi_y_center})")
    print(f"LG channel width a = {tier.channel_width_a:.1f} vox")
    print(f"Internal noise σ = {args.internal_noise_sigma}")
    print()

    base_dir = Path(args.dataset_dir)
    if not base_dir.exists():
        print(f"Dataset directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    noMAR_dir = base_dir / "noMAR_recon"
    if not noMAR_dir.exists():
        print(f"noMAR_recon/ not found in {base_dir}", file=sys.stderr)
        sys.exit(1)

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    # Channel templates (tier-aware)
    U = calculate_lg_channels(tier)   # (NUM_CHANNELS, roi_size²)

    # Determine MAR source
    if args.self_test:
        mar_lp_root = noMAR_dir / "LP"
        mar_la_root = noMAR_dir / "LA"
        mar_label   = "noMAR_recon [self-test — ΔAUC=0 by definition]"
        ils_mode    = False
        recon_pipeline = "N/A (self-test)"
        print("  *** SELF-TEST MODE: ΔAUC = 0 by definition ***\n")
    else:
        mar_root = Path(args.mar_output_dir)
        if not mar_root.exists():
            print(f"--mar-output-dir not found: {mar_root}", file=sys.stderr)
            sys.exit(1)
        mar_lp_root = mar_root / "LP"
        mar_la_root = mar_root / "LA"
        mar_label   = f"MAR (submission: {mar_root})"
        ils_mode    = True
        recon_pipeline = args.reconstruction_pipeline or "not declared"

    results = {}
    for cond, lp_root, la_root, display in [
        ("noMAR", noMAR_dir / "LP", noMAR_dir / "LA",
         "noMAR_recon (reference FBP, no correction)"),
        ("MAR",   mar_lp_root,       mar_la_root,       mar_label),
    ]:
        print(f"\nProcessing: {display}")
        lp_rois, la_rois = load_all_rois_parallel(lp_root, la_root, tier, n_workers)

        # Project through channel templates
        lp_feat = (U @ lp_rois.T).T   # (N, NUM_CHANNELS)
        la_feat = (U @ la_rois.T).T

        stats = compute_cho_performance(
            lp_feat, la_feat,
            n_boot=args.n_boot,
            internal_noise_sigma=args.internal_noise_sigma,
        )

        # 50/50 bias
        bias_5050 = compute_50_50_auc(lp_feat, la_feat,
                                       internal_noise_sigma=args.internal_noise_sigma)
        stats["bias_5050"] = bias_5050

        # Wilcoxon test on hold-out scores
        stats["wilcoxon"] = compute_wilcoxon_test(stats["_s_lp_ho"], stats["_s_la_ho"])

        # Noise sweep (optional, but always run for noMAR to build the sensitivity matrix)
        if args.noise_sigma_sweep or cond == "noMAR":
            stats["noise_sweep"] = compute_noise_sweep(lp_feat, la_feat)

        results[cond] = stats
        print(f"  AUC (LOO hold-out) : {stats['AUC']:.4f}")
        print(f"  95% CI             : [{stats['CI_95'][0]:.4f}, {stats['CI_95'][1]:.4f}]")
        print(f"  Bias (resub−LOO)   : {stats['Bias']:+.4f}")
        print(f"  Bias_5050          : {bias_5050['Bias_5050']:+.4f}")
        print(f"  Wilcoxon p (1-tail): {stats['wilcoxon']['p_one_sided']:.4f}  "
              f"({'sig p<0.05' if stats['wilcoxon']['significant_p05'] else 'n.s.'})")

    # ΔAUC and paired bootstrap CI
    delta_auc = results["MAR"]["AUC"] - results["noMAR"]["AUC"]
    s_lp_noMAR = results["noMAR"]["_s_lp_ho"]
    s_la_noMAR = results["noMAR"]["_s_la_ho"]
    s_lp_MAR   = results["MAR"]["_s_lp_ho"]
    s_la_MAR   = results["MAR"]["_s_la_ho"]

    rng_d = np.random.default_rng(99999)
    delta_boots = np.zeros(args.n_boot)
    for b in range(args.n_boot):
        idx = rng_d.integers(0, NUM_REALIZATIONS, size=NUM_REALIZATIONS)
        a_n = mw_auc(s_lp_noMAR[idx], s_la_noMAR[idx])
        a_m = mw_auc(s_lp_MAR[idx],   s_la_MAR[idx])
        delta_boots[b] = a_m - a_n
    delta_ci_lo = float(np.percentile(delta_boots, 2.5))
    delta_ci_hi = float(np.percentile(delta_boots, 97.5))

    # ΔAUC paired Wilcoxon
    d_delta = (s_lp_MAR - s_la_MAR) - (s_lp_noMAR - s_la_noMAR)
    try:
        _, p_delta = wilcoxon(d_delta, alternative='two-sided', zero_method='wilcox')
    except Exception:
        _, p_delta = wilcoxon(d_delta, zero_method='wilcox')

    # Print final summary
    print()
    print("=" * 66)
    print(f"ASTM WKXXXXX v1.0.0 — CHO Result  [{tier.tier_id}]")
    print("=" * 66)
    r_n = results["noMAR"]
    r_m = results["MAR"]
    print(f"  AUC_noMAR  : {r_n['AUC']:.4f}  "
          f"(Bias {r_n['Bias']:+.4f}; "
          f"CI [{r_n['CI_95'][0]:.4f}, {r_n['CI_95'][1]:.4f}])")
    print(f"  AUC_MAR    : {r_m['AUC']:.4f}  "
          f"(Bias {r_m['Bias']:+.4f}; "
          f"CI [{r_m['CI_95'][0]:.4f}, {r_m['CI_95'][1]:.4f}])")
    print(f"  ΔAUC       : {delta_auc:+.4f}  "
          f"(95% CI [{delta_ci_lo:+.4f}, {delta_ci_hi:+.4f}])")
    print(f"  Wilcoxon (MAR>noMAR): {r_m['wilcoxon']['p_one_sided']:.4f}  "
          f"({'sig p<0.05' if r_m['wilcoxon']['significant_p05'] else 'n.s.'})")
    if args.noise_sigma_sweep:
        sf = r_n.get("noise_sweep", {}).get("sigmoid_fit", {})
        if sf.get("fit_ok"):
            print(f"  Sigmoid fit (noMAR): σ₀ = {sf['sigma_0']:.1f}  "
                  f"A = {sf['A']:.4f}  k = {sf['k']:.4f}  R² = {sf['r_squared']:.4f}")
    if not ils_mode:
        print("  *** SELF-TEST: ΔAUC = 0 by definition ***")
    else:
        print(f"  Pipeline: {recon_pipeline}")
    print("=" * 66)

    # Build JSON output
    def _clean_stats(s: dict) -> dict:
        out = {}
        for k, v in s.items():
            if k.startswith("_"):
                continue
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, tuple):
                out[k] = list(v)
            elif isinstance(v, dict):
                out[k] = _clean_stats(v)
            else:
                out[k] = v
        return out

    out = {
        "generator_version":      "v6.0.0",
        "script_version":         "run_cho_analysis_v6_0",
        "standard_reference":     "ASTM-WKXXXXX-v1.0.0",
        "observer":               "2D CHO, single slice",
        "tier":                   tier.to_dict(),
        "lesion_slice_index":     LESION_SLICE_INDEX,
        "num_channels":           NUM_CHANNELS,
        "internal_noise_sigma":   args.internal_noise_sigma,
        "ils_mode":               ils_mode,
        "reconstruction_pipeline": recon_pipeline,
        "noMAR":                  _clean_stats(results["noMAR"]),
        "MAR":                    _clean_stats(results["MAR"]),
        "delta_AUC":              float(delta_auc),
        "delta_AUC_CI_95":        [delta_ci_lo, delta_ci_hi],
        "delta_wilcoxon_p_twosided": float(p_delta),
        "self_test":              not ils_mode,
    }

    results_path = Path(args.results_file)
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written → {results_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
