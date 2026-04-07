#!/usr/bin/env python3
"""
run_cho_analysis_v7_0.py
========================
ASTM WKXXXXX Rev 04 — Reference 2D CHO Analysis (Fan-Beam Dataset)

Changes from run_cho_analysis_v6_0.py
--------------------------------------
[SC]  Single canonical configuration — hardcoded v5.3.0/Rev 04 ROI parameters
      (121×121, centre (281,256), channel width a=7.5 vox). No tier dependency.
[FB]  References fan-beam generator v7.0.0 dataset (geometry-independent — CHO
      operates on reconstructed DICOM images regardless of projection geometry).
[AT]  AUC equivalence tolerance relaxed from ±0.001 to ±0.005 (§8.3, Rev 04).
[RR]  Supports reduced realization count (20 for screening, ≥40 for formal).
[T2]  50/50 estimation bias (2-fold CV) — informative diagnostic.
[T3]  Wilcoxon signed-rank test — informative diagnostic.
[T4]  Noise sensitivity sweep — informative diagnostic.
[T5]  Sigmoid AUC fit — informative diagnostic.

All CHO mathematics (Laguerre-Gauss channels, Hotelling template, LOO,
bootstrap) are identical to v5.3.0/v6.0.0. The observer is geometry-
independent: it operates on HU-calibrated DICOM images, not sinograms.

Normative Parameters (§A1.5, Rev 04)
--------------------------------------
  Observer      : 2D CHO — slice 128 ONLY (§A1.5.3)
  ROI           : 121×121 centred at (281, 256) (§A1.5.4)
  Channels      : 10 Laguerre-Gauss, a = 7.5 voxels (§A1.5.1)
  Covariance    : LA only, pooled (§A1.5.2(a))
  Regularisation: Tikhonov λ = 0.01 × trace(K) / 10 (§A1.5.2(c))
  Internal noise: σ = 15 (normative default, §A1.5.2(d))
  Metric        : Mann-Whitney AUC (LOO hold-out) (§A1.6)
  Tolerance     : ±0.005 AUC (§8.3, Rev 04)

Usage
-----
    # ILS mode:
    python run_cho_analysis_v7_0.py \\
        --dataset-dir ./astm_reference_dataset \\
        --mar-output-dir ./mar_recon \\
        --internal-noise-sigma 15

    # Self-test (ΔAUC = 0 by definition):
    python run_cho_analysis_v7_0.py \\
        --dataset-dir ./astm_reference_dataset \\
        --self-test

    # Screening mode (20 realizations):
    python run_cho_analysis_v7_0.py \\
        --dataset-dir ./astm_reference_dataset_20 \\
        --self-test --realizations 20
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

# ═══════════════════════════════════════════════════════════════════════════════
# Canonical constants (ASTM WKXXXXX Rev 04 — hardcoded, no tier dependency)
# ═══════════════════════════════════════════════════════════════════════════════

LESION_SLICE_INDEX: int = 128     # zero-indexed (§A1.4(a))
NUM_REALIZATIONS_DEFAULT: int = 40

# CHO ROI (§A1.5.4)
ROI_SIZE: int = 121               # 121×121 voxels
ROI_CENTER_X: int = 281           # lesion centre x (§A1.4(c))
ROI_CENTER_Y: int = 256           # phantom centre y

# Channel parameters (§A1.5.1)
NUM_CHANNELS: int = 10            # Laguerre-Gauss channels
CHANNEL_WIDTH_A: float = 7.5      # 1.5 × r_lesion = 1.5 × 5 = 7.5 voxels

# Bootstrap
N_BOOT: int = 1000

# Noise sweep (informative diagnostic)
DEFAULT_SIGMA_SWEEP: list[float] = [0, 5, 10, 15, 20, 25, 30, 40, 50, 65, 80]

# AUC equivalence tolerance (§8.3, Rev 04)
AUC_TOLERANCE: float = 0.005


# ═══════════════════════════════════════════════════════════════════════════════
# Channel templates
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_lg_channels() -> np.ndarray:
    """
    Generate 10 2D Laguerre-Gauss channel templates.

    u_n(x,y) = L_n(2π r²/a²) · exp(−π r²/a²), L2-normalised over the ROI.

    Returns: U, shape (NUM_CHANNELS, ROI_SIZE²)
    """
    half = ROI_SIZE // 2
    print(f"  LG channels: ROI={ROI_SIZE}×{ROI_SIZE}, "
          f"a={CHANNEL_WIDTH_A:.1f} vox  ... ", end="", flush=True)

    x = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)
    r2 = xx**2 + yy**2
    arg = 2.0 * np.pi * r2 / CHANNEL_WIDTH_A**2
    env = np.exp(-np.pi * r2 / CHANNEL_WIDTH_A**2)

    rows = []
    for n in range(NUM_CHANNELS):
        u2d = genlaguerre(n, 0)(arg) * env
        norm = np.linalg.norm(u2d)
        rows.append((u2d / norm).flatten())

    print("done.")
    return np.vstack(rows)  # (NUM_CHANNELS, ROI_SIZE²)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_slice_roi(folder_path: Path) -> np.ndarray:
    """
    Load the ROI from LESION_SLICE_INDEX of a DICOM realization folder.

    Reads only slice_{LESION_SLICE_INDEX+1:04d}.dcm. 3D integration is
    prohibited (§A1.5.3).

    Returns: roi, shape (ROI_SIZE²,), float64
    """
    fname = folder_path / f"slice_{LESION_SLICE_INDEX + 1:04d}.dcm"
    if not fname.exists():
        raise FileNotFoundError(
            f"Expected slice not found: {fname}\n"
            f"  Ensure filenames are 1-indexed (slice_0001.dcm … slice_0256.dcm)."
        )
    dcm = pydicom.dcmread(str(fname))
    img = dcm.pixel_array.astype(np.float64)
    img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

    half = ROI_SIZE // 2
    ys = ROI_CENTER_Y - half
    ye = ROI_CENTER_Y + half + 1
    xs = ROI_CENTER_X - half
    xe = ROI_CENTER_X + half + 1
    roi = img[ys:ye, xs:xe]
    return roi.flatten()


def _worker_load(folder_str: str) -> np.ndarray:
    return load_slice_roi(Path(folder_str))


def load_all_rois_parallel(
    lp_root: Path,
    la_root: Path,
    num_realizations: int,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ROIs from all realizations. Returns (lp_rois, la_rois)."""
    lp_folders = [
        str(lp_root / f"realization_{i + 1:03d}")
        for i in range(num_realizations)
    ]
    la_folders = [
        str(la_root / f"realization_{i + 1:03d}")
        for i in range(num_realizations)
    ]
    all_folders = lp_folders + la_folders

    rois: dict[str, np.ndarray] = {}
    if n_workers == 1:
        for i, f in enumerate(all_folders):
            print(f"  Loading {i + 1}/{len(all_folders)}...", end="\r")
            rois[f] = _worker_load(f)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futs = {exe.submit(_worker_load, f): f for f in all_folders}
            done = 0
            for fut in as_completed(futs):
                f = futs[fut]
                rois[f] = fut.result()
                done += 1
                print(f"  Loaded {done}/{len(all_folders)}...",
                      end="\r", flush=True)

    print(f"  Loaded {len(all_folders)} realizations.              ")
    lp_rois = np.vstack([rois[f] for f in lp_folders])
    la_rois = np.vstack([rois[f] for f in la_folders])
    return lp_rois, la_rois


# ═══════════════════════════════════════════════════════════════════════════════
# Core CHO statistics (identical to v5.3.0 / v6.0.0)
# ═══════════════════════════════════════════════════════════════════════════════

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
    test_lp: np.ndarray,
    test_la: np.ndarray,
    internal_noise_var: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Hotelling template on training features; score test features."""
    K_la = np.cov(train_la, rowvar=False)
    lam = 0.01 * np.trace(K_la) / NUM_CHANNELS
    K_reg = K_la + (lam + internal_noise_var) * np.eye(NUM_CHANNELS)
    w = np.linalg.solve(
        K_reg,
        np.mean(train_lp, axis=0) - np.mean(train_la, axis=0),
    )
    return test_lp @ w, test_la @ w


def compute_cho_performance(
    lp_features: np.ndarray,
    la_features: np.ndarray,
    num_realizations: int,
    n_boot: int = N_BOOT,
    boot_seed: int = 12345,
    internal_noise_sigma: float = 0.0,
) -> dict:
    """
    Resubstitution AUC, LOO hold-out AUC, bias, and bootstrap CI.

    Returns dict with keys: AUC, Bias, CI_95, _s_lp_ho, _s_la_ho
    """
    N = num_realizations
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
        te_lp = lp_features[i:i + 1]
        te_la = la_features[i:i + 1]
        s_lp_ho[i], s_la_ho[i] = (
            v[0] for v in _fit_and_score(
                tr_lp, tr_la, te_lp, te_la,
                internal_noise_var=int_var,
            )
        )
    auc_ho = mw_auc(s_lp_ho, s_la_ho)
    bias = auc_resub - auc_ho

    # Bootstrap 95% CI
    rng = np.random.default_rng(boot_seed)
    boot_aucs = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot_aucs[b] = mw_auc(s_lp_ho[idx], s_la_ho[idx])
    ci_lo = float(np.percentile(boot_aucs, 2.5))
    ci_hi = float(np.percentile(boot_aucs, 97.5))

    return {
        "AUC": float(auc_ho),
        "Bias": float(bias),
        "CI_95": (ci_lo, ci_hi),
        "_s_lp_ho": s_lp_ho,
        "_s_la_ho": s_la_ho,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced diagnostics (informative — not normative)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_50_50_auc(
    lp_features: np.ndarray,
    la_features: np.ndarray,
    num_realizations: int,
    internal_noise_sigma: float = 0.0,
    seed: int = 42,
) -> dict:
    """2-fold cross-validation estimation bias (50/50 split)."""
    N = num_realizations
    int_var = internal_noise_sigma ** 2
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    mid = N // 2

    tr_lp_a, te_lp_a = lp_features[idx[mid:]], lp_features[idx[:mid]]
    tr_la_a, te_la_a = la_features[idx[mid:]], la_features[idx[:mid]]
    s_lp_a, s_la_a = _fit_and_score(tr_lp_a, tr_la_a, te_lp_a, te_la_a, int_var)

    tr_lp_b, te_lp_b = lp_features[idx[:mid]], lp_features[idx[mid:]]
    tr_la_b, te_la_b = la_features[idx[:mid]], la_features[idx[mid:]]
    s_lp_b, s_la_b = _fit_and_score(tr_lp_b, tr_la_b, te_lp_b, te_la_b, int_var)

    auc_5050 = mw_auc(
        np.concatenate([s_lp_a, s_lp_b]),
        np.concatenate([s_la_a, s_la_b]),
    )
    s_lp_rs, s_la_rs = _fit_and_score(
        lp_features, la_features, lp_features, la_features, int_var,
    )
    auc_resub = mw_auc(s_lp_rs, s_la_rs)

    return {
        "AUC_resub": float(auc_resub),
        "AUC_5050": float(auc_5050),
        "Bias_5050": float(auc_resub - auc_5050),
    }


def compute_wilcoxon_test(s_lp_ho: np.ndarray, s_la_ho: np.ndarray) -> dict:
    """One-tailed paired Wilcoxon signed-rank test: H₁: median(d) > 0."""
    d = s_lp_ho - s_la_ho
    try:
        stat, p = wilcoxon(d, alternative='greater', zero_method='wilcox')
        p_one_sided = float(p)
    except TypeError:
        stat, p_two = wilcoxon(d, zero_method='wilcox')
        p_one_sided = (
            float(p_two) / 2.0
            if float(np.sum(d > 0)) >= len(d) / 2
            else 1.0 - float(p_two) / 2.0
        )
    return {
        "wilcoxon_stat": float(stat),
        "p_one_sided": p_one_sided,
        "significant_p05": bool(p_one_sided < 0.05),
    }


def fit_auc_sigmoid(sigmas: list[float], aucs: list[float]) -> dict:
    """Fit AUC(σ) = A / (1 + exp(k(σ − σ₀))) + 0.5."""
    sigmas_arr = np.asarray(sigmas, dtype=float)
    aucs_arr = np.asarray(aucs, dtype=float)

    def _sigmoid(s, A, k, s0):
        return A / (1.0 + np.exp(k * (s - s0))) + 0.5

    A0 = float(np.max(aucs_arr) - 0.5)
    s0_0 = float(sigmas_arr[np.argmin(np.abs(aucs_arr - (0.5 + A0 / 2)))])

    try:
        popt, _ = curve_fit(
            _sigmoid, sigmas_arr, aucs_arr,
            p0=[max(A0, 0.01), -0.05, s0_0],
            bounds=([0, -5, 0], [0.5, -1e-6, 200]),
            maxfev=5000,
        )
        A, k, s0 = popt
        ss_res = float(np.sum((aucs_arr - _sigmoid(sigmas_arr, *popt)) ** 2))
        ss_tot = float(np.sum((aucs_arr - np.mean(aucs_arr)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        return {"A": float(A), "k": float(k), "sigma_0": float(s0),
                "fit_ok": True, "r_squared": float(r2)}
    except Exception as exc:
        return {"A": None, "k": None, "sigma_0": None,
                "fit_ok": False, "r_squared": None, "error": str(exc)}


def compute_noise_sweep(
    lp_features: np.ndarray,
    la_features: np.ndarray,
    num_realizations: int,
    sigma_range: list[float] | None = None,
    n_boot_sweep: int = 200,
) -> dict:
    """Vaishnav Sensitivity Matrix: AUC at each internal_noise_sigma."""
    if sigma_range is None:
        sigma_range = DEFAULT_SIGMA_SWEEP

    print(f"  Noise sweep ({len(sigma_range)} σ values)...", flush=True)
    sweep_points = []
    for sigma in sigma_range:
        stats = compute_cho_performance(
            lp_features, la_features, num_realizations,
            n_boot=n_boot_sweep,
            internal_noise_sigma=float(sigma),
        )
        sweep_points.append({
            "sigma": sigma,
            "AUC": stats["AUC"],
            "CI_95": list(stats["CI_95"]),
        })
        print(f"    σ={sigma:4.0f}  AUC={stats['AUC']:.4f}", flush=True)

    sigmas = [p["sigma"] for p in sweep_points]
    aucs = [p["AUC"] for p in sweep_points]
    return {
        "sweep_points": sweep_points,
        "sigmoid_fit": fit_auc_sigmoid(sigmas, aucs),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASTM WKXXXXX Rev 04 — Reference 2D CHO Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-dir", required=True,
                        help="Root directory of the v7.0.0 dataset")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mar-output-dir",
                      help="(ILS mode) Lab MAR reconstructions directory")
    mode.add_argument("--self-test", action="store_true",
                      help="(Validation) Score noMAR_recon vs itself (ΔAUC=0)")

    parser.add_argument("--reconstruction-pipeline", default="", metavar="DESC",
                        help="Free-text pipeline description (ILS mode)")
    parser.add_argument("--internal-noise-sigma", type=float, default=15.0,
                        help="Vaishnav internal observer noise σ (default: 15)")
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

    print(f"ASTM WKXXXXX Rev 04 — 2D CHO Analysis")
    print(f"ROI: {ROI_SIZE}×{ROI_SIZE}  centre=({ROI_CENTER_X},{ROI_CENTER_Y})")
    print(f"LG channel width a = {CHANNEL_WIDTH_A:.1f} vox")
    print(f"Internal noise σ = {args.internal_noise_sigma}")
    print(f"Realizations: {num_real} per condition "
          f"({'SCREENING' if num_real < 40 else 'FORMAL'})")
    print(f"AUC equivalence tolerance: ±{AUC_TOLERANCE}")
    print()

    if num_real < 40:
        print("WARNING: Screening mode (< 40 realizations). "
              "Results are not reportable under §10.2.\n")

    base_dir = Path(args.dataset_dir)
    if not base_dir.exists():
        print(f"Dataset directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    noMAR_dir = base_dir / "noMAR_recon"
    if not noMAR_dir.exists():
        print(f"noMAR_recon/ not found in {base_dir}", file=sys.stderr)
        sys.exit(1)

    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    # Channel templates
    U = calculate_lg_channels()  # (NUM_CHANNELS, ROI_SIZE²)

    # MAR source
    if args.self_test:
        mar_lp_root = noMAR_dir / "LP"
        mar_la_root = noMAR_dir / "LA"
        mar_label = "noMAR_recon [self-test — ΔAUC=0 by definition]"
        ils_mode = False
        recon_pipeline = "N/A (self-test)"
        print("  *** SELF-TEST MODE: ΔAUC = 0 by definition ***\n")
    else:
        mar_root = Path(args.mar_output_dir)
        if not mar_root.exists():
            print(f"--mar-output-dir not found: {mar_root}", file=sys.stderr)
            sys.exit(1)
        mar_lp_root = mar_root / "LP"
        mar_la_root = mar_root / "LA"
        mar_label = f"MAR (submission: {mar_root})"
        ils_mode = True
        recon_pipeline = args.reconstruction_pipeline or "not declared"

    results: dict[str, dict] = {}
    for cond, lp_root, la_root, display in [
        ("noMAR", noMAR_dir / "LP", noMAR_dir / "LA",
         "noMAR_recon (reference FBP)"),
        ("MAR", mar_lp_root, mar_la_root, mar_label),
    ]:
        print(f"\nProcessing: {display}")
        lp_rois, la_rois = load_all_rois_parallel(
            lp_root, la_root, num_real, n_workers,
        )

        # Project through channel templates
        lp_feat = (U @ lp_rois.T).T  # (N, NUM_CHANNELS)
        la_feat = (U @ la_rois.T).T

        stats = compute_cho_performance(
            lp_feat, la_feat, num_real,
            n_boot=args.n_boot,
            internal_noise_sigma=args.internal_noise_sigma,
        )

        # Informative diagnostics
        stats["bias_5050"] = compute_50_50_auc(
            lp_feat, la_feat, num_real,
            internal_noise_sigma=args.internal_noise_sigma,
        )
        stats["wilcoxon"] = compute_wilcoxon_test(
            stats["_s_lp_ho"], stats["_s_la_ho"],
        )
        if args.noise_sigma_sweep or cond == "noMAR":
            stats["noise_sweep"] = compute_noise_sweep(
                lp_feat, la_feat, num_real,
            )

        results[cond] = stats
        print(f"  AUC (LOO hold-out) : {stats['AUC']:.4f}")
        print(f"  95% CI             : "
              f"[{stats['CI_95'][0]:.4f}, {stats['CI_95'][1]:.4f}]")
        print(f"  Bias (resub−LOO)   : {stats['Bias']:+.4f}")
        print(f"  Bias_5050          : "
              f"{stats['bias_5050']['Bias_5050']:+.4f}")
        print(f"  Wilcoxon p (1-tail): "
              f"{stats['wilcoxon']['p_one_sided']:.4f}  "
              f"({'sig p<0.05' if stats['wilcoxon']['significant_p05'] else 'n.s.'})")

    # ΔAUC and paired bootstrap CI
    delta_auc = results["MAR"]["AUC"] - results["noMAR"]["AUC"]
    s_lp_n = results["noMAR"]["_s_lp_ho"]
    s_la_n = results["noMAR"]["_s_la_ho"]
    s_lp_m = results["MAR"]["_s_lp_ho"]
    s_la_m = results["MAR"]["_s_la_ho"]

    rng_d = np.random.default_rng(99999)
    delta_boots = np.zeros(args.n_boot)
    for b in range(args.n_boot):
        idx = rng_d.integers(0, num_real, size=num_real)
        a_n = mw_auc(s_lp_n[idx], s_la_n[idx])
        a_m = mw_auc(s_lp_m[idx], s_la_m[idx])
        delta_boots[b] = a_m - a_n
    delta_ci_lo = float(np.percentile(delta_boots, 2.5))
    delta_ci_hi = float(np.percentile(delta_boots, 97.5))

    # Paired Wilcoxon on ΔAUC
    d_delta = (s_lp_m - s_la_m) - (s_lp_n - s_la_n)
    try:
        _, p_delta = wilcoxon(d_delta, alternative='two-sided',
                              zero_method='wilcox')
    except Exception:
        _, p_delta = wilcoxon(d_delta, zero_method='wilcox')

    # Summary
    print()
    print("=" * 66)
    print(f"ASTM WKXXXXX Rev 04 — CHO Result")
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
    print(f"  Wilcoxon (ΔAUC ≠ 0): p={float(p_delta):.4f}")

    if num_real < 40:
        print("  *** SCREENING MODE — not reportable under §10.2 ***")
    if not ils_mode:
        print("  *** SELF-TEST: ΔAUC = 0 by definition ***")
    else:
        print(f"  Pipeline: {recon_pipeline}")
    print("=" * 66)

    # JSON output
    def _clean(s: dict) -> dict:
        out = {}
        for k, v in s.items():
            if k.startswith("_"):
                continue
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, tuple):
                out[k] = list(v)
            elif isinstance(v, dict):
                out[k] = _clean(v)
            else:
                out[k] = v
        return out

    out = {
        "generator_version": "v7.0.0",
        "script_version": "run_cho_analysis_v7_0",
        "standard_reference": "ASTM-WKXXXXX-Rev04",
        "acquisition_geometry": "fan-beam (SID=570mm, SDD=1040mm)",
        "observer": "2D CHO, single slice (§A1.5.3)",
        "roi_size": ROI_SIZE,
        "roi_center": [ROI_CENTER_X, ROI_CENTER_Y],
        "channel_width_a": CHANNEL_WIDTH_A,
        "num_channels": NUM_CHANNELS,
        "lesion_slice_index": LESION_SLICE_INDEX,
        "internal_noise_sigma": args.internal_noise_sigma,
        "num_realizations": num_real,
        "auc_tolerance": AUC_TOLERANCE,
        "ils_mode": ils_mode,
        "reconstruction_pipeline": recon_pipeline,
        "noMAR": _clean(results["noMAR"]),
        "MAR": _clean(results["MAR"]),
        "delta_AUC": float(delta_auc),
        "delta_AUC_CI_95": [delta_ci_lo, delta_ci_hi],
        "delta_wilcoxon_p_twosided": float(p_delta),
        "self_test": not ils_mode,
        "screening_mode": num_real < 40,
    }

    results_path = Path(args.results_file)
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written → {results_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
