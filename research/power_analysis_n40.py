#!/usr/bin/env python3
"""
Model-based power / precision analysis for the N = 40 minimum.

Supports ASTM WKXXXXX §10.2 and §17.1.6 with a *pre-pilot, informational*
estimate of how the test result's precision and bias scale with the number of
realizations N. The pilot precision study (§17.1.6) provides the definitive,
data-driven statement; this script only establishes that N = 40 is a defensible
choice ahead of that study.

Method
------
A Monte Carlo that drives the **exact reference estimation procedure** — the
LOO hold-out Hotelling template fit and Mann-Whitney AUC imported from
`run_cho_analysis_v7_0.py` — on simulated channel features, so the finite-N
sampling SD and the resubstitution-minus-hold-out bias are measured on the same
code labs run, not on a re-derivation.

Model (transparent; assumptions stated)
---------------------------------------
- 10 channel features per realization (NUM_CHANNELS).
- Lesion-absent features ~ MVN(0, I_10); lesion-present = same-shape noise + a
  signal vector s placed in one channel. Signal magnitude is calibrated so the
  large-N AUC equals a target operating point (noMAR target = 0.8294, the locked
  baseline).
- Pairing across conditions: the MAR condition reuses the SAME per-realization
  noise draws as noMAR (only the signal magnitude differs), reproducing the
  realization-level correlation that the paired ΔAUC bootstrap exploits and that
  determines SD(ΔAUC).
- Internal-noise regularization is set to 0 here. This is deliberately
  CONSERVATIVE: the σ_internal = 15 term in the reference pipeline stabilizes the
  covariance inverse, which *reduces* both estimation variance and resub−LOO
  bias. Reporting the unregularized values therefore upper-bounds the SD and bias
  the regularized estimator achieves. Identity noise covariance is the
  transparent baseline; absolute values depend on the (unmodeled) real channel
  covariance, so the figures characterize scaling and order of magnitude, not a
  substitute for the pilot.

Targets checked (from §17.1.6)
------------------------------
  (b) within-condition SD(ΔAUC) ≤ 0.05
  (c) AUC estimation bias (resubstitution − hold-out) ≤ 0.02
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_cho_analysis_v7_0 import mw_auc, _fit_and_score  # noqa: E402

P = 10                       # channels (NUM_CHANNELS)
TARGET_AUC_NOMAR = 0.8294    # locked baseline
DELTA_REPRESENTATIVE = -0.23 # LI-MAR floor; SD(ΔAUC) is ~insensitive to the mean
N_GRID = (20, 40, 80)
M_TRIALS = 2000
SEED = 20260530


def signal_for_auc(auc: float) -> float:
    """Single-channel signal magnitude giving large-N AUC = Φ(m/√2)."""
    return float(np.sqrt(2.0) * norm.ppf(auc))


def _loo_scores(lp: np.ndarray, la: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    s_lp = np.zeros(n)
    s_la = np.zeros(n)
    for i in range(n):
        tr_lp = np.delete(lp, i, axis=0)
        tr_la = np.delete(la, i, axis=0)
        a, b = _fit_and_score(tr_lp, tr_la, lp[i:i + 1], la[i:i + 1], 0.0)
        s_lp[i], s_la[i] = a[0], b[0]
    return s_lp, s_la


def _resub_auc(lp: np.ndarray, la: np.ndarray) -> float:
    a, b = _fit_and_score(lp, la, lp, la, 0.0)
    return mw_auc(a, b)


def run(n: int, m_trials: int, rng: np.random.Generator) -> dict:
    s_no = np.zeros(P); s_no[0] = signal_for_auc(TARGET_AUC_NOMAR)
    s_mar = np.zeros(P); s_mar[0] = signal_for_auc(TARGET_AUC_NOMAR + DELTA_REPRESENTATIVE)

    deltas = np.empty(m_trials)
    auc_no = np.empty(m_trials)
    bias_no = np.empty(m_trials)
    for t in range(m_trials):
        e_lp = rng.standard_normal((n, P))   # shared across conditions (pairing)
        e_la = rng.standard_normal((n, P))
        lp_no, la_no = e_lp + s_no, e_la
        lp_mar, la_mar = e_lp + s_mar, e_la

        slp_no, sla_no = _loo_scores(lp_no, la_no, n)
        slp_m, sla_m = _loo_scores(lp_mar, la_mar, n)
        a_no = mw_auc(slp_no, sla_no)
        a_mar = mw_auc(slp_m, sla_m)

        deltas[t] = a_mar - a_no
        auc_no[t] = a_no
        bias_no[t] = _resub_auc(lp_no, la_no) - a_no

    return {
        "N": n,
        "mean_AUC_noMAR": float(np.mean(auc_no)),
        "SD_AUC_noMAR": float(np.std(auc_no, ddof=1)),
        "SD_dAUC": float(np.std(deltas, ddof=1)),
        "mean_bias": float(np.mean(bias_no)),
        "p95_bias": float(np.percentile(bias_no, 95)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"Model-based power analysis — {M_TRIALS} trials/point, "
          f"target AUC_noMAR={TARGET_AUC_NOMAR}, representative ΔAUC={DELTA_REPRESENTATIVE}")
    print(f"{'N':>4}  {'mean AUC_noMAR':>15}  {'SD(ΔAUC)':>10}  "
          f"{'mean bias':>10}  {'p95 bias':>9}")
    rows = []
    for n in N_GRID:
        r = run(n, M_TRIALS, rng)
        rows.append(r)
        print(f"{r['N']:>4}  {r['mean_AUC_noMAR']:>15.4f}  {r['SD_dAUC']:>10.4f}  "
              f"{r['mean_bias']:>10.4f}  {r['p95_bias']:>9.4f}")

    out_path = Path(__file__).resolve().parent / "power_analysis_n40_results.json"
    out_path.write_text(json.dumps(rows, indent=2))

    r40 = next(r for r in rows if r["N"] == 40)
    print()
    print(f"At N=40:  SD(ΔAUC) = {r40['SD_dAUC']:.4f}  (§17.1.6(b) target ≤ 0.05 — "
          f"{'PASS' if r40['SD_dAUC'] <= 0.05 else 'FAIL'})")
    print(f"At N=40:  mean bias = {r40['mean_bias']:.4f}  (§17.1.6(c) target ≤ 0.02 — "
          f"{'PASS' if r40['mean_bias'] <= 0.02 else 'FAIL'})")
    print("\nConservative (σ_internal=0, identity covariance); pilot is definitive.")


if __name__ == "__main__":
    main()
