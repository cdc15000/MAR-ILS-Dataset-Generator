"""
Channelized Hotelling observer and antler-plot AUC∞ (Vaishnav et al. 2020, §2.D–2.E, §3).

- Template w = Σ⁻¹Δμ with Σ = (Σ₁ + Σ₂)/2 (pooled, sample covariances).
- SNR² = (t̄₂ − t̄₁)² / (½(σ²_t1 + σ²_t2)); AUC = Φ(SNR/√2) (Vaishnav Eq. 2,
  the default). signed=True keeps the sign of t̄₂ − t̄₁ instead, for comparison.
- Resubstitution: template and SNR from the same N_T images per class.
- Hold-out: template from N_T training images, SNR on a separate test set.
- For each of n_reps random draws, AUC against 1/N_T is fitted by a straight
  line for each method; AUC∞ = (⟨intercept_resub⟩ + ⟨intercept_holdout⟩)/2.
- Error: bias b = |⟨int_r⟩ − ⟨int_h⟩|/2, variance v = SD of the mean of the
  per-draw AUC∞; ε = √(b² + v²).

internal_noise_c > 0 adds Abbey and Barrett's channel internal noise,
K_ε = c·diag(Σ) (informational comparison only; normative c = 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

DEFAULT_NT: tuple[int, ...] = (25, 50, 60, 75, 80, 90, 100, 125, 250, 500, 800, 1000)
DEFAULT_REPS: int = 50


def channel_outputs(rois: np.ndarray, channels: np.ndarray) -> np.ndarray:
    """rois (N, s, s) → channel outputs (N, n_channels)."""
    return rois.reshape(rois.shape[0], -1) @ channels


def template(v1: np.ndarray, v2: np.ndarray, internal_noise_c: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Hotelling template and internal-noise covariance from absent (v1) and present (v2) outputs."""
    sigma = 0.5 * (np.cov(v1, rowvar=False) + np.cov(v2, rowvar=False))
    k_eps = internal_noise_c * np.diag(np.diag(sigma))
    w = np.linalg.solve(sigma + k_eps, v2.mean(0) - v1.mean(0))
    return w, k_eps


def auc_from_template(w: np.ndarray, k_eps: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                      signed: bool = False) -> float:
    t1, t2 = v1 @ w, v2 @ w
    var = 0.5 * (t1.var(ddof=1) + t2.var(ddof=1)) + float(w @ k_eps @ w)
    diff = t2.mean() - t1.mean()
    snr = (diff if signed else abs(diff)) / np.sqrt(var)
    return float(norm.cdf(snr / np.sqrt(2.0)))


def resub_auc(v1: np.ndarray, v2: np.ndarray, c: float = 0.0, signed: bool = False) -> float:
    w, k = template(v1, v2, c)
    return auc_from_template(w, k, v1, v2, signed)


def holdout_auc(v1_tr, v2_tr, v1_te, v2_te, c: float = 0.0, signed: bool = False) -> float:
    w, k = template(v1_tr, v2_tr, c)
    return auc_from_template(w, k, v1_te, v2_te, signed)


@dataclass
class AntlerResult:
    n_t: np.ndarray
    resub: np.ndarray          # (n_reps, len(n_t))
    holdout: np.ndarray        # (n_reps, len(n_t))
    int_resub: np.ndarray      # (n_reps,)
    int_holdout: np.ndarray    # (n_reps,)
    auc_inf: float
    bias: float
    variance: float
    error: float

    def as_dict(self) -> dict:
        return {
            "N_T": self.n_t.tolist(),
            "resub_mean": self.resub.mean(0).tolist(),
            "resub_sd": self.resub.std(0, ddof=1).tolist(),
            "holdout_mean": self.holdout.mean(0).tolist(),
            "holdout_sd": self.holdout.std(0, ddof=1).tolist(),
            "intercept_resub": float(self.int_resub.mean()),
            "intercept_holdout": float(self.int_holdout.mean()),
            "AUC_inf": self.auc_inf, "bias": self.bias,
            "variance": self.variance, "error": self.error,
        }


def _intercept(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[1])


def antler(v1_train: np.ndarray, v2_train: np.ndarray, v1_test: np.ndarray, v2_test: np.ndarray,
           n_t: tuple[int, ...] = DEFAULT_NT, n_reps: int = DEFAULT_REPS,
           seed: int = 0, internal_noise_c: float = 0.0, signed: bool = False) -> AntlerResult:
    """Antler-plot AUC∞ from training pools and a separate test set (channel outputs)."""
    n_max = min(len(v1_train), len(v2_train))
    n_t = np.array([n for n in n_t if n <= n_max])
    rng = np.random.default_rng(seed)
    r = np.empty((n_reps, n_t.size))
    h = np.empty((n_reps, n_t.size))
    for k in range(n_reps):
        for j, n in enumerate(n_t):
            i1 = rng.choice(len(v1_train), n, replace=False)
            i2 = rng.choice(len(v2_train), n, replace=False)
            a, b = v1_train[i1], v2_train[i2]
            w, ke = template(a, b, internal_noise_c)
            r[k, j] = auc_from_template(w, ke, a, b, signed)
            h[k, j] = auc_from_template(w, ke, v1_test, v2_test, signed)
    x = 1.0 / n_t
    ir = np.array([_intercept(x, r[k]) for k in range(n_reps)])
    ih = np.array([_intercept(x, h[k]) for k in range(n_reps)])
    per_draw = 0.5 * (ir + ih)
    auc_inf = float(per_draw.mean())
    bias = float(abs(ir.mean() - ih.mean()) / 2.0)
    variance = float(per_draw.std(ddof=1) / np.sqrt(n_reps))
    return AntlerResult(n_t, r, h, ir, ih, auc_inf, bias, variance, float(np.hypot(bias, variance)))
