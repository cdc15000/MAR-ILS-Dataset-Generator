"""
Physics-based CT acquisition noise model (Vaishnav 2020).
"""

from __future__ import annotations

import numpy as np

from mar_ils_core.constants import SCATTER_FRAC, SIGMA_E_COUNTS


def apply_noise(
    sino_clean: np.ndarray,
    I0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Physics-based CT acquisition noise.

      I_meas = Poisson(I0 * exp(-p) + S) + N(0, sigma_e^2)
      p_meas = -ln(max(I_meas, 0.1) / I0)
    """
    S = SCATTER_FRAC * I0
    I_expected = I0 * np.exp(-sino_clean) + S
    I_measured = rng.poisson(I_expected).astype(np.float64)
    I_measured += rng.normal(0.0, SIGMA_E_COUNTS, size=I_measured.shape)
    I_measured = np.maximum(I_measured, 0.1)
    return (-np.log(I_measured / I0)).astype(np.float32)


# Floor on scatter-corrected counts before the log (v8.0.0).
SCATTER_CORRECTED_FLOOR_COUNTS: float = 1.0


def apply_noise_scatter_corrected(
    sino_clean: np.ndarray,
    I0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Physics-based CT acquisition noise with the known scatter subtracted
    before the log (v8.0.0). Same measurement model as ``apply_noise``.

      I_meas = Poisson(I0 * exp(-p) + S) + N(0, sigma_e^2)
      p_meas = -ln(max(I_meas - S, 1) / I0)
    """
    S = SCATTER_FRAC * I0
    I_expected = I0 * np.exp(-sino_clean) + S
    I_measured = rng.poisson(I_expected).astype(np.float64)
    I_measured += rng.normal(0.0, SIGMA_E_COUNTS, size=I_measured.shape)
    I_primary = np.maximum(I_measured - S, SCATTER_CORRECTED_FLOOR_COUNTS)
    return (-np.log(I_primary / I0)).astype(np.float32)
