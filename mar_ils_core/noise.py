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
