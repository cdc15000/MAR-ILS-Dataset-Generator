"""
Dense difference-of-Gaussians (D-DOG) channels, Abbey and Barrett,
J Opt Soc Am A 18(3):473-488, 2001, Eq. 35 and §4:

    C_j(rho) = exp(-½(rho/(Q·sigma_j))²) - exp(-½(rho/sigma_j)²),
    sigma_j = sigma_0·alpha^j,

rho in cycles/pixel. The paper does not state the index range; j = 1…10 is
used (its square channels are indexed from 1). Channel scaling does not
affect the Hotelling observer, so no normalization is applied.
"""
from __future__ import annotations

import numpy as np

SIGMA0: float = 0.005
ALPHA: float = 1.4
Q: float = 1.67
N_CHANNELS: int = 10


def ddog_channels(size: int, first_index: int = 1, sigma0: float = SIGMA0,
                  alpha: float = ALPHA, q: float = Q, n: int = N_CHANNELS) -> np.ndarray:
    """Spatial channels, shape (size*size, n), centred on the ROI centre pixel."""
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    rho = np.hypot(fx, fy)
    out = np.empty((size * size, n))
    for k in range(n):
        s = sigma0 * alpha ** (first_index + k)
        spec = np.exp(-0.5 * (rho / (q * s)) ** 2) - np.exp(-0.5 * (rho / s) ** 2)
        spatial = np.fft.fftshift(np.real(np.fft.ifft2(spec)))   # origin at size//2
        out[:, k] = spatial.ravel()
    return out
