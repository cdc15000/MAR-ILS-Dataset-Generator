"""
Control methods applied to the stored sinograms of a dataset (Rev 06 §17.7, §A1.9).

The dataset stores the line integrals p of each realization as float32
(ingestion spec §4). The functions here take that array and return the
corrected sinogram or the corrected image, using the reference
reconstruction (vaishnav_sim.fbp) and the reference implementation of the
control methods (mar_v9). Laboratories in the interlaboratory study and the
coordinator run this same code, so their images are expected to be equal.

Improving control (MAR 2, §A1.9): metal segmented in the uncorrected μ image
by the fixed threshold τ, its trace found, and the trace replaced by linear
interpolation across channels.

Degrading control (MAR 1, §17.7(b)): rays whose detected counts fall below
the fixed count threshold are replaced in the same way. Counts are recovered
from the stored line integrals as I₀·exp(−p).

The thresholds are those of the dataset record and shall not be changed
(§17.7(c)).
"""
from __future__ import annotations

import numpy as np

from mar_ils_core import mar_v9
from mar_ils_core import vaishnav_sim as vs

TAU_MU_CM = 1.9529025691374509          # Rev 06 §A1.9(b)
TAU_COUNTS = 1124.2100350620863         # MAR 1, degrading control
CONTROLS = ("improving", "degrading")


def improving_sinogram(p: np.ndarray) -> np.ndarray:
    """§A1.9(a) to (d): the inpainted line integrals, float64."""
    p = np.asarray(p, dtype=np.float64)
    return mar_v9.interpolate_trace(p, mar_v9.mask_trace(vs.fbp(p) > TAU_MU_CM))


def degrading_sinogram(p: np.ndarray, i0: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return mar_v9.interpolate_trace(p, i0 * np.exp(-p) < TAU_COUNTS)


def improving_hu(p: np.ndarray) -> np.ndarray:
    """§A1.9(e): the corrected image in HU, 512 × 512, before DICOM rounding."""
    return vs.to_hu(vs.fbp(improving_sinogram(p)))


def degrading_hu(p: np.ndarray, i0: float) -> np.ndarray:
    return vs.to_hu(vs.fbp(degrading_sinogram(p, i0)))


def control_hu(p: np.ndarray, control: str, i0: float) -> np.ndarray:
    if control == "improving":
        return improving_hu(p)
    if control == "degrading":
        return degrading_hu(p, i0)
    raise ValueError(f"unknown control {control!r}; choose from {', '.join(CONTROLS)}")
