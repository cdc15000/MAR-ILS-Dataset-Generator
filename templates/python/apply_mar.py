"""
Laboratory MAR implementation: REPLACE THE BODIES OF THESE TWO FUNCTIONS.

The harness calls apply_mar() once per realization, and calls reconstruct()
for the reconstruction check (draft standard §12.2).

As supplied, apply_mar() applies no MAR: it reconstructs the sinogram with
the reference reconstruction, so its output matches the supplied noMAR image
of the same realization to within DICOM rounding. Run it once that way to
check the harness before plugging in your method.

These templates serve Condition 2 (the laboratory's own MAR). Under
Condition 1 the laboratory works from the text of the standard and the
ingestion specification only (protocol §A7).
"""
from __future__ import annotations

import numpy as np
import reference_fbp


def reconstruct(p: np.ndarray, geo) -> np.ndarray:
    """Reconstruction used under draft standard §14.3, returning HU (512 x 512).

    Either the supplied reference reconstruction (as here) or your own, which
    must then pass the reconstruction check (run_mar_harness.py --recon-check).
    """
    return reference_fbp.to_hu(reference_fbp.fbp(p, geo))


def apply_mar(p: np.ndarray, geo) -> np.ndarray:
    """Apply MAR to one realization.

    Args:
        p:   float64 array (984, 888), line integrals indexed [view, channel]
             (ingestion spec §4.1). Counts, if needed: geo.I0 * exp(-p).
        geo: Geometry (run_mar_harness.py) with SID, SDD, view and channel
             angles in radians, and the acquisition constants.

    Returns:
        float array (512, 512) in HU per §A1.1(h), indexed [row, column]
        with pixel (256, 256) at the isocenter (ingestion spec §6.2).
    """
    # ── REPLACE WITH YOUR MAR PIPELINE ──
    # A sinogram-domain method corrects p here and then calls reconstruct().
    return reconstruct(p, geo)
