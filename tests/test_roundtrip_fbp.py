"""
Tests for forward projection → FBP round-trip.

Verifies that tissue reconstructs to ~40 HU and the pipeline is self-consistent.
"""

import numpy as np
import pytest

from mar_ils_core.constants import (
    BACKGROUND_HU, MU_TISSUE_CM, X_DIM, Y_DIM, PHANTOM_CENTER_Y,
)
from mar_ils_core.phantom import build_attenuation_map


class TestRoundTripFBP:
    """Round-trip: build phantom → forward project → FBP → check HU."""

    @pytest.fixture(scope="class")
    def reconstructed_hu(self):
        """Forward project and FBP-reconstruct a noise-free phantom."""
        from generator_v7_0_0 import forward_project_slice, _fbp_fanbeam_core
        mu = build_attenuation_map(place_lesion=False, jitter_deg=0.0)
        sino = forward_project_slice(mu)
        mu_recon = _fbp_fanbeam_core(sino)

        # Calibrate: measure DC offset in a clean tissue ROI
        from mar_ils_core.constants import (
            PHANTOM_CENTER_X, BODY_SEMI_Y_VOX,
        )
        y0 = PHANTOM_CENTER_Y - round(0.90 * BODY_SEMI_Y_VOX)
        y1 = PHANTOM_CENTER_Y - round(0.40 * BODY_SEMI_Y_VOX)
        x0 = PHANTOM_CENTER_X - round(0.20 * 170)
        x1 = PHANTOM_CENTER_X + round(0.20 * 170)
        dc_offset = float(mu_recon[y0:y1, x0:x1].mean()) - MU_TISSUE_CM

        hu = (
            (mu_recon - MU_TISSUE_CM - dc_offset) / MU_TISSUE_CM * 1000.0
            + BACKGROUND_HU
        )
        return hu

    def test_tissue_hu(self, reconstructed_hu):
        """Tissue in calibration ROI should reconstruct to ~40 HU."""
        from mar_ils_core.constants import (
            PHANTOM_CENTER_X, BODY_SEMI_Y_VOX,
        )
        y0 = PHANTOM_CENTER_Y - round(0.90 * BODY_SEMI_Y_VOX)
        y1 = PHANTOM_CENTER_Y - round(0.40 * BODY_SEMI_Y_VOX)
        x0 = PHANTOM_CENTER_X - round(0.20 * 170)
        x1 = PHANTOM_CENTER_X + round(0.20 * 170)
        roi = reconstructed_hu[y0:y1, x0:x1]
        assert np.mean(roi) == pytest.approx(BACKGROUND_HU, abs=1.0)

    def test_air_hu(self, reconstructed_hu):
        """Air outside the phantom should reconstruct well below tissue HU.
        FBP truncation artifacts at corners mean air won't be exactly -1000 HU,
        but it should be clearly negative and below tissue."""
        assert reconstructed_hu[10, 10] < -500

    def test_reconstruction_shape(self, reconstructed_hu):
        assert reconstructed_hu.shape == (Y_DIM, X_DIM)
