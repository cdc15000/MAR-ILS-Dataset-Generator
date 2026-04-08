"""
Tests for normative constants — guards the locked metrology baseline.
"""

import numpy as np
import pytest

from mar_ils_core.constants import (
    # Volume
    X_DIM, Y_DIM, Z_DIM, VOXEL_MM, VOXEL_CM,
    # Fan-beam
    SID_MM, SDD_MM, N_ANGLES, N_DET,
    GAMMA_MAX_RAD, DELTA_GAMMA_RAD,
    DET_FAN_ANGLES_RAD, COS_DET_FAN,
    ANGLES_DEG, ANGLES_RAD,
    # Physics
    MU_AIR_CM, MU_TISSUE_CM, MU_IRON_CM,
    BACKGROUND_HU, METAL_HU,
    # Phantom
    BODY_SEMI_X_VOX, BODY_SEMI_Y_VOX,
    METAL_RADIUS_VOX, LESION_RADIUS_VOX,
    LESION_CENTER_X, LESION_SLICE_INDEX,
    # Lesion
    LESION_DELTA_HU, MU_LESION_CM,
    # Noise
    NUM_REALIZATIONS_DEFAULT, BASE_SEED,
    # CHO
    ROI_SIZE, ROI_CENTER_X, ROI_CENTER_Y,
    NUM_CHANNELS, CHANNEL_WIDTH_A,
    AUC_TOLERANCE,
)


class TestLockedBaselines:
    """These values are locked per CLAUDE.md metrology baseline (2026-04-07)."""

    def test_mu_lesion_cm(self):
        expected = MU_TISSUE_CM * (1 + 12 / 1000)
        assert MU_LESION_CM == pytest.approx(expected, rel=1e-10)

    def test_num_realizations(self):
        assert NUM_REALIZATIONS_DEFAULT == 40

    def test_internal_noise_sigma_default(self):
        # Normative default is sigma=15, specified via CLI, not a constant
        # but AUC_TOLERANCE is locked
        assert AUC_TOLERANCE == 0.005

    def test_lesion_contrast(self):
        assert LESION_DELTA_HU == 12.0

    def test_tissue_hu(self):
        assert BACKGROUND_HU == 40.0

    def test_fan_beam_geometry(self):
        assert SID_MM == 570.0
        assert SDD_MM == 1040.0

    def test_base_seed(self):
        assert BASE_SEED == 20260314


class TestVolumeGeometry:
    def test_dimensions(self):
        assert X_DIM == 512
        assert Y_DIM == 512
        assert Z_DIM == 256

    def test_voxel_size(self):
        assert VOXEL_MM == 0.5
        assert VOXEL_CM == pytest.approx(0.05)


class TestFanBeamGeometry:
    def test_n_angles(self):
        assert N_ANGLES == 720

    def test_n_det(self):
        assert N_DET == 512

    def test_angles_array_shape(self):
        assert ANGLES_DEG.shape == (720,)
        assert ANGLES_RAD.shape == (720,)

    def test_angles_range(self):
        assert ANGLES_DEG[0] == pytest.approx(0.0)
        assert ANGLES_DEG[-1] < 360.0

    def test_det_fan_angles_shape(self):
        assert DET_FAN_ANGLES_RAD.shape == (512,)
        assert COS_DET_FAN.shape == (512,)

    def test_det_fan_angles_symmetric(self):
        assert DET_FAN_ANGLES_RAD[0] == pytest.approx(-DET_FAN_ANGLES_RAD[-1], abs=1e-10)

    def test_gamma_max_positive(self):
        assert GAMMA_MAX_RAD > 0
        assert np.degrees(GAMMA_MAX_RAD) == pytest.approx(12.98, abs=0.1)


class TestPhysicsConstants:
    def test_mu_ordering(self):
        assert MU_AIR_CM < MU_TISSUE_CM < MU_IRON_CM

    def test_mu_lesion_slightly_above_tissue(self):
        assert MU_LESION_CM > MU_TISSUE_CM
        assert (MU_LESION_CM - MU_TISSUE_CM) / MU_TISSUE_CM * 1000 == pytest.approx(12.0)

    def test_metal_hu(self):
        assert METAL_HU == 3000.0


class TestPhantomGeometry:
    def test_body_semi_axes(self):
        assert BODY_SEMI_X_VOX == 170
        assert BODY_SEMI_Y_VOX == 120

    def test_metal_radius(self):
        assert METAL_RADIUS_VOX == 10

    def test_lesion_radius(self):
        assert LESION_RADIUS_VOX == 5

    def test_lesion_center(self):
        assert LESION_CENTER_X == 281

    def test_lesion_slice(self):
        assert LESION_SLICE_INDEX == 128


class TestCHOConstants:
    def test_roi(self):
        assert ROI_SIZE == 121
        assert ROI_CENTER_X == 281
        assert ROI_CENTER_Y == 256

    def test_channels(self):
        assert NUM_CHANNELS == 10
        assert CHANNEL_WIDTH_A == 7.5


class TestDerivedConsistency:
    """Verify derived constants are consistent with their inputs."""

    def test_voxel_cm_from_mm(self):
        assert VOXEL_CM == VOXEL_MM / 10.0

    def test_delta_gamma_from_gamma_max(self):
        assert DELTA_GAMMA_RAD == pytest.approx(2.0 * GAMMA_MAX_RAD / N_DET)

    def test_lesion_center_geometry(self):
        # Lesion is at 256 + metal_radius + gap + lesion_radius
        # = 256 + 10 + 10 + 5 = 281
        assert LESION_CENTER_X == 281
