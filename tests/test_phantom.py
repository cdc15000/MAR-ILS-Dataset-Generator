"""
Tests for phantom construction.
"""

import numpy as np
import pytest

from mar_ils_core.constants import (
    X_DIM, Y_DIM,
    PHANTOM_CENTER_X, PHANTOM_CENTER_Y,
    MU_AIR_CM, MU_TISSUE_CM, MU_IRON_CM, MU_LESION_CM,
    LESION_CENTER_X,
)
from mar_ils_core.phantom import (
    build_body_mask,
    build_metal_mask,
    build_lesion_mask,
    build_attenuation_map,
)


class TestMaskShapes:
    def test_body_mask_shape(self, coordinate_grids):
        yy, xx = coordinate_grids
        mask = build_body_mask(yy, xx)
        assert mask.shape == (Y_DIM, X_DIM)
        assert mask.dtype == bool

    def test_metal_mask_shape(self, coordinate_grids):
        yy, xx = coordinate_grids
        mask = build_metal_mask(yy, xx)
        assert mask.shape == (Y_DIM, X_DIM)

    def test_lesion_mask_shape(self, coordinate_grids):
        yy, xx = coordinate_grids
        mask = build_lesion_mask(yy, xx)
        assert mask.shape == (Y_DIM, X_DIM)


class TestMaskProperties:
    def test_body_contains_center(self, coordinate_grids):
        yy, xx = coordinate_grids
        mask = build_body_mask(yy, xx)
        assert mask[PHANTOM_CENTER_Y, PHANTOM_CENTER_X]

    def test_body_excludes_corners(self, coordinate_grids):
        yy, xx = coordinate_grids
        mask = build_body_mask(yy, xx)
        assert not mask[0, 0]
        assert not mask[0, X_DIM - 1]
        assert not mask[Y_DIM - 1, 0]

    def test_metal_at_center(self, coordinate_grids):
        yy, xx = coordinate_grids
        mask = build_metal_mask(yy, xx)
        assert mask[PHANTOM_CENTER_Y, PHANTOM_CENTER_X]

    def test_metal_not_at_lesion(self, coordinate_grids):
        yy, xx = coordinate_grids
        metal = build_metal_mask(yy, xx)
        assert not metal[PHANTOM_CENTER_Y, LESION_CENTER_X]

    def test_lesion_at_expected_location(self, coordinate_grids):
        yy, xx = coordinate_grids
        mask = build_lesion_mask(yy, xx)
        assert mask[PHANTOM_CENTER_Y, LESION_CENTER_X]
        # Lesion should not be at phantom center (that's where metal is)
        assert not mask[PHANTOM_CENTER_Y, PHANTOM_CENTER_X]

    def test_metal_inside_body(self, coordinate_grids):
        yy, xx = coordinate_grids
        body = build_body_mask(yy, xx)
        metal = build_metal_mask(yy, xx)
        # All metal pixels should also be body pixels
        assert np.all(body[metal])

    def test_lesion_inside_body(self, coordinate_grids):
        yy, xx = coordinate_grids
        body = build_body_mask(yy, xx)
        lesion = build_lesion_mask(yy, xx)
        assert np.all(body[lesion])

    def test_lesion_does_not_overlap_metal(self, coordinate_grids):
        yy, xx = coordinate_grids
        metal = build_metal_mask(yy, xx)
        lesion = build_lesion_mask(yy, xx)
        assert not np.any(metal & lesion)


class TestAttenuationMap:
    def test_no_lesion(self):
        mu = build_attenuation_map(place_lesion=False)
        assert mu.shape == (Y_DIM, X_DIM)
        assert mu.dtype == np.float32

    def test_with_lesion(self):
        mu = build_attenuation_map(place_lesion=True)
        assert mu.shape == (Y_DIM, X_DIM)

    def test_tissue_value(self):
        mu = build_attenuation_map(place_lesion=False)
        # Check a point inside body but not metal
        assert mu[PHANTOM_CENTER_Y, PHANTOM_CENTER_X + 50] == pytest.approx(
            MU_TISSUE_CM, rel=1e-4
        )

    def test_metal_value(self):
        mu = build_attenuation_map(place_lesion=False)
        assert mu[PHANTOM_CENTER_Y, PHANTOM_CENTER_X] == pytest.approx(
            MU_IRON_CM, rel=1e-4
        )

    def test_air_value(self):
        mu = build_attenuation_map(place_lesion=False)
        assert mu[0, 0] == pytest.approx(MU_AIR_CM, rel=1e-4)

    def test_lesion_value(self):
        mu = build_attenuation_map(place_lesion=True)
        assert mu[PHANTOM_CENTER_Y, LESION_CENTER_X] == pytest.approx(
            MU_LESION_CM, rel=1e-4
        )

    def test_lesion_absent_when_not_placed(self):
        mu = build_attenuation_map(place_lesion=False)
        # Without lesion, this should be tissue
        assert mu[PHANTOM_CENTER_Y, LESION_CENTER_X] == pytest.approx(
            MU_TISSUE_CM, rel=1e-4
        )

    def test_jitter_preserves_shape(self):
        mu = build_attenuation_map(place_lesion=False, jitter_deg=5.0)
        assert mu.shape == (Y_DIM, X_DIM)
        assert mu.dtype == np.float32
