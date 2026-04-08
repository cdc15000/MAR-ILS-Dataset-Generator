"""Shared fixtures for MAR ILS tests."""

import numpy as np
import pytest

from mar_ils_core.constants import X_DIM, Y_DIM


@pytest.fixture
def coordinate_grids():
    """Standard (yy, xx) meshgrids for 512x512 volume."""
    return np.mgrid[0:Y_DIM, 0:X_DIM]
