"""Shared fixtures for MAR ILS tests."""

import numpy as np
import pytest

from mar_ils_core.constants import X_DIM, Y_DIM


@pytest.fixture
def coordinate_grids():
    """Standard (yy, xx) meshgrids for 512x512 volume."""
    return np.mgrid[0:Y_DIM, 0:X_DIM]


# The v6 to v8 scripts live in legacy/; the older tests import them by name.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "legacy"))
