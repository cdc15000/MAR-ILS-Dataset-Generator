"""
Tests for the v8.0.0 seeding scheme and the null-signal reference radius.
"""

import numpy as np

import generator_v8_0_0 as gen
from mar_ils_core.constants import LESION_CENTER_X, LESION_RADIUS_VOX, PHANTOM_CENTER_X


def _draw(*key):
    return gen.slice_rng(*key).standard_normal(64)


class TestSliceRng:
    def test_lp_and_la_streams_differ(self):
        assert not np.array_equal(_draw(0, "LP", 3, 128), _draw(0, "LA", 3, 128))

    def test_slices_differ(self):
        assert not np.array_equal(_draw(0, "LA", 3, 127), _draw(0, "LA", 3, 128))

    def test_replicates_differ(self):
        assert not np.array_equal(_draw(0, "LP", 3, 128), _draw(1, "LP", 3, 128))

    def test_reproducible(self):
        np.testing.assert_array_equal(_draw(2, "LP", 5, 128), _draw(2, "LP", 5, 128))


class TestNullSignalRadius:
    def test_trace_covers_lesion(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "algorithms" / "v7"))
        from reference_null_signal_v7 import NULL_RADIUS_VOX
        assert NULL_RADIUS_VOX == (LESION_CENTER_X - PHANTOM_CENTER_X) + LESION_RADIUS_VOX + 1
        assert NULL_RADIUS_VOX == 31
