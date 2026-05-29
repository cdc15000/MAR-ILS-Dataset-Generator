import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "algorithms" / "v7"))
import reference_li_mar_v7 as li  # noqa: E402
from mar_ils_core.phantom import build_metal_mask  # noqa: E402


class TestLinearInterpMetal:
    def test_fills_metal_columns_linearly(self):
        # one view (row), detector bins 0..9; bins 4,5 are metal (W<0.5)
        sino = np.array([[0.0, 1.0, 2.0, 3.0, 99.0, 99.0, 6.0, 7.0, 8.0, 9.0]])
        W = np.ones_like(sino)
        W[0, 4] = 0.0
        W[0, 5] = 0.0
        out = li.linear_interp_metal(sino, W)
        # bins 4,5 linearly interpolated between clean neighbours 3.0 and 6.0
        assert out[0, 4] == pytest.approx(4.0)
        assert out[0, 5] == pytest.approx(5.0)

    def test_metal_free_view_unchanged(self):
        sino = np.array([[0.0, 1.0, 2.0, 3.0]])
        W = np.ones_like(sino)  # all clean
        out = li.linear_interp_metal(sino, W)
        assert np.array_equal(out, sino)


class TestDetectMetalMask:
    def test_finds_high_hu_disc(self):
        hu = np.full((512, 512), 40.0)
        yy, xx = np.mgrid[0:512, 0:512]
        disc = (xx - 256) ** 2 + (yy - 256) ** 2 <= 10 ** 2
        hu[disc] = 3000.0
        mask = li.detect_metal_mask(hu)
        assert np.array_equal(mask, disc)

    def test_threshold_is_configurable(self):
        hu = np.full((4, 4), 1500.0)
        assert li.detect_metal_mask(hu, thresh=1000.0).all()
        assert not li.detect_metal_mask(hu, thresh=2000.0).any()


class TestMetalTraceWeights:
    def test_centred_rod_flags_central_detector_bins(self):
        yy, xx = np.mgrid[0:512, 0:512]
        metal_mask = build_metal_mask(yy, xx)  # centred 10-vox iron rod
        W = li.metal_trace_weights(metal_mask)
        assert W.shape == (720, 512)            # (N_ANGLES, N_DET)
        assert ((W == 0.0) | (W == 1.0)).all()  # binary
        # A centred rod projects to the central detector region for every view,
        # leaving the detector edges clean.
        assert (W < 0.5).any()                   # some metal-traced rays
        assert W[0, 0] == 1.0 and W[0, -1] == 1.0  # edges clean
        # Metal-traced bins cluster near detector centre.
        metal_cols = np.where(W[0] < 0.5)[0]
        assert abs(metal_cols.mean() - 256) < 60
