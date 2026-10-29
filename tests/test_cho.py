"""
Tests for CHO analysis components.
"""

import numpy as np
import pytest

from mar_ils_core.constants import (
    NUM_CHANNELS,
    ROI_SIZE,
)


class TestLGChannels:
    @pytest.fixture(scope="class")
    def channels(self):
        from run_cho_analysis_v7_0 import calculate_lg_channels
        return calculate_lg_channels()

    def test_shape(self, channels):
        assert channels.shape == (NUM_CHANNELS, ROI_SIZE ** 2)

    def test_normalized(self, channels):
        """Each channel should be L2-normalised."""
        for n in range(NUM_CHANNELS):
            norm = np.linalg.norm(channels[n])
            assert norm == pytest.approx(1.0, abs=1e-10)

    def test_first_channel_positive_center(self, channels):
        """L_0 Gaussian should be positive at center."""
        center = ROI_SIZE * (ROI_SIZE // 2) + ROI_SIZE // 2
        assert channels[0, center] > 0


class TestMWAUC:
    def test_perfect_separation(self):
        from run_cho_analysis_v7_0 import mw_auc
        lp = np.array([10.0, 11.0, 12.0])
        la = np.array([1.0, 2.0, 3.0])
        assert mw_auc(lp, la) == 1.0

    def test_perfect_overlap(self):
        from run_cho_analysis_v7_0 import mw_auc
        scores = np.array([1.0, 2.0, 3.0])
        assert mw_auc(scores, scores) == 0.5

    def test_reversed_gives_zero(self):
        from run_cho_analysis_v7_0 import mw_auc
        lp = np.array([1.0, 2.0, 3.0])
        la = np.array([10.0, 11.0, 12.0])
        assert mw_auc(lp, la) == 0.0

    def test_range(self):
        from run_cho_analysis_v7_0 import mw_auc
        rng = np.random.default_rng(42)
        lp = rng.normal(1, 1, 50)
        la = rng.normal(0, 1, 50)
        auc = mw_auc(lp, la)
        assert 0.0 <= auc <= 1.0


class TestCHOPerformance:
    def test_self_test_delta_auc_zero(self):
        """When LP and LA features are identical, ΔAUC should be 0."""
        from run_cho_analysis_v7_0 import compute_cho_performance
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (20, NUM_CHANNELS))
        result = compute_cho_performance(
            features, features, 20,
            n_boot=100,
            internal_noise_sigma=15.0,
        )
        assert result["AUC"] == pytest.approx(0.5, abs=0.15)


class TestRefitBootstrapV8:
    """run_cho_analysis_v8_0: LOO point estimate equals v7; refit bootstrap."""

    @pytest.fixture(scope="class")
    def feats(self):
        rng = np.random.default_rng(0)
        la = rng.normal(0.0, 30.0, size=(40, NUM_CHANNELS))
        lp = rng.normal(0.0, 30.0, size=(40, NUM_CHANNELS))
        lp[:, 0] += 15.0
        return lp, la

    def test_loo_matches_v7(self, feats):
        from run_cho_analysis_v7_0 import compute_cho_performance
        from run_cho_analysis_v8_0 import loo_auc
        lp, la = feats
        v7 = compute_cho_performance(lp, la, 40, n_boot=10, internal_noise_sigma=15)
        assert loo_auc(lp, la, 225.0) == v7["AUC"]

    def test_heldout_copies_excluded(self, feats):
        """A resample made of one index repeated has no training data left."""
        from run_cho_analysis_v8_0 import loo_auc
        lp, la = feats
        idx = np.array([3, 3, 7, 7, 7, 9])
        auc = loo_auc(lp, la, 225.0, idx)
        assert 0.0 <= auc <= 1.0

    def test_self_test_delta_zero_every_draw(self, feats):
        from run_cho_analysis_v8_0 import refit_bootstrap
        lp, la = feats
        b = refit_bootstrap({"noMAR": (lp, la), "MAR": (lp, la)}, 225.0, n_boot=20)
        np.testing.assert_array_equal(b["MAR"] - b["noMAR"], 0.0)

    def test_normal_interval_contains_estimate(self):
        from run_cho_analysis_v8_0 import _ci
        lo, hi = _ci(0.8, np.array([0.6, 0.7, 0.65, 0.72]))
        assert lo < 0.8 < hi
