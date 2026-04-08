"""
Tests for the physics-based noise model.
"""

import numpy as np
import pytest

from mar_ils_core.noise import apply_noise


class TestNoiseModel:
    def test_output_shape(self):
        sino = np.ones((10, 20), dtype=np.float32)
        rng = np.random.default_rng(42)
        out = apply_noise(sino, I0=1e5, rng=rng)
        assert out.shape == (10, 20)
        assert out.dtype == np.float32

    def test_deterministic_with_seed(self):
        sino = np.ones((10, 20), dtype=np.float32) * 0.5
        out1 = apply_noise(sino, I0=1e5, rng=np.random.default_rng(123))
        out2 = apply_noise(sino, I0=1e5, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        sino = np.ones((10, 20), dtype=np.float32) * 0.5
        out1 = apply_noise(sino, I0=1e5, rng=np.random.default_rng(1))
        out2 = apply_noise(sino, I0=1e5, rng=np.random.default_rng(2))
        assert not np.array_equal(out1, out2)

    def test_mean_close_to_clean(self):
        """With high I0, noisy sinogram mean should be close to clean.
        Note: scatter fraction (5%) biases the mean downward from the
        clean value, so we use a wider tolerance."""
        sino = np.full((100, 100), 0.5, dtype=np.float32)
        rng = np.random.default_rng(42)
        out = apply_noise(sino, I0=1e7, rng=rng)
        # Scatter shifts the mean; check it's in the right ballpark
        assert np.mean(out) == pytest.approx(0.5, abs=0.1)

    def test_noise_increases_with_lower_I0(self):
        """Lower I0 should produce more noise (higher std)."""
        sino = np.full((200, 200), 0.5, dtype=np.float32)
        out_high = apply_noise(sino, I0=1e6, rng=np.random.default_rng(42))
        out_low = apply_noise(sino, I0=1e3, rng=np.random.default_rng(42))
        assert np.std(out_low) > np.std(out_high)

    def test_no_negative_infinity(self):
        """Output should not contain -inf (from log of negative)."""
        sino = np.full((50, 50), 2.0, dtype=np.float32)  # high attenuation
        rng = np.random.default_rng(42)
        out = apply_noise(sino, I0=1e4, rng=rng)
        assert np.all(np.isfinite(out))
