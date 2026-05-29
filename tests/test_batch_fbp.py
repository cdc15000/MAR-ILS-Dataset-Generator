"""
Bit-exactness regression test for batch vs. per-slice fan-beam FBP.

CLAUDE.md documents a normative claim (2026-04-07):

    "Bit-exactness: Batch FBP matches per-slice FBP within float32 precision
     (max 0.0005 HU difference)."

The generator uses the batch backprojection kernel (`_backproject_batch_jit`,
geometry computed once for all slices) in production workers, but the per-slice
kernel (`_fbp_fanbeam_core`) is the analytic reference exercised by
`test_roundtrip_fbp`. If the two ever diverge, the locked baseline AUC could
shift without any test noticing. This test guards that equivalence.

The batch kernel is Numba-only, so the whole module is skipped when numba is
not installed (the CI `numba` job installs it; the default `test` job does not).
"""

import numpy as np
import pytest

import generator_v7_0_0 as g
from mar_ils_core.phantom import build_attenuation_map, build_metal_mask

pytestmark = pytest.mark.skipif(
    not g._HAS_NUMBA, reason="batch backprojection kernel requires numba"
)

# Tolerance from the CLAUDE.md bit-exactness claim.
MAX_HU_DIFF = 0.0005


def _mu_to_hu(mu_recon: np.ndarray) -> np.ndarray:
    """Convert raw reconstructed mu (cm^-1) to HU (no DC offset; relative only)."""
    return (
        (mu_recon - g.MU_TISSUE_CM) / g.MU_TISSUE_CM * 1000.0 + g.BACKGROUND_HU
    ).astype(np.float32)


def _batch_reconstruct(sinograms: np.ndarray) -> np.ndarray:
    """Reconstruct a (Z, N_ANGLES, N_DET) stack via the batch kernel.

    Mirrors the filter/transpose/scale steps in
    `generator_v7_0_0.reconstruct_realization_batch` so the comparison
    isolates the backprojection kernel (the only place geometry handling
    differs between batch and per-slice paths).
    """
    n_slices, n_proj, n_det = sinograms.shape

    weighted = sinograms.astype(np.float64) * g.COS_DET_FAN[np.newaxis, np.newaxis, :]
    ramp = np.abs(np.fft.rfftfreq(n_det))
    filtered = np.fft.irfft(
        np.fft.rfft(weighted, axis=2) * ramp[np.newaxis, np.newaxis, :],
        n=n_det, axis=2,
    )
    filtered_t = np.ascontiguousarray(filtered.transpose(1, 2, 0))

    scale = np.pi / n_proj / (g.SID_CM * g.DELTA_GAMMA_RAD)
    recon_yxz = g._backproject_batch_jit(
        filtered_t, g.ANGLES_RAD, g.SID_VOX, g.DELTA_GAMMA_RAD,
        float(g.DET_FAN_ANGLES_RAD[0]), g.X_DIM / 2.0, g.Y_DIM / 2.0,
        g.X_DIM, g.Y_DIM, n_slices,
    )
    return recon_yxz * scale  # (Y, X, Z) mu map


class TestBatchVsPerSliceFBP:
    """Batch backprojection must match per-slice within float32 precision."""

    @pytest.fixture(scope="class")
    def slices(self):
        """A small noise-free stack: lesion-present and lesion-absent phantoms."""
        mu_lp = build_attenuation_map(place_lesion=True, jitter_deg=0.0)
        mu_la = build_attenuation_map(place_lesion=False, jitter_deg=0.0)
        sinos = np.stack([
            g.forward_project_slice(mu_lp),
            g.forward_project_slice(mu_la),
            g.forward_project_slice(mu_lp),
        ]).astype(np.float32)
        return sinos

    def test_bit_exact_recon(self, slices):
        per_slice = np.stack([
            _mu_to_hu(g._fbp_fanbeam_core(slices[z].astype(np.float64)))
            for z in range(slices.shape[0])
        ])

        batch_mu = _batch_reconstruct(slices)
        batch = np.stack([
            _mu_to_hu(batch_mu[:, :, z]) for z in range(slices.shape[0])
        ])

        # Exclude the metal core: production hard-sets every metal-mask pixel to
        # exactly METAL_HU in write_dicom_slice, so float32 accumulation-order
        # differences in the raw (~10000 HU) rod reconstruction never reach the
        # DICOM output. The claim applies to the diagnostic field of view.
        yy, xx = np.mgrid[0:g.Y_DIM, 0:g.X_DIM]
        non_metal = ~build_metal_mask(yy, xx)

        diff = np.abs(per_slice - batch)[:, non_metal]
        max_diff = float(np.max(diff))
        assert max_diff < MAX_HU_DIFF, (
            f"batch vs per-slice FBP differ by {max_diff:.6f} HU "
            f"(claim: < {MAX_HU_DIFF} HU)"
        )
