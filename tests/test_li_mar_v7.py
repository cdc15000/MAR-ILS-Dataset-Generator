import json as _json
import sys
from pathlib import Path

import h5py
import numpy as np
import pydicom
import pytest
from pydicom.uid import generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "algorithms" / "v7"))
import reference_li_mar_v7 as li  # noqa: E402
from mar_ils_core.phantom import build_metal_mask, build_attenuation_map  # noqa: E402
from mar_ils_core.dicom_utils import write_dicom_slice  # noqa: E402


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


class TestLiMarSlice:
    @pytest.fixture(scope="class")
    def slice_inputs(self):
        from generator_v7_0_0 import forward_project_slice, fbp_reconstruct_slice
        mu = build_attenuation_map(place_lesion=True, jitter_deg=0.0)
        sino = forward_project_slice(mu).astype(np.float64)
        nomar_hu = fbp_reconstruct_slice(sino, dc_offset_cm=0.0)
        return sino, nomar_hu

    def test_returns_finite_corrected_slice_and_mask(self, slice_inputs):
        sino, nomar_hu = slice_inputs
        hu_corr, metal_mask = li.li_mar_slice(sino, nomar_hu, dc_offset_cm=0.0)
        assert hu_corr.shape == (512, 512)
        assert np.isfinite(hu_corr).all()
        assert metal_mask.dtype == bool
        assert metal_mask.sum() > 0          # rod detected
        assert metal_mask.sum() < 2000       # but not the whole image

    def test_metal_trace_is_replaced(self, slice_inputs):
        # After LI, the metal region reconstructs to ~tissue, far below the
        # ~10000 HU raw metal in the noMAR image. Confirms the trace was filled.
        sino, nomar_hu = slice_inputs
        hu_corr, metal_mask = li.li_mar_slice(sino, nomar_hu, dc_offset_cm=0.0)
        assert hu_corr[metal_mask].mean() < nomar_hu[metal_mask].mean()


class TestLoadDcOffset:
    def test_reads_value(self, tmp_path):
        # Mirrors the generator's nested structure: noise_model.dc_offset_cm.
        (tmp_path / "generator_provenance.json").write_text(
            _json.dumps({"noise_model": {"dc_offset_cm": -0.029}})
        )
        assert li.load_dc_offset(tmp_path) == pytest.approx(-0.029)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            li.load_dc_offset(tmp_path)

    def test_missing_key_raises(self, tmp_path):
        # provenance present but without noise_model.dc_offset_cm
        (tmp_path / "generator_provenance.json").write_text(
            _json.dumps({"noise_model": {"I0_calibrated": 310853}})
        )
        with pytest.raises(KeyError):
            li.load_dc_offset(tmp_path)


class TestDiscoverRealizations:
    def test_counts_h5_files(self, tmp_path):
        d = tmp_path / "sinograms" / "LP"
        d.mkdir(parents=True)
        for n in (1, 2, 3):
            (d / f"realization_{n:03d}.h5").touch()
        assert li.discover_realizations(tmp_path, "LP") == 3

    def test_zero_when_absent(self, tmp_path):
        assert li.discover_realizations(tmp_path, "LP") == 0


def _build_tiny_dataset(root: Path):
    """One-slice (slice_index 0) LP realization: sinogram H5 + noMAR DICOM."""
    from generator_v7_0_0 import forward_project_slice, fbp_reconstruct_slice
    mu = build_attenuation_map(place_lesion=True, jitter_deg=0.0)
    sino = forward_project_slice(mu).astype(np.float32)        # (720, 512)
    nomar_hu = fbp_reconstruct_slice(sino, dc_offset_cm=0.0)   # (512, 512)

    sdir = root / "sinograms" / "LP"
    sdir.mkdir(parents=True)
    with h5py.File(str(sdir / "realization_001.h5"), "w") as f:
        f.create_dataset("line_integrals", data=sino[np.newaxis, :, :])  # (1,720,512)

    ndir = root / "noMAR_recon" / "LP" / "realization_001"
    yy, xx = np.mgrid[0:512, 0:512]
    write_dicom_slice(
        nomar_hu, 0, output_dir=ndir, realization_idx=0, condition_label="LP",
        study_uid=generate_uid(), series_uid=generate_uid(),
        metal_mask=build_metal_mask(yy, xx),
    )
    (root / "generator_provenance.json").write_text(
        _json.dumps({"noise_model": {"dc_offset_cm": 0.0}})
    )


class TestProcessRealization:
    def test_writes_readable_corrected_dicom(self, tmp_path):
        _build_tiny_dataset(tmp_path)
        out = tmp_path / "li_mar_recon"
        li.process_realization(
            tmp_path, out, "LP", 0, dc_offset_cm=0.0, slice_index=0,
        )
        dcm = out / "LP" / "realization_001" / "slice_0001.dcm"
        assert dcm.exists()
        ds = pydicom.dcmread(str(dcm))
        hu = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        assert hu.shape == (512, 512)
        # metal hard-set to 3000 HU by the writer. Check the rod *core* (always
        # detected); the geometric boundary can blur below the 2000 HU threshold.
        yy, xx = np.mgrid[0:512, 0:512]
        core = (xx - 256) ** 2 + (yy - 256) ** 2 <= 5 ** 2
        assert hu[256, 256] == pytest.approx(3000.0)
        assert np.allclose(hu[core], 3000.0)

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            li.process_realization(tmp_path, tmp_path / "o", "LP", 0, 0.0, slice_index=0)

    def test_wrong_sinogram_shape_raises(self, tmp_path):
        # Build a valid tiny dataset, then overwrite the sinogram with a
        # geometry-mismatched one (wrong detector count).
        _build_tiny_dataset(tmp_path)
        sdir = tmp_path / "sinograms" / "LP"
        with h5py.File(str(sdir / "realization_001.h5"), "w") as f:
            f.create_dataset("line_integrals", data=np.zeros((1, 720, 256), np.float32))
        with pytest.raises(ValueError, match="geometry mismatch"):
            li.process_realization(
                tmp_path, tmp_path / "o", "LP", 0, 0.0, slice_index=0,
            )


class TestMain:
    def test_runs_over_tiny_dataset(self, tmp_path):
        _build_tiny_dataset(tmp_path)          # LP only
        out = tmp_path / "li_mar_recon"
        li.main(
            ["--dataset-dir", str(tmp_path), "--output-dir", str(out),
             "--realizations", "1"],
            slice_index=0,
        )
        assert (out / "LP" / "realization_001" / "slice_0001.dcm").exists()

    def test_dc_offset_override_skips_provenance(self, tmp_path):
        _build_tiny_dataset(tmp_path)
        (tmp_path / "generator_provenance.json").unlink()  # force reliance on override
        out = tmp_path / "li_mar_recon"
        li.main(
            ["--dataset-dir", str(tmp_path), "--output-dir", str(out),
             "--realizations", "1", "--dc-offset-cm", "0.0"],
            slice_index=0,
        )
        assert (out / "LP" / "realization_001" / "slice_0001.dcm").exists()
