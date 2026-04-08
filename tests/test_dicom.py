"""
Tests for DICOM utilities and CP-2575 compliance.
"""


import numpy as np
import pydicom
import pytest

from mar_ils_core.constants import (
    X_DIM, Y_DIM, VOXEL_MM, METAL_HU,
    TAG_MAR_SEQ, TAG_MAR_APPLIED,
)
from mar_ils_core.dicom_utils import write_dicom_slice, add_mar_macro


class TestDICOMWrite:
    @pytest.fixture
    def dicom_path(self, tmp_path):
        """Write a test DICOM and return its path."""
        hu = np.full((Y_DIM, X_DIM), 40.0, dtype=np.float32)
        metal_mask = np.zeros((Y_DIM, X_DIM), dtype=bool)
        metal_mask[256, 256] = True

        from pydicom.uid import generate_uid
        write_dicom_slice(
            hu, z=0,
            output_dir=tmp_path,
            realization_idx=0,
            condition_label="test",
            study_uid=generate_uid(),
            series_uid=generate_uid(),
            metal_mask=metal_mask,
        )
        return tmp_path / "slice_0001.dcm"

    def test_file_created(self, dicom_path):
        assert dicom_path.exists()

    def test_readable(self, dicom_path):
        ds = pydicom.dcmread(str(dicom_path))
        assert ds.Rows == Y_DIM
        assert ds.Columns == X_DIM

    def test_pixel_spacing(self, dicom_path):
        ds = pydicom.dcmread(str(dicom_path))
        assert float(ds.PixelSpacing[0]) == VOXEL_MM

    def test_rescale_slope_intercept(self, dicom_path):
        ds = pydicom.dcmread(str(dicom_path))
        assert float(ds.RescaleSlope) == 1
        assert float(ds.RescaleIntercept) == 0

    def test_metal_hardset(self, dicom_path):
        ds = pydicom.dcmread(str(dicom_path))
        pixels = ds.pixel_array
        # Metal pixel at (256, 256) should be 3000
        assert pixels[256, 256] == int(METAL_HU)

    def test_tissue_value(self, dicom_path):
        ds = pydicom.dcmread(str(dicom_path))
        pixels = ds.pixel_array
        # Non-metal pixel should be 40 (the input HU)
        assert pixels[100, 100] == 40


class TestCP2575:
    def test_mar_macro_present(self, dicom_path):
        ds = pydicom.dcmread(str(dicom_path))
        assert TAG_MAR_SEQ in ds

    def test_mar_applied_no(self, dicom_path):
        ds = pydicom.dcmread(str(dicom_path))
        mar_seq = ds[TAG_MAR_SEQ]
        mar_applied = mar_seq.value[0][TAG_MAR_APPLIED].value
        assert mar_applied == "NO"

    def test_add_mar_macro_standalone(self):
        ds = pydicom.Dataset()
        add_mar_macro(ds, applied="YES")
        assert TAG_MAR_SEQ in ds
        assert ds[TAG_MAR_SEQ].value[0][TAG_MAR_APPLIED].value == "YES"

    @pytest.fixture
    def dicom_path(self, tmp_path):
        hu = np.full((Y_DIM, X_DIM), 40.0, dtype=np.float32)
        metal_mask = np.zeros((Y_DIM, X_DIM), dtype=bool)
        metal_mask[256, 256] = True
        from pydicom.uid import generate_uid
        write_dicom_slice(
            hu, z=0,
            output_dir=tmp_path,
            realization_idx=0,
            condition_label="test",
            study_uid=generate_uid(),
            series_uid=generate_uid(),
            metal_mask=metal_mask,
        )
        return tmp_path / "slice_0001.dcm"
