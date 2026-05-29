"""
DICOM utilities for MAR ILS framework.

Includes CP-2575 Metal Artifact Reduction Macro injection and
standard CT DICOM writing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid

from mar_ils_core.constants import (
    X_DIM, Y_DIM, VOXEL_MM, METAL_HU,
    TAG_MAR_SEQ, TAG_MAR_APPLIED,
)


def add_mar_macro(ds: Dataset, applied: str = "NO") -> None:
    """Inject DICOM 2026b CP-2575 Metal Artifact Reduction Macro."""
    mar_item = Dataset()
    mar_item.add_new(TAG_MAR_APPLIED, 'CS', applied)
    ds.add_new(TAG_MAR_SEQ, 'SQ', [mar_item])


def write_dicom_slice(
    hu: np.ndarray,
    z: int,
    *,
    output_dir: Path,
    realization_idx: int,
    condition_label: str,
    study_uid: str,
    series_uid: str,
    metal_mask: np.ndarray,
    dataset_version: str = "v7.0.0",
    standard_ref: str = "ASTM-WKXXXXX-Rev04",
) -> None:
    """Write one 2D HU array as DICOM. Metal hard-set to 3000 HU (§A1.3(d,f))."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hu = hu.copy()
    hu[metal_mask] = METAL_HU
    hu_clipped = np.clip(hu, -1024, 32767).astype(np.int16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'

    # Encoding (Explicit VR Little Endian) is derived from
    # file_meta.TransferSyntaxUID at save time via enforce_file_format=True.
    ds = FileDataset(str(output_dir), {}, file_meta=file_meta, preamble=b'\0' * 128)

    now = datetime.now(timezone.utc)
    ds.ContentDate = now.strftime('%Y%m%d')
    ds.ContentTime = now.strftime('%H%M%S.%f')
    ds.Modality = 'CT'
    ds.Manufacturer = 'ASTM WKXXXXX ILS'
    ds.StudyDescription = f'MAR ILS {dataset_version}'
    ds.SeriesDescription = condition_label
    ds.ProtocolName = f'{standard_ref}-{condition_label}'
    ds.ConvolutionKernel = 'RAM-LAK'
    ds.KVP = '60'
    ds.ExposureTime = '0'
    ds.SliceThickness = str(VOXEL_MM)
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Rows = Y_DIM
    ds.Columns = X_DIM
    ds.PixelSpacing = [VOXEL_MM, VOXEL_MM]
    ds.ImagePositionPatient = [0.0, 0.0, float(z * VOXEL_MM)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SliceLocation = float(z * VOXEL_MM)
    ds.InstanceNumber = z + 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.PixelData = hu_clipped.tobytes()

    add_mar_macro(ds)

    ds.save_as(str(output_dir / f'slice_{z + 1:04d}.dcm'), enforce_file_format=True)
