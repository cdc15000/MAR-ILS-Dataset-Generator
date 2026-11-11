"""
DICOM CT images for the v9 dataset (ingestion spec §6).

One single-frame 512 × 512 image per realization: HU per Rev 06 §A1.1(h),
rounded to whole HU, signed 16 bits, Rescale Slope 1 and Intercept 0, with the
DICOM 2026b CP-2575 Metal Artifact Reduction Macro.

Nothing written here depends on the realization's class, level, pool or
index: one Study and one Series UID per dataset, one Content Date and Time
for all files, Instance Number 1, no Series Description or Protocol Name.
The master files can therefore be distributed under coded names unchanged.
"""
from __future__ import annotations

import secrets
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

IMAGE_PX = 512
PIXEL_MM = 300.0 / IMAGE_PX
TAG_MAR_SEQUENCE = 0x00189390
TAG_MAR_APPLIED = 0x00189391


def fixed_length_uid() -> str:
    """A 2.25 UID with a random 38-digit integer (below 2^128), always 43 characters,
    so every file of a dataset has the same size whatever its UID."""
    return "2.25." + str(10 ** 37 + secrets.randbelow(9 * 10 ** 37))


def hu_to_pixels(hu: np.ndarray) -> np.ndarray:
    """Stored values: HU rounded to the nearest integer, clipped to signed 16 bits."""
    return np.clip(np.rint(hu), -32768, 32767).astype("<i2")


def write_ct(hu: np.ndarray, path: Path, *, study_uid: str, series_uid: str,
             content_date: str, content_time: str, mar_applied: bool) -> None:
    pixels = hu_to_pixels(hu)
    if pixels.shape != (IMAGE_PX, IMAGE_PX):
        raise ValueError(f"image shape {pixels.shape}, expected (512, 512)")

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = fixed_length_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.ContentDate = content_date
    ds.ContentTime = content_time
    ds.Modality = "CT"
    ds.InstanceNumber = 1
    ds.ImagePositionPatient = [-150.0, -150.0, 0.0]     # center of pixel (0, 0)
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [PIXEL_MM, PIXEL_MM]
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.PixelData = pixels.tobytes()

    item = Dataset()
    item.add_new(TAG_MAR_APPLIED, "CS", "YES" if mar_applied else "NO")
    ds.add_new(TAG_MAR_SEQUENCE, "SQ", [item])

    ds.save_as(str(path), enforce_file_format=True)


def read_hu(path: Path) -> np.ndarray:
    import pydicom
    ds = pydicom.dcmread(str(path))
    return ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
