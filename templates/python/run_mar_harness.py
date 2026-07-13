#!/usr/bin/env python3
"""
MAR ILS Lab Harness — Python
ASTM WKXXXXX Rev 05

Turnkey harness for applying a lab's MAR algorithm to the ILS sinogram
dataset.  The lab implements one function — apply_mar() — in apply_mar.py.
This harness handles all HDF5 reading, DICOM writing, directory layout,
and CP-2575 MAR macro injection.

Usage:
    python run_mar_harness.py \
        --dataset-dir ./astm_reference_dataset \
        --output-dir  ./mar_recon
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid

# ── Lab-supplied MAR implementation ──────────────────────────────────────────
from apply_mar import apply_mar


# ── Geometry container ───────────────────────────────────────────────────────

@dataclass
class FanBeamGeometry:
    """All acquisition geometry needed to reconstruct from fan-beam sinograms."""
    n_slices: int          # 256
    n_angles: int          # 720
    n_det: int             # 512
    voxel_mm: float        # 0.5
    sid_mm: float          # 570.0  (source-to-isocenter)
    sdd_mm: float          # 1040.0 (source-to-detector)
    gamma_max_deg: float   # ~12.97
    delta_gamma_deg: float # ~0.0507
    angles_deg: np.ndarray        # (720,) — source rotation angles
    det_fan_angles_deg: np.ndarray  # (512,) — fan angle per detector element


# ── HDF5 reader ─────────────────────────────────────────────────────────────

def read_sinogram(h5_path: Path) -> tuple[np.ndarray, FanBeamGeometry]:
    """Read sinogram and geometry from an ILS HDF5 file.

    Returns:
        sinogram: float32 array of shape (n_slices, n_angles, n_det)
        geometry: FanBeamGeometry with all reconstruction parameters
    """
    with h5py.File(str(h5_path), "r") as f:
        sinogram = f["line_integrals"][:].astype(np.float32)
        g = f["geometry"].attrs
        geo = FanBeamGeometry(
            n_slices=int(g["n_slices"]),
            n_angles=int(g["n_angles"]),
            n_det=int(g["n_det"]),
            voxel_mm=float(g["voxel_mm"]),
            sid_mm=float(g["SID_mm"]),
            sdd_mm=float(g["SDD_mm"]),
            gamma_max_deg=float(g["gamma_max_deg"]),
            delta_gamma_deg=float(g["delta_gamma_deg"]),
            angles_deg=np.asarray(g["angles_deg"], dtype=np.float64),
            det_fan_angles_deg=np.asarray(g["det_fan_angles_deg"], dtype=np.float64),
        )
    return sinogram, geo


# ── DICOM writer ─────────────────────────────────────────────────────────────

TAG_MAR_SEQ = 0x00189390
TAG_MAR_APPLIED = 0x00189391

def write_dicom_slice(
    hu_slice: np.ndarray,
    z: int,
    output_dir: Path,
    voxel_mm: float,
    study_uid: str,
    series_uid: str,
) -> None:
    """Write one 512x512 HU slice as a DICOM CT file with CP-2575 MAR macro."""
    hu_clipped = np.clip(hu_slice, -1024, 32767).astype(np.int16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"

    ds = FileDataset(
        str(output_dir), {}, file_meta=file_meta, preamble=b"\0" * 128
    )
    ds.Modality = "CT"
    ds.Manufacturer = "ASTM WKXXXXX ILS"
    ds.StudyDescription = "MAR ILS Lab Submission"
    ds.SeriesDescription = "MAR"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Rows = hu_clipped.shape[0]
    ds.Columns = hu_clipped.shape[1]
    ds.PixelSpacing = [voxel_mm, voxel_mm]
    ds.ImagePositionPatient = [0.0, 0.0, float(z * voxel_mm)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SliceLocation = float(z * voxel_mm)
    ds.SliceThickness = str(voxel_mm)
    ds.InstanceNumber = z + 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.PixelData = hu_clipped.tobytes()

    # DICOM 2026b CP-2575 Metal Artifact Reduction Macro
    mar_item = Dataset()
    mar_item.add_new(TAG_MAR_APPLIED, "CS", "YES")
    ds.add_new(TAG_MAR_SEQ, "SQ", [mar_item])

    ds.save_as(
        str(output_dir / f"slice_{z + 1:04d}.dcm"), enforce_file_format=True
    )


def write_realization(
    hu_volume: np.ndarray,
    output_dir: Path,
    voxel_mm: float,
) -> None:
    """Write a full (n_slices, 512, 512) HU volume as numbered DICOM slices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    for z in range(hu_volume.shape[0]):
        write_dicom_slice(
            hu_volume[z], z, output_dir, voxel_mm, study_uid, series_uid
        )


# ── Main harness ─────────────────────────────────────────────────────────────

def process_condition(
    sinogram_dir: Path,
    output_dir: Path,
    condition: str,
) -> None:
    """Process all realizations for one condition (LP or LA)."""
    h5_files = sorted(sinogram_dir.glob("realization_*.h5"))
    if not h5_files:
        print(f"  No sinogram files found in {sinogram_dir}", file=sys.stderr)
        return

    print(f"  {condition}: {len(h5_files)} realizations")
    for h5_path in h5_files:
        tag = h5_path.stem  # e.g. "realization_001"
        print(f"    {tag} ... ", end="", flush=True)

        sinogram, geo = read_sinogram(h5_path)

        # ── Lab's MAR reconstruction ──
        hu_volume = apply_mar(sinogram, geo)

        if hu_volume.shape != (geo.n_slices, 512, 512):
            raise ValueError(
                f"apply_mar returned shape {hu_volume.shape}, "
                f"expected ({geo.n_slices}, 512, 512)"
            )

        realization_dir = output_dir / condition / tag
        write_realization(hu_volume, realization_dir, geo.voxel_mm)
        print("done")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAR ILS Lab Harness — apply MAR to sinogram dataset"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Path to the ILS sinogram dataset (contains sinograms/LP and sinograms/LA)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("mar_recon"),
        help="Output directory for MAR-reconstructed DICOMs (default: ./mar_recon)",
    )
    args = parser.parse_args()

    sino_root = args.dataset_dir / "sinograms"
    if not sino_root.exists():
        print(f"Error: {sino_root} not found", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"MAR ILS Harness")
    print(f"  Dataset:  {args.dataset_dir}")
    print(f"  Output:   {args.output_dir}")
    print()

    for condition in ("LP", "LA"):
        process_condition(sino_root / condition, args.output_dir, condition)

    print()
    print(f"Done. Submit {args.output_dir}/ to the ILS coordinator.")


if __name__ == "__main__":
    main()
