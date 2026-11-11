#!/usr/bin/env python3
"""
MAR ILS laboratory harness, Python (ASTM WK101378 draft Rev 06, v9 dataset).

Reads the coded sinograms of one replicate dataset in the format of the
sinogram ingestion specification (version 2.0), calls apply_mar() from
apply_mar.py for each, and writes one DICOM CT image per realization, named
by its code, with the CP-2575 MAR Macro (protocol §A8.3). It then writes the
SHA-256 manifest of the run directory (protocol §A8.2).

For Condition 2 (the laboratory's own MAR). Under Condition 1 the harness is
not used (protocol §A7).

Usage:
    # Reconstruction check, draft standard §12.2 (protocol Step 2.1):
    python run_mar_harness.py --recon-check FAMILIARIZATION/recon_check/recon_check.h5

    # One run directory per assigned replicate dataset:
    python run_mar_harness.py REP04 C2_R1
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from apply_mar import apply_mar, reconstruct
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

IMAGE_PX = 512
PIXEL_MM = 300.0 / IMAGE_PX
RECON_CHECK_TOL_HU = 1.0


# ── Ingestion spec §4 ────────────────────────────────────────────────────────

@dataclass
class Geometry:
    sid_mm: float
    sdd_mm: float
    n_views: int
    n_channels: int
    delta_gamma_rad: float
    gamma0_rad: float
    betas_rad: np.ndarray
    gammas_rad: np.ndarray
    I0: float
    scatter_counts: float
    electronic_variance_counts2: float
    log_floor_counts: float


def read_sinogram(h5_path: Path) -> tuple[np.ndarray, Geometry]:
    """Line integrals (n_views, n_channels) as float64, indexed [view, channel]."""
    with h5py.File(h5_path, "r") as f:
        d = f["line_integrals"]
        p = d[()].astype(np.float64)
        a = d.attrs
        geo = Geometry(
            sid_mm=float(a["SID_mm"]), sdd_mm=float(a["SDD_mm"]),
            n_views=int(a["n_views"]), n_channels=int(a["n_channels"]),
            delta_gamma_rad=float(a["delta_gamma_rad"]), gamma0_rad=float(a["gamma0_rad"]),
            betas_rad=np.asarray(a["betas_rad"], dtype=np.float64),
            gammas_rad=np.asarray(a["gammas_rad"], dtype=np.float64),
            I0=float(a["I0"]), scatter_counts=float(a["scatter_counts"]),
            electronic_variance_counts2=float(a["electronic_variance_counts2"]),
            log_floor_counts=float(a["log_floor_counts"]),
        )
    if p.shape != (geo.n_views, geo.n_channels):
        raise ValueError(f"{h5_path}: line_integrals shape {p.shape}, "
                         f"expected ({geo.n_views}, {geo.n_channels})")
    return p, geo


# ── Ingestion spec §6.3 ──────────────────────────────────────────────────────

def write_dicom(hu: np.ndarray, path: Path, study_uid: str, series_uid: str) -> None:
    """One 512 x 512 CT image in HU: signed 16-bit, slope 1, intercept 0, MAR Applied = YES."""
    pixels = np.clip(np.rint(hu), -32768, 32767).astype("<i2")

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.InstanceNumber = 1
    ds.ImagePositionPatient = [-150.0, -150.0, 0.0]   # center of pixel (0, 0)
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

    # DICOM 2026b CP-2575 Metal Artifact Reduction Macro, written by numeric tag
    item = Dataset()
    item.add_new(0x00189391, "CS", "YES")
    ds.add_new(0x00189390, "SQ", [item])

    ds.save_as(str(path), enforce_file_format=True)


def write_manifest(run_dir: Path) -> Path:
    """SHA-256 of every file in the run directory, beside it, in sha256sum format."""
    out = run_dir.parent / f"{run_dir.name}_checksums_sha256.txt"
    lines = []
    for f in sorted(run_dir.iterdir()):
        if f.is_file():
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n")
    out.write_text("".join(lines))
    return out


# ── Modes ────────────────────────────────────────────────────────────────────

def recon_check(check_h5: Path) -> int:
    p, geo = read_sinogram(check_h5)
    with h5py.File(check_h5, "r") as f:
        ref = f["reference_image_hu"][()]
    hu = np.asarray(reconstruct(p, geo), dtype=np.float64)
    if hu.shape != ref.shape:
        print(f"reconstruct() returned {hu.shape}, expected {ref.shape}", file=sys.stderr)
        return 1
    worst = float(np.max(np.abs(hu - ref)))
    passed = worst <= RECON_CHECK_TOL_HU
    print(f"Reconstruction check: largest difference {worst:.6f} HU "
          f"({'PASS' if passed else 'FAIL'}, tolerance {RECON_CHECK_TOL_HU} HU)")
    return 0 if passed else 1


def run(dataset_dir: Path, run_dir: Path) -> int:
    sinograms = sorted((dataset_dir / "sinograms").glob("*.h5"))
    if not sinograms:
        print(f"No sinograms in {dataset_dir / 'sinograms'}", file=sys.stderr)
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    study_uid, series_uid = generate_uid(), generate_uid()
    print(f"{dataset_dir} -> {run_dir}: {len(sinograms)} realizations")
    for n, h5_path in enumerate(sinograms, 1):
        code = h5_path.stem
        p, geo = read_sinogram(h5_path)
        hu = np.asarray(apply_mar(p, geo))
        if hu.shape != (IMAGE_PX, IMAGE_PX):
            raise ValueError(f"apply_mar returned {hu.shape} for {code}, expected (512, 512)")
        write_dicom(hu, run_dir / f"{code}.dcm", study_uid, series_uid)
        if n % 100 == 0 or n == len(sinograms):
            print(f"  {n}/{len(sinograms)}")
    print(f"Manifest: {write_manifest(run_dir)}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="MAR ILS laboratory harness (v9 dataset)")
    ap.add_argument("dataset_dir", nargs="?", type=Path, help="replicate dataset directory, e.g. REP04")
    ap.add_argument("run_dir", nargs="?", type=Path, help="run directory to write, e.g. C2_R1")
    ap.add_argument("--recon-check", type=Path, metavar="RECON_CHECK_H5",
                    help="run the reconstruction check of draft standard §12.2 and exit")
    a = ap.parse_args()
    if a.recon_check:
        sys.exit(recon_check(a.recon_check))
    if a.dataset_dir is None or a.run_dir is None:
        ap.error("give dataset_dir and run_dir, or --recon-check")
    sys.exit(run(a.dataset_dir, a.run_dir))


if __name__ == "__main__":
    main()
