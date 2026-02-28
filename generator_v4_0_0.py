#!/usr/bin/env python3
"""
mar_ils_generator_v4.0.0.py
============================
MAR Interlaboratory Study — Standardized Reconstructed Digital Volumetric Dataset Generator
Compliant with: ASTM WKXXXXX Standard Test Method for Quantitative Evaluation of
Metal Artifact Reduction Performance in Tomographic Imaging Systems, Revision 02

Revision history
----------------
v3.0.1  Original release (non-compliant; see change log below)
v4.0.0  Revision 02 compliance update — changes from v3.0.1:

    CRITICAL FIXES
    [C1] Lesion HU corrected from 42 to 120 (§A1.4(e), §A1.3(c)).
         v3.0.1 used 2 HU contrast (42 − 40 HU), producing per-voxel CNR ≈ 0.07
         and AUC values statistically indistinguishable from chance.
         Revision 02 mandates 120 HU (80 HU contrast, CNR ≈ 2.7).

    [C2] SKE noise-exclusion rule implemented (§10.1.2, §A1.4(f)).
         v3.0.1 applied Gaussian noise to all body voxels including the lesion mask,
         corrupting the lesion interior and violating the SKE homogeneity requirement.
         Noise is now applied only to voxels that are:
             (a) inside the body ellipse, AND
             (b) outside the metal mask, AND
             (c) outside the lesion mask.
         The lesion is restored to HU_LESION_SKE after noise and artifact application,
         mirroring the existing metal restoration step.

    NORMATIVE FIXES
    [N1] Artifact template peak-normalised to 400 HU (§A1.7.4).
         v3.0.1 used implicit scaling with no guaranteed peak value.
         The template is now explicitly rescaled so max(|template|) within body
         (excluding metal mask) equals ARTIFACT_PEAK_HU = 400.

    [N2] Deterministic seed scheme enforced (§A1.7.3).
         v3.0.1 used per-run random seeds without recording them.
         Seed for realization i is now BASE_SEED + i, written to metadata CSV.
         BASE_SEED is a fixed module constant; modifying it produces a
         non-compliant dataset and invalidates checksums.

    [N3] Noise application domain explicitly conditioned on three-mask exclusion (§A1.2(f)).
         Implemented as part of [C2]; documented separately for clarity.

    [N4] Lesion centre coordinate verified to (281, 256) voxels (§A1.4(c)).
         The offset calculation now uses integer arithmetic referencing the
         normative formula directly, with an assertion guard.

    [N5] DICOM version tag updated to reflect Revision 02 (§16.1(d)).
         ProtocolName (0018,1030) now encodes "ASTM-WKXXXXX-Rev02-Reference".
         ConvolutionKernel (0018,1210) records artifact peak HU for traceability.

    ADMINISTRATIVE
    [A1] Module docstring and inline comments updated throughout.
    [A2] Compliance assertions added at module level and in slice constructor.
    [A3] Metadata CSV updated: adds seed_used, artifact_peak_hu, lesion_hu,
         noise_sigma, ske_noise_excluded columns.

Usage
-----
    python mar_ils_generator_v4.0.0.py [--output-dir ./dataset] [--dry-run]

Outputs (per run)
-----------------
    <output_dir>/
        noMAR/
            LP/  realization_001/ ... realization_020/   (256 DICOM files each)
            LA/  realization_001/ ... realization_020/
        MAR_ready/
            LP/  realization_001/ ... realization_020/
            LA/  realization_001/ ... realization_020/
        checksums_sha256.txt
        dataset_metadata.csv
        generator_provenance.json

Notes
-----
The CHO analysis, Mann-Whitney AUC estimation, bootstrap CI computation,
and Tikhonov-regularised covariance are NOT part of this generator.
They are the reference CHO implementation deliverable specified in §8.3.
This file generates DICOM input volumes only.

Author  : ASTM F04 Subcommittee Working Draft
Standard: ASTM WKXXXXX Revision 02
Date    : 2026-02-28
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
import scipy.ndimage

# ═══════════════════════════════════════════════════════════════════════════════
# §A1.1  VOLUME GEOMETRY  (normative — do not modify)
# ═══════════════════════════════════════════════════════════════════════════════
X_DIM: int = 512          # voxels
Y_DIM: int = 512          # voxels
Z_DIM: int = 256          # slices
VOXEL_MM: float = 0.5     # mm isotropic

# ═══════════════════════════════════════════════════════════════════════════════
# §A1.2  PHANTOM CROSS-SECTION  (normative — do not modify)
# ═══════════════════════════════════════════════════════════════════════════════
BODY_SEMI_X_MM: float = 85.0    # mm  → 170 voxels
BODY_SEMI_Y_MM: float = 60.0    # mm  → 120 voxels
BODY_CENTER_X: int = 256        # voxels (image centre)
BODY_CENTER_Y: int = 256        # voxels (image centre)
HU_BACKGROUND: int = 40         # HU  soft-tissue equivalent
HU_AIR: int = -1000             # HU  outside body ellipse
NOISE_SIGMA_HU: float = 30.0    # HU  Gaussian noise std dev

# ═══════════════════════════════════════════════════════════════════════════════
# §A1.3  METAL ROD  (normative — do not modify)
# ═══════════════════════════════════════════════════════════════════════════════
METAL_RADIUS_MM: float = 5.0    # mm  → radius 10 voxels, diameter 10 mm
HU_METAL: int = 3000            # HU  (restored last in slice construction)

METAL_CENTER_X: int = BODY_CENTER_X   # 256 voxels
METAL_CENTER_Y: int = BODY_CENTER_Y   # 256 voxels
METAL_RADIUS_VOX: int = round(METAL_RADIUS_MM / VOXEL_MM)   # 10 voxels

# Metal boundary (positive-x face) in voxels
METAL_BOUNDARY_X: int = METAL_CENTER_X + METAL_RADIUS_VOX   # 266 voxels

# ═══════════════════════════════════════════════════════════════════════════════
# §A1.4  LESION ROD  (normative — do not modify)
# ═══════════════════════════════════════════════════════════════════════════════
LESION_RADIUS_MM: float = 2.5   # mm  → radius 5 voxels, diameter 5 mm
#
# [C1] HU_LESION_SKE corrected from 42 → 120 (§A1.4(e), Revision 02)
# v3.0.1 value 42 produced 2 HU contrast → per-voxel CNR ≈ 0.07 → AUC ≈ 0.5
# 120 HU produces 80 HU contrast → per-voxel CNR ≈ 2.7 → measurable ΔAUC
#
HU_LESION_SKE: int = 120        # HU  fixed SKE value (no internal noise)

LESION_RADIUS_VOX: int = round(LESION_RADIUS_MM / VOXEL_MM)   # 5 voxels

# §A1.4(c) normative formula:
# x_lesion = x_metal + r_metal + 5 mm_gap + r_lesion
# expressed in voxels:
_GAP_MM: float = 5.0
_GAP_VOX: int = round(_GAP_MM / VOXEL_MM)   # 10 voxels
LESION_CENTER_X: int = (
    METAL_CENTER_X + METAL_RADIUS_VOX + _GAP_VOX + LESION_RADIUS_VOX
)   # 256 + 10 + 10 + 5 = 281 voxels
LESION_CENTER_Y: int = BODY_CENTER_Y   # 256 voxels

# Compliance assertion — catches any future constant drift
assert LESION_CENTER_X == 281, (
    f"LESION_CENTER_X={LESION_CENTER_X} but normative value is 281 (§A1.4(c)). "
    "Do not modify METAL or LESION radius/gap constants."
)

# ═══════════════════════════════════════════════════════════════════════════════
# §A1.7  BACKGROUND VARIABILITY  (normative — do not modify)
# ═══════════════════════════════════════════════════════════════════════════════
JITTER_MAX_DEG: float = 15.0    # °  uniform jitter range [−15, +15]
ARTIFACT_PEAK_HU: float = 400.0 # HU  [N1] normative peak magnitude

# [N2] Deterministic seed scheme (§A1.7.3)
# Seed for realization i (0-indexed) = BASE_SEED + i
# BASE_SEED is fixed and must not be modified; changing it invalidates checksums.
BASE_SEED: int = 20260228       # YYYYMMDD of Revision 02 date

# ═══════════════════════════════════════════════════════════════════════════════
# STUDY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
NUM_REALIZATIONS: int = 20      # §10.2 minimum
DATASET_VERSION: str = "v4.0.0"
STANDARD_REF: str = "ASTM-WKXXXXX-Rev02"


# ═══════════════════════════════════════════════════════════════════════════════
# MASK CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def _make_coordinate_grids() -> tuple[np.ndarray, np.ndarray]:
    """Return (yy, xx) integer coordinate grids for the XY plane."""
    yy, xx = np.mgrid[0:Y_DIM, 0:X_DIM]
    return yy, xx


def _body_mask(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Boolean mask: True inside the body ellipse (§A1.2)."""
    semi_x_vox = BODY_SEMI_X_MM / VOXEL_MM
    semi_y_vox = BODY_SEMI_Y_MM / VOXEL_MM
    return (
        ((xx - BODY_CENTER_X) / semi_x_vox) ** 2
        + ((yy - BODY_CENTER_Y) / semi_y_vox) ** 2
    ) <= 1.0


def _circle_mask(
    yy: np.ndarray, xx: np.ndarray,
    cx: int, cy: int, r_vox: int
) -> np.ndarray:
    """Boolean mask: True inside a circle of radius r_vox centred at (cx, cy)."""
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r_vox ** 2


# ═══════════════════════════════════════════════════════════════════════════════
# ARTIFACT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_artifact_template(
    body_mask: np.ndarray,
    metal_mask: np.ndarray,
) -> np.ndarray:
    """
    Generate the 2D base artifact template per §A1.7.4.

    Procedure:
      1. Forward-project phantom (background + metal) using approximate Radon
         over 180 equally-spaced angles.
      2. Simulate photon starvation: sinogram bins above the 99th percentile
         of non-zero values → replaced with 2 % of the 50th percentile.
      3. Reconstruct both original and corrupted sinograms using FBP (Ram-Lak).
      4. Difference = corrupted_recon − original_recon.
      5. Zero outside body, zero inside metal.
      6. [N1] Rescale so max(|template|) within valid region = ARTIFACT_PEAK_HU.

    Returns
    -------
    template : ndarray shape (Y_DIM, X_DIM), float32
    """
    # Build phantom for projection
    phantom = np.full((Y_DIM, X_DIM), HU_AIR, dtype=np.float32)
    phantom[body_mask] = HU_BACKGROUND
    phantom[metal_mask] = HU_METAL

    n_angles = 180
    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)

    def _radon_fwd(image: np.ndarray) -> np.ndarray:
        """Approximate Radon transform via rotation + column sum."""
        sino = np.zeros((n_angles, X_DIM), dtype=np.float64)
        for i, ang in enumerate(angles):
            rotated = scipy.ndimage.rotate(
                image, -ang, reshape=False, order=1,
                mode="constant", cval=HU_AIR
            )
            sino[i] = rotated.sum(axis=0)
        return sino

    def _fbp(sino: np.ndarray) -> np.ndarray:
        """Minimal filtered back-projection (Ram-Lak in frequency domain)."""
        n_proj, n_det = sino.shape
        # Ram-Lak filter
        freq = np.fft.rfftfreq(n_det)
        ramp = np.abs(freq)
        filtered = np.zeros_like(sino)
        for i in range(n_proj):
            spec = np.fft.rfft(sino[i])
            filtered[i] = np.fft.irfft(spec * ramp, n=n_det)
        # Back-project
        recon = np.zeros((Y_DIM, X_DIM), dtype=np.float64)
        for i, ang in enumerate(angles):
            proj_img = np.tile(filtered[i], (Y_DIM, 1))
            recon += scipy.ndimage.rotate(
                proj_img, ang, reshape=False, order=1,
                mode="constant", cval=0.0
            )
        return (recon * np.pi / n_angles).astype(np.float32)

    sino_clean = _radon_fwd(phantom)

    # Photon-starvation simulation
    non_zero = sino_clean[sino_clean != 0]
    thresh = np.percentile(non_zero, 99)
    floor = 0.02 * np.percentile(non_zero, 50)
    sino_corrupted = sino_clean.copy()
    sino_corrupted[sino_corrupted > thresh] = floor

    recon_clean = _fbp(sino_clean)
    recon_corrupt = _fbp(sino_corrupted)

    template = recon_corrupt - recon_clean

    # Zero outside body and inside metal
    template[~body_mask] = 0.0
    template[metal_mask] = 0.0

    # [N1] Normalise peak to ARTIFACT_PEAK_HU (§A1.7.4 step 6)
    peak = np.max(np.abs(template))
    if peak > 0:
        template *= ARTIFACT_PEAK_HU / peak
    else:
        warnings.warn(
            "Artifact template has zero peak — sinogram simulation may have failed.",
            RuntimeWarning, stacklevel=2
        )

    return template.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# SLICE CONSTRUCTION  (the core compliance-critical function)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_slice(
    z: int,
    *,
    lesion_present: bool,
    mar_condition: bool,      # True = MAR_ready (no artifact added)
    artifact_template: np.ndarray,   # pre-rotated for this realization
    body_mask: np.ndarray,
    metal_mask: np.ndarray,
    lesion_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Construct one axial slice in strict §10.1.2 construction order:

        Step 1  Background fill (air and soft tissue)
        Step 2  Lesion placement — BEFORE noise (SKE: fixed HU, no noise)  [C2]
        Step 3  Artifact addition (noMAR condition only)
        Step 4  Gaussian noise applied ONLY outside lesion AND metal masks  [C2][N3]
        Step 5  Metal restoration (always last)
        Step 6  Lesion restoration (always last, after noise)               [C2]

    The order of steps 5 and 6 matches the normative requirement that both
    the metal and lesion voxels are restored to their fixed values AFTER
    noise application, guaranteeing homogeneous interiors for both structures.

    Parameters
    ----------
    z               : slice index (0-indexed; unused geometrically but kept for
                      future z-dependent extensions)
    lesion_present  : LP (True) or LA (False) condition
    mar_condition   : if True, artifact template is NOT added (MAR-ready series)
    artifact_template : 2D rotated template for this realization, shape (Y, X)
    body_mask       : bool (Y, X)
    metal_mask      : bool (Y, X)
    lesion_mask     : bool (Y, X)
    rng             : numpy Generator (seeded per realization per §A1.7.3)

    Returns
    -------
    sl : ndarray shape (Y_DIM, X_DIM), float32
    """
    sl = np.full((Y_DIM, X_DIM), float(HU_AIR), dtype=np.float32)

    # Step 1 — background
    sl[body_mask] = HU_BACKGROUND

    # Step 2 — lesion (placed before noise to define the region; restored after noise)
    # The actual SKE guarantee is enforced by the restoration in Step 6.
    # We mark the region here so Step 4 knows to exclude it.
    if lesion_present:
        sl[lesion_mask] = float(HU_LESION_SKE)   # temporary; overwritten in Step 6

    # Step 3 — artifact (noMAR condition only)
    if not mar_condition:
        sl += artifact_template      # additive in image domain (§A1.7.4)
        # Re-clip to avoid negative air voxels bleeding into body statistics
        sl[~body_mask] = HU_AIR

    # Step 4 — Gaussian noise, excluded from lesion and metal masks [C2][N3]
    # This is the compliance-critical fix vs. v3.0.1.
    # v3.0.1 applied noise to ALL body voxels, corrupting the lesion interior.
    noise_target_mask = body_mask & (~metal_mask) & (~lesion_mask)
    noise = rng.normal(0.0, NOISE_SIGMA_HU, size=(Y_DIM, X_DIM)).astype(np.float32)
    sl[noise_target_mask] += noise[noise_target_mask]

    # Step 5 — metal restoration (always; overrides noise and artifact)
    sl[metal_mask] = float(HU_METAL)

    # Step 6 — lesion restoration [C2]
    # Restores the lesion to its exact SKE value after noise has been applied
    # to surrounding voxels. This mirrors the metal restoration pattern and
    # guarantees a homogeneous lesion interior regardless of noise field.
    if lesion_present:
        sl[lesion_mask] = float(HU_LESION_SKE)

    return sl


# ═══════════════════════════════════════════════════════════════════════════════
# DICOM WRITING
# ═══════════════════════════════════════════════════════════════════════════════

def _make_dicom(
    pixel_array: np.ndarray,
    series_uid: str,
    sop_uid: str,
    instance_number: int,
    series_description: str,
    realization_index: int,
    condition_label: str,
    seed_used: int,
) -> FileDataset:
    """Construct a minimal but standards-traceable DICOM CT slice."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(
        filename_or_obj=None,
        dataset={},
        file_meta=file_meta,
        is_implicit_VR=False,
        is_little_endian=True,
    )
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    # Patient / Study
    ds.PatientName = "ASTM^WKXXXXX^REV02"
    ds.PatientID = f"MAR-ILS-REV02-R{realization_index:03d}"
    ds.StudyDescription = "ASTM-WKXXXXX-MAR-ILS"                     # (0008,1030)
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = datetime.now(timezone.utc).strftime("%Y%m%d")
    ds.StudyTime = datetime.now(timezone.utc).strftime("%H%M%S")
    ds.AccessionNumber = "WKXXXXX-Rev02"          # SH max 16 chars; full ref in ProtocolName

    # Series
    ds.Modality = "CT"
    ds.SeriesDescription = series_description
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = realization_index

    # [N5] Revision 02 traceability tags (§16.1(d))
    ds.ProtocolName = f"ASTM-WKXXXXX-Rev02-Reference-{DATASET_VERSION}"  # (0018,1030)
    ds.ConvolutionKernel = f"FBP-AP{int(ARTIFACT_PEAK_HU)}"   # SH max 16; "FBP-AP400" = 9 chars
    ds.KVP = "120"

    # Instance
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = str(instance_number + 1)
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]

    # Image geometry
    ds.Rows = Y_DIM
    ds.Columns = X_DIM
    ds.PixelSpacing = [VOXEL_MM, VOXEL_MM]
    ds.SliceThickness = VOXEL_MM
    ds.SliceLocation = instance_number * VOXEL_MM
    ds.ImagePositionPatient = [0.0, 0.0, float(instance_number) * VOXEL_MM]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1   # signed
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.RescaleType = "HU"

    # Private ASTM provenance tags (0009,xx01–xx03)
    # Split across three LO tags (max 64 chars each) to avoid VR length warnings.
    # Tag 01: condition and seed          e.g. "Cond=noMAR/LP;Seed=20260228"     (27 chars max)
    # Tag 02: physics parameters          e.g. "LesionHU=120;ArtPeak=400;NoiseSigma=30" (38 chars)
    # Tag 03: compliance flags and rev    e.g. "SKE-NoiseExcluded=1;Rev=02;Std=WKXXXXX" (39 chars)
    block = ds.private_block(0x0009, "ASTM_MAR_ILS", create=True)
    block.add_new(0x01, "LO", f"Cond={condition_label};Seed={seed_used}")
    block.add_new(0x02, "LO",
        f"LesionHU={HU_LESION_SKE};"
        f"ArtPeak={int(ARTIFACT_PEAK_HU)};"
        f"NoiseSigma={int(NOISE_SIGMA_HU)}"
    )
    block.add_new(0x03, "LO", "SKE-NoiseExcluded=1;Rev=02;Std=WKXXXXX")

    # Pixel data
    pixel_data = np.clip(pixel_array, -32768, 32767).astype(np.int16)
    ds.PixelData = pixel_data.tobytes()

    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKSUM
# ═══════════════════════════════════════════════════════════════════════════════

def _sha256_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(output_dir: Path, dry_run: bool = False) -> None:
    """
    Generate the full ASTM WKXXXXX Revision 02 compliant dataset.

    Directory structure
    -------------------
    <output_dir>/
        noMAR/LP/realization_001..020/   256 DICOM files each
        noMAR/LA/realization_001..020/
        MAR_ready/LP/realization_001..020/
        MAR_ready/LA/realization_001..020/
        checksums_sha256.txt
        dataset_metadata.csv
        generator_provenance.json
    """
    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ASTM WKXXXXX Rev02 — MAR ILS Dataset Generator {DATASET_VERSION}")
    print(f"Output directory : {output_dir}")
    print(f"BASE_SEED        : {BASE_SEED}")
    print(f"Lesion HU (SKE)  : {HU_LESION_SKE} HU  [corrected from v3.0.1 value of 42]")
    print(f"Artifact peak    : {ARTIFACT_PEAK_HU} HU")
    print(f"Noise σ          : {NOISE_SIGMA_HU} HU  (excluded from lesion & metal masks)")
    print(f"Realizations     : {NUM_REALIZATIONS} per condition (4 conditions)")
    print()

    # ── Pre-compute static masks ──────────────────────────────────────────────
    yy, xx = _make_coordinate_grids()
    body_mask   = _body_mask(yy, xx)
    metal_mask  = _circle_mask(yy, xx, METAL_CENTER_X, METAL_CENTER_Y, METAL_RADIUS_VOX)
    lesion_mask = _circle_mask(yy, xx, LESION_CENTER_X, LESION_CENTER_Y, LESION_RADIUS_VOX)

    print(f"Lesion centre    : ({LESION_CENTER_X}, {LESION_CENTER_Y}) voxels  "
          f"[normative: (281, 256)]")
    print(f"Metal boundary x : {METAL_BOUNDARY_X} voxels")
    print(f"Gap (boundary to lesion centre) : "
          f"{LESION_CENTER_X - METAL_BOUNDARY_X - LESION_RADIUS_VOX} mm")
    print()

    # ── Build artifact template once ─────────────────────────────────────────
    print("Building artifact template ...", end=" ", flush=True)
    if not dry_run:
        artifact_base = _build_artifact_template(body_mask, metal_mask)
        print(f"peak = {np.max(np.abs(artifact_base)):.1f} HU  ✓")
    else:
        artifact_base = np.zeros((Y_DIM, X_DIM), dtype=np.float32)
        print("(dry-run, skipped)")

    # ── Conditions ───────────────────────────────────────────────────────────
    conditions = [
        ("noMAR",     "LP", True,  False),
        ("noMAR",     "LA", False, False),
        ("MAR_ready", "LP", True,  True),
        ("MAR_ready", "LA", False, True),
    ]

    metadata_rows: list[dict] = []
    all_dicom_paths: list[Path] = []

    # ── Generation loop ───────────────────────────────────────────────────────
    for cond_dir, lp_la, lesion_present, mar_condition in conditions:
        cond_label = f"{cond_dir}/{lp_la}"
        print(f"Generating condition: {cond_label}")

        for r in range(NUM_REALIZATIONS):
            # [N2] Deterministic seed (§A1.7.3): seed = BASE_SEED + realization_index
            seed = BASE_SEED + r
            rng = np.random.default_rng(seed)

            # Per-realization jitter angle: Uniform[−15°, +15°] (§A1.7.2)
            jitter_deg = float(rng.uniform(-JITTER_MAX_DEG, JITTER_MAX_DEG))

            # Rotate artifact template for this realization
            if not dry_run:
                art_rotated = scipy.ndimage.rotate(
                    artifact_base, jitter_deg,
                    reshape=False, order=1,
                    mode="constant", cval=0.0
                ).astype(np.float32)
                art_rotated[~body_mask] = 0.0
                art_rotated[metal_mask] = 0.0
            else:
                art_rotated = artifact_base.copy()

            series_uid = generate_uid()
            real_dir = output_dir / cond_dir / lp_la / f"realization_{r+1:03d}"
            if not dry_run:
                real_dir.mkdir(parents=True, exist_ok=True)

            for z in range(Z_DIM):
                sl = _build_slice(
                    z,
                    lesion_present=lesion_present,
                    mar_condition=mar_condition,
                    artifact_template=art_rotated,
                    body_mask=body_mask,
                    metal_mask=metal_mask,
                    lesion_mask=lesion_mask,
                    rng=rng,
                )

                sop_uid = generate_uid()
                dcm = _make_dicom(
                    pixel_array=sl,
                    series_uid=series_uid,
                    sop_uid=sop_uid,
                    instance_number=z,
                    series_description=(
                        f"{STANDARD_REF}_{cond_dir}_{lp_la}_R{r+1:03d}"
                    ),
                    realization_index=r + 1,
                    condition_label=cond_label,
                    seed_used=seed,
                )

                dcm_path = real_dir / f"slice_{z+1:04d}.dcm"
                if not dry_run:
                    dcm.save_as(str(dcm_path), write_like_original=False)
                    all_dicom_paths.append(dcm_path)

            print(f"  R{r+1:02d}  seed={seed}  jitter={jitter_deg:+.2f}°", flush=True)

            metadata_rows.append({
                "condition":           cond_dir,
                "lp_la":               lp_la,
                "realization":         r + 1,
                "seed_used":           seed,                 # [N2]
                "jitter_deg":          round(jitter_deg, 4),
                "lesion_present":      lesion_present,
                "mar_condition":       mar_condition,
                "lesion_hu":           HU_LESION_SKE,        # [C1]
                "lesion_center_x_vox": LESION_CENTER_X,      # [N4]
                "lesion_center_y_vox": LESION_CENTER_Y,
                "artifact_peak_hu":    ARTIFACT_PEAK_HU,     # [N1]
                "noise_sigma_hu":      NOISE_SIGMA_HU,
                "ske_noise_excluded":  True,                 # [C2]
                "n_slices":            Z_DIM,
                "observer_channels":   "Laguerre-Gauss",     # metadata only
                "n_channels":          10,
                "channel_width_vox":   7.5,
                "tikhonov_lambda":     "0.01*trace(K)/p",
                "auc_estimator":       "Mann-Whitney",
                "generator_version":   DATASET_VERSION,
                "standard_ref":        STANDARD_REF,
            })

        print()

    if dry_run:
        print("Dry run complete — no files written.")
        return

    # ── Checksums (§8.1 / §11.1) ─────────────────────────────────────────────
    print("Computing SHA-256 checksums ...", end=" ", flush=True)
    checksum_path = output_dir / "checksums_sha256.txt"
    with open(checksum_path, "w", encoding="utf-8") as cf:
        cf.write(f"# ASTM WKXXXXX Revision 02 — SHA-256 Checksum Manifest\n")
        cf.write(f"# Generator: {DATASET_VERSION}  Date: "
                 f"{datetime.now(timezone.utc).isoformat()}\n")
        cf.write(f"# BASE_SEED: {BASE_SEED}\n")
        cf.write(f"# Format: <sha256>  <relative_path>\n")
        for p in sorted(all_dicom_paths):
            rel = p.relative_to(output_dir)
            cf.write(f"{_sha256_file(p)}  {rel}\n")
    print(f"{len(all_dicom_paths)} files  ✓")

    # ── Metadata CSV ──────────────────────────────────────────────────────────
    meta_path = output_dir / "dataset_metadata.csv"
    if metadata_rows:
        with open(meta_path, "w", newline="", encoding="utf-8") as mf:
            writer = csv.DictWriter(mf, fieldnames=list(metadata_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metadata_rows)
    print(f"Metadata CSV     : {meta_path}")

    # ── Provenance JSON ───────────────────────────────────────────────────────
    prov = {
        "generator_version":  DATASET_VERSION,
        "standard_reference": STANDARD_REF,
        "generation_utc":     datetime.now(timezone.utc).isoformat(),
        "base_seed":          BASE_SEED,
        "constants": {
            "X_DIM": X_DIM, "Y_DIM": Y_DIM, "Z_DIM": Z_DIM,
            "VOXEL_MM": VOXEL_MM,
            "HU_BACKGROUND": HU_BACKGROUND,
            "HU_LESION_SKE": HU_LESION_SKE,         # [C1]
            "HU_METAL": HU_METAL,
            "LESION_CENTER_X": LESION_CENTER_X,      # [N4]
            "LESION_CENTER_Y": LESION_CENTER_Y,
            "LESION_RADIUS_VOX": LESION_RADIUS_VOX,
            "METAL_CENTER_X": METAL_CENTER_X,
            "METAL_RADIUS_VOX": METAL_RADIUS_VOX,
            "NOISE_SIGMA_HU": NOISE_SIGMA_HU,
            "SKE_NOISE_EXCLUDED_FROM_LESION": True,  # [C2]
            "ARTIFACT_PEAK_HU": ARTIFACT_PEAK_HU,    # [N1]
            "JITTER_MAX_DEG": JITTER_MAX_DEG,
        },
        "compliance_notes": {
            "C1": "HU_LESION_SKE=120 (corrected from v3.0.1 value of 42)",
            "C2": "Noise excluded from lesion mask; lesion restored post-noise",
            "N1": f"Artifact peak normalised to {ARTIFACT_PEAK_HU} HU",
            "N2": f"Deterministic seeds: BASE_SEED + realization_index",
            "N3": "Noise domain: body AND NOT metal AND NOT lesion",
            "N4": f"Lesion centre verified at ({LESION_CENTER_X}, {LESION_CENTER_Y}) vox",
            "N5": "DICOM ProtocolName encodes ASTM-WKXXXXX-Rev02",
        },
        "num_dicom_files": len(all_dicom_paths),
    }
    prov_path = output_dir / "generator_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as pf:
        json.dump(prov, pf, indent=2)

    print(f"Provenance JSON  : {prov_path}")
    print()
    print("=" * 60)
    print("Dataset generation complete.")
    print(f"  Total DICOM files : {len(all_dicom_paths)}")
    print(f"  Checksum manifest : {checksum_path}")
    print()
    print("COMPLIANCE SUMMARY (Revision 02 fixes applied)")
    print("  [C1] Lesion HU = 120 (80 HU contrast, CNR ≈ 2.7)  ✓")
    print("  [C2] SKE: noise excluded from lesion mask           ✓")
    print("  [N1] Artifact peak normalised to 400 HU             ✓")
    print("  [N2] Deterministic seed scheme enforced             ✓")
    print("  [N3] Noise domain: body ∩ ¬metal ∩ ¬lesion         ✓")
    print(f"  [N4] Lesion centre = ({LESION_CENTER_X}, {LESION_CENTER_Y}) vox              ✓")
    print("  [N5] DICOM Rev02 traceability tags                  ✓")
    print()
    print("NOT IN SCOPE for this generator (separate §8.3 deliverable):")
    print("  — Reference CHO implementation (Laguerre-Gauss, 10 channels)")
    print("  — Tikhonov-regularised covariance estimation")
    print("  — Mann-Whitney AUC + bootstrap CI")
    print("  — ±0.001 AUC numerical equivalence validation")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"ASTM WKXXXXX Rev02 MAR ILS Dataset Generator {DATASET_VERSION}"
        )
    )
    parser.add_argument(
        "--output-dir", default="./astm_mar_ils_dataset",
        help="Root directory for output dataset (default: ./astm_mar_ils_dataset)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate constants and masks without writing any files"
    )
    args = parser.parse_args()

    # Pre-flight compliance checks
    assert HU_LESION_SKE == 120, \
        f"HU_LESION_SKE must be 120 per §A1.4(e) Rev02; got {HU_LESION_SKE}"
    assert LESION_CENTER_X == 281, \
        f"LESION_CENTER_X must be 281 per §A1.4(c); got {LESION_CENTER_X}"
    assert ARTIFACT_PEAK_HU == 400.0, \
        f"ARTIFACT_PEAK_HU must be 400 per §A1.7.4; got {ARTIFACT_PEAK_HU}"
    assert JITTER_MAX_DEG == 15.0, \
        f"JITTER_MAX_DEG must be 15 per §A1.7.2; got {JITTER_MAX_DEG}"
    assert X_DIM == 512 and Y_DIM == 512 and Z_DIM == 256, \
        "Volume dimensions must be 512×512×256 per §A1.1"
    assert VOXEL_MM == 0.5, \
        f"VOXEL_MM must be 0.5 per §A1.1; got {VOXEL_MM}"

    generate_dataset(Path(args.output_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()