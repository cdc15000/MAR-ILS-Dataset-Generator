#!/usr/bin/env python3
"""
MAR ILS Dataset Generator  —  v3.0
====================================
Generates a synthetic CT DICOM dataset for a Metal Artifact Reduction (MAR)
interlaboratory study (ILS), following the framework of Vaishnav et al.
(Medical Physics, 47(8), 2020).

Key design decisions aligned with Vaishnav et al.
--------------------------------------------------
* Signal geometry  : CYLINDRICAL rod running the full z-extent (like a
                     "removable rod" physical phantom insert), not a sphere.
                     This ensures the lesion is present in EVERY slice of the
                     LP series, matching Vaishnav's rod-signal design and
                     making it immediately visible when any slice is opened.
* Signal physics   : The lesion REPLACES the background rather than being
                     added to it.  Inside the lesion cylinder, HU is set to
                     a fixed value HU_BACKGROUND + HU_LESION_DELTA with NO
                     noise.  This produces a homogeneous disc against a noisy
                     background — exactly what the CHO is optimised for and
                     what Vaishnav describes as "a homogeneous disk."
* Signal contrast  : HU_LESION_DELTA = 80 HU (lesion = 120 HU).  In the
                     soft-tissue window (C=40, W=400) this maps to ~70% grey
                     — clearly visible but not saturated.  The secondary DICOM
                     window (C=80, W=160) centres the display range on the
                     lesion for even clearer inspection.
* Artifact model   : Photon-starvation filtered backprojection (FBP) on a
                     noiseless 2D phantom.  Computed ONCE and added to every
                     slice in the noMAR condition.
* Four folders     : lesion_absent/present × noMAR/MAR, per realization.
* Independent noise: Each realization uses seed = BASE_SEED + i.
* Lesion location  : Cylinder axis at X=276, Y=256 — 5 mm beyond the metal
                     rod boundary in the +x direction.

Changelog
---------
v3.0  • Sphere → cylinder (lesion present in all 256 slices)
      • Lesion set by replacement (homogeneous) not addition (noisy)
      • HU_LESION_DELTA corrected to 80 HU (was 200; was saturating window)
      • Dual DICOM window presets (soft-tissue + lesion-centred)
      • SliceLocation tag added for correct DICOM viewer ordering
      • SeriesDescription encodes condition + lesion location for labs
      • Fixed duplicate vol.copy() / lesion application logic
      • Fixed conditions 5-tuple unpacking
v2.0  • Artifact template via FBP (was: no artifacts)
      • Four MAR-condition folders per realization (was: two)
      • Body phantom + correct window/level (earlier versions)
"""

import os, shutil, struct, zipfile, hashlib, csv
import numpy as np
from datetime import datetime, date
from scipy.ndimage import rotate

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DICOM_DIR      = os.path.join(SCRIPT_DIR, "DICOM")
PDF_FILE       = os.path.join(SCRIPT_DIR, "MAR_ILS_Lab_Instructions.pdf")
METADATA_FILE  = os.path.join(SCRIPT_DIR, "metadata.csv")
CHECKSUMS_FILE = os.path.join(SCRIPT_DIR, "checksums_sha256.txt")
ZIP_FILE       = os.path.join(SCRIPT_DIR, "MAR_ILS_Archive.zip")

# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
Z_DIM, Y_DIM, X_DIM = 256, 512, 512
VOXEL_MM = 0.5

# Body phantom (ellipse)
BODY_SEMI_X = int(170 / VOXEL_MM)   # 340 vox
BODY_SEMI_Y = int(120 / VOXEL_MM)   # 240 vox

# Metal rod — cylindrical, full z-extent, centred in FOV
METAL_CENTER_X   = 256
METAL_CENTER_Y   = 256
METAL_RADIUS_MM  = 5.0
METAL_RADIUS_VOX = int(round(METAL_RADIUS_MM / VOXEL_MM))   # 10 vox

# Lesion — CYLINDRICAL rod, full z-extent, 5 mm beyond metal boundary in +x
# (matches Vaishnav "removable rod" geometry)
METAL_BOUNDARY_X  = METAL_CENTER_X + METAL_RADIUS_VOX       # 266
LESION_OFFSET_VOX = int(round(5.0  / VOXEL_MM))             # 10 vox
LESION_CENTER_X   = METAL_BOUNDARY_X + LESION_OFFSET_VOX    # 276
LESION_CENTER_Y   = 256
LESION_RADIUS_MM  = 2.5
LESION_RADIUS_VOX = int(round(LESION_RADIUS_MM / VOXEL_MM)) # 5 vox

# ─────────────────────────────────────────────────────────────────────────────
# HU values
# ─────────────────────────────────────────────────────────────────────────────
HU_AIR          = -1000
HU_BACKGROUND   =    40   # soft tissue
HU_METAL        =  3000
HU_LESION       = HU_BACKGROUND + 80   # 120 HU — homogeneous, no noise
# In W=400, C=40 window (range −160 to 240):
#   background 40 HU → 50% grey
#   lesion    120 HU → 70% grey  ← clearly visible, not saturated

# ─────────────────────────────────────────────────────────────────────────────
# Noise & realizations
# ─────────────────────────────────────────────────────────────────────────────
NOISE_STD_HU     = 30.0
BASE_SEED        = 42
NUM_REALIZATIONS = 20

# ─────────────────────────────────────────────────────────────────────────────
# Artifact simulation
# ─────────────────────────────────────────────────────────────────────────────
N_PROJECTION_ANGLES = 180
ARTIFACT_PEAK_HU    = 400

# ─────────────────────────────────────────────────────────────────────────────
# DICOM identifiers
# ─────────────────────────────────────────────────────────────────────────────
UID_ROOT     = "1.2.826.0.1.3680043.10.999"
STUDY_DATE   = date.today().strftime("%Y%m%d")
STUDY_TIME   = "120000"
STUDY_UID    = f"{UID_ROOT}.1.{STUDY_DATE}"
PATIENT_NAME = "Synthetic^MAR^ILS"
PATIENT_ID   = "MAR_ILS_001"


# ─────────────────────────────────────────────────────────────────────────────
# DICOM writer
# ─────────────────────────────────────────────────────────────────────────────

def _tag(group, element, vr, value):
    length = len(value)
    if vr in ("OB", "OW", "SQ", "UC", "UN", "UR", "UT"):
        return struct.pack("<HH2sHI", group, element,
                          vr.encode("ascii"), 0, length) + value
    return struct.pack("<HH2sH", group, element,
                      vr.encode("ascii"), length) + value


def write_ct_dicom(path, pixel_2d, slice_idx, series_uid,
                   sop_uid, series_description="MAR ILS"):
    rows, cols = pixel_2d.shape
    z_mm = slice_idx * VOXEL_MM
    px   = np.clip(pixel_2d, -32768, 32767).astype(np.int16).tobytes()
    if len(px) % 2:
        px += b"\x00"

    def ui(v): b=v.encode(); return b if len(b)%2==0 else b+b"\x00"
    def ds(v): b=v.encode(); return b if len(b)%2==0 else b+b" "
    def lo(v): b=v.encode(); return b if len(b)%2==0 else b+b" "
    def cs(v): b=v.encode(); return b if len(b)%2==0 else b+b" "
    def is_(v):b=v.encode(); return b if len(b)%2==0 else b+b" "
    def us(v): return struct.pack("<H", v)

    CT_SOP  = b"1.2.840.10008.5.1.4.1.1.2"
    TSYN    = b"1.2.840.10008.1.2.1"
    if len(CT_SOP)%2: CT_SOP += b"\x00"
    if len(TSYN)%2:   TSYN   += b"\x00"

    meta = (
        _tag(0x0002,0x0001,"OB",b"\x00\x01") +
        _tag(0x0002,0x0002,"UI",CT_SOP) +
        _tag(0x0002,0x0003,"UI",ui(sop_uid)) +
        _tag(0x0002,0x0010,"UI",TSYN)
    )
    meta = _tag(0x0002,0x0000,"UL",struct.pack("<I",len(meta))) + meta

    ds_block = (
        _tag(0x0008,0x0016,"UI",ui("1.2.840.10008.5.1.4.1.1.2")) +
        _tag(0x0008,0x0018,"UI",ui(sop_uid)) +
        _tag(0x0008,0x0020,"DA",ds(STUDY_DATE)) +
        _tag(0x0008,0x0030,"TM",ds(STUDY_TIME)) +
        _tag(0x0008,0x0060,"CS",cs("CT")) +
        _tag(0x0008,0x0070,"LO",lo("Synthetic ILS Generator")) +
        _tag(0x0008,0x103E,"LO",lo(series_description[:64])) +
        _tag(0x0010,0x0010,"PN",ds(PATIENT_NAME)) +
        _tag(0x0010,0x0020,"LO",lo(PATIENT_ID)) +
        _tag(0x0018,0x0050,"DS",ds(str(VOXEL_MM))) +
        _tag(0x0018,0x0088,"DS",ds(str(VOXEL_MM))) +
        _tag(0x0020,0x000D,"UI",ui(STUDY_UID)) +
        _tag(0x0020,0x000E,"UI",ui(series_uid)) +
        _tag(0x0020,0x0013,"IS",is_(str(slice_idx+1))) +
        _tag(0x0020,0x0032,"DS",ds(f"0\\0\\{z_mm:.4f}")) +
        _tag(0x0020,0x0037,"DS",ds("1\\0\\0\\0\\1\\0")) +
        _tag(0x0020,0x1041,"DS",ds(f"{z_mm:.4f}")) +       # SliceLocation
        _tag(0x0028,0x0002,"US",us(1)) +
        _tag(0x0028,0x0004,"CS",cs("MONOCHROME2")) +
        _tag(0x0028,0x0010,"US",us(rows)) +
        _tag(0x0028,0x0011,"US",us(cols)) +
        _tag(0x0028,0x0030,"DS",ds(f"{VOXEL_MM}\\{VOXEL_MM}")) +
        _tag(0x0028,0x0100,"US",us(16)) +
        _tag(0x0028,0x0101,"US",us(16)) +
        _tag(0x0028,0x0102,"US",us(15)) +
        _tag(0x0028,0x0103,"US",us(1)) +
        # Two window presets: soft-tissue (C=40,W=400) + lesion (C=80,W=160)
        _tag(0x0028,0x1050,"DS",ds("40\\80")) +
        _tag(0x0028,0x1051,"DS",ds("400\\160")) +
        _tag(0x0028,0x1052,"DS",ds("0")) +
        _tag(0x0028,0x1053,"DS",ds("1")) +
        _tag(0x7FE0,0x0010,"OW",px)
    )

    with open(path, "wb") as f:
        f.write(b"\x00"*128 + b"DICM" + meta + ds_block)


# ─────────────────────────────────────────────────────────────────────────────
# Masks  (2D; applied to every slice)
# ─────────────────────────────────────────────────────────────────────────────

def build_masks_2d():
    """Return 2-D boolean arrays for body, metal and lesion cross-sections."""
    y, x = np.ogrid[:Y_DIM, :X_DIM]
    body   = (((x-X_DIM//2)/BODY_SEMI_X)**2 + ((y-Y_DIM//2)/BODY_SEMI_Y)**2) <= 1.0
    metal  = (x-METAL_CENTER_X)**2 + (y-METAL_CENTER_Y)**2 <= METAL_RADIUS_VOX**2
    lesion = (x-LESION_CENTER_X)**2 + (y-LESION_CENTER_Y)**2 <= LESION_RADIUS_VOX**2
    return body, metal, lesion


# ─────────────────────────────────────────────────────────────────────────────
# Artifact template  (computed once via FBP)
# ─────────────────────────────────────────────────────────────────────────────

def _radon(image, angles_deg):
    N = image.shape[0]
    sino = np.zeros((N, len(angles_deg)), dtype=np.float32)
    for i, ang in enumerate(angles_deg):
        rot = rotate(image.astype(np.float32), ang, reshape=False,
                     order=1, mode='constant', cval=0)
        sino[:, i] = rot.sum(axis=0)
    return sino


def _fbp(sinogram, angles_deg):
    n_r, n_ang = sinogram.shape
    freqs = np.fft.rfftfreq(n_r)
    ramp  = np.abs(freqs); ramp[0] = 0.0
    filtered = np.zeros_like(sinogram)
    for i in range(n_ang):
        filtered[:, i] = np.fft.irfft(np.fft.rfft(sinogram[:, i]) * ramp, n=n_r)
    N = n_r
    recon = np.zeros((N, N), dtype=np.float64)
    for i, ang in enumerate(angles_deg):
        proj_2d  = np.tile(filtered[:, i], (N, 1))
        recon   += rotate(proj_2d.astype(np.float32), -ang, reshape=False,
                          order=1, mode='constant', cval=0)
    return (recon * np.pi / (2*n_ang)).astype(np.float32)


def build_artifact_template(body_mask, metal_mask):
    """
    Simulate metal streak artifacts via photon-starvation FBP.
    Returns a 2-D float32 array (same shape as one slice) to be ADDED to
    every slice of the noMAR volumes.  Computed once; identical for all
    slices (metal is a cylinder → same cross-section every z).
    """
    print("  Computing FBP artifact template (once) …", flush=True)
    angles = np.linspace(0, 180, N_PROJECTION_ANGLES, endpoint=False)

    phantom = np.zeros((Y_DIM, X_DIM), np.float32)
    phantom[body_mask]  = float(HU_BACKGROUND)
    phantom[metal_mask] = float(HU_METAL)

    sino_clean = _radon(phantom, angles)

    # Threshold at 99th percentile separates metal-contaminated rays cleanly
    nz = sino_clean[sino_clean > 0]
    threshold    = np.percentile(nz, 99.0)
    starvation   = np.percentile(nz, 50) * 0.02   # near noise floor

    corrupted = sino_clean.copy()
    corrupted[sino_clean > threshold] = starvation

    recon_clean   = _fbp(sino_clean, angles)
    recon_corrupt = _fbp(corrupted,  angles)

    template = recon_corrupt - recon_clean
    template[metal_mask] = 0.0
    template[~body_mask] = 0.0

    peak = np.abs(template[body_mask & ~metal_mask]).max()
    if peak > 0:
        template *= ARTIFACT_PEAK_HU / peak

    print(f"  Artifact template: dark={template.min():.0f} HU, "
          f"bright={template.max():.0f} HU", flush=True)
    return template


# ─────────────────────────────────────────────────────────────────────────────
# Volume generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_volumes():
    """
    For each realization i (1…N):
      • Draw noise vol from rng(BASE_SEED+i); zero outside body.
      • noMAR vol  = base + artifact_template + noise; metal → HU_METAL.
      • MAR vol    = base + noise;                      metal → HU_METAL.
      • LP variant: lesion pixels SET TO HU_LESION (homogeneous — no noise).
      • LA variant: no change to lesion region.
    Four output folders per realization:
      lesion_absent_NN_noMAR, lesion_present_NN_noMAR,
      lesion_absent_NN_MAR,   lesion_present_NN_MAR.
    """
    if os.path.exists(DICOM_DIR):
        shutil.rmtree(DICOM_DIR)
    os.makedirs(DICOM_DIR)

    body_mask, metal_mask, lesion_mask = build_masks_2d()
    artifact_2d = build_artifact_template(body_mask, metal_mask)

    # Base 2-D slice (no noise, no lesion)
    base_2d = np.full((Y_DIM, X_DIM), HU_AIR, dtype=np.float32)
    base_2d[body_mask]  = HU_BACKGROUND
    base_2d[metal_mask] = HU_METAL

    # Artifact is the same every slice (cylinder → constant cross-section)
    art_2d = artifact_2d   # shape (Y_DIM, X_DIM)

    # Lesion annotation for series description
    desc_lx = f"X={LESION_CENTER_X},Y={LESION_CENTER_Y},r={LESION_RADIUS_VOX}vox,allZ"

    total = NUM_REALIZATIONS
    for i in range(1, total+1):
        rng   = np.random.default_rng(BASE_SEED + i)
        # Independent noise volume: (Z, Y, X), only inside body
        noise_3d = rng.normal(0.0, NOISE_STD_HU,
                              (Z_DIM, Y_DIM, X_DIM)).astype(np.float32)
        noise_3d[:, ~body_mask] = 0.0

        conditions = [
            ("lesion_absent_noMAR",  f"{UID_ROOT}.3.{i:02d}.0", False, True,
             f"LA noMAR R{i:02d} | artifacts | no lesion"),
            ("lesion_present_noMAR", f"{UID_ROOT}.3.{i:02d}.1", True,  True,
             f"LP noMAR R{i:02d} | artifacts | lesion {desc_lx}"),
            ("lesion_absent_MAR",    f"{UID_ROOT}.4.{i:02d}.0", False, False,
             f"LA MAR   R{i:02d} | clean     | no lesion"),
            ("lesion_present_MAR",   f"{UID_ROOT}.4.{i:02d}.1", True,  False,
             f"LP MAR   R{i:02d} | clean     | lesion {desc_lx}"),
        ]

        for cond_name, series_uid, add_lesion, add_artifacts, series_desc in conditions:
            folder = os.path.join(DICOM_DIR, f"{cond_name}_{i:02d}")
            os.makedirs(folder)

            for z in range(Z_DIM):
                # Start from base 2-D slice
                sl = base_2d.copy()

                # Add artifacts (noMAR only)
                if add_artifacts:
                    sl = sl + art_2d

                # Add noise (always)
                sl = sl + noise_3d[z]

                # Restore metal (overrides noise + artifacts)
                sl[metal_mask] = HU_METAL

                # Lesion: REPLACE with homogeneous HU (no noise inside lesion)
                # This is the key correction: set, don't add, and use fixed HU
                if add_lesion:
                    sl[lesion_mask] = HU_LESION

                sop_uid = f"{series_uid}.{z+1}"
                fname   = os.path.join(folder, f"slice{z+1:03d}.dcm")
                write_ct_dicom(fname, sl, z, series_uid, sop_uid, series_desc)

        print(f"  [{i}/{total}] Realization {i:02d} done "
              f"(4 series × {Z_DIM} slices)", flush=True)

    print(f"DICOM written to: {DICOM_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Metadata CSV
# ─────────────────────────────────────────────────────────────────────────────

def generate_metadata_csv():
    rows = [
        ("Matrix (Z, Y, X)",            f"{Z_DIM}, {Y_DIM}, {X_DIM}", "voxels", ""),
        ("Voxel size",                   f"{VOXEL_MM}",                "mm",     "isotropic"),
        ("Body phantom semi-axis X/Y",  f"{BODY_SEMI_X*VOXEL_MM:.0f} / {BODY_SEMI_Y*VOXEL_MM:.0f}",
                                                                        "mm",     "elliptical cross-section"),
        ("Metal geometry",               "cylinder, full z-extent",    "",       ""),
        ("Metal center X, Y",           f"{METAL_CENTER_X}, {METAL_CENTER_Y}", "voxels", ""),
        ("Metal diameter",              f"{METAL_RADIUS_MM*2:.0f}",    "mm",     ""),
        ("Metal HU",                    f"{HU_METAL}",                 "HU",     "fixed"),
        ("Lesion geometry",              "cylinder, full z-extent",    "",       "matches Vaishnav rod-signal"),
        ("Lesion center X, Y",          f"{LESION_CENTER_X}, {LESION_CENTER_Y}", "voxels",
                                                                                  "5 mm beyond metal boundary"),
        ("Lesion diameter",             f"{LESION_RADIUS_MM*2:.0f}",  "mm",     ""),
        ("Lesion HU",                   f"{HU_LESION}",               "HU",     "homogeneous; no noise inside"),
        ("Lesion delta S",              f"{HU_LESION-HU_BACKGROUND}", "HU",     "signal-background difference"),
        ("Lesion present in slice",      "ALL (1–256)",                "",       "cylindrical geometry"),
        ("Background HU",               f"{HU_BACKGROUND}",           "HU",     "soft tissue"),
        ("Air HU",                      f"{HU_AIR}",                  "HU",     "outside body"),
        ("Noise std (inside body)",     f"{NOISE_STD_HU:.0f}",        "HU",     "Gaussian; absent inside lesion"),
        ("Base RNG seed",               f"{BASE_SEED}",               "",       "realization i → seed BASE+i"),
        ("Artifact model",               "Photon-starvation FBP",     "",       ""),
        ("Projection angles",           f"{N_PROJECTION_ANGLES}",     "",       "uniform 0–180°"),
        ("Artifact peak magnitude",     f"{ARTIFACT_PEAK_HU}",        "HU",     ""),
        ("Realizations",                f"{NUM_REALIZATIONS}",         "",       "per condition"),
        ("DICOM Window 1 (C/W)",         "40 / 400",                  "HU",     "soft tissue"),
        ("DICOM Window 2 (C/W)",         "80 / 160",                  "HU",     "lesion-centred"),
        ("CHO channel type",             "Laguerre-Gauss or DDOG",    "",       "10 channels recommended"),
        ("AUC estimator",                "Mann-Whitney",              "",       "mid-rank tie handling"),
        ("Reference",                    "Vaishnav et al., Med Phys 47(8) 2020", "", ""),
    ]
    with open(METADATA_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Parameter","Value","Units","Notes"])
        w.writerows(rows)
    print(f"Metadata CSV: {METADATA_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 checksums
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def generate_checksums():
    with open(CHECKSUMS_FILE, "w") as out:
        out.write(f"# MAR ILS SHA-256  {datetime.now().isoformat()}\n\n")
        for root, dirs, files in os.walk(DICOM_DIR):
            dirs.sort()
            for fname in sorted(files):
                if fname.lower().endswith(".dcm"):
                    fp  = os.path.join(root, fname)
                    rel = os.path.relpath(fp, SCRIPT_DIR)
                    out.write(f"{sha256_file(fp)}  {rel}\n")
    print(f"Checksums: {CHECKSUMS_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Lab instructions PDF
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf():
    doc = SimpleDocTemplate(PDF_FILE, pagesize=letter,
                            topMargin=0.75*inch, bottomMargin=0.75*inch,
                            leftMargin=inch, rightMargin=inch)
    styles = getSampleStyleSheet()
    T  = ParagraphStyle("T",  parent=styles["Title"],   fontSize=16, spaceAfter=6)
    H1 = ParagraphStyle("H1", parent=styles["Heading1"],fontSize=13, spaceBefore=14, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"],fontSize=11, spaceBefore=8,  spaceAfter=3)
    B  = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=10, leading=14,     spaceAfter=6)
    W  = ParagraphStyle("W",  parent=styles["Normal"],  fontSize=10, leading=13,
                         backColor=colors.lightyellow, borderPadding=(4,6,4,6), spaceAfter=8)

    def tbl(data, widths):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#003366")),
            ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
            ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,colors.HexColor("#EEF2F7")]),
            ("GRID",           (0,0),(-1,-1), 0.4, colors.grey),
            ("TOPPADDING",     (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 4),
        ]))
        return t

    story = [
        Paragraph("MAR ILS Lab Instructions", T),
        Paragraph("Metal Artifact Reduction Interlaboratory Study — v3.0",
                  styles["Heading2"]),
        Paragraph(f"Generated {date.today().isoformat()}", styles["Italic"]),
        HRFlowable(width="100%",thickness=1,color=colors.grey,spaceAfter=12),
        Paragraph("1. Purpose", H1),
        Paragraph(
            "This protocol describes how participating laboratories should process "
            "the synthetic CT dataset and compute ΔAUC = AUC(MAR) − AUC(noMAR) "
            "following the Vaishnav et al. (2020) framework for objective MAR "
            "performance assessment.", B),
        Paragraph("2. Signal Design (Vaishnav et al. alignment)", H1),
        Paragraph(
            "The lesion signal is a CYLINDRICAL rod running the full z-extent of "
            "the volume (matching the 'removable rod' geometry of a physical MAR "
            "phantom). It is placed 5 mm beyond the metal boundary in +x. Inside "
            "the cylinder the HU is fixed at 120 HU (homogeneous — no noise), "
            "against a noisy soft-tissue background of 40±30 HU. The signal is "
            "ABSENT from the lesion-absent (LA) series and PRESENT in all 256 "
            "slices of the lesion-present (LP) series. This means the lesion is "
            "visible in ANY slice when LP and LA are compared side-by-side.", B),
        Paragraph("3. Archive Contents", H1),
        tbl([
            ["Folder pattern",                "Contents"],
            ["lesion_absent_NN_noMAR/",       "Tissue + metal + artifacts, no lesion"],
            ["lesion_present_NN_noMAR/",      "Tissue + metal + artifacts + cylindrical lesion rod"],
            ["lesion_absent_NN_MAR/",         "Tissue + metal, artifacts removed (ideal MAR)"],
            ["lesion_present_NN_MAR/",        "Tissue + metal, no artifacts + cylindrical lesion rod"],
            ["metadata.csv",                  "All phantom parameters"],
            ["checksums_sha256.txt",          "Per-file SHA-256 manifest"],
            ["MAR_ILS_Lab_Instructions.pdf",  "This document"],
        ], [2.8*inch, 3.8*inch]),
        Spacer(1,8),
        Paragraph("4. Key Parameters", H1),
        tbl([
            ["Parameter","Value","Notes"],
            ["Lesion geometry","Cylinder, all slices","Present in slices 1–256 of LP series"],
            ["Lesion centre (X,Y)","276, 256","5 mm beyond metal boundary in +x"],
            ["Lesion diameter","5 mm","10 voxels @ 0.5 mm/vox"],
            ["Lesion HU","120","Homogeneous; delta = 80 HU above background"],
            ["Background HU","40 ± 30","Soft tissue with Gaussian noise"],
            ["Metal HU","3000","10 mm cylindrical rod"],
            ["Artifact peak","±400 HU","FBP photon-starvation model"],
            ["Window 1 (default)","C=40, W=400","Soft-tissue preset; background=50% grey"],
            ["Window 2 (lesion)","C=80, W=160","Lesion at 120 HU → 75% grey; use this to see lesion clearly"],
        ], [2.2*inch,1.8*inch,2.6*inch]),
        Spacer(1,8),
        Paragraph("5. How to Verify the Lesion Visually", H1),
        Paragraph(
            "Open any slice from a lesion_present series and compare to the same "
            "slice from a lesion_absent series. In your DICOM viewer select Window "
            "preset 2 (C=80, W=160). The lesion appears as a uniform grey disc at "
            "image coordinates X=276, Y=256, approximately 10 pixels in diameter. "
            "It is present in ALL 256 slices. In the noMAR series it may be "
            "partially obscured by dark/bright streak artifacts radiating from "
            "the metal rod (white dot at X=256, Y=256).", B),
        Paragraph("6. Procedure", H1),
    ]
    for title, body in [
        ("Step 1 — Verify checksums",
         "Check every DICOM file against checksums_sha256.txt before processing."),
        ("Step 2 — Process noMAR series",
         "Load the _noMAR series through your MAR-disabled pipeline. Record all settings."),
        ("Step 3 — Process MAR series",
         "Load the _MAR series with MAR enabled. All other parameters must be identical."),
        ("Step 4 — Apply 3-D CHO",
         "Apply the 3-D channelized Hotelling observer to LP and LA pairs for each "
         "condition. Extract a 121×121 pixel ROI centred on X=276, Y=256 in each slice, "
         "then form the 3-D volume ROI. Compute AUC(noMAR) and AUC(MAR) via Mann-Whitney."),
        ("Step 5 — Compute ΔAUC",
         "ΔAUC = AUC(MAR) − AUC(noMAR). Report mean, SD, and 95% bootstrap CI."),
    ]:
        story.append(KeepTogether([Paragraph(title,H2), Paragraph(body,B)]))

    story.append(Paragraph("7. Compliance Requirements", H1))
    for txt in [
        "Do NOT modify the dataset. Any alteration invalidates the result.",
        "Processing parameters must be identical between MAR and noMAR conditions.",
        "Use 3-D (volumetric) CHO — 2-D slice-by-slice implementations are prohibited.",
        "Use 64-bit floating-point arithmetic throughout CHO computation.",
        "Report all deviations from this protocol in your submission.",
    ]:
        story.append(Paragraph(f"▶  {txt}", W))

    doc.build(story)
    print(f"PDF: {PDF_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# ZIP archive
# ─────────────────────────────────────────────────────────────────────────────

def create_zip():
    for fp in [PDF_FILE, METADATA_FILE, CHECKSUMS_FILE]:
        if not os.path.isfile(fp):
            raise RuntimeError(f"Missing: {fp}")
    dcm_files = sorted(
        os.path.join(r, f)
        for r, _, files in os.walk(DICOM_DIR)
        for f in files if f.lower().endswith(".dcm")
    )
    expected = NUM_REALIZATIONS * 4 * Z_DIM
    if len(dcm_files) != expected:
        raise RuntimeError(f"Expected {expected} DICOMs, found {len(dcm_files)}")
    print(f"Verified {len(dcm_files)} DICOM files — zipping …")
    with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in [PDF_FILE, METADATA_FILE, CHECKSUMS_FILE]:
            zf.write(fp, os.path.basename(fp))
        for fp in dcm_files:
            zf.write(fp, os.path.relpath(fp, SCRIPT_DIR))
    archive_sha = sha256_file(ZIP_FILE)
    with open(ZIP_FILE+".sha256","w") as f:
        f.write(f"{archive_sha}  {os.path.basename(ZIP_FILE)}\n")
    mb = os.path.getsize(ZIP_FILE)/(1024**2)
    print(f"Archive: {ZIP_FILE}  ({mb:.1f} MB)  SHA-256: {archive_sha}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== MAR ILS Dataset Generator v3.0 ===\n")
    print("[1/5] PDF …");       generate_pdf()
    print("\n[2/5] Volumes …"); generate_volumes()
    print("\n[3/5] CSV …");     generate_metadata_csv()
    print("\n[4/5] Checksums …");generate_checksums()
    print("\n[5/5] ZIP …");     create_zip()
    print("\n=== Done ===")
