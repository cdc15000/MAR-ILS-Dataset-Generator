#!/usr/bin/env python3
"""
MAR ILS Dataset Generator  —  v3.0.1
====================================
Generates a synthetic CT DICOM dataset for a Metal Artifact Reduction (MAR)
interlaboratory study (ILS), following the framework of Vaishnav et al.
(Medical Physics, 47(8), 2020) and aligned with ASTM WKXXXXX.
"""

import os, shutil, struct, hashlib, csv
import numpy as np
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import rotate

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # noqa: N801
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable
            desc = kwargs.get("desc", "")
            total = kwargs.get("total", "?")
            print(f"{desc} ({total} items) …", flush=True)
        def __iter__(self): return iter(self._it)
        def __enter__(self): return self
        def __exit__(self, *a): print()
        def update(self, n=1): pass
        def set_postfix_str(self, s): pass

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

# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
Z_DIM, Y_DIM, X_DIM = 256, 512, 512
VOXEL_MM = 0.5

BODY_SEMI_X = int(170 / VOXEL_MM)
BODY_SEMI_Y = int(120 / VOXEL_MM)

METAL_CENTER_X   = 256
METAL_CENTER_Y   = 256
METAL_RADIUS_MM  = 5.0
METAL_RADIUS_VOX = int(round(METAL_RADIUS_MM / VOXEL_MM))

METAL_BOUNDARY_X  = METAL_CENTER_X + METAL_RADIUS_VOX
LESION_OFFSET_VOX = int(round(5.0  / VOXEL_MM))
LESION_CENTER_X   = METAL_BOUNDARY_X + LESION_OFFSET_VOX
LESION_CENTER_Y   = 256
LESION_RADIUS_MM  = 2.5
LESION_RADIUS_VOX = int(round(LESION_RADIUS_MM / VOXEL_MM))

# ─────────────────────────────────────────────────────────────────────────────
# HU values
# ─────────────────────────────────────────────────────────────────────────────
HU_AIR        = -1000
HU_BACKGROUND =    40

# TASK DIFFICULTY TUNING: 
# 42 HU Lesion - 40 HU Background = 2 HU Contrast (Whisper Contrast)
HU_TARGET_SKE =    42   
HU_METAL      =  3000

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
# PDF font names
# ─────────────────────────────────────────────────────────────────────────────
PDF_FONT      = "Helvetica"
PDF_FONT_BOLD = "Helvetica-Bold"

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
        _tag(0x0008,0x1030,"LO",lo("ASTM-WKXXXXX-MAR-ILS")) +
        _tag(0x0008,0x103E,"LO",lo(series_description[:64])) +
        _tag(0x0010,0x0010,"PN",ds(PATIENT_NAME)) +
        _tag(0x0010,0x0020,"LO",lo(PATIENT_ID)) +
        _tag(0x0018,0x0050,"DS",ds(str(VOXEL_MM))) +
        _tag(0x0018,0x0088,"DS",ds(str(VOXEL_MM))) +
        _tag(0x0018,0x1030,"LO",lo("v3.0.1-Reference")) +
        _tag(0x0018,0x1210,"LO",lo("FBP-Synthetic-Noise")) +
        _tag(0x0020,0x000D,"UI",ui(STUDY_UID)) +
        _tag(0x0020,0x000E,"UI",ui(series_uid)) +
        _tag(0x0020,0x0013,"IS",is_(str(slice_idx+1))) +
        _tag(0x0020,0x0032,"DS",ds(f"0\\0\\{z_mm:.4f}")) +
        _tag(0x0020,0x0037,"DS",ds("1\\0\\0\\0\\1\\0")) +
        _tag(0x0020,0x1041,"DS",ds(f"{z_mm:.4f}")) +
        _tag(0x0028,0x0002,"US",us(1)) +
        _tag(0x0028,0x0004,"CS",cs("MONOCHROME2")) +
        _tag(0x0028,0x0010,"US",us(rows)) +
        _tag(0x0028,0x0011,"US",us(cols)) +
        _tag(0x0028,0x0030,"DS",ds(f"{VOXEL_MM}\\{VOXEL_MM}")) +
        _tag(0x0028,0x0100,"US",us(16)) +
        _tag(0x0028,0x0101,"US",us(16)) +
        _tag(0x0028,0x0102,"US",us(15)) +
        _tag(0x0028,0x0103,"US",us(1)) +
        _tag(0x0028,0x1050,"DS",ds("40\\40")) +
        _tag(0x0028,0x1051,"DS",ds("400\\100")) +
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
    print("  Computing FBP artifact template (once) …", flush=True)
    angles = np.linspace(0, 180, N_PROJECTION_ANGLES, endpoint=False)

    phantom = np.zeros((Y_DIM, X_DIM), np.float32)
    phantom[body_mask]  = float(HU_BACKGROUND)
    phantom[metal_mask] = float(HU_METAL)

    sino_clean = _radon(phantom, angles)

    nz = sino_clean[sino_clean > 0]
    threshold  = np.percentile(nz, 99.0)
    starvation = np.percentile(nz, 50) * 0.02

    corrupted = sino_clean.copy()
    corrupted[sino_clean > threshold] = starvation

    recon_clean   = _fbp(sino_clean, angles)
    recon_corrupt = _fbp(corrupted,  angles)

    template = recon_corrupt - recon_clean
    template[metal_mask] = 0.0
    template[~body_mask] = 0.0

    peak_val = np.abs(template[body_mask & ~metal_mask]).max()
    if peak_val > 1e-6:
        template *= ARTIFACT_PEAK_HU / peak_val

    print(f"  Artifact template: dark={template.min():.0f} HU, "
          f"bright={template.max():.0f} HU", flush=True)
    return template

# ─────────────────────────────────────────────────────────────────────────────
# Volume generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_volumes():
    if os.path.exists(DICOM_DIR):
        shutil.rmtree(DICOM_DIR)
    os.makedirs(DICOM_DIR)

    body_mask, metal_mask, lesion_mask = build_masks_2d()
    
    # We build the base template once...
    base_art_2d = build_artifact_template(body_mask, metal_mask)

    base_2d = np.full((Y_DIM, X_DIM), HU_AIR, dtype=np.float32)
    base_2d[body_mask]  = HU_BACKGROUND
    base_2d[metal_mask] = HU_METAL

    desc_lx = f"X={LESION_CENTER_X},Y={LESION_CENTER_Y},r={LESION_RADIUS_VOX}vox,allZ"

    total = NUM_REALIZATIONS
    for i in range(1, total+1):
        rng = np.random.default_rng(BASE_SEED + i)
        noise_3d = rng.normal(0.0, NOISE_STD_HU,
                              (Z_DIM, Y_DIM, X_DIM)).astype(np.float32)
        noise_3d[:, ~body_mask] = 0.0
        
        # --- JITTER IMPLEMENTATION ---
        # Randomly rotate the artifact template by +/- 15 degrees.
        # This breaks static background cancellation in the CHO.
        jitter_deg = rng.uniform(-15.0, 15.0)
        art_2d = rotate(base_art_2d, jitter_deg, reshape=False, order=1, mode='constant', cval=0.0)
        
        # Clean up any interpolation bleed from the rotation
        art_2d[~body_mask] = 0.0
        art_2d[metal_mask] = 0.0
        # -----------------------------

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
                sl = base_2d.copy()
                
                # PHYSICS STACK CORRECTION:
                # 1. Place the underlying physical phantom density (SKE Lesion)
                if add_lesion: 
                    sl[lesion_mask] = HU_TARGET_SKE
                
                # 2. Add the scanner physics (Jittered artifacts and noise pass THROUGH the lesion)
                if add_artifacts: 
                    sl = sl + art_2d
                sl = sl + noise_3d[z]
                
                # 3. Restore the dense metal core (overrides noise and streaks)
                sl[metal_mask] = HU_METAL

                sop_uid = f"{series_uid}.{z+1}"
                fname   = os.path.join(folder, f"slice{z+1:03d}.dcm")
                write_ct_dicom(fname, sl, z, series_uid, sop_uid, series_desc)

        print(f"  [{i}/{total}] Realization {i:02d} done (Jitter: {jitter_deg:+.2f}°) "
              f"(4 series × {Z_DIM} slices)", flush=True)

    print(f"DICOM written to: {DICOM_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Metadata CSV
# ─────────────────────────────────────────────────────────────────────────────
def generate_metadata_csv():
    rows = [
        ("Generator version",            "3.0.1",                       ""),
        ("Matrix (Z, Y, X)",            f"{Z_DIM}, {Y_DIM}, {X_DIM}", "voxels", ""),
        ("Voxel size",                   f"{VOXEL_MM}",                "mm",     "isotropic"),
        ("Body phantom semi-axis X/Y",  f"{BODY_SEMI_X*VOXEL_MM:.0f} / {BODY_SEMI_Y*VOXEL_MM:.0f}", "mm", "elliptical cross-section"),
        ("Metal geometry",               "cylinder, full z-extent",    "",       ""),
        ("Metal center X, Y",           f"{METAL_CENTER_X}, {METAL_CENTER_Y}", "voxels", ""),
        ("Metal diameter",              f"{METAL_RADIUS_MM*2:.0f}",    "mm",     ""),
        ("Metal HU",                    f"{HU_METAL}",                 "HU",     "fixed"),
        ("Lesion geometry",              "cylinder, full z-extent",    "",       "matches Vaishnav rod-signal"),
        ("Lesion center X, Y",          f"{LESION_CENTER_X}, {LESION_CENTER_Y}", "voxels", "5 mm beyond metal boundary"),
        ("Lesion diameter",             f"{LESION_RADIUS_MM*2:.0f}",  "mm",     ""),
        ("Lesion HU (HU_TARGET_SKE)",   f"{HU_TARGET_SKE}",           "HU",     "fixed SKE replacement; 42 HU whisper contrast"),
        ("Lesion delta S",              f"{HU_TARGET_SKE-HU_BACKGROUND}", "HU", "signal-background difference"),
        ("Lesion present in slice",      "ALL (1–256)",                "",       "cylindrical geometry"),
        ("Background HU",               f"{HU_BACKGROUND}",           "HU",     "soft tissue"),
        ("Air HU",                      f"{HU_AIR}",                  "HU",     "outside body"),
        ("Noise std (inside body)",     f"{NOISE_STD_HU:.0f}",        "HU",     "Gaussian; noise added OVER lesion"),
        ("Base RNG seed",               f"{BASE_SEED}",               "",       "realization i → seed BASE+i"),
        ("Artifact model",               "Photon-starvation FBP",     "",       ""),
        ("Artifact Jitter",              "+/- 15 degrees",            "",       "Randomly rotated per realization to prevent CHO cancellation"),
        ("Projection angles",           f"{N_PROJECTION_ANGLES}",     "",       "uniform 0–180°"),
        ("Artifact peak magnitude",     f"{ARTIFACT_PEAK_HU}",        "HU",     ""),
        ("Artifact scaling stability",   "peak_val > 1e-6 guard",     "",       "v3.0.1 improvement"),
        ("Realizations",                f"{NUM_REALIZATIONS}",         "",       "per condition"),
        ("DICOM Study Description",      "ASTM-WKXXXXX-MAR-ILS",      "",       "tag 0008,1030; v3.0.1"),
        ("DICOM Protocol Name",          "v3.0.1-Reference",          "",       "tag 0018,1030; v3.0.1"),
        ("DICOM Convolution Kernel",     "FBP-Synthetic-Noise",       "",       "tag 0018,1210; v3.0.1"),
        ("DICOM Window 1 (C/W)",         "40 / 400",                  "HU",     "soft tissue"),
        ("DICOM Window 2 (C/W)",         "40 / 100",                  "HU",     "lesion-centred low contrast"),
        ("CHO channel type",             "Laguerre-Gauss or DDOG",    "",       "10 channels recommended"),
        ("AUC estimator",                "Mann-Whitney",              "",       "mid-rank tie handling"),
        ("Reference",                    "Vaishnav et al., Med Phys 47(8) 2020", "", ""),
        ("ASTM standard",                "ASTM WKXXXXX",              "",       ""),
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

def _hash_one(fp):
    return os.path.relpath(fp, SCRIPT_DIR), sha256_file(fp)

def generate_checksums(dcm_files=None, num_workers=None):
    if dcm_files is None:
        dcm_files = sorted(
            os.path.join(r, f)
            for r, _, files in os.walk(DICOM_DIR)
            for f in files if f.lower().endswith(".dcm")
        )

    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_hash_one, fp): fp for fp in dcm_files}
        with tqdm(total=len(futures), desc="  Checksums", unit="file",
                  dynamic_ncols=True) as bar:
            for fut in as_completed(futures):
                rel, digest = fut.result()
                results[rel] = digest
                bar.update()

    with open(CHECKSUMS_FILE, "w") as out:
        out.write(f"# MAR ILS SHA-256  {datetime.now().isoformat()}\n\n")
        for rel in sorted(results):
            out.write(f"{results[rel]}  {rel}\n")

    print(f"  Checksums written → {CHECKSUMS_FILE}")

# ─────────────────────────────────────────────────────────────────────────────
# Lab instructions PDF (v3.0.1 Restored & Wrapped)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf():
    doc = SimpleDocTemplate(PDF_FILE, pagesize=letter,
                            topMargin=0.75*inch, bottomMargin=0.75*inch,
                            leftMargin=inch, rightMargin=inch)
    styles = getSampleStyleSheet()
    
    # Custom Styles using registered DejaVu fonts
    T  = ParagraphStyle("T",  parent=styles["Title"], fontName=PDF_FONT_BOLD, fontSize=16, spaceAfter=6)
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=PDF_FONT_BOLD, fontSize=13, spaceBefore=14, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=PDF_FONT_BOLD, fontSize=11, spaceBefore=8,  spaceAfter=3)
    H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName=PDF_FONT,      fontSize=10, spaceBefore=4,  spaceAfter=6)
    B  = ParagraphStyle("B",  parent=styles["Normal"], fontName=PDF_FONT, fontSize=10, leading=14, spaceAfter=6)
    TC = ParagraphStyle("TC",      parent=styles["Normal"], fontName=PDF_FONT,      fontSize=9, leading=11)
    TC_BOLD = ParagraphStyle("TC_BOLD", parent=styles["Normal"], fontName=PDF_FONT_BOLD, fontSize=9, leading=11, textColor=colors.white)
    W  = ParagraphStyle("W",  parent=styles["Normal"], fontName=PDF_FONT, fontSize=10, leading=13,
                         backColor=colors.lightyellow, borderPadding=(4,6,4,6), spaceAfter=8)

    story = []

    # Title Section
    story.append(Paragraph("MAR ILS Lab Instructions", T))
    story.append(Paragraph(f"Metal Artifact Reduction Interlaboratory Study — v3.0.1 (ASTM WKXXXXX)", H3))
    story.append(Paragraph(f"Generated {date.today().isoformat()}", styles["Italic"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey, spaceAfter=12))

    # 1. Purpose
    story.append(Paragraph("1. Purpose", H1))
    story.append(Paragraph(
        "This protocol describes how participating laboratories should process the synthetic CT dataset "
        "and compute ΔAUC = AUC(MAR) − AUC(noMAR) following the <i>Vaishnav, et al.</i> (2020) framework for "
        "objective MAR performance assessment. This study is aligned with ASTM WKXXXXX requirements.", B))

    # 2. Signal Design
    story.append(Paragraph("2. Signal Design", H1))
    story.append(Paragraph(
        f"The lesion signal is a CYLINDRICAL rod running the full z-extent of the volume (matching the "
        f"'removable rod' geometry of a physical MAR phantom). It is placed 5 mm beyond the metal boundary in +x. "
        f"Inside the cylinder the HU is fixed at {HU_TARGET_SKE} HU (Signal-Known-Exactly), against a noisy "
        f"soft-tissue background of {HU_BACKGROUND}±{NOISE_STD_HU:.0f} HU. The signal is ABSENT from the "
        f"lesion-absent (LA) series and PRESENT in all {Z_DIM} slices of the lesion-present (LP) series.", B))

    # 3. Archive Contents
    story.append(Paragraph("3. Folder Contents", H1))
    content_data = [
        ["Folder pattern", "Contents"],
        ["lesion_absent_NN_noMAR/", "Tissue + metal + artifacts, no lesion"],
        ["lesion_present_NN_noMAR/", "Tissue + metal + artifacts + cylindrical lesion rod"],
        ["lesion_absent_NN_MAR/", "Tissue + metal, artifacts removed (ideal MAR)"],
        ["lesion_present_NN_MAR/", "Tissue + metal, no artifacts + cylindrical lesion rod"],
        ["metadata.csv", "All phantom parameters (v3.0.1)"],
        ["checksums_sha256.txt", "Per-file SHA-256 manifest"],
        ["MAR_ILS_Lab_Instructions.pdf", "This document"]
    ]
    # Wrap folder contents table
    wrapped_content = [
        [Paragraph(cell, TC_BOLD) for cell in content_data[0]],
        *[[Paragraph(cell, TC) for cell in row] for row in content_data[1:]]
    ]
    t1 = Table(wrapped_content, colWidths=[2.8*inch, 3.8*inch])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,colors.HexColor("#EEF2F7")]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))

    # 4. Key Parameters (WITH WRAPPING)
    story.append(Paragraph("4. Key Parameters", H1))
    param_data = [
        ["Parameter", "Value", "Notes"],
        ["Lesion geometry", "Cylinder, all slices", "Present in slices 1–256 of LP series"],
        ["Lesion centre (X,Y)", f"{LESION_CENTER_X}, {LESION_CENTER_Y}", "5 mm beyond metal boundary in +x"],
        ["Lesion HU (SKE)", f"{HU_TARGET_SKE} HU", "Fixed intensity; 2 HU whisper contrast above background"],
        ["Background HU", f"{HU_BACKGROUND} ± {NOISE_STD_HU:.0f}", "Soft tissue with Gaussian noise"],
        ["Metal HU", f"{HU_METAL}", "10 mm cylindrical rod"],
        ["Artifact Jitter", "±15° random", "Rotates per realization to block static background cancellation"],
        ["DICOM Tags", "ASTM-WKXXXXX", "StudyDescription (0008,1030) set for traceability"],
        ["Window 2 (lesion)", "C=40, W=100", "Optimized for 42 HU detectability; use this to see lesion clearly"]
    ]
    # Wrap Key Parameters table
    wrapped_params = [
        [Paragraph(cell, TC_BOLD) for cell in param_data[0]],
        *[[Paragraph(cell, TC) for cell in row] for row in param_data[1:]]
    ]
    t2 = Table(wrapped_params, colWidths=[1.8*inch, 1.2*inch, 3.5*inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,colors.HexColor("#EEF2F7")]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))

    # 5. Visual Verification
    story.append(Paragraph("5. How to Verify the Lesion Visually", H1))
    story.append(Paragraph(
        f"Open any slice from a lesion_present series. Select Window preset 2 (C=40, W=100). "
        f"The lesion appears as a faint grey disc at image coordinates X={LESION_CENTER_X}, Y={LESION_CENTER_Y}. "
        f"Due to the low contrast (2 HU), it may be partially obscured by streak artifacts and noise. "
        f"Check ALL slices; the lesion is present throughout the 3D volume.", B))

    # 6. Procedure
    story.append(Paragraph("6. Procedure", H1))
    steps = [
        ("Step 1 — Verify checksums", "Check every DICOM file against checksums_sha256.txt before processing."),
        ("Step 2 — Process noMAR series", "Load the _noMAR series through your MAR-disabled pipeline. Record all settings."),
        ("Step 3 — Process MAR series", "Load the _MAR series with MAR enabled. All other parameters must be identical."),
        ("Step 4 — Apply 3-D CHO", "Apply the 3-D channelized Hotelling observer to LP and LA pairs for each condition. "
                                   "Extract a 121×121 pixel ROI centred on the lesion. Compute AUC via Mann-Whitney."),
        ("Step 5 — Compute ΔAUC", "ΔAUC = AUC(MAR) − AUC(noMAR). Report mean, SD, and 95% bootstrap CI.")
    ]
    for title, body in steps:
        story.append(KeepTogether([Paragraph(title, H2), Paragraph(body, B)]))

    # 7. Compliance
    story.append(Paragraph("7. Compliance Requirements", H1))
    compliance = [
        "Do NOT modify the dataset. Any alteration invalidates the result.",
        "Processing parameters must be identical between MAR and noMAR conditions.",
        "Use 3-D (volumetric) CHO — 2-D slice-by-slice implementations are prohibited.",
        "Use 64-bit floating-point arithmetic throughout CHO computation.",
        "Report all deviations from this protocol in your submission."
    ]
    for txt in compliance:
        story.append(Paragraph(f"▶  {txt}", W))

    doc.build(story)
    print(f"PDF: {PDF_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== MAR ILS Dataset Generator v3.0.1 ===")
    print("[1/4] PDF …");         generate_pdf()
    print("\n[2/4] Volumes …");   generate_volumes()
    print("\n[3/4] CSV …");       generate_metadata_csv()
    print("\n[4/4] Checksums …"); generate_checksums()
    print("\n=== Done ===")