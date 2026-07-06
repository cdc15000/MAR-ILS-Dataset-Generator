"""
Normative constants for ASTM WKXXXXX Rev 05 (Fan-Beam, v7.0.0).

This module is the single source of truth for all physics, geometry, and
observer parameters. Both generator_v7_0_0.py and run_cho_analysis_v7_0.py
import from here.
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Volume geometry (§A1.1)
# ═══════════════════════════════════════════════════════════════════════════════

X_DIM: int = 512
Y_DIM: int = 512
Z_DIM: int = 256
VOXEL_MM: float = 0.5
VOXEL_CM: float = VOXEL_MM / 10.0  # 0.05 cm
PHANTOM_CENTER_X: int = 256
PHANTOM_CENTER_Y: int = 256

# ═══════════════════════════════════════════════════════════════════════════════
# Fan-beam geometry [Rev 05] (§A1.1(f,g))
# ═══════════════════════════════════════════════════════════════════════════════

SID_MM: float = 570.0       # source-to-isocenter distance
SDD_MM: float = 1040.0      # source-to-detector distance
SID_CM: float = SID_MM / 10.0
SDD_CM: float = SDD_MM / 10.0
SID_VOX: float = SID_MM / VOXEL_MM   # 1140.0 voxels
SDD_VOX: float = SDD_MM / VOXEL_MM   # 2080.0 voxels
N_ANGLES: int = 720          # full 360° rotation
N_DET: int = 512             # equi-angular detector elements

# Detector fan angles
_FOV_HALF_MM: float = X_DIM * VOXEL_MM / 2.0  # 128 mm
GAMMA_MAX_RAD: float = float(np.arcsin(_FOV_HALF_MM / SID_MM))
DELTA_GAMMA_RAD: float = 2.0 * GAMMA_MAX_RAD / N_DET
DET_FAN_ANGLES_RAD: np.ndarray = (
    (np.arange(N_DET) - N_DET / 2.0 + 0.5) * DELTA_GAMMA_RAD
)
COS_DET_FAN: np.ndarray = np.cos(DET_FAN_ANGLES_RAD)

# Source rotation angles
ANGLES_DEG: np.ndarray = np.linspace(0.0, 360.0, N_ANGLES, endpoint=False)
ANGLES_RAD: np.ndarray = np.deg2rad(ANGLES_DEG)

# Ray-tracing parameters (forward projection)
_DIAG_VOX: float = float(np.sqrt(X_DIM**2 + Y_DIM**2)) / 2.0
_RAY_T_START: float = SID_VOX - _DIAG_VOX - 10.0
_RAY_T_END: float = SID_VOX + _DIAG_VOX + 10.0
_RAY_STEP: float = 0.4   # voxel step along each ray (sub-voxel accuracy)
_RAY_T_VALS: np.ndarray = np.arange(_RAY_T_START, _RAY_T_END, _RAY_STEP)
_N_RAY_SAMPLES: int = len(_RAY_T_VALS)

# ═══════════════════════════════════════════════════════════════════════════════
# Physical constants (NIST XCOM, 60 keV monochromatic)
# ═══════════════════════════════════════════════════════════════════════════════

MU_AIR_CM: float = 0.000196
MU_TISSUE_CM: float = 0.2059
MU_IRON_CM: float = 2.408
BACKGROUND_HU: float = 40.0
METAL_HU: float = 3000.0

# ═══════════════════════════════════════════════════════════════════════════════
# Phantom geometry (§A1.2–§A1.4)
# ═══════════════════════════════════════════════════════════════════════════════

BODY_SEMI_X_VOX: int = round(85.0 / VOXEL_MM)   # 170
BODY_SEMI_Y_VOX: int = round(60.0 / VOXEL_MM)   # 120
METAL_RADIUS_VOX: int = round(5.0 / VOXEL_MM)    # 10
LESION_RADIUS_VOX: int = round(2.5 / VOXEL_MM)   # 5
LESION_CENTER_X: int = 281    # §A1.4(c): 256 + 10 + 10 + 5
LESION_SLICE_INDEX: int = 128

# ═══════════════════════════════════════════════════════════════════════════════
# Lesion contrast (§A1.4(e))
# ═══════════════════════════════════════════════════════════════════════════════

LESION_DELTA_HU: float = 12.0
MU_LESION_CM: float = MU_TISSUE_CM * (1.0 + LESION_DELTA_HU / 1000.0)

# ═══════════════════════════════════════════════════════════════════════════════
# Noise model
# ═══════════════════════════════════════════════════════════════════════════════

SCATTER_FRAC: float = 0.05
SIGMA_E_COUNTS: float = 5.0
NOISE_SIGMA_TARGET_HU: float = 30.0
JITTER_MAX_DEG: float = 15.0

# ═══════════════════════════════════════════════════════════════════════════════
# Study parameters
# ═══════════════════════════════════════════════════════════════════════════════

NUM_REALIZATIONS_DEFAULT: int = 40
BASE_SEED: int = 20260314

# ═══════════════════════════════════════════════════════════════════════════════
# CHO observer parameters (§A1.5, Rev 05)
# ═══════════════════════════════════════════════════════════════════════════════

ROI_SIZE: int = 121               # 121×121 voxels
ROI_CENTER_X: int = 281           # lesion centre x (§A1.4(c))
ROI_CENTER_Y: int = 256           # phantom centre y
NUM_CHANNELS: int = 10            # Laguerre-Gauss channels
CHANNEL_WIDTH_A: float = 7.5      # 1.5 × r_lesion = 1.5 × 5 = 7.5 voxels
N_BOOT: int = 1000
AUC_TOLERANCE: float = 0.005      # ±0.005 AUC (§8.4, Rev 05)
DEFAULT_SIGMA_SWEEP: list[float] = [0, 5, 10, 15, 20, 25, 30, 40, 50, 65, 80]

# ═══════════════════════════════════════════════════════════════════════════════
# DICOM 2026b CP-2575 tags
# ═══════════════════════════════════════════════════════════════════════════════

TAG_MAR_SEQ: int = 0x00189390           # Metal Artifact Reduction Sequence
TAG_MAR_APPLIED: int = 0x00189391       # Metal Artifact Reduction Applied
