#!/usr/bin/env python3
"""
tier_config.py
==============
ASTM WKXXXXX v1.0.0 — Imaging Scenario Tier Registry

Three tiers for regulatory 510(k) performance declarations:
  T1_AB   — Large adult-body Co-Cr rod  (mandatory anchor tier)
  T2_SB   — Small pediatric-body SS-316L rod  (supplemental)
  T3_HEAD — Circular head phantom Ti-6Al-4V rod  (supplemental)

All tiers share the same acquisition geometry (360 angles, 512 detectors,
0.5 mm voxel, 60 keV monochromatic, parallel-beam) and realization count
(40 LP + 40 LA).  ΔAUC values are therefore directly comparable across tiers
and against the v5.3.0 ASTM baseline (AUC_noMAR = 0.7063).

Usage
-----
    from tier_config import TIER_REGISTRY, TierConfig
    tc = TIER_REGISTRY["T1_AB"]
    print(tc)

    from tier_config import validate_tier_registry
    validate_tier_registry()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ─── Normative acquisition constants (shared across all tiers) ─────────────────
VOXEL_MM: float = 0.5           # mm per voxel (isotropic)
VOXEL_CM: float = VOXEL_MM / 10.0  # 0.05 cm
N_ANGLES: int   = 360
N_DET:    int   = 512

PHANTOM_CENTER_X: int = 256
PHANTOM_CENTER_Y: int = 256

LESION_SLICE_INDEX: int = 128   # zero-indexed central slice
NUM_REALIZATIONS:   int = 40

# Physical constants (NIST XCOM, 60 keV monochromatic)
MU_AIR_CM:    float = 0.000196
MU_TISSUE_CM: float = 0.2059    # cm⁻¹
BACKGROUND_HU:        float = 40.0
METAL_HU_RESTORE:     float = 3000.0
SCATTER_FRAC:         float = 0.05
SIGMA_E_COUNTS:       float = 5.0
NOISE_SIGMA_TARGET_HU: float = 30.0

# Sinogram-domain lesion contrast (same physics for all tiers, §A1.4)
LESION_DELTA_HU: float = 12.0
MU_LESION_SCALE: float = 1.0 + LESION_DELTA_HU / 1000.0  # ≈ 1.012


# ─── TierConfig dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class TierConfig:
    """
    Frozen dataclass encoding all geometry and observer parameters for one
    ASTM WKXXXXX imaging scenario tier.

    Immutable base fields are specified at construction.  Derived fields
    (blockage_frac, channel_width_a, etc.) are computed once in
    ``__post_init__`` via ``object.__setattr__``.
    """

    # ── Identity ───────────────────────────────────────────────────────────────
    tier_id:     str   # "T1_AB", "T2_SB", or "T3_HEAD"
    description: str   # human-readable label for PDF reports

    # ── Body geometry ──────────────────────────────────────────────────────────
    body_semi_x_mm:   float  # column semi-axis (mm); = body_semi_y_mm for circle
    body_semi_y_mm:   float  # row semi-axis (mm)
    is_circular_body: bool   # True → treat body as a circle (head tier)

    # ── Metal insert ───────────────────────────────────────────────────────────
    metal_radius_mm: float  # rod radius (mm)
    metal_mu_cm:     float  # linear attenuation coefficient at 60 keV (cm⁻¹)
    metal_material:  str    # label for reports ("CoCr", "SS316L", "Ti6Al4V")

    # ── Lesion disc ────────────────────────────────────────────────────────────
    lesion_semi_major_mm: float  # semi-axis along displacement direction (mm)
    lesion_semi_minor_mm: float  # semi-axis perpendicular to displacement (mm)
    lesion_center_x:      int    # voxel column; row is always PHANTOM_CENTER_Y

    # ── CHO observer ──────────────────────────────────────────────────────────
    roi_size: int  # square ROI side length in voxels (must be odd)

    # ── Derived fields (init=False; set in __post_init__) ─────────────────────
    metal_radius_vox:      int   = field(init=False)
    body_semi_x_vox:       int   = field(init=False)
    body_semi_y_vox:       int   = field(init=False)
    lesion_semi_major_vox: int   = field(init=False)
    lesion_semi_minor_vox: int   = field(init=False)
    l_nominal_vox:         int   = field(init=False)
    l_nominal_mm:          float = field(init=False)
    gap_vox:               int   = field(init=False)
    gap_mm:                float = field(init=False)
    blockage_frac:         float = field(init=False)
    channel_width_a:       float = field(init=False)   # LG channel width (voxels)
    mu_lesion_cm:          float = field(init=False)
    roi_x_center:          int   = field(init=False)   # = lesion_center_x
    roi_y_center:          int   = field(init=False)   # = PHANTOM_CENTER_Y

    def __post_init__(self) -> None:
        def _s(name: str, val) -> None:
            object.__setattr__(self, name, val)

        mrv  = round(self.metal_radius_mm      / VOXEL_MM)
        bxv  = round(self.body_semi_x_mm       / VOXEL_MM)
        byv  = round(self.body_semi_y_mm       / VOXEL_MM)
        lmav = round(self.lesion_semi_major_mm / VOXEL_MM)
        lmiv = round(self.lesion_semi_minor_mm / VOXEL_MM)

        _s("metal_radius_vox",      mrv)
        _s("body_semi_x_vox",       bxv)
        _s("body_semi_y_vox",       byv)
        _s("lesion_semi_major_vox", lmav)
        _s("lesion_semi_minor_vox", lmiv)

        l_vox = self.lesion_center_x - PHANTOM_CENTER_X
        _s("l_nominal_vox", l_vox)
        _s("l_nominal_mm",  float(l_vox) * VOXEL_MM)

        gap_v = l_vox - mrv - lmav
        _s("gap_vox", gap_v)
        _s("gap_mm",  float(gap_v) * VOXEL_MM)

        # Angular blockage fraction: fraction of 180° projection angles whose
        # ray through the iso-centre passes through the metal rod.
        # Formula: (2/π) × arcsin(R_metal / L_nominal)
        ratio = min(float(mrv) / float(l_vox), 1.0)
        _s("blockage_frac", (2.0 / math.pi) * math.asin(ratio))

        # LG channel width: 1.5 × lesion semi-major axis (voxels)
        _s("channel_width_a", 1.5 * float(lmav))

        # μ_lesion: +12 HU sinogram-domain contrast (same for all tiers)
        _s("mu_lesion_cm", MU_TISSUE_CM * MU_LESION_SCALE)

        # CHO ROI centred on the lesion
        _s("roi_x_center", self.lesion_center_x)
        _s("roi_y_center", PHANTOM_CENTER_Y)

    # ── ROI slice helpers ──────────────────────────────────────────────────────

    def roi_x_bounds(self) -> tuple[int, int]:
        """(x_start, x_end) for numpy index img[y_start:y_end, x_start:x_end]."""
        half = self.roi_size // 2
        return self.roi_x_center - half, self.roi_x_center + half + 1

    def roi_y_bounds(self) -> tuple[int, int]:
        """(y_start, y_end) for numpy index img[y_start:y_end, ...]."""
        half = self.roi_size // 2
        return self.roi_y_center - half, self.roi_y_center + half + 1

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-serializable dict of all fields (for HDF5 attrs and TDP JSON)."""
        return {
            "tier_id":               self.tier_id,
            "description":           self.description,
            "body_semi_x_mm":        self.body_semi_x_mm,
            "body_semi_y_mm":        self.body_semi_y_mm,
            "is_circular_body":      self.is_circular_body,
            "metal_radius_mm":       self.metal_radius_mm,
            "metal_mu_cm":           self.metal_mu_cm,
            "metal_material":        self.metal_material,
            "lesion_semi_major_mm":  self.lesion_semi_major_mm,
            "lesion_semi_minor_mm":  self.lesion_semi_minor_mm,
            "lesion_center_x":       self.lesion_center_x,
            "roi_size":              self.roi_size,
            "metal_radius_vox":      self.metal_radius_vox,
            "body_semi_x_vox":       self.body_semi_x_vox,
            "body_semi_y_vox":       self.body_semi_y_vox,
            "lesion_semi_major_vox": self.lesion_semi_major_vox,
            "lesion_semi_minor_vox": self.lesion_semi_minor_vox,
            "l_nominal_vox":         self.l_nominal_vox,
            "l_nominal_mm":          self.l_nominal_mm,
            "gap_vox":               self.gap_vox,
            "gap_mm":                self.gap_mm,
            "blockage_frac":         self.blockage_frac,
            "channel_width_a":       self.channel_width_a,
            "mu_lesion_cm":          self.mu_lesion_cm,
            "roi_x_center":          self.roi_x_center,
            "roi_y_center":          self.roi_y_center,
        }

    def __str__(self) -> str:
        body_shape = (
            f"circle r={self.body_semi_x_mm:.0f} mm ({self.body_semi_x_vox} vox)"
            if self.is_circular_body
            else (
                f"ellipse {self.body_semi_x_mm:.0f}×{self.body_semi_y_mm:.0f} mm "
                f"({self.body_semi_x_vox}×{self.body_semi_y_vox} vox)"
            )
        )
        xs, xe = self.roi_x_bounds()
        ys, ye = self.roi_y_bounds()
        return (
            f"TierConfig(tier_id={self.tier_id!r})\n"
            f"  description  : {self.description}\n"
            f"  body         : {body_shape}\n"
            f"  metal        : r={self.metal_radius_mm:.1f} mm ({self.metal_radius_vox} vox)"
            f"  μ={self.metal_mu_cm:.3f} cm⁻¹  [{self.metal_material}]\n"
            f"  lesion       : {self.lesion_semi_major_mm:.1f}×"
            f"{self.lesion_semi_minor_mm:.1f} mm  @ x={self.lesion_center_x}\n"
            f"  l_nominal    : {self.l_nominal_mm:.1f} mm ({self.l_nominal_vox} vox)"
            f"   gap={self.gap_mm:.1f} mm ({self.gap_vox} vox)\n"
            f"  blockage     : {self.blockage_frac*100:.2f}%\n"
            f"  CHO ROI      : {self.roi_size}×{self.roi_size} vox  "
            f"x=[{xs}:{xe}]  y=[{ys}:{ye}]  a={self.channel_width_a:.1f} vox"
        )


# ─── Tier Registry ─────────────────────────────────────────────────────────────

TIER_REGISTRY: dict[str, TierConfig] = {

    "T1_AB": TierConfig(
        tier_id="T1_AB",
        description="Large adult-body Co-Cr rod — mandatory 510(k) anchor tier",
        # Body: same geometry as ASTM WKXXXXX v5.3.0 baseline (170×120 vox)
        body_semi_x_mm=85.0,
        body_semi_y_mm=60.0,
        is_circular_body=False,
        # Metal: Co-Cr alloy (20 vox radius, ~41% blockage)
        metal_radius_mm=10.0,   # 20 vox
        metal_mu_cm=4.2,        # Co-Cr at 60 keV (μ_Fe = 2.408, μ_CoCr ≈ 4.2)
        metal_material="CoCr",
        # Lesion: 6×4 vox ellipse, 3.5 mm edge-to-edge gap
        lesion_semi_major_mm=3.0,   # 6 vox
        lesion_semi_minor_mm=2.0,   # 4 vox
        lesion_center_x=289,    # l_nominal = 33 vox = 16.5 mm; gap = 7 vox = 3.5 mm
        roi_size=145,
    ),

    "T2_SB": TierConfig(
        tier_id="T2_SB",
        description="Small pediatric-body SS-316L rod — supplemental tier",
        # Body: half linear scale of T1_AB (85×60 vox)
        body_semi_x_mm=42.5,
        body_semi_y_mm=30.0,
        is_circular_body=False,
        # Metal: 316L stainless steel (10 vox radius, ~40% blockage)
        metal_radius_mm=5.0,    # 10 vox
        metal_mu_cm=2.8,        # 316L SS at 60 keV
        metal_material="SS316L",
        # Lesion: 4×3 vox ellipse, 1.5 mm edge-to-edge gap
        lesion_semi_major_mm=2.0,   # 4 vox
        lesion_semi_minor_mm=1.5,   # 3 vox
        lesion_center_x=273,    # l_nominal = 17 vox = 8.5 mm; gap = 3 vox = 1.5 mm
        roi_size=97,
    ),

    "T3_HEAD": TierConfig(
        tier_id="T3_HEAD",
        description="Circular head phantom Ti-6Al-4V rod — supplemental tier",
        # Body: circular (r = 100 mm = 200 vox), representing cranial anatomy
        body_semi_x_mm=100.0,
        body_semi_y_mm=100.0,   # = body_semi_x_mm for circular body
        is_circular_body=True,
        # Metal: Ti-6Al-4V alloy (16 vox radius, ~16% blockage)
        metal_radius_mm=8.0,    # 16 vox
        metal_mu_cm=1.5,        # Ti-6Al-4V at 60 keV
        metal_material="Ti6Al4V",
        # Lesion: 3×2 vox ellipse, 22.5 mm edge-to-edge gap (low-blockage)
        lesion_semi_major_mm=1.5,   # 3 vox
        lesion_semi_minor_mm=1.0,   # 2 vox
        lesion_center_x=320,    # l_nominal = 64 vox = 32.0 mm; gap = 45 vox = 22.5 mm
        roi_size=73,
    ),
}


# ─── Validation ────────────────────────────────────────────────────────────────

def validate_tier_registry() -> None:
    """
    Run normative assertions on all registered tiers and print a summary table.

    Raises AssertionError on any violation.  Call this at startup in any script
    that imports TIER_REGISTRY to catch accidental modifications.
    """
    print("ASTM WKXXXXX v1.0.0 — Tier Registry Validation")
    print("=" * 68)

    _expected: dict[str, dict] = {
        "T1_AB":   {
            "l_nominal_mm": 16.5, "blockage_lo": 0.40, "blockage_hi": 0.43,
            "lesion_center_x": 289, "roi_size": 145,
        },
        "T2_SB":   {
            "l_nominal_mm": 8.5,  "blockage_lo": 0.39, "blockage_hi": 0.41,
            "lesion_center_x": 273, "roi_size": 97,
        },
        "T3_HEAD": {
            "l_nominal_mm": 32.0, "blockage_lo": 0.15, "blockage_hi": 0.17,
            "lesion_center_x": 320, "roi_size": 73,
        },
    }

    hdr = (f"{'tier_id':<10} {'l_nom_mm':>10} {'blockage%':>10} "
           f"{'lesion_cx':>10} {'ROI':>5} {'a_vox':>7}")
    print(hdr)
    print("-" * 56)

    for tid, tc in TIER_REGISTRY.items():
        exp = _expected[tid]

        assert abs(tc.l_nominal_mm - exp["l_nominal_mm"]) < 0.01, (
            f"{tid}: l_nominal_mm={tc.l_nominal_mm:.3f} ≠ {exp['l_nominal_mm']}"
        )
        assert exp["blockage_lo"] <= tc.blockage_frac <= exp["blockage_hi"], (
            f"{tid}: blockage_frac={tc.blockage_frac:.4f} outside "
            f"[{exp['blockage_lo']}, {exp['blockage_hi']}]"
        )
        assert tc.lesion_center_x == exp["lesion_center_x"], (
            f"{tid}: lesion_center_x={tc.lesion_center_x} ≠ {exp['lesion_center_x']}"
        )
        assert tc.roi_size == exp["roi_size"], (
            f"{tid}: roi_size={tc.roi_size} ≠ {exp['roi_size']}"
        )
        assert tc.roi_size % 2 == 1, f"{tid}: roi_size must be odd, got {tc.roi_size}"
        assert tc.gap_vox >= 1, (
            f"{tid}: gap_vox={tc.gap_vox} — lesion overlaps or touches metal edge"
        )
        xs, xe = tc.roi_x_bounds()
        ys, ye = tc.roi_y_bounds()
        assert 0 <= xs and xe <= 512, f"{tid}: ROI x [{xs},{xe}] out of [0,512]"
        assert 0 <= ys and ye <= 512, f"{tid}: ROI y [{ys},{ye}] out of [0,512]"

        print(
            f"{tid:<10} {tc.l_nominal_mm:>10.1f} {tc.blockage_frac*100:>10.2f} "
            f"{tc.lesion_center_x:>10} {tc.roi_size:>5} {tc.channel_width_a:>7.1f}"
        )

    print()
    print("T1_AB  — mandatory anchor tier for all 510(k) submissions.")
    print("T2_SB  — supplemental (claim-specific; small body / pediatric).")
    print("T3_HEAD — supplemental (claim-specific; cranial implant).")
    print("All assertions passed.")


if __name__ == "__main__":
    validate_tier_registry()
    print()
    for tc in TIER_REGISTRY.values():
        print(tc)
        print()
