# De Man Calibration Sweep — Passability Memo

**Date:** 2026-03-16
**Author:** CHO Analysis Pipeline (automated)
**Subject:** Tier recalibration to De Man robust reconstruction range (15%–25% blockage)

---

## 1. Objective

Recalibrate all three ASTM WKXXXXX tiers from their current angular blockage
levels to the midpoint of the De Man robust reconstruction range (22.5%).
Hypothesis: reducing blockage from ~40% to ~23% moves tiers out of the
"information-theoretic event horizon" and creates an achievable path to
SUPERIORITY (DAUC > 0) for sinogram-domain MAR algorithms.

## 2. Geometric Optimization Results

| Tier    | Old R (mm) | Old Block% | New R (mm) | New Block% | New Gap (vox) | Clinical Floor |
|---------|-----------|-----------|-----------|-----------|--------------|----------------|
| T1_AB   | 10.0      | 41.45     | 5.5       | 21.63     | 16           | N/A            |
| T2_SB   | 5.0       | 40.04     | 3.0       | 22.96     | 7            | OK (diam=6mm > 2mm) |
| T3_HEAD | 8.0       | 16.09     | 11.0      | 22.34     | 39           | N/A            |

Formula: `blockage = (2/pi) * arcsin(R_metal / L_nominal)`

All tiers now sit within 21.6%–23.0% blockage.
T2_SB rod diameter = 6.0 mm, well above the 2.0 mm pediatric clinical floor.

## 3. CHO Analysis — T2_SB (De Man-Calibrated)

Regenerated 40 LP + 40 LA realizations at the new T2_SB geometry
(R=6 vox, blockage=22.96%). Ran iMAR and FS-iMAR, both with
`--internal-noise-sigma 15`.

### 3.1 Results

| Condition  | AUC (LOO) | 95% CI            | DAUC   | 95% CI DAUC           | Outcome       |
|------------|----------|-------------------|--------|-----------------------|---------------|
| noMAR      | 0.8019   | [0.7362, 0.8806]  | —      | —                     | (baseline)    |
| iMAR       | 0.5394   | [0.5181, 0.5731]  | -0.2625| [-0.3381, -0.1913]    | INFERIOR      |
| FS-iMAR    | 0.5400   | [0.5125, 0.5775]  | -0.2619| [-0.3331, -0.1925]    | INFERIOR      |

### 3.2 Comparison with Old 40% Blockage T2_SB

| Metric         | Old (40% block) | New (23% block) | Direction |
|----------------|----------------|----------------|-----------|
| AUC_noMAR      | 0.5700         | 0.8019         | +0.2319 (better) |
| AUC_FS-iMAR    | 0.5106         | 0.5400         | +0.0294 (marginal) |
| DAUC (FS-iMAR) | -0.0594        | -0.2619        | -0.2025 (MUCH WORSE) |

## 4. Analysis — Why the De Man Hypothesis Failed

The De Man recalibration dramatically improved the noMAR baseline
(AUC: 0.57 -> 0.80) because at 23% blockage, metal streaks are mild
and the 12 HU lesion is readily detectable by the CHO without any correction.

However, all sinogram-domain MAR algorithms (iMAR, FS-iMAR) still replace
the metal-trace rays (~23% of projections) with an analytic tissue prior
set uniformly to 40 HU — a prior that contains **no lesion information**.
This replacement destroys the lesion signal carried by those rays.

**At high blockage (40%):** The metal-corrupted rays carry nearly zero
useful lesion signal (overwhelmed by metal attenuation and scatter noise).
Replacing them with a flat prior loses little; DAUC ~ -0.06 (INDETERMINATE).

**At low blockage (23%):** The metal-corrupted rays still carry significant
lesion signal (metal shadow is weaker, photon counts higher). Replacing them
with a flat prior destroys **real information**. Meanwhile, the noMAR baseline
is already excellent. Net result: DAUC ~ -0.26 (INFERIOR).

### The Fundamental Paradox

```
Blockage ↓  →  noMAR quality ↑↑  →  MAR algorithms destroy more signal than they recover
Blockage ↑  →  noMAR quality ↓↓  →  MAR algorithms cannot recover destroyed information
```

There is no "sweet spot" where sinogram inpainting MAR algorithms demonstrate
SUPERIORITY via task-based lesion detectability. The ASTM framework correctly
captures this via DAUC: these algorithms trade one form of degradation (streaks)
for another (information loss in the lesion ROI).

## 5. Recommendation

**The De Man recalibration does NOT provide an achievable path to SUPERIORITY.**

The original high-blockage tier geometry (T2_SB at 40%) is actually the better
standard design for two reasons:

1. **Clinical relevance:** The 40% blockage regime represents the challenging
   cases where MAR is most clinically needed (large implants near diagnostic ROIs).
   Testing at 23% blockage evaluates a regime where clinicians rarely invoke MAR.

2. **Metrological sensitivity:** At 40% blockage, the DAUC spread between algorithms
   is narrow (-0.05 to -0.07), making the metric sensitive to algorithmic differences.
   At 23% blockage, all algorithms collapse to DAUC ~ -0.26, losing discriminative power.

### Recommended Standard Language

For ASTM WKXXXXX Rev 03, retain the current tier geometries and define outcomes:

- **SUPERIOR:** DAUC > 0, CI_lower > 0 — MAR improved lesion detectability
- **NON-INFERIOR:** DAUC CI contains 0 — MAR did not degrade detectability
- **INDETERMINATE:** DAUC < 0 but CI contains 0 — insufficient evidence
- **INFERIOR:** DAUC < 0, CI_upper < 0 — MAR degraded lesion detectability

The standard should note that NON-INFERIOR is the realistic best-case outcome
for sinogram-domain MAR at the normative tier geometries. SUPERIORITY requires
fundamentally different algorithmic approaches (e.g., deep learning with
lesion-aware priors, or dual-energy decomposition).

## 6. Tier Config Status

`tier_config.py` has been updated with the De Man-calibrated radii for
experimental documentation purposes. **Revert to original values before
any normative dataset regeneration.**

Original values for restoration:
- T1_AB: `metal_radius_mm=10.0`  (20 vox, 41.45% blockage)
- T2_SB: `metal_radius_mm=5.0`   (10 vox, 40.04% blockage)
- T3_HEAD: `metal_radius_mm=8.0` (16 vox, 16.09% blockage)
