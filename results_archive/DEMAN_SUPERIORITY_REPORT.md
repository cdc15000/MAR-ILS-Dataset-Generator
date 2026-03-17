# De Man Superiority Sprint — Final Report

**Date:** 2026-03-16
**Tier:** T2_SB (Pediatric, SS-316L, 23% blockage — De Man calibrated)
**Baseline:** AUC_noMAR = 0.8019 (σ_internal = 15)

---

## 1. Executive Summary

**The ΔAUC > 0 barrier was NOT broken.** All classical MAR algorithms tested are
definitively INFERIOR at 23% blockage. The best result was MBIR-Superiority
(prior-free, binary-weighted) at ΔAUC = −0.245, which is statistically
indistinguishable from sinogram-inpainting approaches (iMAR ΔAUC = −0.263).

> **ERRATUM (2026-03-16 late):** An earlier version of this report claimed
> MBIR achieved ΔAUC = −0.006 (INDETERMINATE). That result was caused by a
> 20× HU amplification bug in the MBIR→HU conversion (dividing by
> MU_WATER×VOXEL_CM = 0.010295 instead of MU_WATER = 0.2059). The corrected
> results below supersede all previous MBIR numbers.

---

## 2. Algorithms Tested

### 2.1 Task 1 — "Lesion-Safe" FS-iMAR (Partial Correction Blend)

Standard FS-iMAR was analysed through its Fourier decomposition. The CHO's
primary Laguerre-Gauss channel (a=6.0 vox) peaks at f ≈ 0.027 cycles/vox.
At the default σ=1.5, the HP filter passes only 3.2% of this frequency into
the detail layer — the lesion's detection-relevant signal is almost entirely
in the LP band, which comes from the lesion-free iMAR base.

**Resolution:** Rather than tuning σ (which cannot fix the DC contrast problem),
implemented a partial-correction blending approach:

    combined = (1−α) × noMAR + α × iMAR

Swept α ∈ {0.02, 0.05, 0.10, 0.15, 0.20, 0.30}.

### 2.2 Task 2 — MBIR Variants

Two MBIR approaches tested:

1. **MBIR-iMAR** (original): Stage 1 iMAR prior → Stage 2 PWLS with
   exponential weights W=exp(−y). n_iter=30, λ_TV=5×10⁻⁵.

2. **MBIR-Superiority** (prior-free): Warm start from noMAR FBP (preserves
   100% lesion signal), binary weights W=1.0 clean / W=0.05 metal.
   n_iter=40, λ_TV=2×10⁻⁵. Barzilai-Borwein adaptive step sizing.

### 2.3 Additional — Edge-Preserving Denoisers

Bilateral filter (σ_s=2.0 vox, σ_r ∈ {10, 15, 25, 40} HU) and median filter
(k=3) applied directly to noMAR images. Hypothesis: reduce streak variance
while preserving lesion edges.

---

## 3. Results (CORRECTED)

| Algorithm                     | AUC_MAR  | ΔAUC     | 95% CI                | Outcome         |
|-------------------------------|----------|----------|-----------------------|-----------------|
| noMAR (baseline)              | 0.8019   | —        | —                     | —               |
| **MBIR-Superiority (n=40)**   | **0.5569** | **−0.245** | **[−0.313, −0.181]** | **INFERIOR** |
| MBIR-iMAR (n=30, corrected)   | 0.5556   | −0.246   | [−0.313, −0.182]     | INFERIOR        |
| Blend α=0.02                  | 0.8019   | +0.000   | [−0.027, +0.027]     | trivial ≈noMAR  |
| Blend α=0.05                  | 0.7881   | −0.014   | [−0.059, +0.033]     | INDETERMINATE   |
| Blend α=0.10                  | 0.7381   | −0.064   | [−0.125, −0.003]     | INFERIOR        |
| Bilateral σr=10               | 0.6925   | −0.109   | [−0.168, −0.048]     | INFERIOR        |
| Bilateral σr=15               | 0.6950   | −0.107   | [−0.164, −0.049]     | INFERIOR        |
| Bilateral σr=25               | 0.6875   | −0.114   | [−0.179, −0.049]     | INFERIOR        |
| Bilateral σr=40               | 0.6869   | −0.115   | [−0.183, −0.047]     | INFERIOR        |
| Median k=3                    | 0.6100   | −0.192   | [−0.258, −0.122]     | INFERIOR        |
| iMAR                          | 0.5394   | −0.263   | [−0.338, −0.191]     | INFERIOR        |
| FS-iMAR (σ=1.5)               | 0.5400   | −0.262   | [−0.333, −0.193]     | INFERIOR        |

---

## 4. Analysis

### 4.1 The HU Conversion Bug

The earlier "near-parity" MBIR result (ΔAUC = −0.006) was caused by a unit
mismatch in the MBIR→HU conversion:

- MBIR Stage 2 operates in **μ (cm⁻¹)** units
- The HU formula used `x / (MU_WATER × VOXEL_CM)` — correct for FBP output
  (which is in μ·VOXEL_CM units) but **wrong** for MBIR output
- `MU_WATER = 0.2059`, `VOXEL_CM = 0.05`, so `MU_WATER × VOXEL_CM = 0.010295`
- Dividing by 0.010295 instead of 0.2059 = **20× amplification** of all
  deviations from water
- The CHO's fixed internal noise σ=15 is NOT scaled, so 20× amplified images
  have 20× larger signal relative to σ, artificially boosting AUC

**Fix:** `x_hu = (x / MU_WATER - 1.0) × 1000.0` (applied to both
`reference_mbir_superiority.py` and `reference_mbir_imar.py` Stage 2).

### 4.2 Why ALL Algorithms Fail at 23% Blockage

At 23% blockage, noMAR FBP is near-optimal (AUC = 0.80). The metal streaks
are mild enough that the CHO can "see through" them. Any algorithm that
modifies pixel values in the lesion ROI faces a fundamental trade-off:

1. **Sinogram inpainting** (iMAR, FS-iMAR): Replaces 23% of projections
   with a lesion-free prior. The lesion signal in those rays is destroyed.
   Result: AUC ≈ 0.54 (ΔAUC ≈ −0.26).

2. **MBIR** (both variants): The iterative solver converges to a compromise
   between measured data and regularization. Even with ultra-low TV and
   binary weights, the PWLS objective smooths the 12 HU lesion below
   detectability. Result: AUC ≈ 0.56 (ΔAUC ≈ −0.25).

3. **Partial blend**: Linear interpolation between noMAR (lesion + streaks)
   and iMAR (no lesion + no streaks). Monotonically decreasing AUC for α > 0.

4. **Edge-preserving denoisers**: 12 HU lesion is indistinguishable from
   30 HU noise. All filters smooth the lesion away.

### 4.3 The Blockage-Detectability Paradox

| Blockage | noMAR AUC | Best MAR AUC | Best ΔAUC | Best Algorithm |
|----------|-----------|--------------|-----------|----------------|
| 40% (original) | 0.5700 | 0.5144 | −0.056 | iMAR |
| 23% (De Man)   | 0.8019 | 0.5569 | −0.245 | MBIR-Superiority |

The paradox: lowering blockage makes noMAR **better** but makes MAR **worse
relative to noMAR**. At 40% blockage, ΔAUC = −0.056 (mild harm). At 23%
blockage, ΔAUC = −0.245 (severe harm). MAR algorithms are designed to fix
severe artifacts; when artifacts are mild, the "cure" is worse than the disease.

---

## 5. Implications for the ASTM Standard

1. **At normative blockage (~40%):** MAR is mildly INFERIOR (ΔAUC ≈ −0.06).
   The standard's non-inferiority test should use this regime.

2. **At De Man blockage (~23%):** MAR is catastrophically INFERIOR (ΔAUC ≈ −0.25).
   This regime is unsuitable for demonstrating MAR superiority.

3. **No classical MAR algorithm achieves ΔAUC > 0** at any tested blockage.
   The 12 HU noise-limited task is genuinely difficult — a MAR algorithm would
   need to reduce streak noise without touching the lesion, which requires
   knowledge of the lesion location (defeating the purpose of detection).

---

## 6. Deliverables

1. **Reference implementation**: `algorithms/reference_mbir_superiority.py`
   (prior-free, binary-weighted MBIR — corrected HU conversion)
2. **Bug fix**: `algorithms/reference_mbir_imar.py` Stage 2 HU conversion corrected
3. **JSON results**: `deman_mbir_superiority_results.json`,
   `deman_mbir_imar_corrected_results.json`, blend/denoise results
4. **Visualizations**: `superiority_achieved_roi.png`, `deman_dauc_comparison.png`
5. **This report**: `DEMAN_SUPERIORITY_REPORT.md` (corrected)

---

## 7. Conclusion

The ΔAUC > 0 barrier has NOT been broken. With correct HU conversion, ALL
classical MAR algorithms are INFERIOR at 23% blockage — including the
prior-free binary-weighted MBIR that was specifically designed to maximize
lesion preservation.

The fundamental finding: **at low blockage, noMAR FBP is near-optimal and
unbeatable by classical MAR approaches.** The 12 HU lesion contrast is too
close to the 30 HU noise floor for any algorithm to improve detection without
prior knowledge of the lesion location. Superiority may only be achievable
at higher blockage levels where the noMAR baseline is degraded enough that
streak reduction provides net benefit — but at those levels, the information
destruction from sinogram inpainting is also most severe.

This is a fundamental limitation of task-based MAR assessment with subtle
lesions, not a deficiency of any particular algorithm.
