# algorithms/v7/ — fan-beam reference MAR

Reference MAR implementations compatible with the **v7.0.0 fan-beam** dataset
(ASTM Rev 05). Unlike the parallel-beam references in `algorithms/`, these
operate on the (256, 720, 512) fan-beam sinograms and reuse the validated
`forward_project_slice` / `fbp_reconstruct_slice` from `generator_v7_0_0.py`.

## reference_li_mar_v7.py — Linear-Interpolation MAR

Parameter-free LI-MAR. **Non-normative**: the ASTM type test measures the
*lab's* algorithm. This reference exists as a CHO-pipeline negative control
(designated control, ΔAUC ≈ −0.23) and a reproducible delta-AUC anchor — the
floor every commercial MAR is expected to beat.

It detects metal by thresholding the existing noMAR reconstruction, fills the
metal trace in the sinogram by linear interpolation along the detector axis,
reconstructs with fan-beam FBP, and writes `slice_0129.dcm` only (what the CHO
reads).

### Usage

```bash
python algorithms/v7/reference_li_mar_v7.py \
    --dataset-dir ./astm_reference_dataset \
    --output-dir  ./li_mar_recon
# then score it:
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./li_mar_recon \
    --internal-noise-sigma 15
```

Options: `--realizations N` (default auto-detect), `--metal-hu-thresh 2000`,
`--dc-offset-cm <float>` (default read from `generator_provenance.json`).

### delta-AUC anchor

Measured on the locked reference dataset (N=40, σ=15), 2026-05-29:

| Quantity | Value | 95% CI |
|---|---|---|
| AUC_noMAR | **0.8294** | [0.7612, 0.9025] |
| AUC_MAR (LI-MAR) | **0.5994** | [0.5687, 0.6613] |
| **ΔAUC (LI-MAR − noMAR)** | **−0.2300** | [−0.2969, −0.1475] |

Wilcoxon (ΔAUC ≠ 0): p < 0.0001. AUC_noMAR reproduces the locked baseline
(0.8294) exactly, confirming the regenerated dataset matches the reference.

**Interpretation.** LI-MAR *significantly degrades* detectability (ΔAUC ≈ −0.23),
establishing the **floor** — a MAR algorithm that cannot clearly beat this (ideally
ΔAUC ≥ 0) is not helping.

The mechanism was verified to be genuine secondary artifacts, not an
implementation artifact of over-aggressive trace removal:

- The metal trace covers only **20 of 512 detector bins per view** (matching the
  r=10-voxel rod's projected width), so LI-MAR does *not* interpolate away the
  lesion's own projections — the lesion sits well outside the replaced trace.
- Yet across the 40 LP realizations the lesion-region signal variance inflates
  **~10×** under LI-MAR (≈ ±1.8 HU noMAR → ≈ ±17 HU LI-MAR), i.e. crude per-view
  interpolation injects new streak artifacts near the lesion that swamp the
  ~12 HU signal and drive detection toward chance.

Directionally consistent with the earlier parallel-beam study (LI-MAR below noMAR
in both); the magnitude differs (≈ −0.057 there vs −0.230 here), as expected given
the different geometry and operating point.

This is a manual step (a full dataset is ~35 GB), not part of CI. To reproduce,
run the two commands above and read `cho_results.json`.
