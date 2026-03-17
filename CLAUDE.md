# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **MAR ILS Dataset Generator** — a reference implementation for generating a standardized, synthetic CT dataset for Metal Artifact Reduction (MAR) Interlaboratory Studies (ILS), compliant with **ASTM WKXXXXX Revision 03** and **IEC 60601-2-44 Ed. 4**. The framework follows the task-based signal detection approach from Vaishnav et al. (Medical Physics, 47(8), 2020).

## Commands

### Environment Setup
```bash
source mar-ils/bin/activate   # activate virtual environment (Python 3.10)
pip install -r requirements.txt
```

### Generate Dataset (v5.3.0 — current normative reference)
```bash
python generator_v5_3_0.py                                     # all cores, ./astm_mar_ils_dataset_v5
python generator_v5_3_0.py --output-dir ./my_dataset           # custom output directory
python generator_v5_3_0.py --workers 8                         # limit to 8 workers
python generator_v5_3_0.py --workers 1                         # serial (for debugging)
python generator_v5_3_0.py --dry-run                           # validate config without writing
python generator_v5_3_0.py --no-pdf                            # skip lab instructions PDF
```

### Run CHO Analysis (v5.3.0 — current normative reference, 2D single-slice)
```bash
# ILS mode (lab submits reconstructed DICOMs):
python run_cho_analysis_v5_3.py \
    --dataset-dir ./astm_mar_ils_dataset_v5 \
    --mar-output-dir ./mar_recon \
    --reconstruction-pipeline "Vendor FBP, kernel B30f"

# Self-test (pipeline validation — ΔAUC = 0 by definition):
python run_cho_analysis_v5_3.py \
    --dataset-dir ./astm_mar_ils_dataset_v5 \
    --self-test

# Parallel realization loading (M3 optimization):
python run_cho_analysis_v5_3.py --workers 0 ...   # 0 = os.cpu_count()
```

### Visualize Sinograms
```bash
python view_sinograms.py sinograms/LP/realization_001.h5
python view_sinograms.py sinograms/LP/realization_001.h5 --slice 128
python view_sinograms.py sinograms/LP/realization_001.h5 sinograms/LA/realization_001.h5 --compare
```

## Architecture

### Scripts (versioned by filename — use the highest version)
- **`generator_v5_3_0.py`** — Physics-based sinogram dataset generator. **Current normative 2D reference.** Single-disc lesion at slice 128, ~12 HU physics contrast via sinogram domain only (no post-FBP hard-set). 40 LP + 40 LA realizations.
- **`run_cho_analysis_v5_3.py`** — Reference 2D CHO implementation. **Current normative reference.** Loads only `slice_0129.dcm`; 3D integration prohibited. Use `--internal-noise-sigma 15`.
- **`view_sinograms.py`** — Interactive matplotlib viewer for HDF5 sinogram files.
- `generator_v5_2_0.py` — Superseded (3D full-cylinder lesion causes AUC ceiling; lesion 120 HU).
- `generator_v5_1_0.py` — Superseded (no normative HU values).
- `generator_v5_0_0.py` / `run_cho_analysis_v5_0.py` — Superseded (2D CHO correct but v5.0.0 generator).

### Generator Pipeline (`generator_v5_3_0.py`)
1. **I₀ Calibration** — 3-FBP analytic approach: noise-free full phantom (DC offset), noise-free body-only phantom, 5 noisy MC draws at I₀_ref=1e5; analytic scaling σ ∝ 1/√I₀.
2. **Phantom Construction** — Analytic elliptical body (170×120 voxels), centered iron rod (10 voxel radius), optional lesion disc (5 voxel radius) at (281, 256) in **slice 128 only** [X1].
3. **Forward Projection** — Parallel-beam, 60 keV monochromatic, 360 angles × 512 detectors. LP realizations: 2 forward projections (with and without lesion); LA: 1 forward projection. Noise model: `I_meas = Poisson(I₀·exp(−∫μ dl) + S) + N(0,σ_e²)`.
4. **FBP Reconstruction** — Rotation-sum approximation + Ram-Lak filter + DC offset correction + `+BACKGROUND_HU` (40 HU). Produces `noMAR_recon/` DICOM series.
5. **Natural Reconstruction** — No post-FBP hard-set overrides. Lesion contrast (~12 HU) emerges purely from sinogram-domain `MU_LESION_CM`. Metal mask hard-set to 3000 HU [R3] (always last, only override remaining).
6. **UID Consistency** — `StudyInstanceUID` and `SeriesInstanceUID` generated once per realization worker; shared across all 256 DICOM slices [X4].
7. **Output** — 40 realizations × 2 conditions (LP/LA) as HDF5 sinograms + DICOM reconstructions + checksums + metadata + PDF instructions. Parallel via `ProcessPoolExecutor` (80 tasks).

### CHO Analysis Pipeline (`run_cho_analysis_v5_3.py`)
1. Loads **only `slice_0129.dcm`** (LESION_SLICE_INDEX = 128, 1-indexed) from each of 40 LP and 40 LA realization folders. 3D integration across z is **prohibited** (§A1.5.3).
2. Projects each 121×121 ROI through 10 2D Laguerre-Gauss channels → feature vector of shape `(40, 10)` per condition.
3. Fits Hotelling template with Tikhonov regularisation (λ = 0.01 × trace(K)/10) using **LA covariance only** [§A1.5.2(a)].
4. Computes LOO hold-out AUC (Mann-Whitney) with 1000-resample bootstrap CI and paired ΔAUC bootstrap CI.
5. Writes `cho_results.json`.

### Why single-slice 2D observer:
The 3D volumetric CHO accumulates √256 = 16× coherent z-integration gain when the lesion spans all slices. With the 120 HU contrast of v5.2.0 this drives d′ ≈ 367 and AUC → 1.000 for **both** noMAR and MAR conditions, making ΔAUC unmeasurable. The v5.3.0 solution: (a) restrict lesion to one slice — eliminates z-integration gain; (b) pure sinogram-domain contrast at ~12 HU with no post-FBP hard-set — establishes genuine noise-limited detection. The Vaishnav Transition (2026-03-14) removed all hard-set pixel overrides, calibrated MU_LESION_CM for ~12 HU FBP contrast, expanded to 40 realizations per condition, and confirmed AUC_noMAR = 0.7063 with `--internal-noise-sigma 15`.

### Dataset Structure (v5.3.0)
```
<output_dir>/
    sinograms/
        LP/  realization_001.h5 ... realization_040.h5   ← lab deliverable (HDF5)
        LA/  realization_001.h5 ... realization_040.h5
    noMAR_recon/
        LP/  realization_001/ ... realization_040/        ← 256 DICOMs each
        LA/  realization_001/ ... realization_040/
    checksums_sha256.txt
    dataset_metadata.csv
    generator_provenance.json
    MAR_ILS_Lab_Instructions.pdf
```

### HDF5 Sinogram Format
Each `realization_NNN.h5` contains:
- `line_integrals`: float32 `(256, 360, 512)` — measured line integrals in neper
- `geometry` attrs: `n_slices`, `n_angles`, `n_det`, `voxel_mm`, `angles_deg`
- `noise_params` attrs: `I0`, `scatter_frac`, `sigma_e_counts`, `seed`, `jitter_deg`, `place_lesion`, `lesion_slice_index` (128), `lesion_z_extent` (1 for LP, 0 for LA)

### Lab Submission Structure (for CHO analysis)
Labs reconstruct MAR-corrected images and submit:
```
mar_recon/
    LP/  realization_001/ ... (DICOM slices named slice_NNNN.dcm, 1-indexed)
    LA/  realization_001/ ...
```
CHO analysis reads **only** `slice_0129.dcm` from each folder.

## Key Constants (Normative — Do Not Modify)
| Parameter | Value |
|---|---|
| Volume | 512×512×256 voxels, 0.5 mm isotropic |
| Acquisition | 360 angles, 512 detectors, parallel-beam, 60 keV |
| Realizations | **40 LP + 40 LA** |
| Lesion z-extent | **Slice 128 only** (single disc, not cylinder) [X1] |
| CHO observer | **2D, slice 128 only** — 3D integration PROHIBITED [§A1.5.3] |
| CHO ROI center | (281, 256), 121×121 voxels |
| Background HU | 40 HU (§10.1.1, §A1.2(d)) [R1] |
| Lesion μ (STS) | `MU_TISSUE_CM × (1 + 12/1000)` → ~12 HU FBP contrast [X2] |
| Lesion contrast | **~12 HU** (sinogram-domain only, no post-FBP hard-set) [X2] |
| Metal HU | 3000 HU (§A1.3(d,f), §10.1.2 step 5) [R3] |
| Noise σ target | 30 HU in soft tissue |
| μ soft tissue | 0.2059 cm⁻¹ |
| μ iron | 2.408 cm⁻¹ |
| Internal noise σ | **15** (Vaishnav CHO regularisation — normative default) |

## ASTM METROLOGY BASELINE (LOCKED 2026-03-14)

These parameters were validated via the Vaishnav Transition calibration sweep and represent the immutable project floor for all ILS reporting. Do not modify without Task Group approval and a full regeneration + CHO re-validation.

| Parameter | Locked Value | Notes |
|---|---|---|
| `MU_LESION_CM` | `MU_TISSUE_CM × (1 + 12/1000)` | ~12 HU effective FBP contrast; sinogram-domain only |
| `NUM_REALIZATIONS` | **40** LP + **40** LA | Per condition; 80 total tasks |
| `--internal-noise-sigma` | **15** | Vaishnav internal observer noise; normative CHO default |
| `BASELINE_AUC_noMAR` | **0.7063** | LOO hold-out; 95% CI [0.6575, 0.7844] |
| Reference dataset | `./astm_reference_dataset/` | Gold-standard; do not overwrite |

**Rationale:** The Vaishnav Transition removed all post-FBP hard-set pixel overrides, forcing the lesion signal to emerge through genuine Radon inversion against a 30 HU noise floor. This establishes a human-correlated task-based assessment: the CHO operates on physically realistic noisy images, not artificially clean lesion ROIs. The resulting AUC_noMAR = 0.7063 confirms the task is noise-limited and metrologically sensitive to MAR algorithm quality.

**Canonical run command:**
```bash
python run_cho_analysis_v5_3.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./mar_recon \
    --internal-noise-sigma 15 \
    --reconstruction-pipeline "Vendor FBP, kernel B30f"
```

## Versioning Convention
Scripts embed version in filename (e.g., `_v5_3_0`). Use the highest-numbered version. The v5.3.0 generator is the current 2D normative reference. As of the Vaishnav Transition (2026-03-14), it operates in natural-reconstruction mode: no post-FBP hard-set overrides, ~12 HU sinogram-domain lesion contrast, 40 realizations per condition.
