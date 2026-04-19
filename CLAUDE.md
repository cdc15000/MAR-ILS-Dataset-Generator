# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **MAR ILS Dataset Generator** — a reference implementation for generating a standardized, synthetic CT dataset for Metal Artifact Reduction (MAR) Interlaboratory Studies (ILS), compliant with **ASTM WKXXXXX Revision 04** and **IEC 60601-2-44 Ed. 4**. The framework follows the task-based signal detection approach from Vaishnav et al. (Medical Physics, 47(8), 2020).

## Project Root

The persistent working directory is `~/projects/mar-ils/`. A convenience symlink exists at `/tmp/mar-ils` → `~/projects/mar-ils/`.

## Commands

### Environment Setup
```bash
cd ~/projects/mar-ils
source mar-ils/bin/activate   # activate virtual environment (Python 3.10)
pip install -r requirements.txt
pip install numba              # optional: ~24x speedup via JIT + batch geometry
```

### Generate Dataset (v7.0.0 — current normative reference, fan-beam)
```bash
python generator_v7_0_0.py                                     # all cores, ./astm_reference_dataset
python generator_v7_0_0.py --output-dir ./my_dataset           # custom output directory
python generator_v7_0_0.py --workers 8                         # limit to 8 workers
python generator_v7_0_0.py --workers 1                         # serial (for debugging)
python generator_v7_0_0.py --realizations 20                   # screening mode (pilot eval)
python generator_v7_0_0.py --dry-run                           # validate config without writing
python generator_v7_0_0.py --no-pdf                            # skip lab instructions PDF
```

### Run CHO Analysis (v7.0.0 — current normative reference, 2D single-slice)
```bash
# ILS mode (lab submits reconstructed DICOMs):
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./mar_recon \
    --internal-noise-sigma 15

# Self-test (pipeline validation — ΔAUC = 0 by definition):
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --self-test

# Screening mode (20 realizations — informative only):
python run_cho_analysis_v7_0.py \
    --dataset-dir ./my_dataset_20 \
    --self-test --realizations 20
```

### Visualize Sinograms
```bash
python view_sinograms.py sinograms/LP/realization_001.h5
python view_sinograms.py sinograms/LP/realization_001.h5 --slice 128
```

## Architecture

### Scripts (versioned by filename — use the highest version)
- **`generator_v7_0_0.py`** — Fan-beam physics-based sinogram dataset generator. **Current normative reference.** Single canonical configuration (ASTM WKXXXXX Rev 04): fan-beam (SID=570mm, SDD=1040mm), 720 angles, iron rod, circular lesion, ~12 HU sinogram-domain contrast. 40 LP + 40 LA realizations (or 20 for screening).
- **`run_cho_analysis_v7_0.py`** — Reference 2D CHO implementation. **Current normative reference.** Hardcoded v5.3.0/Rev 04 ROI parameters (121×121, centre (281,256), channel width 7.5). AUC equivalence tolerance ±0.005.
- **`patch_2026b_metadata.py`** — One-time utility to inject DICOM 2026b CP-2575 MAR metadata into existing datasets.
- `legacy/generator_v6_0_0.py` — Research tier framework (T1_AB/T2_SB/T3_HEAD). Not normative. Uses parallel-beam geometry and tier-specific parameters. Retained for multi-tier research.
- `legacy/run_cho_analysis_v6_0.py` — Tier-aware CHO. Not normative. Uses tier_config.py for ROI parameters.
- `tier_config.py` — Three-tier registry (v6 research framework only).

### Generator Pipeline (`generator_v7_0_0.py`)
1. **I₀ Calibration** — 3-FBP analytic approach using fan-beam geometry: noise-free full phantom (DC offset), noise-free body-only phantom, 5 noisy MC draws at I₀_ref=1e5; analytic scaling σ ∝ 1/√I₀.
2. **Phantom Construction** — Analytic elliptical body (170×120 voxels), centered iron rod (10 voxel radius), optional circular lesion disc (5 voxel radius) at (281, 256) in **slice 128 only**.
3. **Forward Projection** — **Fan-beam** (SID=570mm, SDD=1040mm), 60 keV monochromatic, **720 angles × 512 equi-angular detectors, full 360° rotation**. Ray-driven integration via sub-voxel sampling. LP: 2 forward projections; LA: 1. Noise: `I_meas = Poisson(I₀·exp(−∫μ dl) + S) + N(0,σ_e²)`. **Numba JIT**: hand-rolled bilinear interpolation replaces `scipy.ndimage.map_coordinates`.
4. **FBP Reconstruction** — **Fan-beam FBP**: cosine pre-weighting + Ram-Lak filter + (SID/L)² distance-weighted backprojection + DC offset correction + `+BACKGROUND_HU` (40 HU). **Batch mode** (Numba): all 256 slices backprojected with geometry computed once per (pixel, angle) pair (~4x fewer arctan2 calls). Produces `noMAR_recon/` DICOM series.
5. **Natural Reconstruction** — No post-FBP hard-set overrides. Lesion contrast (~12 HU) emerges purely from sinogram-domain `MU_LESION_CM`. Metal mask hard-set to 3000 HU (always last, only override remaining).
6. **UID Consistency** — `StudyInstanceUID` and `SeriesInstanceUID` generated once per realization worker.
7. **Output** — 40 (or 20) realizations × 2 conditions (LP/LA) as HDF5 sinograms + DICOM reconstructions + checksums + metadata + PDF instructions.

### CHO Analysis Pipeline (`run_cho_analysis_v7_0.py`)
1. Loads **only `slice_0129.dcm`** (LESION_SLICE_INDEX = 128, 1-indexed) from each realization. 3D integration prohibited (§A1.5.3).
2. Projects each 121×121 ROI through 10 2D Laguerre-Gauss channels → feature vector of shape `(N, 10)`.
3. Fits Hotelling template with Tikhonov regularisation (λ = 0.01 × trace(K)/10) using **LA covariance only**.
4. Computes LOO hold-out AUC (Mann-Whitney) with 1000-resample bootstrap CI and paired ΔAUC bootstrap CI.
5. Informative diagnostics: Wilcoxon signed-rank test, 50/50 estimation bias, noise sensitivity sweep, sigmoid AUC fit.
6. Writes `cho_results.json`.

### Dataset Structure (v7.0.0)
```
<output_dir>/
    sinograms/
        LP/  realization_001.h5 ... realization_040.h5   ← lab deliverable (HDF5)
        LA/  realization_001.h5 ... realization_040.h5
    noMAR_recon/
        LP/  realization_001/ ... realization_040/        ← 256 DICOMs each
        LA/  realization_001/ ... realization_040/
    checksums_sha256.txt
    generator_provenance.json
    MAR_ILS_Lab_Instructions.pdf
```

### HDF5 Sinogram Format (v7.0.0 — fan-beam)
Each `realization_NNN.h5` contains:
- `line_integrals`: float32 `(256, 720, 512)` — fan-beam line integrals in neper
- `geometry` attrs: `type`='fan-beam', `SID_mm`=570, `SDD_mm`=1040, `n_angles`=720, `n_det`=512, `gamma_max_deg`, `delta_gamma_deg`, `angles_deg`, `det_fan_angles_deg`
- `noise_params` attrs: `I0`, `scatter_frac`, `sigma_e_counts`, `seed`, `jitter_deg`, `place_lesion`, `lesion_slice_index` (128), `lesion_z_extent`

### Lab Submission Structure (for CHO analysis)
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
| Acquisition | **Fan-beam, SID=570 mm, SDD=1040 mm, 720 angles, 512 equi-angular detectors, full 360°** |
| Realizations | **40 LP + 40 LA** (screening: 20+20) |
| Lesion z-extent | **Slice 128 only** (single disc) |
| CHO observer | **2D, slice 128 only** — 3D integration PROHIBITED |
| CHO ROI | (281, 256), 121×121 voxels |
| Channel width a | 7.5 voxels |
| Background HU | 40 HU |
| Lesion contrast | **~12 HU** (sinogram-domain only, no post-FBP hard-set) |
| Metal HU | 3000 HU |
| Noise σ target | 30 HU in soft tissue |
| μ soft tissue | 0.2059 cm⁻¹ |
| μ iron | 2.408 cm⁻¹ |
| Internal noise σ | **15** (normative default) |
| AUC tolerance | **±0.005** |

## ASTM METROLOGY BASELINE (LOCKED 2026-04-07)

| Parameter | Status | Value |
|---|---|---|
| `MU_LESION_CM` | **Locked** | `MU_TISSUE_CM × (1 + 12/1000)` |
| `NUM_REALIZATIONS` | **Locked** | **40** LP + **40** LA |
| `--internal-noise-sigma` | **Locked** | **15** |
| `BASELINE_AUC_noMAR` | **Locked** | **0.8294**, CI [0.7612, 0.9025] (N=40, σ=15) |
| Acquisition geometry | **Locked** | Fan-beam, SID=570mm, SDD=1040mm |
| Calibrated I₀ | **Locked** | 310,853 |
| DC offset | **Locked** | −0.029 cm⁻¹ (≈ −141 HU) |
| Lesion contrast | **Locked** | 12.0 HU (exact) |
| Tissue HU | **Locked** | 40.0 HU (exact) |

Historical baselines (superseded):
- Parallel-beam (v5.3.0): AUC_noMAR = 0.7063
- Fan-beam screening (N=20): AUC_noMAR = 0.6600, CI [0.6225, 0.7551]

### Fan-Beam Calibration Validation (2026-04-06)
- Tissue HU reconstructs to exactly 40.0 HU in calibration ROI
- Lesion contrast reconstructs to exactly 12.0 HU
- Noise σ calibrates to 30 HU target (52.9 HU at I₀=1e5, scaled to 30 HU at I₀=310,853)
- Screening AUC_noMAR=0.6600 at σ=15 (N=20); CI encompasses parallel-beam baseline of 0.7063
- Noise sweep: AUC ranges from 0.60 (σ=0) to 0.71 (σ=30), confirming regularisation dependence

### Numba Acceleration (2026-04-07)
- **Forward projection**: Numba @njit with hand-rolled bilinear interpolation replaces `scipy.ndimage.map_coordinates`. ~24x speedup (0.29s compiled vs ~7s NumPy).
- **FBP backprojection**: Numba @njit eliminates Python loop overhead and temporary array allocation. ~5x speedup per slice.
- **Batch geometry sharing**: All 256 slices backprojected together; geometry (arctan2, L², detector index) computed once per (pixel, angle) pair, applied to all slices. ~4x additional speedup.
- **Combined**: 61s per realization (single-threaded worker), down from ~1300s (NumPy). ~24x total speedup.
- **N=40 estimated wall time**: ~10 minutes with 8 workers (M3 MacBook Pro, 18+ GB RAM).
- **Bit-exactness**: Batch FBP matches per-slice FBP within float32 precision (max 0.0005 HU difference).
- **Fallback**: All Numba code is conditional (`if _HAS_NUMBA`). Without numba installed, generator falls back to original NumPy implementation.
- **Thread management**: Workers set `numba.set_num_threads(1)`; parallelism is across workers via `ProcessPoolExecutor`. Calibration uses all cores.
- **N=40 generated**: 2026-04-07, 80 realizations (35 GB), SHA-256 verified, baseline AUC locked at 0.8294.

**Canonical run command:**
```bash
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./mar_recon \
    --internal-noise-sigma 15
```

### DICOM 2026b Compliance (CP-2575, 2026-04-07)
- All DICOM output includes the **Metal Artifact Reduction Macro** (PS3.3 C.8.15.3.15).
- `(0018,9390)` Metal Artifact Reduction Sequence (SQ, 1 item) → `(0018,9391)` Metal Artifact Reduction Applied (CS) = `"NO"`.
- Tags are written via hex (`ds.add_new(0x00189390, 'SQ', [...])`) because pydicom's bundled dictionary does not yet include the 2026b additions. Verification:
  ```python
  import pydicom
  ds = pydicom.dcmread('path/to/slice.dcm')
  print(ds[0x00189390].value[0][0x00189391].value)  # → "NO"
  ```
- `patch_2026b_metadata.py` retrofits existing datasets (idempotent, regenerates checksums).

## Regulatory Framework (Layered Approach)

| Layer | Document | Role |
|---|---|---|
| 1 | IEC 60601-2-44 Ed. 4 §203.6.7.101 | MAR method available (.1), user information (.2), DICOM recording (.3) |
| 2 | ASTM FXXXX (bound to §203.6.7.101.1 via post-publication Corrigendum/Amendment, deferred until ASTM FXXXX publishes) | TYPE TEST — measures ΔAUC |
| 3 | FDA guidance | Acceptance criteria (non-degradation threshold, superiority claim) |

See `docs_and_references/IEC_203_6_7_101_compliance_statement_proposal.md` (post-publication Amendment proposal) and `docs_and_references/FDA_guidance_framework.md` for draft regulatory text.

## Versioning Convention
Scripts embed version in filename. **v7.0.0** is the current normative reference (fan-beam, single canonical config, ASTM Rev 04). v6.0.0 is the research tier framework (parallel-beam, three tiers).
