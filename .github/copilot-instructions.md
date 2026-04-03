# Copilot Instructions for MAR ILS Dataset Generator

## Quick Setup

```bash
source mar-ils/bin/activate       # Activate venv (Python 3.10)
pip install -r requirements.txt   # Install dependencies
```

**Key Dependencies:** numpy, scipy, pydicom, h5py, torch, reportlab, matplotlib

---

## High-Level Architecture

This is the **MAR ILS Dataset Generator** — a standardized evaluation framework for Metal Artifact Reduction (MAR) algorithms in CT imaging, compliant with **ASTM WKXXXXX v1.0.0** and **IEC 60601-2-44 Ed. 4**. It implements task-based signal detection following Vaishnav et al. (Medical Physics, 2020).

### Three-Layer Architecture

1. **Dataset Generation** (`generator_v6_0_0.py`)
   - Generates synthetic CT sinograms using forward projection (60 keV monochromatic, 360 angles, 512 detectors)
   - Physics-based noise model: Poisson + electronic noise
   - Outputs 40 LP (lesion present) + 40 LA (lesion absent) realizations per tier as HDF5 + DICOMs

2. **MAR Algorithm Testing** (`/algorithms/reference_*.py`)
   - Reference implementations: iMAR, MBIR, Spectral MAR, POCS-TV, etc.
   - Labs reconstruct the sinograms with their own MAR pipeline and submit DICOMs

3. **Statistical Analysis** (`run_cho_analysis_v6_0.py`)
   - Channelized Hotelling Observer (CHO) with Laguerre-Gauss channels
   - Computes ΔAUC (change in detectability) via leave-one-out CV + bootstrap CI
   - Evaluates MAR algorithm effectiveness

### Tiered Framework (v6.0.0)

Three imaging scenarios with different phantom geometries, metal inserts, and physical parameters:

| Tier | Description | Body | Metal | Status |
|------|-------------|------|-------|--------|
| **T1_AB** | Adult body (mandatory) | Ellipse 170×120 mm | Cobalt-Chrome (Co-Cr) rod r=10 mm | Baseline |
| **T2_SB** | Small body (pediatric) | Ellipse 126×94 mm | Stainless Steel (SS-316L) rod r=6.5 mm | Recent focus |
| **T3_HEAD** | Head/neuro | Circle Ø=200 mm | Titanium (Ti-6Al-4V) rod r=8 mm | Supplemental |

All tiers share:
- Same acquisition (360°, 512 detectors, 0.5 mm voxel, parallel-beam)
- Same realization count (40 LP + 40 LA)
- Lesion: single-slice disc (slice 128 only, ~12 HU sinogram contrast)
- CHO observer: 2D on slice 128 (3D integration prohibited per ASTM §A1.5.3)

### Key Computational Constraints

- **Sinogram shape:** (256 slices, 360 angles, 512 detectors) — ~47 MB per realization in HDF5
- **DICOM output:** 256 slices per realization → ~120 MB per realization
- **Parallelization:** Dataset generation uses `ProcessPoolExecutor` (80 realization tasks)
- **CHO bootstrap:** 1000 resamples × leave-one-out (LOO) → compute time ~30s per dataset

---

## Script Versioning & Usage

**Scripts embed version in filename** (e.g., `generator_v6_0_0.py`). **Always use the highest version number.** v6.0.0 is the current normative reference with full tier support.

### Generate Dataset

```bash
# Default: T1_AB tier, all CPU cores, output to ./astm_mar_ils_dataset_v6
python generator_v6_0_0.py

# Custom tier
python generator_v6_0_0.py --tier T2_SB --output-dir ./t2_sb_dataset

# Contrast sensitivity sweep: scale lesion μ by factor
python generator_v6_0_0.py --tier T1_AB --contrast-factor 0.5

# Sweep mode: only write slice_0129.dcm per realization (256× DICOM storage reduction)
python generator_v6_0_0.py --tier T1_AB --sweep-mode

# Control parallelism
python generator_v6_0_0.py --workers 8        # Limit to 8 workers
python generator_v6_0_0.py --workers 1        # Serial (debugging)

# Dry-run (validate config, no output files)
python generator_v6_0_0.py --dry-run --tier T2_SB

# Skip PDF generation
python generator_v6_0_0.py --no-pdf
```

### Run CHO Analysis

```bash
# ILS mode: evaluate lab-submitted MAR reconstructions
python run_cho_analysis_v6_0.py \
    --dataset-dir ./astm_mar_ils_dataset_v6 \
    --mar-output-dir ./lab_reconstructions \
    --tier T1_AB

# Self-test (sanity check): runs noMAR vs. noMAR—ΔAUC must be ≈0
python run_cho_analysis_v6_0.py \
    --dataset-dir ./astm_mar_ils_dataset_v6 \
    --self-test \
    --tier T1_AB

# Parallel realization loading
python run_cho_analysis_v6_0.py \
    --dataset-dir ./astm_mar_ils_dataset_v6 \
    --mar-output-dir ./lab_reconstructions \
    --workers 0  # 0 = os.cpu_count()
    --tier T1_AB

# Noise sensitivity sweep (analyzes AUC across 11 internal-noise-sigma values)
python run_cho_analysis_v6_0.py \
    --dataset-dir ./astm_mar_ils_dataset_v6 \
    --mar-output-dir ./lab_reconstructions \
    --noise-sweep \
    --tier T1_AB
```

### Visualize Sinograms

```bash
python view_sinograms.py sinograms/LP/realization_001.h5

# View specific slice
python view_sinograms.py sinograms/LP/realization_001.h5 --slice 100

# Compare LP vs. LA side-by-side
python view_sinograms.py sinograms/LP/realization_001.h5 sinograms/LA/realization_001.h5 --compare
```

---

## Data Structures & I/O

### HDF5 Sinogram Format

Each `realization_NNN.h5` in `sinograms/{LP,LA}/` contains:

```
/line_integrals         ← float32 (256, 360, 512) in neper
/tier_config            ← group with tier-specific parameters (attrs)
  .attrs['tier_id']                ← "T1_AB", "T2_SB", "T3_HEAD"
  .attrs['lesion_slice_index']     ← 128 (zero-indexed)
  .attrs['lesion_z_extent']        ← 1 if LP, 0 if LA
  .attrs['place_lesion']           ← boolean
  .attrs['contrast_factor']        ← scales MU_LESION_CM (default 1.0)
/geometry (attrs)
  .attrs['n_slices']               ← 256
  .attrs['n_angles']               ← 360
  .attrs['n_det']                  ← 512
  .attrs['voxel_mm']               ← 0.5
  .attrs['angles_deg']             ← array [0, 1, 2, ..., 359]
/noise_params (attrs)
  .attrs['I0']                     ← photon flux at reference (1e5)
  .attrs['sigma_e_counts']         ← 5.0 (electronic noise)
  .attrs['scatter_frac']           ← 0.05
  .attrs['seed']                   ← random seed for reproducibility
  .attrs['jitter_deg']             ← angular jitter (typically 0.0)
```

### DICOM Output

```
noMAR_recon/{LP,LA}/realization_NNN/
  slice_0001.dcm  ← 1-indexed (DICOM convention)
  slice_0002.dcm
  ...
  slice_0256.dcm
```

- **SliceLocation:** varies by slice index
- **StudyInstanceUID / SeriesInstanceUID:** generated once per realization (constant across all 256 slices)
- **WindowCenter / WindowWidth:** dual presets (soft tissue + lesion-centered)
- **PixelData:** -1024 to +3000 HU (background ≈40 HU, iron ≈3000 HU)

### Lab Submission Format (for CHO analysis)

Labs reconstruct MAR-corrected images and submit:

```
mar_recon/{LP,LA}/realization_NNN/
  slice_0001.dcm  ← 1-indexed DICOM slices
  ...
  slice_0256.dcm
```

CHO analysis **reads only `slice_0129.dcm`** (LESION_SLICE_INDEX = 128, 1-indexed).

---

## Key Constants & Normative Values

**ASTM METROLOGY BASELINE (Locked 2026-03-14)** — Do not modify without Task Group approval.

| Parameter | Normative Value | Tier Scope | Notes |
|-----------|-----------------|-----------|-------|
| Voxel size | 0.5 mm (isotropic) | All | 512×512×256 volume |
| Acquisition | 360°, 512 detectors, parallel-beam, 60 keV | All | Monochromatic |
| Realizations | 40 LP + 40 LA | All | Per tier, 80 tasks total |
| Lesion z-extent | Slice 128 only (disc) | All | Single-slice (not cylinder) |
| CHO observer | 2D, slice 128 only | All | 3D integration PROHIBITED (§A1.5.3) |
| CHO ROI size | Tier-specific | All | From `TierConfig.roi_size` |
| CHO channels | Laguerre-Gauss (10 channels) | All | Channel width from `TierConfig.channel_width_a` |
| Lesion contrast | ~12 HU (sinogram-domain) | All | No post-FBP hard-set overrides |
| Background HU | 40 HU | All | From FBP calibration |
| Metal HU | 3000 HU | All | Hard-set in post-FBP (always last) |
| Noise target | 30 HU in soft tissue | All | Via I₀ calibration |
| Internal observer noise σ | 15 HU | Analysis | Vaishnav CHO regularization |
| Baseline AUC (noMAR) | 0.7063 [0.6575, 0.7844] | T1_AB (v5.3.0) | Reference 95% CI |

### Understanding the Physics

- **I₀ Calibration:** 3-pass FBP approach calibrates Poisson noise to achieve 30 HU soft-tissue noise (§A1.2).
- **Lesion Contrast:** ~12 HU emerges purely from sinogram-domain linear attenuation (`MU_LESION_CM = MU_TISSUE_CM × 1.012`). No post-FBP pixel hard-set.
- **Noise Model:** `I_measured = Poisson(I₀·exp(−∫μ dl) + S) + N(0, σ_e²)` where S is scatter fraction.
- **Why 2D CHO:** 3D volumetric observer on v5.2.0 (full-cylinder lesion, 120 HU) saturated to AUC=1.000 due to z-integration gain (16×). v6.0.0 uses single-slice restriction to measure genuine noise-limited detection.

---

## Code Organization

### `tier_config.py`
- **TierConfig dataclass:** Frozen, immutable config for each tier.
- **Tier registry:** `TIER_REGISTRY["T1_AB"]`, `TIER_REGISTRY["T2_SB"]`, `TIER_REGISTRY["T3_HEAD"]`.
- **Normative constants:** Physical properties (μ values), phantom dimensions, lesion parameters.
- **Derived fields:** Computed in `__post_init__` via immutable pattern (e.g., blockage fraction, channel width).

### `generator_v6_0_0.py`
1. **I₀ Calibration (3 passes):** Noise-free full phantom (DC), clean tissue ROI, 5 MC draws.
2. **Phantom Construction:** Analytic ellipse/circle, centered metal rod, optional lesion disc at (281, 256) in slice 128.
3. **Forward Projection:** Radon transform (360 angles, 512 detectors) with noise injection.
4. **FBP Reconstruction:** Rotation-sum + Ram-Lak filter + DC offset correction.
5. **Parallel Execution:** 80 realization tasks via `ProcessPoolExecutor`.
6. **Output:** HDF5 sinograms, DICOM reconstructions, checksums, metadata, lab instructions PDF.

### `run_cho_analysis_v6_0.py`
1. **Load Data:** 40 LP + 40 LA DICOMs from lab submissions.
2. **Feature Extraction:** 2D Laguerre-Gauss channels (10 channels, slice 128 only).
3. **CHO Fitting:** Hotelling template with Tikhonov regularization (λ = 0.01 × trace(K) / 10).
4. **AUC Estimation:** Leave-one-out (LOO) hold-out, 1000-resample bootstrap.
5. **Extended Metrics (v6.0.0):** 50/50 bias, Wilcoxon signed-rank test, noise sensitivity sweep, sigmoid fit.
6. **JSON Output:** `cho_results.json` with all results.

### `algorithms/reference_*.py`
- **Baseline implementations:** iMAR, MBIR, Spectral MAR, POCS-TV, NMAR, etc.
- **MAR algorithm testing:** Loads noMAR DICOMs, applies MAR, writes MAR-corrected DICOMs for CHO analysis.

### Utility Scripts
- **`generate_tdp_report.py`** — Generates technical dossier PDF comparing algorithms across tiers.
- **`generate_detectability_curves.py`** — Produces AUC vs. contrast/noise plots.
- **`sweep_vaishnav_matrix.py`** — Batch sensitivity sweeps (contrast, noise levels, etc.).
- **`apply_mar_sir.py`** — Standalone Selective Inpainting Reconstruction (SIR) implementation.

---

## Key Conventions

### Naming & Paths

- **Realization folders:** `realization_001` through `realization_040` (zero-padded 3 digits).
- **DICOM slices:** `slice_0001.dcm` through `slice_0256.dcm` (1-indexed, zero-padded 4 digits).
- **Sinogram HDF5:** `realization_001.h5` format inside `sinograms/{LP,LA}/`.
- **Output directory:** Default is `./astm_mar_ils_dataset_v6` (configurable via `--output-dir`).

### Lesion Placement

- **Lesion position (2D):** Center at (281, 256) in the 512×512 FOV.
- **Lesion slice (3D):** Slice 128 (zero-indexed, middle of 256-slice volume).
- **ROI for CHO:** Extracted from DICOM around lesion center using `TierConfig.roi_size`.

### Command-Line Conventions

- **--tier:** Required. One of `T1_AB`, `T2_SB`, `T3_HEAD`.
- **--output-dir:** Optional. Default: `./astm_mar_ils_dataset_v6`.
- **--workers:** Optional. Integer ≥ 1, or 0 for `os.cpu_count()`. Default: all cores.
- **--dry-run:** Optional flag. Validates config without writing files.
- **--sweep-mode:** Optional flag. Generator only writes slice_0129.dcm per realization (256× storage reduction).
- **--contrast-factor:** Optional float. Scales lesion attenuation (1.0 = normative 12 HU).
- **--no-pdf:** Optional flag. Skip PDF generation.

### Reproducibility

- **Random seeds:** Each realization has a deterministic seed stored in HDF5 `/noise_params.attrs['seed']`.
- **Rerunning generation:** Using the same `--tier`, same `--contrast-factor`, and same seed yields identical sinograms.
- **DICOM UIDs:** StudyInstanceUID and SeriesInstanceUID are generated once per realization worker (constant across 256 slices).

### Tier-Specific Differences

**TierConfig attributes vary per tier:**

```python
tc = TIER_REGISTRY["T1_AB"]
print(tc.body_semi_x_mm, tc.body_semi_y_mm)     # 85.0, 60.0 mm
print(tc.metal_radius_mm)                       # 10.0 mm
print(tc.lesion_semi_a_mm, tc.lesion_semi_b_mm) # 2.5, 2.5 mm (circle)
print(tc.roi_size)                              # 121×121 pixels
print(tc.channel_width_a)                       # 3.0 (Laguerre-Gauss channel width)
```

All tiers use same I₀ calibration approach, noise model, and acquisition geometry.

---

## Common Tasks

### Generate a Fresh Dataset for T2_SB
```bash
python generator_v6_0_0.py --tier T2_SB --output-dir ./my_t2_sb_dataset
```

### Run CHO Analysis on Your MAR Algorithm
```bash
# Assuming lab reconstructions are in ./my_mar_recon/LP and ./my_mar_recon/LA
python run_cho_analysis_v6_0.py \
    --dataset-dir ./my_t2_sb_dataset \
    --mar-output-dir ./my_mar_recon \
    --tier T2_SB
```
Results in `cho_results.json` with ΔAUC and 95% CI.

### Compare Algorithm Performance via TDP Report
```bash
python generate_tdp_report.py \
    --cho-results ./algo1_results.json ./algo2_results.json \
    --output ./TDP_Comparison.pdf
```

### Debug a Single Realization
```bash
# Serial generation for deterministic output
python generator_v6_0_0.py --tier T1_AB --workers 1 --output-dir ./debug_dataset

# Visualize its sinogram
python view_sinograms.py ./debug_dataset/sinograms/LP/realization_001.h5
```

### Sensitivity Sweep (Contrast Variation)
```bash
python sweep_vaishnav_matrix.py \
    --tier T1_AB \
    --contrast-factors 0.5 1.0 1.5 2.0 \
    --output-dir ./sweep_results
```

---

## Important Notes for Contributors

1. **Versioning**: Always embed version in script filename. Maintain backward compatibility by keeping superseded versions (they document the evolution).

2. **Tier Registry**: When adding a new tier, update `TIER_REGISTRY` in `tier_config.py` and validate with `validate_tier_registry()`.

3. **Normative Constants**: The locked metrology baseline (2026-03-14) includes `MU_LESION_CM`, `NUM_REALIZATIONS`, and target AUC ranges. Changes require ASTM Task Group approval and full regeneration + re-validation.

4. **DICOM Compliance**: Always set `SliceLocation`, `StudyInstanceUID`, `SeriesInstanceUID`, and dual window presets when generating DICOM files.

5. **HDF5 Metadata**: Store all realization parameters in HDF5 attributes (`/tier_config`, `/noise_params`, `/geometry`) for reproducibility and post-hoc analysis.

6. **CHO Observer**: Strictly enforce 2D analysis on slice 128 only. 3D integration is prohibited by ASTM §A1.5.3.

7. **Bootstrap and LOO**: All AUC metrics use 1000-resample bootstrap with leave-one-out hold-out. Report 95% CI alongside point estimates.

---

## References

- **ASTM WKXXXXX** — Draft standard for MAR algorithm performance assessment via task-based ILS.
- **IEC 60601-2-44 Ed. 4** — Medical electrical equipment safety & performance standards (CT).
- **Vaishnav et al. (2020)** — "CT metal artifact reduction algorithms: Toward a framework for objective performance." *Medical Physics*, 47(8), 3858-3866.
- **CLAUDE.md** — Detailed internal documentation on generator pipeline, physics constants, and design rationale.

---

*Last updated: 2026-04-03. For questions or contributions, see GitHub issues or contact the ASTM F04 subcommittee.*
