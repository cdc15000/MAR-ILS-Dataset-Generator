# Copilot Instructions for MAR ILS Dataset Generator

## Quick Setup

```bash
source mar-ils/bin/activate       # Activate venv (Python 3.10)
pip install -r requirements.txt   # Install dependencies
pip install numba                 # Optional: ~24× speedup (JIT + batch geometry)
```

**Key dependencies:** numpy, scipy, pydicom, h5py, reportlab, tqdm, matplotlib.
`numba` is optional (~24× speedup); `torch` is optional and only required by
`algorithms/reference_dlsc_imar.py`.

> The authoritative, detailed reference is **`CLAUDE.md`** at the repo root. This
> file is a Copilot-oriented summary; when the two disagree, `CLAUDE.md` wins.

---

## High-Level Architecture

The **MAR ILS Dataset Generator** produces a standardized, synthetic CT dataset
for Metal Artifact Reduction (MAR) Interlaboratory Studies (ILS), compliant with
**ASTM WKXXXXX Revision 05** and **IEC 60601-2-44 Ed. 4**. It implements the
task-based signal detection approach of Vaishnav et al. (*Medical Physics*,
47(8), 2020).

### Three-stage workflow

1. **Dataset generation** (`generator_v7_0_0.py`)
   - Physics-based **fan-beam** forward projection (SID = 570 mm, SDD = 1040 mm,
     720 angles, 512 equi-angular detectors, full 360°), 60 keV monochromatic.
   - Noise model: `I_meas = Poisson(I₀·exp(−∫μ dl) + S) + N(0, σ_e²)`.
   - Outputs **40 LP (lesion present) + 40 LA (lesion absent)** realizations as
     HDF5 sinograms + fan-beam FBP DICOM reconstructions (screening: 20 + 20).

2. **MAR algorithm application** (labs, optionally `algorithms/reference_*.py`)
   - A participating lab reconstructs the sinograms with its own MAR pipeline and
     submits DICOMs. **Note:** the `algorithms/reference_*.py` implementations
     currently target the *legacy v6 parallel-beam* framework
     (`run_cho_analysis_v6_0.py`, `tier_config`); they are not yet ported to the
     v7 fan-beam geometry.

3. **Statistical analysis** (`run_cho_analysis_v7_0.py`)
   - 2D Channelized Hotelling Observer (CHO) with 10 Laguerre-Gauss channels on
     **slice 128 only** (3D integration shall not be performed, §A1.5.3).
   - Computes ΔAUC via leave-one-out CV + 1000-resample bootstrap CI.

### Single canonical configuration (v7.0.0)

v7.0.0 uses **one normative configuration** (ASTM Rev 05). There are **no tiers**
in v7 — the three-tier framework (T1_AB / T2_SB / T3_HEAD, parallel-beam) belongs
to the **legacy v6** research framework under `legacy/` and `tier_config.py`,
which is retained for multi-tier research but is **not normative**.

---

## Script Versioning & Usage

**Scripts embed the version in the filename.** **v7.0.0 is the current normative
reference** (fan-beam, single config, ASTM Rev 05). Always use the highest version
number. v6.0.0 (`legacy/`) is the non-normative parallel-beam tiered framework.

### Generate the dataset

```bash
python generator_v7_0_0.py                            # all cores, ./astm_reference_dataset
python generator_v7_0_0.py --output-dir ./my_dataset  # custom output directory
python generator_v7_0_0.py --workers 8                # limit workers
python generator_v7_0_0.py --workers 1                # serial (debugging)
python generator_v7_0_0.py --realizations 20          # screening mode (pilot)
python generator_v7_0_0.py --dry-run                  # validate config, no writes
python generator_v7_0_0.py --no-pdf                   # skip lab-instructions PDF
```

### Run CHO analysis

```bash
# ILS mode (evaluate a lab's MAR reconstructions):
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./mar_recon \
    --internal-noise-sigma 15

# Self-test (pipeline validation — ΔAUC = 0 by definition):
python run_cho_analysis_v7_0.py --dataset-dir ./astm_reference_dataset --self-test

# Screening mode (20 realizations — informative only):
python run_cho_analysis_v7_0.py --dataset-dir ./my_dataset_20 --self-test --realizations 20
```

### Visualize sinograms

```bash
python view_sinograms.py sinograms/LP/realization_001.h5
python view_sinograms.py sinograms/LP/realization_001.h5 --slice 128
```

---

## Code Organization

### `mar_ils_core/` — shared library (single source of truth)
- `constants.py` — normative ASTM Rev 05 parameters (geometry, μ values, ROI,
  HU targets, DICOM tag numbers).
- `phantom.py` — analytic attenuation map (`build_attenuation_map`) and body /
  metal / lesion masks.
- `noise.py` — Poisson + electronic noise model (`apply_noise`).
- `dicom_utils.py` — CT DICOM writer + CP-2575 MAR Macro injection.

`generator_v7_0_0.py` and `run_cho_analysis_v7_0.py` both import from this package.

### `generator_v7_0_0.py` pipeline
1. **I₀ calibration** — 3-FBP analytic approach (fan-beam): noise-free full
   phantom (DC offset), noise-free body-only, 5 noisy MC draws at I₀_ref = 1e5;
   analytic scaling σ ∝ 1/√I₀.
2. **Phantom construction** — analytic ellipse body, centred iron rod, optional
   lesion disc at (281, 256) in **slice 128 only**.
3. **Forward projection** — fan-beam, ray-driven sub-voxel sampling. Numba JIT
   with hand-rolled bilinear interpolation when available.
4. **FBP reconstruction** — cosine pre-weight + Ram-Lak + (SID/L)²
   distance-weighted backprojection + DC offset correction + `+BACKGROUND_HU`.
   Numba **batch** kernel backprojects all 256 slices with geometry computed once.
5. **Natural reconstruction** — lesion contrast (~12 HU) emerges purely from
   sinogram-domain `MU_LESION_CM`; no post-FBP hard-set except metal → 3000 HU.
6. **Output** — HDF5 sinograms, DICOM series, SHA-256 checksums, provenance JSON,
   lab-instructions PDF.

### `run_cho_analysis_v7_0.py` pipeline
1. Loads **only `slice_0129.dcm`** (LESION_SLICE_INDEX = 128, 1-indexed) per realization.
2. Projects each 121×121 ROI through 10 2D Laguerre-Gauss channels → `(N, 10)`.
3. Fits Hotelling template with Tikhonov regularization (λ = 0.01 × trace(K)/10)
   using **LA covariance only**.
4. LOO hold-out AUC (Mann-Whitney) + 1000-resample bootstrap CI + paired ΔAUC CI.
5. Writes `cho_results.json`.

### Numba acceleration
All Numba code is conditional on `_HAS_NUMBA`; without numba the generator falls
back to NumPy. Workers set `numba.set_num_threads(1)` (parallelism is across
`ProcessPoolExecutor` workers). The batch FBP kernel is bit-exact vs per-slice
within float32 precision in the diagnostic FOV (regression-guarded by
`tests/test_batch_fbp.py`, which requires numba).

---

## Tests, Lint & CI

```bash
python -m pytest tests/ -v                 # full suite
pip install numba && python -m pytest -v   # also exercises JIT + batch FBP paths
ruff check mar_ils_core/ tests/ generator_v7_0_0.py run_cho_analysis_v7_0.py
```

CI (`.github/workflows/ci.yml`) runs four jobs: `lint` (ruff), `test` (NumPy
fallback), `test-numba` (JIT paths), and `dry-run` (`generator_v7_0_0.py --dry-run`).

---

## Data Structures & I/O

### HDF5 sinogram format (fan-beam)
Each `sinograms/{LP,LA}/realization_NNN.h5` contains:
- `line_integrals`: float32 `(256, 720, 512)` line integrals in neper.
- `geometry` attrs: `type='fan-beam'`, `SID_mm=570`, `SDD_mm=1040`, `n_angles=720`,
  `n_det=512`, `gamma_max_deg`, `delta_gamma_deg`, `angles_deg`, `det_fan_angles_deg`.
- `noise_params` attrs: `I0`, `scatter_frac`, `sigma_e_counts`, `seed`, `jitter_deg`,
  `place_lesion`, `lesion_slice_index` (128), `lesion_z_extent`.

### DICOM output / lab submission
```
noMAR_recon/{LP,LA}/realization_NNN/slice_0001.dcm … slice_0256.dcm   (1-indexed)
mar_recon/{LP,LA}/realization_NNN/slice_0001.dcm  … slice_0256.dcm   (lab submission)
```
CHO analysis **reads only `slice_0129.dcm`**. All DICOMs include the CP-2575
Metal Artifact Reduction Macro (`(0018,9390)` → `(0018,9391)` = `"NO"` for noMAR).

---

## Key Constants (Normative — Do Not Modify)

| Parameter | Value |
|---|---|
| Volume | 512×512×256 voxels, 0.5 mm isotropic |
| Acquisition | Fan-beam, SID = 570 mm, SDD = 1040 mm, 720 angles, 512 equi-angular detectors, full 360° |
| Realizations | 40 LP + 40 LA (screening: 20 + 20) |
| Lesion z-extent | Slice 128 only (single disc) |
| CHO observer | 2D, slice 128 only — 3D integration shall not be performed |
| CHO ROI | (281, 256), 121×121 voxels |
| Channel width a | 7.5 voxels |
| Background HU | 40 HU |
| Lesion contrast | ~12 HU (sinogram-domain only) |
| Metal HU | 3000 HU |
| Noise σ target | 30 HU in soft tissue |
| μ soft tissue | 0.2059 cm⁻¹ |
| μ iron | 2.408 cm⁻¹ |
| Internal noise σ | 15 (normative default) |
| AUC tolerance | ±0.005 |

**ASTM metrology baseline (locked 2026-04-07):** `MU_LESION_CM = MU_TISSUE_CM ×
(1 + 12/1000)`, N = 40 LP + 40 LA, σ = 15, **`BASELINE_AUC_noMAR = 0.8294`**,
CI [0.7612, 0.9025], calibrated I₀ = 310,853. Changes require ASTM Task Group
approval and full regeneration + re-validation.

---

## Notes for Contributors

1. **Versioning:** embed the version in the script filename; keep superseded
   versions to document evolution.
2. **Normative constants:** the locked metrology baseline (`MU_LESION_CM`,
   `NUM_REALIZATIONS`, `BASELINE_AUC_noMAR`, geometry) must not change without
   Task Group approval. Touching the projection / FBP / calibration code risks
   perturbing the bit-exact locked AUC.
3. **DICOM compliance:** always set `SliceLocation`, `StudyInstanceUID`,
   `SeriesInstanceUID`, and include the CP-2575 MAR Macro (via `dicom_utils`).
4. **HDF5 metadata:** store all realization parameters in HDF5 attributes for
   reproducibility.
5. **CHO observer:** strictly 2D on slice 128. 3D integration shall not be performed (§A1.5.3).
6. **Bootstrap / LOO:** report 95% CI alongside every AUC point estimate.

---

## References

- **ASTM WKXXXXX Rev 05** — draft standard for MAR performance assessment via task-based ILS.
- **IEC 60601-2-44 Ed. 4** — medical electrical equipment safety & performance (CT).
- **Vaishnav et al. (2020)** — *Medical Physics*, 47(8), 3858-3866.
- **`CLAUDE.md`** — authoritative internal documentation (pipeline, physics, design rationale).

---

*Last updated: 2026-07-06. Reflects the v7.0.0 fan-beam normative reference (ASTM Rev 05).*
