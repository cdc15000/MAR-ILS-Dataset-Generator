![Axial pelvic CT of a patient with bilateral metal-on-metal total hip arthroplasties, before and after orthopaedic metal artefact reduction.](docs_and_references/images/wellenberg_2019_fig8_ab.jpg)<br>
<sub>*Adapted from Wellenberg et al., Skeletal Radiology 48:1775 (2019) [[DOI](https://doi.org/10.1007/s00256-019-03206-z)] · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*</sub>

# MAR ILS Dataset Generator and Evaluation Framework

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![Standard: ASTM WKXXXXX Rev 06 draft](https://img.shields.io/badge/Standard-ASTM%20WKXXXXX%20Rev%2006%20draft-orange)
![Standard: IEC 60601-2-44 Ed. 4](https://img.shields.io/badge/Standard-IEC%2060601--2--44%20Ed.%204-green)
![DICOM 2026b](https://img.shields.io/badge/DICOM-2026b%20CP--2575-purple)

## Contents

[**Project Status**](#project-status) |
[**Overview**](#overview) |
[**v9 Redesign**](#october-2026-v9-redesign) |
[**Setup**](#setup) |
[**Quick Start (v7)**](#quick-start-v700-superseded) |
[**Usage**](#usage) |
[**Dataset**](#dataset) |
[**References**](#references) |
[**Citation**](#citation) |
[**License**](#license) |
[**Governance**](#governance) |
[**Technical Contact**](#technical-contact)

For per-version status, performance work, tooling, DICOM 2026b compliance details, repository map, and regulatory positioning, see [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

---

## Project Status

| Phase | Description | Status |
| :--- | :--- | :--- |
| 1 | Concept development (Vaishnav framework adaptation) | Complete |
| 2 | Draft ASTM test method (WKXXXXX Rev 01–06) | Rev 06 drafted October 2026 |
| 3 | Reference methodology (parallel-beam v5/v6, fan-beam v7; v9 redesign after Vaishnav et al. 2020) | v9 built and validated October 2026 |
| 4 | ILS reference implementation (v9 dataset writer, blinding tooling, lab instructions, DICOM 2026b) | v9 dataset writer pending |
| 5 | ASTM interlaboratory study (Form 01 package ready, awaiting IEC 60601-2-44 Ed. 4 publication, forecast November 2026; dataset DOI reserved; lab recruitment pending) | In progress |
| 6 | Journal publication (*Medical Physics* or equivalent) | Planned |
| 7 | IEC incorporation (§203.6.7.101.1 binding amendment, post-ASTM-FXXXX publication) | Planned |

---

## Overview

This is the **reference evaluation framework** for conducting standardized **Interlaboratory Studies (ILS)** on CT **Metal Artifact Reduction (MAR)** algorithms. Following the task-based signal detection framework of **Vaishnav et al. (Medical Physics, 2020)**, this toolset provides:

1.  **Standardized Dataset Generation:** v9.0.0 simulates Vaishnav's numerical head phantom (2D, polychromatic 120 kV fan beam, analytic projections). v7.0.0 (simplified torso phantom) is superseded.
2.  **Reference MAR Algorithms:** the two sinogram-inpainting methods of Vaishnav et al. as known-answer controls (`mar_ils_core/mar_v9.py`); the v7 LI-MAR reference ([`algorithms/v7/`](algorithms/v7/)) and legacy parallel-beam implementations are retained for the record.
3.  **Objective Statistical Scoring:** lesion detectability change ($\Delta AUC_\infty$) from a **Channelized Hotelling Observer (CHO)** with dense difference-of-Gaussians channels and antler-plot extrapolation, as in Vaishnav et al.
4.  **AI-Native Diagnostics:** Integrated **Model Context Protocol (MCP)** servers for automated data auditing and visualization.

The [project wiki](https://github.com/cdc15000/MAR-ILS-Dataset-Generator/wiki) describes the [Technical Foundation](https://github.com/cdc15000/MAR-ILS-Dataset-Generator/wiki/Technical-Foundation) of the CHO and the [Phantom Design](https://github.com/cdc15000/MAR-ILS-Dataset-Generator/wiki/Phantom-Design). Technical Foundation also explains [why the method uses a task-based ΔAUC](https://github.com/cdc15000/MAR-ILS-Dataset-Generator/wiki/Technical-Foundation#why-a-task-based-metric-rather-than-fundamental-image-quality-metrics) rather than HU accuracy, noise or other fundamental image-quality metrics.

---

## October 2026: v9 Redesign

An audit in October 2026 found defects in the v7.0.0 generator (shared noise between paired realizations, a lesion that rotated with the artifact jitter, and uncorrected scatter). With those fixed, the v7 LI-MAR result of ΔAUC ≈ −0.23 does not hold, and it is withdrawn. The method was then rebuilt to follow Vaishnav et al. (Med Phys 2020) closely:

- **Phantom:** the paper's head phantom (bone skull, iron fillings, cochlear and deep-brain implants, rotatable lumpy inset), 2D, 512 × 512 over 30 cm.
- **Acquisition:** fan beam (SID 541 mm, SDD 949 mm, 888 channels, 984 views), 120 kV spectrum (SpekPy), Poisson noise, constant scatter and electronic noise, from the references the paper cites.
- **Observer:** CHO with ten dense difference-of-Gaussians channels, pooled covariance, no internal noise; AUC∞ by antler-plot extrapolation; 1000 training and 1000 test images per class.
- **Test levels:** signal amplitudes 1.020, 1.025 and 1.030 at dose 1 × 10⁶ (the E691 materials).
- **Validation (pre-registered, five independent replicate datasets):** pipeline identity, null-signal, graded-harm and control-sign checks all pass. The improving control gives ΔAUC∞ = +0.215, +0.238 and +0.254 at the three levels; the degrading control gives −0.061; replicate SD 0.004 to 0.008. An attempt to reproduce the paper's published curves numerically matched the qualitative behaviour but not the size of the MAR benefit.

v9 code: [`mar_ils_core/vaishnav_phantom.py`](mar_ils_core/vaishnav_phantom.py), [`vaishnav_sim.py`](mar_ils_core/vaishnav_sim.py), [`ddog.py`](mar_ils_core/ddog.py), [`cho_v9.py`](mar_ils_core/cho_v9.py), [`mar_v9.py`](mar_ils_core/mar_v9.py), [`generator_v9_0_0.py`](generator_v9_0_0.py) (ROI mode; the dataset writer for sinograms and DICOM is next), [`run_cho_analysis_v9_0.py`](run_cho_analysis_v9_0.py), [`tests/test_v9.py`](tests/test_v9.py).

```bash
# Simulate centre ROIs for FBP, MAR 1 and MAR 2 at one level (small N shown)
python generator_v9_0_0.py --out roi.npz --dose 1e6 --amplitude 1.025 \
    --lump-scale 0.0588 --n-train 100 --n-test 100 \
    --methods fbp mar1 mar2 --tau-counts 1124 --tau-mu 1.953

# AUC∞ per method and ΔAUC∞ against FBP
python run_cho_analysis_v9_0.py roi.npz --out cho.json
```

The v8 files (`generator_v8_0_0.py`, `run_cho_analysis_v8_0.py`, `pilot_gate_v8_0.py`) are the intermediate fix of the v7 design and are kept for the record. The v7 sections below describe the superseded design.

---

## Setup

### Requirements

* **Python** 3.11 or newer
* **~40 GB free disk** for a full $N=40$ dataset (~35 GB of HDF5 sinograms + DICOM reconstructions)
* **16+ GB RAM** recommended; 8 CPU cores gets full-$N$ generation under 30 minutes. On an 8 GB machine, use `--workers 2`: more workers exhaust RAM and run slower.
* macOS, Linux, or Windows (paths below use POSIX)

### Installation

```bash
git clone https://github.com/cdc15000/MAR-ILS-Dataset-Generator.git
cd MAR-ILS-Dataset-Generator

python3.11 -m venv mar-ils
source mar-ils/bin/activate

pip install -r requirements.txt
pip install numba    # optional but strongly recommended; ~24x speedup
```

### Verify the install

```bash
python generator_v7_0_0.py --dry-run
```

A clean `--dry-run` prints the reference constants and exits 0 without writing any files.

For the superseded v7.0.0 dataset, the self-test in [Quick Start](#quick-start-v700-superseded) reports AUC_noMAR = 0.8294, 95% CI [0.7612, 0.9025]. The archived reference record is in [`docs_and_references/baseline/`](docs_and_references/baseline/).

---

## Quick Start (v7.0.0, superseded)

```bash
# Generate the canonical fan-beam dataset
python generator_v7_0_0.py --output-dir ./astm_reference_dataset

# Self-test (validates the CHO pipeline; ΔAUC = 0 by definition)
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --self-test --internal-noise-sigma 15

# MAR scoring mode (score a MAR reconstruction set)
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./mar_recon \
    --internal-noise-sigma 15
```

---

## Usage

### 1. Generate the reference dataset

```bash
python generator_v7_0_0.py [options]
```

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--output-dir PATH` | `./astm_reference_dataset` | Target directory for the dataset |
| `--workers N` | all cores | Process-pool size (use `1` for debugging) |
| `--realizations N` | `40` | Normative $N=40$; `20` for screening pilot |
| `--dry-run` | off | Print config and exit without writing any files |
| `--no-pdf` | off | Skip the lab instructions PDF |

### 2. Run CHO analysis

**Self-test:** pipeline validation (ΔAUC = 0 by construction):

```bash
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --self-test --internal-noise-sigma 15
```

**MAR scoring mode:** score a MAR reconstruction set against the noMAR baseline:

```bash
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./mar_recon \
    --internal-noise-sigma 15
```

`--internal-noise-sigma 15` is the normative default; do not change it for results scored against the $N=40$ baseline (AUC_noMAR = 0.8294).

### 3. Apply the reference LI-MAR (v7, superseded)

A parameter-free linear-interpolation MAR for the v7 fan-beam dataset, retained for the record. Its earlier ΔAUC ≈ −0.23 came from defects in the v7 generator and is withdrawn; see [October 2026: v9 Redesign](#october-2026-v9-redesign).

```bash
# Produce a CHO-ready LI-MAR reconstruction set (slice_0129.dcm per realization)
python algorithms/v7/reference_li_mar_v7.py \
    --dataset-dir ./astm_reference_dataset \
    --output-dir  ./li_mar_recon

# Score it against the noMAR baseline
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./li_mar_recon \
    --internal-noise-sigma 15
```

LI-MAR output is tagged Metal Artifact Reduction Applied = `YES`, as the ILS protocol requires for MAR-corrected submissions.

See [`algorithms/v7/README.md`](algorithms/v7/README.md) for the measured ΔAUC anchor on the reference $N=40$ dataset.

### 4. Visualize sinograms

```bash
python view_sinograms.py sinograms/LP/realization_001.h5
python view_sinograms.py sinograms/LP/realization_001.h5 --slice 128
```

---

## Dataset

### Produced by the generator

```
<output_dir>/
├── sinograms/
│   ├── LP/  realization_001.h5 ... realization_040.h5   # lesion-present
│   └── LA/  realization_001.h5 ... realization_040.h5   # lesion-absent
├── noMAR_recon/
│   ├── LP/  realization_001/ ... realization_040/        # 256 DICOMs each
│   └── LA/  realization_001/ ... realization_040/
├── checksums_sha256.txt           # SHA-256 integrity manifest
├── generator_provenance.json      # reference constants + git SHA
└── MAR_ILS_Lab_Instructions.pdf   # one-page instructions for the labeled reference dataset
```

Total size: **~35 GB** for $N=40$; **~18 GB** for $N=20$ screening.

Every DICOM file the generator writes includes the DICOM 2026b CP-2575 Metal Artifact Reduction Macro with Metal Artifact Reduction Applied `(0018,9391)` = `NO`. The generator has written the macro natively since April 7, 2026, so no separate patch step is needed. `patch_2026b_metadata.py` is kept only to retrofit datasets generated before that date.

### Distribution

The canonical $N=40$ dataset will be distributed to ILS laboratories through Zenodo under DOI [10.5281/zenodo.23144489](https://doi.org/10.5281/zenodo.23144489), licensed CC BY 4.0. The DOI is reserved; the record will be published, and the link will resolve, before the dataset is distributed to laboratories. Until then, generate the dataset locally with the command above.

### HDF5 sinogram format

Each `realization_NNN.h5` contains:

| Dataset / attr | Shape / type | Content |
| :--- | :--- | :--- |
| `line_integrals` | `(256, 720, 512)` float32 | Fan-beam line integrals, neper |
| `@geometry.SID_mm` / `@SDD_mm` | scalar | 570 / 1040 |
| `@geometry.n_angles` / `@n_det` | scalar | 720 / 512 |
| `@geometry.angles_deg` / `@det_fan_angles_deg` | 1-D array | Projection + detector geometry samples |
| `@noise_params.I0` / `sigma_e_counts` / `seed` | scalar | Noise calibration (I₀ = 310,853) |
| `@lesion_slice_index` | scalar | `128`, the only slice containing the lesion disc |

### MAR reconstruction layout (for CHO scoring)

`run_cho_analysis_v7_0.py` reads MAR reconstructions in this structure:

```
mar_recon/
├── LP/  realization_001/ ... (slice_NNNN.dcm, 1-indexed)
└── LA/  realization_001/ ...
```

Only `slice_0129.dcm` is read by `run_cho_analysis_v7_0.py`. The CHO is 2D on slice 128; 3D integration shall not be performed per §A1.5.3 of the draft standard.

ILS laboratories receive a coded dataset and submission instructions from the study coordinator.

---

## References

1. **Vaishnav, J.Y., Ghammraoui, B., Leifer, M., Zeng, R., Jiang, L., and Myers, K.J.** "CT metal artifact reduction algorithms: Toward a framework for objective performance assessment." *Medical Physics*, August 2020, Vol. 47, No. 8, pp. 3344–3355. [[DOI](https://doi.org/10.1002/mp.14231)]

2. **Gjesteby, L., De Man, B., Jin, Y., Paganetti, H., Verburg, J., Giantsoudi, D., and Wang, G.** "Metal Artifact Reduction in CT: Where Are We After Four Decades?" *IEEE Access*, 2016, Vol. 4, pp. 5826–5849. [[DOI](https://doi.org/10.1109/ACCESS.2016.2608621)]

3. **Barrett, H.H., and Myers, K.J.** *Foundations of Image Science*, Chapter 14 — Image Quality, pp. 913–1000. Wiley-Interscience, Hoboken (NJ), 2004. [[Publisher](https://www.wiley.com/en-us/Foundations+of+Image+Science-p-9780471153009)]

4. **American Association of Physicists in Medicine.** *AAPM Report No. 233 — Performance Evaluation of CT Systems*, 2019. [[AAPM / issuu](https://issuu.com/aapmdocs/docs/tg-233_final_8ec461f2715a5e?mode=embed&viewMode=singlePage&backgroundColor=eeeeee)]

---

## Citation

If you use this framework in your research, please cite:

> Cocchiaraley, C.D. *MAR ILS Dataset Generator and Evaluation Framework*, v7.0.0.
> ASTM Work Item WKXXXXX: Standard Test Method for Evaluation of Metal Artifact Reduction (MAR) Performance in Tomographic Imaging Systems Using a Model Observer.
> https://github.com/cdc15000/MAR-ILS-Dataset-Generator

To cite the reference dataset:

> Cocchiaraley, C. *MAR ILS Reference Dataset v7.0.0: Synthetic Fan-Beam CT Sinograms and Reference Reconstructions for a Metal Artifact Reduction Interlaboratory Study*. Zenodo, 2026. https://doi.org/10.5281/zenodo.23144489

The dataset DOI is reserved and resolves once the Zenodo record is published.

---

## License

All code and documentation in this repository are released under the **Apache License, Version 2.0**; see [`LICENSE.md`](LICENSE.md) for the full text.

The single third-party figure (axial pelvic CT at the top of this README) is reproduced from Wellenberg et al. 2019 under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) and is not covered by the Apache 2.0 license; attribution is given in the figure caption.

---

## Governance

Change control, version freezing, and the approval process for normative parameters are documented in [`GOVERNANCE.md`](GOVERNANCE.md).

---

## Technical Contact

**Christopher D. Cocchiaraley** Consumer Member, ASTM International Committee F04  
Executor of the Estate of Veronica M. Cocchiaraley  
ORCID: [0009-0006-4995-566X](https://orcid.org/0009-0006-4995-566X)
