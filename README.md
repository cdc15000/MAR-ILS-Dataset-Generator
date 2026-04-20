![Axial pelvic CT of a patient with bilateral metal-on-metal total hip arthroplasties, before and after orthopaedic metal artefact reduction.](docs_and_references/images/wellenberg_2019_fig8_ab.jpg)<br>
<sub>*Adapted from Wellenberg et al., Skeletal Radiology 48:1775 (2019) [[DOI](https://doi.org/10.1007/s00256-019-03206-z)] · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*</sub>

# MAR ILS Dataset Generator and Evaluation Framework

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Standard: ASTM WKXXXXX Rev 04](https://img.shields.io/badge/Standard-ASTM%20WKXXXXX%20Rev%2004-orange)
![Standard: IEC 60601-2-44 Ed. 4](https://img.shields.io/badge/Standard-IEC%2060601--2--44%20Ed.%204-green)
![DICOM 2026b](https://img.shields.io/badge/DICOM-2026b%20CP--2575-purple)

## Contents

- [Overview](#overview)
- [April 2026 Status: Metrology Baseline Locked](#april-2026-status-metrology-baseline-locked)
  - [ILS Reference Baseline (N=40)](#ils-reference-baseline-n40)
- [Performance Breakthrough: Numba Acceleration](#performance-breakthrough-numba-acceleration)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dataset](#dataset)
- [AI-Integrated Laboratory (MCP)](#ai-integrated-laboratory-mcp)
- [Validation & Audit Tools](#validation--audit-tools)
- [Deliverables & Repository Structure](#deliverables--repository-structure)
- [Metadata Standard: DICOM 2026b Compliant](#metadata-standard-dicom-2026b-compliant)
  - [Verification (hex-tag method)](#verification-hex-tag-method)
- [Regulatory Framework](#regulatory-framework)
- [References](#references)
- [License](#license)
- [Technical Contact](#technical-contact)

---

## Overview

This is the **reference evaluation framework** for conducting standardized **Interlaboratory Studies (ILS)** on CT **Metal Artifact Reduction (MAR)** algorithms. Following the task-based signal detection framework of **Vaishnav et al. (Medical Physics, 2020)**, this toolset provides:

1.  **Standardized Dataset Generation:** Physics-based fan-beam sinogram synthesis (v7.0.0).
2.  **Reference MAR Algorithms:** Implementations of iMAR, MBIR, and Dual-Energy Spectral MAR.
3.  **Objective Statistical Scoring:** Quantifying lesion detectability ($\Delta AUC$) using a **Channelized Hotelling Observer (CHO)** with normative internal noise ($\sigma=15$).
4.  **AI-Native Diagnostics:** Integrated **Model Context Protocol (MCP)** servers for automated data auditing and visualization.

---

## April 2026 Status: Metrology Baseline Locked

The framework has officially transitioned from parallel-beam research tiers to a **Normative Fan-Beam Geometry** (SID=570mm, SDD=1040mm). 

### ILS Reference Baseline ($N=40$)
The following values are established as the normative floor for ASTM WKXXXXX Rev 04:

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Baseline AUC_noMAR** | **0.8294** | **LOCKED** |
| **95% Confidence Interval** | [0.7612, 0.9025] | Verified |
| **Lesion Contrast** | 12.0 HU | Normative |
| **Integrity Check** | SHA-256 | Validated |

---

## Performance Breakthrough: Numba Acceleration
The v7.0.0 engine features a JIT-compiled **Batch Backprojector** optimized for Apple Silicon (M3) and modern x86_64 architectures, providing a **24x speedup** over standard NumPy implementations.

* **Throughput:** ~61s per realization (256 slices).
* **Efficiency:** Full $N=40$ dataset generation in <30 minutes on standard workstations.

---

## Setup

### Requirements

* **Python** 3.10 or newer
* **~40 GB free disk** for a full $N=40$ dataset (~35 GB of HDF5 sinograms + DICOM reconstructions)
* **16+ GB RAM** recommended; 8 CPU cores gets full-$N$ generation under 30 minutes
* macOS, Linux, or Windows (paths below use POSIX)

### Installation

```bash
git clone https://github.com/cdc15000/MAR-ILS-Dataset-Generator.git
cd MAR-ILS-Dataset-Generator

python3.10 -m venv mar-ils
source mar-ils/bin/activate

pip install -r requirements.txt
pip install numba    # optional but strongly recommended — ~24x speedup
```

### Verify the install

```bash
python generator_v7_0_0.py --dry-run
```

A clean `--dry-run` prints the locked constants and exits 0 without writing any files.

---

## Quick Start

```bash
# Generate the canonical fan-beam dataset
python generator_v7_0_0.py --output-dir ./astm_reference_dataset

# Self-test (validates CHO pipeline — ΔAUC = 0 by definition)
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --self-test --internal-noise-sigma 15

# ILS mode (score your MAR algorithm)
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

**Self-test** — pipeline validation (ΔAUC = 0 by construction):

```bash
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --self-test --internal-noise-sigma 15
```

**ILS mode** — score a submitted MAR reconstruction set against the noMAR baseline:

```bash
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./mar_recon \
    --internal-noise-sigma 15
```

`--internal-noise-sigma 15` is the locked normative default — do not change it for submissions scored against the $N=40$ baseline (AUC_noMAR = 0.8294).

### 3. Visualize sinograms

```bash
python view_sinograms.py sinograms/LP/realization_001.h5
python view_sinograms.py sinograms/LP/realization_001.h5 --slice 128
```

### 4. Patch legacy datasets with DICOM 2026b metadata

```bash
python patch_2026b_metadata.py --dataset-dir ./astm_reference_dataset
```

Idempotent; regenerates `checksums_sha256.txt` after injecting the MAR Macro (§ [Verification](#verification-hex-tag-method)).

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
├── generator_provenance.json      # locked constants + git SHA
└── MAR_ILS_Lab_Instructions.pdf   # one-page operational instructions for labs
```

Total size: **~35 GB** for $N=40$; **~18 GB** for $N=20$ screening.

### HDF5 sinogram format

Each `realization_NNN.h5` contains:

| Dataset / attr | Shape / type | Content |
| :--- | :--- | :--- |
| `line_integrals` | `(256, 720, 512)` float32 | Fan-beam line integrals, neper |
| `@geometry.SID_mm` / `@SDD_mm` | scalar | 570 / 1040 |
| `@geometry.n_angles` / `@n_det` | scalar | 720 / 512 |
| `@geometry.angles_deg` / `@det_fan_angles_deg` | 1-D array | Projection + detector geometry samples |
| `@noise_params.I0` / `sigma_e_counts` / `seed` | scalar | Noise calibration (I₀ = 310,853) |
| `@lesion_slice_index` | scalar | `128` — the only slice containing the lesion disc |

### Lab submission layout (for CHO scoring)

Participating labs submit reconstructed DICOMs back in this structure:

```
mar_recon/
├── LP/  realization_001/ ... (slice_NNNN.dcm, 1-indexed)
└── LA/  realization_001/ ...
```

Only `slice_0129.dcm` is read by `run_cho_analysis_v7_0.py` — the CHO is 2D on slice 128; 3D integration is prohibited per §A1.5.3 of the draft standard.

---

## AI-Integrated Laboratory (MCP)
This repository includes **Model Context Protocol (MCP)** servers, allowing AI assistants to act as "Laboratory Hands" on your local workstation.

* **Data Inspector (`mcp_data_inspector.py`):** Automatically parses CHO JSON results and summarizes statistical significance.
* **Visualization (`mcp_visualization.py`):** Renders HDF5 sinogram slices and ROI comparisons directly in the AI chat interface.

---

## Validation & Audit Tools
To ensure inter-laboratory consistency, use the following in-memory physics auditors:
* **`plot_spectral_transparency.py`:** Visualizes the "Transparency Jump" between 60 keV and 140 keV.
* **`check_metal_overflow.py`:** Audits the normative **3000 HU metal-ROI hard-set** requirement.

---

## Deliverables & Repository Structure

| Path | Description |
| :--- | :--- |
| `generator_v7_0_0.py` | **Normative** fan-beam dataset generator (Rev 04). |
| `run_cho_analysis_v7_0.py` | **Normative** 2D CHO scoring tool (Rev 04). |
| `patch_2026b_metadata.py` | One-time DICOM 2026b CP-2575 metadata patcher. |
| `docs_and_references/ASTM_MAR_Standard.md` | Draft standard text (Rev 04, machine-readable) |
| `docs_and_references/IEC_203_6_7_101_compliance_statement_proposal.md` | Post-publication Corrigendum/Amendment proposal for §203.6.7.101.1 (deferred until ASTM FXXXX publishes) |
| `docs_and_references/FDA_guidance_framework.md` | Draft FDA guidance for acceptance criteria |
| `/algorithms` | Reference MAR implementations (iMAR, MBIR, Spectral). |
| `/legacy` | Archived v6.0.0 parallel-beam research framework. |

---

## Metadata Standard: DICOM 2026b Compliant

This is the **first reference implementation** of the **DICOM CP-2575 Metal Artifact Reduction Macro** (PS3.3 C.8.15.3.15), finalized in DICOM 2026b.

All DICOM files produced by the generator and the `patch_2026b_metadata.py` utility include:

| Tag | Keyword | Value |
| :--- | :--- | :--- |
| `(0018,9390)` | Metal Artifact Reduction Sequence | Present (1 item) |
| `(0018,9391)` | Metal Artifact Reduction Applied | `NO` |

The `noMAR` reconstructions are tagged `NO` by definition. Laboratories applying MAR algorithms must set `(0018,9391)` to `YES` and optionally populate `(0018,9392)` Metal Artifact Reduction Algorithm with a value from **CID 10036** (e.g., `MAR_IMAR`, `MAR_SPECTRAL`).

### Verification (hex-tag method)

Because pydicom's bundled data dictionary does not yet include the 2026b additions, auditing requires hex-tag access:

```python
import pydicom
ds = pydicom.dcmread('astm_reference_dataset/noMAR_recon/LP/realization_001/slice_0129.dcm')

# Access the MAR Macro via hex tags
mar_seq = ds[0x00189390]                          # Metal Artifact Reduction Sequence
mar_applied = mar_seq.value[0][0x00189391].value   # → "NO"
print(f'(0018,9391) Metal Artifact Reduction Applied = {mar_applied}')
```

---

## Regulatory Framework

| Layer | Document | Role |
|-------|----------|------|
| **1 — What to record** | DICOM CP-2575 (2026b) | MAR metadata tags in DICOM |
| **2 — Must have / describe / record** | IEC 60601-2-44 Ed. 4 §203.6.7.101 (.1 method, .2 user info, .3 DICOM) | Regulatory mandate |
| **3 — How to measure (post-publication binding to §203.6.7.101.1)** | ASTM FXXXX (formerly WKXXXXX Rev 04) — Corrigendum/Amendment submitted after ASTM FXXXX publishes | Quantitative ΔAUC TYPE TEST |
| **4 — Acceptance** | FDA guidance (proposed) | Non-degradation / superiority thresholds |

---

## References

1. **Vaishnav, J.Y., Ghammraoui, B., Leifer, M., Zeng, R., Jiang, L., and Myers, K.J.** "CT metal artifact reduction algorithms: Toward a framework for objective performance assessment." *Medical Physics*, August 2020, Vol. 47, No. 8, pp. 3344–3355. [[DOI](https://doi.org/10.1002/mp.14231)]

2. **Gjesteby, L., De Man, B., Jin, Y., Paganetti, H., Verburg, J., Giantsoudi, D., and Wang, G.** "Metal Artifact Reduction in CT: Where Are We After Four Decades?" *IEEE Access*, 2016, Vol. 4, pp. 5826–5849. [[DOI](https://doi.org/10.1109/ACCESS.2016.2608621)]

3. **Barrett, H.H., and Myers, K.J.** *Foundations of Image Science*, Chapter 14 — Image Quality, pp. 913–1000. Wiley-Interscience, Hoboken (NJ), 2004. [[Publisher](https://www.wiley.com/en-us/Foundations+of+Image+Science-p-9780471153009)]

4. **American Association of Physicists in Medicine.** *AAPM Report No. 233 — Performance Evaluation of CT Systems*, 2019. [[AAPM / issuu](https://issuu.com/aapmdocs/docs/tg-233_final_8ec461f2715a5e?mode=embed&viewMode=singlePage&backgroundColor=eeeeee)]

---

## License

All code and documentation in this repository are released under the **Apache License, Version 2.0** — see [`LICENSE.md`](LICENSE.md) for the full text.

The single third-party figure (axial pelvic CT at the top of this README) is reproduced from Wellenberg et al. 2019 under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) and is not covered by the Apache 2.0 license; attribution is given in the figure caption.

---

## Technical Contact

**Christopher D. Cocchiaraley** Consumer Member, ASTM International Committee F04  
Executor of the Estate of Veronica M. Cocchiaraley
