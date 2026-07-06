![Axial pelvic CT of a patient with bilateral metal-on-metal total hip arthroplasties, before and after orthopaedic metal artefact reduction.](docs_and_references/images/wellenberg_2019_fig8_ab.jpg)<br>
<sub>*Adapted from Wellenberg et al., Skeletal Radiology 48:1775 (2019) [[DOI](https://doi.org/10.1007/s00256-019-03206-z)] · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*</sub>

# MAR ILS Dataset Generator and Evaluation Framework

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Standard: ASTM WKXXXXX Rev 05](https://img.shields.io/badge/Standard-ASTM%20WKXXXXX%20Rev%2005-orange)
![Standard: IEC 60601-2-44 Ed. 4](https://img.shields.io/badge/Standard-IEC%2060601--2--44%20Ed.%204-green)
![DICOM 2026b](https://img.shields.io/badge/DICOM-2026b%20CP--2575-purple)

## Contents

- [Project Status](#project-status)
- [Overview](#overview)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dataset](#dataset)
- [References](#references)
- [Citation](#citation)
- [License](#license)
- [Governance](#governance)
- [Technical Contact](#technical-contact)

For per-version status, performance work, tooling, DICOM 2026b compliance details, repository map, and regulatory positioning, see [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

---

## Project Status

| Phase | Description | Status |
| :--- | :--- | :--- |
| 1 | Concept development (Vaishnav framework adaptation) | Complete |
| 2 | Draft ASTM test method (WKXXXXX Rev 01–05) | Complete |
| 3 | Reference methodology (parallel-beam v5/v6, fan-beam v7) | Complete |
| 4 | ILS reference implementation (locked baseline, lab instructions, DICOM 2026b) | Complete |
| 5 | ASTM interlaboratory study (Form 01 + E691 package designed; lab recruitment pending) | In progress |
| 6 | Journal publication (*Medical Physics* or equivalent) | Planned |
| 7 | IEC incorporation (§203.6.7.101.1 binding amendment, post-ASTM-FXXXX publication) | Planned |

---

## Overview

This is the **reference evaluation framework** for conducting standardized **Interlaboratory Studies (ILS)** on CT **Metal Artifact Reduction (MAR)** algorithms. Following the task-based signal detection framework of **Vaishnav et al. (Medical Physics, 2020)**, this toolset provides:

1.  **Standardized Dataset Generation:** Physics-based fan-beam sinogram synthesis (v7.0.0).
2.  **Reference MAR Algorithms:** A parameter-free linear-interpolation reference (**LI-MAR**) for the v7.0.0 fan-beam geometry ([`algorithms/v7/`](algorithms/v7/)), plus legacy parallel-beam implementations (iMAR, MBIR, Dual-Energy Spectral MAR) retained for the v6 research framework ([`algorithms/`](algorithms/)).
3.  **Objective Statistical Scoring:** Quantifying lesion detectability ($\Delta AUC$) using a **Channelized Hotelling Observer (CHO)** with normative internal noise ($\sigma=15$).
4.  **AI-Native Diagnostics:** Integrated **Model Context Protocol (MCP)** servers for automated data auditing and visualization.

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

### 3. Apply the reference LI-MAR (negative control / ΔAUC anchor)

A parameter-free linear-interpolation MAR for the fan-beam dataset. It is **non-normative** (the type test scores the *lab's* algorithm), but it validates the full MAR→CHO loop and anchors the ΔAUC scale as the designated control (ΔAUC ≈ −0.23) — every commercial MAR is expected to beat it.

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

See [`algorithms/v7/README.md`](algorithms/v7/README.md) for the measured ΔAUC anchor on the locked $N=40$ dataset.

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

Only `slice_0129.dcm` is read by `run_cho_analysis_v7_0.py` — the CHO is 2D on slice 128; 3D integration shall not be performed per §A1.5.3 of the draft standard.

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
> ASTM Work Item WKXXXXX — Standard Test Method for Evaluation of CT Metal Artifact Reduction Algorithms Using a Channelized Hotelling Observer.
> https://github.com/cdc15000/MAR-ILS-Dataset-Generator
> DOI: &lt;pending Zenodo deposit&gt;

---

## License

All code and documentation in this repository are released under the **Apache License, Version 2.0** — see [`LICENSE.md`](LICENSE.md) for the full text.

The single third-party figure (axial pelvic CT at the top of this README) is reproduced from Wellenberg et al. 2019 under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) and is not covered by the Apache 2.0 license; attribution is given in the figure caption.

---

## Governance

Change control, version freezing, and the approval process for normative parameters are documented in [`GOVERNANCE.md`](GOVERNANCE.md).

---

## Technical Contact

**Christopher D. Cocchiaraley** Consumer Member, ASTM International Committee F04  
Executor of the Estate of Veronica M. Cocchiaraley
