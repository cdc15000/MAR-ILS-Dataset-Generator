# MAR ILS Dataset Generator

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Standard: ASTM WKXXXXX](https://img.shields.io/badge/Standard-ASTM%20WKXXXXX-orange)
![Standard: IEC 60601-2-44 Ed. 4](https://img.shields.io/badge/Standard-IEC%2060601--2--44%20Ed.%204-green)

## Overview

This is the **reference evaluation framework** for conducting standardized **Interlaboratory Studies (ILS)** on CT **Metal Artifact Reduction (MAR)** algorithms. Following the task-based signal detection framework of **Vaishnav et al. (Medical Physics, 2020)**, this toolset provides:

1. **Standardized Dataset Generation:** Analytic phantoms across three anatomical tiers.
2. **Reference MAR Algorithms:** Implementations of Iterative Inpainting (iMAR), Model-Based Iterative Recon (MBIR), and Dual-Energy Spectral MAR.
3. **Objective Statistical Scoring:** Quantifying lesion detectability ($\Delta AUC$) using a **Channelized Hotelling Observer (CHO)** with 95% Confidence Intervals.

## Design Philosophy

This generator is intended as a **Phase 1 benchmark** — an analytic phantom with controlled Gaussian noise and a deterministic artifact template. This maximizes reproducibility across laboratories and isolates MAR algorithm performance from complex physics variables.

For ASTM WKXXXXX and IEC 60601-2-44 conformity assessment, this analytic approach is the preferred standard due to its mathematical tractability and signal-known-exactly (SKE) properties. Physics-based simulations (e.g., DukeSim/XCAT) are better suited to Phase 2 clinical generalization work.

## 🚀 March 2026 Update: Spectral Superiority

Recent ILS benchmarking has established Dual-Energy Spectral CT as the definitive path to **Superiority** in high-blockage scenarios.

- **The Information Paradox:** At low blockage (~23%), raw FBP is near-optimal. Applying MAR often degrades performance ($\Delta AUC \approx -0.25$).
- **The Software Floor:** At normative "Stress Test" blockage (~40%), single-energy algorithms (iMAR, MBIR) fail to achieve superiority due to total signal loss in the metal shadow.
- **The Spectral Breakthrough:** By maintaining photon transmission through moderate-density implants, Spectral MAR (140 keV VMI blend) achieved the first formally **SUPERIOR** result in this framework ($\Delta AUC +0.049$).

## 📊 ILS Performance Matrix (40% Blockage)

| Tier | Phantom | Metal | 140 keV Trans. | Result |
|------|---------|-------|----------------|--------|
| T1_AB | Adult Body | Cobalt-Chrome | 5.3% | INDETERMINATE |
| T2_SB | Pediatric | Stainless Steel | 36.8% | **SUPERIOR** |
| T3_HEAD | Neuro/Head | Titanium | 47.4% | INFERIOR (Baseline Paradox) |

## Features

- **Cylindrical lesion geometry** spanning all 256 slices for robust 3D-CHO analysis
- **Physically consistent artifacts** simulated via photon-starvation FBP
- **DICOM compliance** with `SliceLocation` tags and dual window presets (soft-tissue and lesion-centered)
- **Automated validation** via SHA-256 checksums and a generated lab instructions PDF

## Installation

```bash
git clone https://github.com/your-username/MAR-ILS-Dataset-Generator.git
cd MAR-ILS-Dataset-Generator
pip install -r requirements.txt
```

## 🛠️ Executing the ILS Workflow

### 1. Generate Standardized Datasets

Generate fresh benchmark datasets for participating laboratories using the normative 40% blockage configuration.

```bash
python generator_v6_0_0.py --tier T2_SB --sweep-mode
```

### 2. Run Reference Algorithms

Test the reference Spectral MAR implementation:

```bash
python algorithms/reference_spectral_mar.py --input-dir ./astm_mar_ils_t2_sb --tier T2_SB --blend-weight 0.7
```

### 3. Statistical CHO Analysis

Evaluate detectability across 80 realizations to determine the precision and bias of the algorithm.

```bash
python run_cho_analysis_v6_0.py --dataset-dir ./astm_mar_ils_t2_sb --mar-output-dir ./astm_mar_ils_t2_sb/spectral_mar_recon
```

## 📂 Deliverables

- `/algorithms` — Reference implementations for benchmark comparison
- `/DICOM` — 20 realizations × 4 series × 256 slices (20,480 slices total)
- `/results_archive` — Historical ILS data, including the `spectral_traceability.json` hardware requirements
- `MAR_ILS_Lab_Instructions.pdf` — Standardized protocol for participating labs
- `TDP_Comparison_Report.pdf` — Automated Technical Dossier comparing all tested algorithms
- `metadata.csv` — Full phantom and simulation parameters
- `checksums_sha256.txt` — SHA-256 manifest for all DICOM files

## Technical Contact

Christopher D. Cocchiaraley  
Consumer Member, ASTM International Committee F04
