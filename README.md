# MAR ILS Dataset Generator (v3.0)

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Standard: ASTM WKXXXXX](https://img.shields.io/badge/Standard-ASTM%20WKXXXXX-orange)
![Standard: IEC 60601-2-44 Ed. 4](https://img.shields.io/badge/Standard-IEC%2060601--2--44%20Ed.%204-green)

## Overview
This repository contains the **Reference Implementation** for generating a standardized, synthetic CT dataset for use in **Metal Artifact Reduction (MAR) Interlaboratory Studies (ILS)**. The generator follows the task-based signal detection framework established by **Vaishnav et al. (Medical Physics, 47(8), 2020)**.
The resulting dataset provides a deterministic "Ground Truth" for quantifying the change in lesion detectability ($\Delta AUC$) attributable to MAR algorithms using a **Channelized Hotelling Observer (CHO)**.
## Strategic Rationale: Phase 1 vs. Phase 2
This generator is purposely designed as a **Phase 1 Benchmark**.
* **Phase 1 (This Tool):** Utilizes an analytic phantom with controlled Gaussian noise and a deterministic artifact template. This provides maximum reproducibility across different laboratories and isolates the MAR algorithm's performance from complex physics-based variables.
* **Phase 2 (Clinical Generalization):** Advanced physics simulations (e.g., DukeSim/XCAT) may be used subsequently for clinical generalization. However, for **ASTM WKXXXXX** and **IEC 60601-2-44** conformity assessment, this analytic benchmark is the preferred "Gold Standard" due to its mathematical tractability and signal-known-exactly (SKE) properties.
## Features
- **Cylindrical Lesion Geometry:** Ensures lesion presence across all 256 slices for robust 3D-CHO analysis.
- **Physically Consistent Artifacts:** Simulates photon-starvation via Filtered Backprojection (FBP).
- **DICOM Compliance:** Includes `SliceLocation` tags and dual window presets (Soft-tissue and Lesion-centered).
- **Automated Validation:** Generates SHA-256 checksums and a comprehensive Lab Instructions PDF.
## Installation
Ensure you have Python 3.8+ installed.
1. Clone the repository:
```bash
git clone https://github.com/your-username/MAR-ILS-Dataset-Generator.git
cd MAR-ILS-Dataset-Generator
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage
Run the generator script to produce the full ILS archive:
```bash
python generator_v3.py
```
This will generate:
- `/DICOM`: 20 realizations (20,480 slices total)
- `MAR_ILS_Lab_Instructions.pdf`: Standardized protocol for participating labs
- `metadata.csv`: Full phantom and simulation parameters
## Technical Contact
Christopher D. Cocchiaraley
Consumer Member, ASTM International Committee F04
