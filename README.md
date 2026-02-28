# MAR ILS Dataset Generator

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Standard: ASTM WKXXXXX](https://img.shields.io/badge/Standard-ASTM%20WKXXXXX-orange)
![Standard: IEC 60601-2-44 Ed. 4](https://img.shields.io/badge/Standard-IEC%2060601--2--44%20Ed.%204-green)

## Overview

This is the **reference implementation** for generating a standardized synthetic CT dataset for **Metal Artifact Reduction (MAR) Interlaboratory Studies (ILS)**. The generator follows the task-based signal detection framework of **Vaishnav et al. (Medical Physics, 47(8), 2020)** and produces a ground-truth dataset for quantifying lesion detectability (ΔAUC) using a **Channelized Hotelling Observer (CHO)**.

## Design Philosophy

This generator is intended as a **Phase 1 benchmark** — an analytic phantom with controlled Gaussian noise and a deterministic artifact template. This maximizes reproducibility across laboratories and isolates MAR algorithm performance from complex physics variables.

For ASTM WKXXXXX and IEC 60601-2-44 conformity assessment, this analytic approach is the preferred standard due to its mathematical tractability and signal-known-exactly (SKE) properties. Physics-based simulations (e.g., DukeSim/XCAT) are better suited to Phase 2 clinical generalization work.

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

## Usage

```bash
python generator_v3_0_1.py
```

This produces:

- `/DICOM` — 20 realizations × 4 series × 256 slices (20,480 slices total)
- `MAR_ILS_Lab_Instructions.pdf` — standardized protocol for participating labs
- `metadata.csv` — full phantom and simulation parameters
- `checksums_sha256.txt` — SHA-256 manifest for all DICOM files

## Technical Contact

Christopher D. Cocchiaraley<br>
Consumer Member, ASTM International Committee F04
