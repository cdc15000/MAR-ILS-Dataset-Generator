# MAR ILS Dataset Generator & Evaluation Framework

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Standard: ASTM WKXXXXX](https://img.shields.io/badge/Standard-ASTM%20WKXXXXX-orange)
![Standard: IEC 60601-2-44 Ed. 4](https://img.shields.io/badge/Standard-IEC%2060601--2--44%20Ed. 4-green)

## Overview

This is the **reference evaluation framework** for conducting standardized **Interlaboratory Studies (ILS)** on CT **Metal Artifact Reduction (MAR)** algorithms. Following the task-based signal detection framework of **Vaishnav et al. (Medical Physics, 2020)**, this toolset provides:

1.  **Standardized Dataset Generation:** Physics-based fan-beam sinogram synthesis (v7.0.0).
2.  **Reference MAR Algorithms:** Implementations of iMAR, MBIR, and Dual-Energy Spectral MAR.
3.  **Objective Statistical Scoring:** Quantifying lesion detectability ($\Delta AUC$) using a **Channelized Hotelling Observer (CHO)** with normative internal noise ($\sigma=15$).
4.  **AI-Native Diagnostics:** Integrated **Model Context Protocol (MCP)** servers for automated data auditing and visualization.

---

## 🏛️ April 2026 Status: Metrology Baseline Locked

The framework has officially transitioned from parallel-beam research tiers to a **Normative Fan-Beam Geometry** (SID=570mm, SDD=1040mm). 

### 📊 ILS Reference Baseline ($N=40$)
The following values are established as the normative floor for ASTM WKXXXXX Rev 04:

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Baseline AUC_noMAR** | **0.8294** | **LOCKED** |
| **95% Confidence Interval** | [0.7612, 0.9025] | Verified |
| **Lesion Contrast** | 12.0 HU | Normative |
| **Integrity Check** | SHA-256 | Validated |

---

## 🚀 Performance Breakthrough: Numba Acceleration
The v7.0.0 engine features a JIT-compiled **Batch Backprojector** optimized for Apple Silicon (M3) and modern x86_64 architectures, providing a **24x speedup** over standard NumPy implementations.

* **Throughput:** ~61s per realization (256 slices).
* **Efficiency:** Full $N=40$ dataset generation in <30 minutes on standard workstations.

---

## 🤖 AI-Integrated Laboratory (MCP)
This repository includes **Model Context Protocol (MCP)** servers, allowing AI assistants to act as "Laboratory Hands" on your local workstation.

* **Data Inspector (`mcp_data_inspector.py`):** Automatically parses CHO JSON results and summarizes statistical significance.
* **Visualization (`mcp_visualization.py`):** Renders HDF5 sinogram slices and ROI comparisons directly in the AI chat interface.

---

## 🛠️ Validation & Audit Tools
To ensure inter-laboratory consistency, use the following in-memory physics auditors:
* **`plot_spectral_transparency.py`:** Visualizes the "Transparency Jump" between 60 keV and 140 keV.
* **`check_metal_overflow.py`:** Audits the normative **3000 HU metal-ROI hard-set** requirement.

---

## 📂 Deliverables & Repository Structure

| Path | Description |
| :--- | :--- |
| `generator_v7_0_0.py` | **Normative** fan-beam dataset generator (Rev 04). |
| `run_cho_analysis_v7_0.py` | **Normative** 2D CHO scoring tool (Rev 04). |
| `/docs_and_references` | ASTM, IEC, and FDA regulatory draft standards. |
| `/algorithms` | Reference MAR implementations (iMAR, MBIR, Spectral). |
| `/legacy` | Archived v6.0.0 parallel-beam research framework. |

---

## Technical Contact

**Christopher D. Cocchiaraley** Consumer Member, ASTM International Committee F04  
Executor of the Estate of Veronica M. Cocchiaraley