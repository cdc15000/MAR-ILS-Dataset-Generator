````markdown
# MAR ILS Dataset Generator & Evaluation Framework

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Standard: ASTM WKXXXXX](https://img.shields.io/badge/Standard-ASTM%20WKXXXXX-orange)
![Standard: IEC 60601-2-44 Ed. 4](https://img.shields.io/badge/Standard-IEC%2060601--2--44%20Ed.%204-green)

## Overview

This is the **reference evaluation framework** for conducting standardized **Interlaboratory
Studies (ILS)** on CT **Metal Artifact Reduction (MAR)** algorithms. Following the task-based
signal detection framework of **Vaishnav et al. (Medical Physics, 2020)**, this toolset
provides:

1. **Standardized Dataset Generation:** Analytic phantoms across three anatomical tiers.
2. **Reference MAR Algorithms:** Implementations of iMAR, MBIR, and Dual-Energy Spectral MAR.
3. **Objective Statistical Scoring:** Quantifying lesion detectability ($\Delta AUC$) using a
   **Channelized Hotelling Observer (CHO)**.
4. **AI-Native Diagnostics:** Integrated **Model Context Protocol (MCP)** servers for automated
   data auditing and visualization.

---

## 🚀 March 2026 Update: Spectral Superiority

Recent ILS benchmarking has established Dual-Energy Spectral CT as the definitive path to
**Superiority** in high-blockage scenarios.

- **The Software Floor:** At normative "Stress Test" blockage (~40%), single-energy algorithms
  (iMAR, MBIR) often fail to achieve superiority due to total signal loss in the metal shadow.
- **The Spectral Breakthrough:** By exploiting the 6× transmission jump at 140 keV
  (6.1% → 36.8%), Spectral MAR achieved the first formally **SUPERIOR** result in this
  framework ($\Delta AUC = +0.049$).

---

## 📊 ILS Performance Matrix (40% Blockage)

| Tier | Phantom | Metal | 140 keV Trans. | Result |
|:-----|:--------|:------|:---------------|:-------|
| **T1_AB** | Adult Body | Cobalt-Chrome | 5.3% | INDETERMINATE |
| **T2_SB** | Pediatric | Stainless Steel | 36.8% | **SUPERIOR** |
| **T3_HEAD** | Neuro/Head | Titanium | 47.4% | INFERIOR (Baseline Paradox) |

---

## 🤖 AI-Integrated Laboratory (MCP)

This repository includes **Model Context Protocol (MCP)** servers, allowing AI assistants to act
as "Laboratory Hands" on your local workstation.

- **Data Inspector (`mcp_data_inspector.py`):** Automatically parses CHO JSON results and
  summarizes statistical significance.
- **Visualization (`mcp_visualization.py`):** Renders HDF5 sinogram slices and ROI comparisons
  directly in the AI chat interface.

### Setup for AI-Native Auditing

Update your `.claude/mcp.json` (or equivalent) to point to your local virtual environment:

```json
{
  "mcpServers": {
    "data-inspector": {
      "command": "/your/path/mar-ils/bin/python3",
      "args": ["/your/path/mcp_data_inspector.py"]
    }
  }
}
```

---

## 🛠️ Validation & Audit Tools

To ensure inter-laboratory consistency, use the following in-memory physics auditors:

- **`plot_spectral_transparency.py`:** Visualizes the "Transparency Jump" between 60 keV and
  140 keV.
- **`check_metal_overflow.py`:** Audits the normative 3000 HU metal-ROI hard-set requirement
  across realizations.

---

## 📂 Deliverables

| Path | Description |
|------|-------------|
| `/algorithms` | Reference implementations (iMAR, MBIR, Spectral) |
| `/results_archive` | Historical ILS data and `spectral_traceability.json` hardware requirements |
| `TDP_Comparison_Report.pdf` | Automated Technical Dossier of tested algorithms |
| `checksums_sha256.txt` | Manifest for all DICOM files |

---

## Technical Contact

**Christopher D. Cocchiaraley**
Consumer Member, ASTM International Committee F04
Executor of the Estate of Veronica M. Cocchiaraley
````