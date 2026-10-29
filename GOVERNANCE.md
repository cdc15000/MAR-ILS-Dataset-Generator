# Governance

This document describes the change-control process for the MAR ILS Dataset Generator and Evaluation Framework.

## Roles

| Role | Who | Authority |
| :--- | :--- | :--- |
| WKXXXXX sponsor | Christopher D. Cocchiaraley | Controls normative branch; authors revisions |
| ASTM Task Group (F04.15) | Committee members | Approves locked-constant changes; conducts ballot |
| ILS participants | Enrolled laboratories | Submit reconstructed DICOMs; do not modify reference code |

## What constitutes a breaking change

Any modification that would alter the bit-exact reproducibility of the locked metrology baseline is a **breaking change**. This includes, but is not limited to:

- Changing any value in the **ASTM Metrology Baseline** table in CLAUDE.md (AUC, I_0, DC offset, contrast, noise parameters, geometry).
- Modifying forward-projection, FBP reconstruction, or CHO observer numerics in a way that changes output values.
- Changing the number of realizations, ROI coordinates, channel parameters, or regularisation constants.
- Altering the DICOM encoding in a way that changes pixel values read by the CHO script.

Non-breaking changes (documentation, tooling, diagnostics, additional output formats) do not require Task Group approval.

## Version freeze protocol

1. **Lock** — When the metrology baseline is validated (self-test AUC within tolerance, SHA-256 checksums verified), the baseline constants are frozen. The locked values and date are recorded in CLAUDE.md.
2. **Regeneration gate** — Any proposed change to locked constants requires full dataset regeneration (N=40, all realizations) and re-validation of the baseline AUC to within the stated tolerance (currently +/-0.005).
3. **Task Group approval** — Changes to locked constants must be presented to ASTM F04.15 and approved before merging to the normative branch.
4. **Versioning** — Breaking changes increment the major version. Non-breaking enhancements increment the minor version. The version is embedded in script filenames (e.g., `generator_v7_0_0.py`).

## Current baseline

The current design is v9.0.0, following Vaishnav et al. (2020). Its fixed parameters are listed in the [v9.0.0 release notes](https://github.com/cdc15000/MAR-ILS-Dataset-Generator/releases/tag/v9.0.0) and in the README. Key values:

| Parameter | Value |
| :--- | :--- |
| Lump amplitude scale | 0.0588 |
| Dose | 1 × 10⁶ counts per detector bin per view |
| Contrast levels | 1.020, 1.025, 1.030 |
| Images | 1000 training + 1000 test per class and level |
| MAR 1 / MAR 2 thresholds | 1124 counts / 1.953 cm⁻¹ |

The v7.0.0 baseline (AUC_noMAR = 0.8294, N = 40, internal noise 15) is superseded.

## Contribution process

1. Open an issue or discussion describing the proposed change.
2. For non-breaking changes: submit a pull request; the sponsor reviews and merges.
3. For breaking changes: the sponsor brings the proposal to the Task Group. If approved, the sponsor regenerates the dataset, re-validates the baseline, updates the locked-constant table, and merges.

## Standard alignment

This repository tracks **ASTM WKXXXXX Rev 06** (draft, October 2026). When the standard is published as ASTM FXXXX, this governance document and all references will be updated to reflect the published designation.
