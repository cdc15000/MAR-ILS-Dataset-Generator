# Governance

This document describes how changes are made to MAR ILS Dataset Generation and Evaluation, the reference software for ASTM WK101378.

## Roles

| Role | Who | Responsibility |
| :--- | :--- | :--- |
| Technical contact | Christopher D. Cocchiaraley | Maintains the software; proposes and carries out changes |
| CT-MAR Test Method task group | ASTM F04.15.24 members | Reviews changes that affect the test method's results |
| ILS laboratories | Participating laboratories | Run the supplied software without modification; report problems |

## What counts as a change to the method

A change to the method is any change that alters the numbers the test produces. This includes:

- the phantom, the scan simulation, the noise model or the reference reconstruction;
- the observer: channels, scored region, training and test sizes, or the AUC∞ extrapolation;
- the contrast levels, the dose, or the thresholds of the two reference MAR methods;
- the content or format of the datasets distributed with the standard.

Documentation, tooling, diagnostics and new output formats that leave the results unchanged are not changes to the method.

## How a change to the method is made

1. **Proposal.** The change, the reason for it and the checks that will judge it are written down before any run, with the pass and fail rules fixed in advance.
2. **Review.** The proposal goes to the CT-MAR Test Method task group.
3. **Validation.** The pre-registered checks are run on independent datasets. A failed check is recorded as a failure; the rules are not adjusted after the fact.
4. **Release.** The change is released as a new version, with the results of the checks in the release notes. A change that alters results increments the major version; other changes increment the minor version. The version is part of each script's file name (for example, `generator_v9_0_0.py`).

## Current version

The current design is v9.0.0, following Vaishnav et al. (2020). Its parameters are listed in the README and in the [v9.0.0 release notes](https://github.com/cdc15000/MAR-ILS-DataGen-Eval/releases/tag/v9.0.0). Key values:

| Parameter | Value |
| :--- | :--- |
| Lump amplitude scale | 0.0588 |
| Dose | 1 × 10⁶ counts per detector bin per view |
| Contrast levels | 1.020, 1.025, 1.030 |
| Images | 1000 training and 1000 test per class and level |
| MAR 1 / MAR 2 thresholds | 1124 counts / 1.953 cm⁻¹ |

The earlier v7.0.0 design is superseded.

## Contributing

1. Open an issue describing the proposed change.
2. A change that leaves results unchanged: submit a pull request; the technical contact reviews and merges it.
3. A change to the method: follows the process above.

## Standard

This repository follows **ASTM WK101378 Rev 06** (draft). When the standard is published, this document and the other references will be updated to its designation.
