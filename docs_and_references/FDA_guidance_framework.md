# FDA Guidance Framework — MAR Performance Evaluation

## Draft Framework for Acceptance Criteria

**Status:** Conceptual framework for future FDA guidance document
**Date:** 2026-04-05
**Author:** Christopher D. Cocchiaraley (Consumer Member, ASTM F04)
**References:** ASTM WKXXXXX Rev 05, IEC 60601-2-44 Ed. 4

---

### Purpose

This document outlines a proposed framework for FDA acceptance criteria when
evaluating Metal Artifact Reduction (MAR) performance in 510(k) submissions.
The framework builds on:

- **ASTM FXXXX (formerly WKXXXXX Rev 05)** — standardized TYPE TEST that measures ΔAUC
- **IEC 60601-2-44 Ed. 4 §203.6.7.101.1** — post-publication Corrigendum/Amendment binding the compliance statement to the ASTM TYPE TEST, to be submitted after ASTM FXXXX publishes

FDA's role in the layered approach is to establish the acceptance threshold:
what ΔAUC values are acceptable for 510(k) clearance?

---

### Proposed Acceptance Criteria

#### 1. Non-Degradation (Mandatory)

**Criterion:** ΔAUC ≥ −T_r

where T_r is the repeatability-derived non-degradation threshold, defined as:

    T_r = 2.8 × S_r

where S_r is the within-laboratory repeatability standard deviation from the
ASTM E691 interlaboratory study.

**Rationale:** A ΔAUC value within the measurement noise floor cannot be
distinguished from zero effect. The 2.8 × S_r factor corresponds to the 95%
repeatability limit per ASTM E177. Any ΔAUC above −T_r is consistent with
non-degradation at the 95% confidence level.

**Preliminary estimate:** Based on pilot data (to be established), S_r is
expected to be in the range 0.005–0.010 AUC units, giving T_r ≈ 0.014–0.028.
A conservative placeholder of **T_r = 0.02** is proposed pending ILS data.

**Non-degradation requirement:**

> The manufacturer shall demonstrate that ΔAUC ≥ −0.02 (placeholder pending
> ILS precision data). If ΔAUC < −0.02, the manufacturer shall provide
> clinical justification for the observed degradation in lesion detectability.

#### 2. Superiority Claim (Optional)

**Criterion:** The lower bound of the bootstrap 95% confidence interval for
ΔAUC shall exceed 0.0.

> If the manufacturer claims that the MAR function improves lesion
> detectability, the lower bound of the 95% bootstrap CI for ΔAUC must be
> positive (i.e., the CI excludes zero). Additionally, the paired Wilcoxon
> signed-rank test shall yield p < 0.05 (one-tailed).

This is a standard superiority claim framework consistent with FDA's
approach to performance claims in diagnostic imaging.

#### 3. Indeterminate Zone

**ΔAUC in [−T_r, 0] with CI spanning zero:**

The MAR function neither demonstrably improves nor degrades lesion
detectability. This is an acceptable outcome — the MAR function is neutral
with respect to the detection task. The manufacturer may still market the MAR
function for artifact reduction purposes, but shall not claim improved lesion
detectability.

---

### 510(k) Submission Requirements

The following shall be included in any 510(k) submission for a CT system with
MAR functionality:

| Item | Reference | Required |
|------|-----------|----------|
| System identification | §16.1(a) | Yes |
| MAR algorithm name and version | §16.1(b) | Yes |
| All imaging parameters (MAR on/off) | §16.1(c) | Yes |
| Dataset version and SHA-256 verification | §16.1(d) | Yes |
| CHO software version and validation AUC | §16.1(e) | Yes |
| Computational environment | §16.1(f) | Yes |
| Number of realizations per condition | §16.1(g) | Yes |
| Mean ΔAUC (3 decimal places, with sign) | §16.2(a) | Yes |
| Standard deviation of ΔAUC | §16.2(b) | Yes |
| Bootstrap 95% CI | §16.2(c) | Yes |
| AUC estimation bias | §16.2(d) | Yes |
| Individual AUC_MAR and AUC_noMAR | §16.2(e) | Yes |
| Internal noise σ used | §16.1(k) | Yes |
| Non-degradation assessment vs. T_r | This guidance | Yes |
| Superiority claim justification (if any) | This guidance | If claimed |

---

### Relationship to Predicate Devices

For 510(k) submissions claiming substantial equivalence to a predicate device
with MAR:

- If the predicate has existing ΔAUC data: the new device's ΔAUC should be
  within the reproducibility limit R = 2.8 × S_R of the predicate's ΔAUC.
- If the predicate has no ΔAUC data: the new device must independently
  satisfy the non-degradation criterion (ΔAUC ≥ −T_r).

---

### Precedent

This framework follows the established FDA pattern for performance standards
in medical imaging:

- **ASTM F2033** (metallic implant fatigue testing) — ASTM defines the test
  method; FDA guidance defines acceptance criteria for specific implant types
- **IEC 62220-1** (digital X-ray detector DQE) — IEC defines the measurement;
  FDA guidance establishes minimum DQE thresholds for 510(k) submissions
- **ACR CT accreditation** — ACR defines measurement procedures; CMS
  establishes accreditation requirements

In each case, the measurement standard and the acceptance criteria are
independently maintained, allowing either to be updated without disrupting the
other. The same principle applies here: ASTM WKXXXXX defines how to measure
ΔAUC; this guidance defines what values are acceptable.

---

### Timeline Dependencies

| Prerequisite | Status | Impact |
|-------------|--------|--------|
| ASTM WKXXXXX Rev 05 ballot | Pending | Must be balloted before FDA can reference |
| ASTM ILS precision study (≥3 labs) | Not started | Required to establish S_r, S_R, T_r |
| IEC 60601-2-44 Ed. 4 publication | ~Q1 2027 | FDA recognition of IEC standard |
| §203.6.7.101.1 binding Corrigendum/Amendment (post-ASTM-FXXXX-publication) | Pending ASTM publication | Optional — FDA can act independently |

FDA can issue draft guidance referencing the ASTM draft standard before the
ASTM ballot is complete, following the precedent of referencing draft consensus
standards in guidance documents. The guidance would be finalized after the
ASTM standard is published and the ILS precision data are available.
