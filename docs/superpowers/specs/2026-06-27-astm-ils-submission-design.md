# ASTM MAR Type Test — New Work Item & ILS Submission Package Design

**Date:** 2026-06-27
**Author:** Christopher D. Cocchiaraley
**Status:** Approved

---

## Objective

Prepare a complete submission package for registering a new ASTM work item (Form 01) under F04.15 (Material Test Methods) for the MAR Type Test standard, with a concurrent Interlaboratory Study (ILS) under ASTM E691. The single-package approach leverages the maturity of the Rev 05 draft standard, locked metrology baseline, and existing reference implementation to present a near-complete standard with a concrete plan to populate §17 (Precision and Bias).

## Strategic Context

### Regulatory Stack

| Layer | Document | Role | Status |
|---|---|---|---|
| 1 | IEC 60601-2-44 Ed. 4 §203.6.7.101 | Requires MAR availability, user info, DICOM recording | RFDIS; publication forecast Oct 2026 |
| 2 | ASTM FXXXX (this submission) | TYPE TEST measuring ΔAUC | Form 01 not yet submitted |
| 3 | FDA guidance | Acceptance criteria (non-degradation threshold) | Conceptual framework drafted |

### IEC Incorporation Path

The ASTM standard is intended for incorporation by normative reference into IEC 60601-2-44 Ed. 4 §203.6.7.101.1 via a post-publication Amendment. The current compliance statement ("Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS") would be replaced with "Compliance is checked by the TYPE TEST specified in ASTM FXXXX." ISO/IEC Directives Part 2 prohibits referencing non-existent standards, so the Amendment is submitted after ASTM FXXXX publishes. The 3-year IEC transition window (~2029) establishes the deadline.

The IEC linkage appears in the Form 01 rationale text only — the Amendment proposal itself is managed separately through USNC TAG SC 62B and is not attached to the ASTM submission.

### DICOM CP-2575 Precedent

DICOM 2026b CP-2575 (Metal Artifact Reduction Macro) was approved and is already incorporated by reference into IEC 60601-2-44 Ed. 4 §203.6.7.101.3. This establishes that §203.6.7.101 already references external standards by compliance statement. The proposed Amendment to §203.6.7.101.1 is syntactically identical — substituting ASTM FXXXX for DICOM PS3.3 as the referenced standard.

### Committee Selection — F04.15

Terry Woods (FDA CDRH, ASTM) advised in November 2023 that F04 has no CT experts and no CT-related standards, and recommended IEC TC 62B WG30 instead. The sponsor subsequently joined WG30 (Consumer Advocate) and DICOM WG-21 (Consumer Member). F04.15 remains the correct ASTM home for three reasons:

1. **The method evaluates a medical device feature (MAR algorithm), not CT hardware.** F04 covers medical devices; E07/E17 cover industrial CT.
2. **The test is entirely computational.** The expertise needed to evaluate the standard is metrology and signal detection theory, not CT scanner operation.
3. **F2119 provides the direct procedural precedent.** MR artifact evaluation from passive implants → CT artifact reduction evaluation. Same subcommittee, complementary scope.
4. **The ILS recruits CT expertise into F04.15.** Imaging physics labs and FDA CDRH staff joining the task group bring the domain knowledge the subcommittee currently lacks.

The cover memorandum acknowledges Terry Woods' feedback directly and explains the evolution.

---

## Deliverables

| Document | Purpose | Format | Length |
|---|---|---|---|
| Document 1: Form 01 field content | Online wizard entries | Text (paste into ASTM portal) | N/A |
| Document 2: Cover memorandum | Executive summary for F04.15 subcommittee chair | PDF/Word attachment | 2–3 pages |
| Document 3: ILS Protocol | E691 study design | PDF/Word attachment | 8–10 pages |
| Attachment: Rev 05 draft standard | The standard itself | PDF/Word (ASTM manuscript format) | ~60 pages |

---

## Document 1: ASTM Form 01 — Complete Wizard Field Content

### Page 1 — Type

- **Type:** Proposed New ASTM Standard
- **Committee:** F04 (Medical and Surgical Materials and Devices)
- **Subcommittee:** F04.15 (Material Test Methods)

### Page 2 — Copyright

- **Selection:** "I am submitting original material (i.e. it is not copyrighted, patented, pending patent, or published elsewhere)"
- **Check** the IP policy acknowledgment box

**IP note:** The draft standard text (`ASTM_MAR_Standard.md`) is currently in a public GitHub repository. ASTM's IP assignment requires that the document not be reproduced or circulated outside of ASTM Committee/Society activities. Before submission, either (a) make the standard text file private, or (b) confirm with ASTM staff (Kate Chalfin, cc'd on the November 2023 Woods correspondence) that prior open publication of a draft does not conflict with the IP assignment. The reference implementation code under Apache 2.0 is separate — ASTM's copyright covers the standard document, not the software.

### Page 3 — Target

| Field | Value | Rationale |
|---|---|---|
| Target date for Subcommittee ballot | **December 2027** | ILS needs ~12 months; task group review after ILS completion |
| Authorized at a Subcommittee meeting? | **No** | Submission itself requests authorization |
| Authorization date | **06 / 2026** | Current month |
| Emergency / regulatory requirement? | **Yes** | IEC 60601-2-44 Ed. 4 publishes ~Oct 2026 with a 3-year transition window; the ASTM standard must publish within that window for incorporation by normative reference |
| Patented or pending patent? | **No** | |
| Expected target date for approval | **12–18 months** | Select the longest available option in the dropdown; if only "3–6 months" is offered, select it and note the actual 12–18 month timeline in the rationale field — the dropdown constrains presentation, not commitment |
| Technical contact | **I will** | |

### Page 4 — Data

**Type of Standard:** Test Method

**Title** (dropdown prepends "Test Method for"):
> Quantitative Evaluation of Metal Artifact Reduction Performance in Tomographic Imaging Systems Using a Channelized Hotelling Observer

**Proposed Scope:**
> This test method specifies a quantitative procedure for evaluating the performance of Metal Artifact Reduction (MAR) methods implemented in tomographic imaging systems. The measurand is the scalar difference in area under the receiver operating characteristic curve (ΔAUC) derived from a signal-known-exactly, background-known-statistically (SKE/BKS) detection task using a fixed channelized Hotelling observer (CHO) implementation, predefined regions of interest, and specified statistical estimation procedures. The method uses a standardized synthetic digital dataset distributed as fan-beam line-integral sinograms and reference reconstructions, verified by SHA-256 checksums. The method applies to systems producing HU-calibrated reconstructed image data and is structured to support precision and bias evaluation in accordance with ASTM E691 and ISO 5725. It is intended for type testing and conformity assessment and does not establish performance acceptance criteria.

**Keywords:**
> metal artifact reduction; computed tomography; channelized Hotelling observer; model observer; area under the ROC curve; ΔAUC; lesion detectability; interlaboratory study; type test; digital phantom; DICOM; sinogram; fan-beam; signal detection; image quality; ISO 5725

**Rationale:**
> Metal artifact reduction (MAR) algorithms are increasingly incorporated into computed tomography (CT) systems, but no standardized test method exists to objectively quantify their effect on lesion detectability. MAR may improve or degrade diagnostic task performance depending on algorithm design and imaging parameters. IEC 60601-2-44 Ed. 4 (publication forecast Q4 2026) establishes at §203.6.7.101 that CT systems shall have MAR methods available, described to users, and recorded in DICOM metadata (per DICOM 2026b CP-2575, approved). However, the current compliance statement verifies documentation only — not algorithmic performance. This test method provides the missing quantitative measurement. It is intended for incorporation by normative reference into IEC 60601-2-44 Ed. 4 §203.6.7.101.1 via a post-publication Amendment, establishing a TYPE TEST that quantifies the signed change in lesion detectability (ΔAUC) attributable to MAR. Users include CT system manufacturers demonstrating MAR performance for regulatory submissions (510(k), CE marking), regulatory authorities evaluating MAR claims, and research laboratories conducting interlaboratory comparison studies. An E691 interlaboratory study is proposed concurrently with standard development to establish the precision and bias statement (§17).

**Existing Standards:**
> ASTM F2119-24 — Standard Test Method for Evaluation of MR Image Artifacts from Passive Implants. F2119 characterizes the physical extent of image artifacts produced by passive implants under standardized MR scanning conditions. The proposed standard is complementary, not duplicative: it quantifies the observer-based task-detectability impact of an algorithmic countermeasure (MAR) applied within CT imaging systems. The two methods address non-overlapping axes — modality (MR vs. CT) and object of measurement (physical artifact extent vs. algorithmic task impact). No existing ASTM, ISO, IEC, or NEMA standard defines a task-based, model-observer test method for MAR performance evaluation. IEC 60601-2-44 Ed. 4 requires MAR availability but does not specify a test method for performance verification.

**Notify Other:**
> ASTM E07.11 (Computed/Digital Radiography); IEC TC 62 / SC 62B WG 30 (Computed Tomography); USNC TAG SC 62B

### Page 5 — Summary

Review all fields. Collaboration area: **Yes** (provides shared workspace for task group members during ILS coordination).

### Page 6 — Confirm

Submit.

---

## Document 2: Cover Memorandum

**Title:** Cover Memorandum — New Work Item Proposal: Standard Test Method for Quantitative Evaluation of Metal Artifact Reduction Performance

**Addressee:** F04.15 Subcommittee Chair

**Format:** 2–3 page PDF/Word attachment

### Structure

**Section 1: Purpose** (~1 paragraph)

States that the memorandum accompanies a New Work Item Proposal for a standard test method evaluating MAR performance, and that an interlaboratory study under E691 is proposed concurrently. Identifies the sponsor (Christopher D. Cocchiaraley, Consumer Member, Account 2439061).

**Section 2: Clinical Motivation** (~2 paragraphs)

The problem: MAR algorithms may improve or degrade lesion detectability depending on implementation. No standardized test exists to quantify this effect. Current IEC 60601-2-44 Ed. 4 §203.6.7.101.1 requires only documentation inspection — a manufacturer can satisfy compliance with a MAR method that has no measurable effect or that degrades diagnostic performance.

The consequence: patients scanned on systems without effective MAR, or with MAR that degrades detectability near implants, face diagnostic risk that is currently invisible to regulatory conformity assessment.

**Section 3: Technical Approach** (~2 paragraphs)

Summarizes the method without requiring the reader to open Rev 05: standardized digital dataset (fan-beam sinograms + reconstructions, SHA-256 verified), channelized Hotelling observer scoring a binary lesion detection task, ΔAUC as the scalar measurand. Emphasizes that the test is entirely computational — no CT scanner hardware is required. References the Vaishnav et al. (Medical Physics, 2020) framework as the scientific basis.

Notes the DICOM 2026b CP-2575 MAR metadata standard (approved, in production) as complementary infrastructure — the recording mechanism for MAR status already exists; this test method provides the performance measurement.

**Section 4: Maturity of the Draft Standard** (~1 paragraph, bulleted)

- Rev 05 draft standard attached (18 normative sections + 2 annexes)
- Metrology baseline locked (AUC_noMAR = 0.8294, N=40, fan-beam geometry)
- Reference dataset generated: 80 realizations, 35 GB, SHA-256 verified
- Reference CHO implementation: `run_cho_analysis_v7_0.py`, hardcoded normative parameters
- Reference MAR algorithm: LI-MAR v7 (parameter-free linear interpolation)
- 79 metrology guard tests, all passing
- DICOM 2026b compliant output (CP-2575 Metal Artifact Reduction Macro)

**Section 5: Jurisdictional Fit — F04.15** (~2 paragraphs)

Acknowledges Terry Woods' (FDA CDRH) November 2023 feedback that F04 has no CT experts or CT-related standards. Explains why the method's evolution addresses this:

1. The method evaluates a medical device feature (MAR algorithm), not CT hardware — jurisdictionally medical-device, not industrial CT.
2. The test is entirely computational — expertise needed is metrology and signal detection theory, not CT scanner operation.
3. F2119 provides the direct procedural precedent: MR artifact evaluation from passive implants → CT artifact reduction evaluation. Same subcommittee, complementary scope.
4. The ILS will recruit imaging physics laboratories and FDA CDRH staff into the task group, bringing CT domain expertise into F04.15.

Notes that per Dr. Woods' recommendation, the sponsor joined IEC SC 62B WG30 (CT) and DICOM WG-21 (CT). The ASTM standard and IEC incorporation are complementary efforts, not duplicative.

**Section 6: Regulatory Context and Timeline** (~1 paragraph)

Three-layer framework: IEC 60601-2-44 Ed. 4 requires MAR → ASTM FXXXX measures ΔAUC → FDA guidance establishes acceptance criteria. The IEC 3-year transition window (~2029) establishes the deadline for ASTM publication. Target: ILS complete mid-2027, ballot late 2027, F-designation early 2028.

**Section 7: Requested Actions** (~3 bullets)

1. Register the work item and assign a WK number
2. Constitute a task group (or assign to an existing one)
3. Authorize the sponsor to begin E691 ILS lab recruitment under the registered work item

---

## Document 3: ILS Protocol (E691 Study Design)

**Title:** Interlaboratory Study Protocol — ASTM WKXXXXX: Evaluation of Metal Artifact Reduction Performance

**Format:** 8–10 page PDF/Word attachment

### Structure

**Section 1: Objective**

To determine the within-laboratory repeatability (S_r) and between-laboratory reproducibility (S_R) of the ΔAUC measurand defined in ASTM WKXXXXX, in accordance with ASTM E691-22 and ISO 5725-2.

**Section 2: Measurand**

Single measurand: ΔAUC = AUC_MAR − AUC_noMAR, computed by the reference CHO implementation on slice 128 of 40 LP + 40 LA paired realizations, with internal noise σ = 15. Reported to three decimal places.

**Section 3: Test Materials**

- Reference dataset: 80 realizations (40 LP, 40 LA), fan-beam sinograms (HDF5) + noMAR reconstructions (DICOM), ~35 GB, SHA-256 verified
- Distribution channel: Zenodo or figshare (DOI-assigned, persistent)
- Each lab receives an identical, checksummed copy — dataset variability is zero by design

**Section 4: Laboratory Requirements**

Minimum qualifications for participation:
- Computational platform capable of running the reference CHO implementation (Python 3.10+, 16 GB RAM)
- Access to at least one MAR algorithm (proprietary, published, or one of the reference implementations provided in the WKXXXXX repository)
- Principal investigator with expertise in CT reconstruction, medical image quality, or signal detection theory
- Commitment to execute the protocol without modification and return results within the execution window

**Section 5: Number of Laboratories**

- **Target:** 8 laboratories (provides robust E691 statistics)
- **Minimum:** 6 laboratories (E691 §6.2 minimum for meaningful precision estimates)
- **Laboratory tiers** (for recruitment purposes, not for stratified analysis):
  - Tier A: CT system manufacturers with proprietary MAR (target 2–3)
  - Tier B: Academic / government imaging physics labs (target 3–4, including FDA CDRH)
  - Tier C: Independent testing / CRO labs (target 1–2)

**Section 6: Test Conditions**

Each lab executes two conditions, producing two ΔAUC values:

| Condition | MAR Algorithm | Purpose |
|---|---|---|
| Condition 1 (mandatory) | Reference LI-MAR v7 (provided) | Reproducibility baseline — all labs run the identical algorithm; any between-lab variance is attributable to computational environment or CHO implementation differences |
| Condition 2 (mandatory) | Lab's own MAR algorithm | Performance measurement — the lab's proprietary or selected algorithm; this is the value of interest for the precision statement |

The two-condition design separates method reproducibility (Condition 1 should yield near-zero between-lab variance) from measurand precision (Condition 2 captures real-world variance).

**Section 6A: First Task Group Meeting — MAR Integration Approach**

The first task group meeting after work item registration shall include the following agenda item:

> **Agenda: Sinogram Ingestion and MAR Integration Path**
>
> The reference implementation provides lab harness templates (Python, MATLAB, C) in which the lab implements a single function (`apply_mar`) that receives sinogram data and geometry and returns a reconstructed HU volume. The harness handles all HDF5 I/O, DICOM writing, and CP-2575 MAR macro injection.
>
> However, commercial MAR implementations are typically embedded in scanner reconstruction pipelines (C/C++/CUDA) and may not be separable as standalone callable functions. Task group members are asked to advise on which integration path is feasible for their organization:
>
> - **Path A — Library extraction:** Isolate the MAR algorithm as a callable function that operates on the provided sinograms. The lab harness templates support this directly.
> - **Path B — Format bridge:** Convert HDF5 sinograms into the lab's proprietary sinogram format, process through the existing reconstruction pipeline with MAR enabled, export DICOM, and reformat output to match the submission spec.
> - **Path C — Offline reconstruction server:** Import sinograms into an offline reconstruction environment (e.g., vendor workstation software), reconstruct with MAR, export and reformat DICOM output.
>
> The sinogram ingestion specification (`sinogram_ingestion_spec.md`) and lab harness templates (`templates/`) are provided as starting points. The task group should determine whether these materials are sufficient or whether additional format support or interface modifications are needed.

This discussion informs whether the ILS protocol requires revision to accommodate vendor-specific integration constraints before the dataset is distributed.

**Section 7: Execution Protocol**

Per-lab procedure (step-by-step):
1. Download dataset, verify SHA-256 checksums against manifest
2. Condition 1: Apply reference LI-MAR v7 to sinograms → reconstruct → run `run_cho_analysis_v7_0.py` → record ΔAUC
3. Condition 2: Apply lab's own MAR to sinograms → reconstruct → run `run_cho_analysis_v7_0.py` → record ΔAUC
4. Complete the Results Submission Template (all §16 fields from the draft standard)
5. Retain reconstructed DICOM volumes for 18 months (audit provision)
6. Return results package to ILS Coordinator

**Section 8: Repeatability Design**

To estimate within-laboratory repeatability (S_r), each lab performs 3 independent replicate runs of Condition 1 (reference LI-MAR). Since the dataset and algorithm are deterministic, variance arises from floating-point non-determinism across runs (if any) and computational environment effects. If Condition 1 replicates show zero variance (expected for deterministic code on a single machine), S_r for Condition 1 is reported as < ε (below measurement resolution) and the repeatability estimate is derived from Condition 2 replicates where the lab runs its own MAR 3 times with any stochastic elements.

**Section 9: Statistical Analysis Plan**

Per E691 §15–17:
- Compute cell means (x̄) and cell standard deviations (s) for each lab × condition
- Compute within-laboratory repeatability standard deviation (S_r)
- Compute between-laboratory reproducibility standard deviation (S_R)
- Compute repeatability limit r = 2.8 × S_r
- Compute reproducibility limit R = 2.8 × S_R
- Apply Cochran's test and Grubbs' test for outlier detection (E691 §16)
- Compute Mandel's h and k statistics for consistency assessment
- Report all results in an ASTM Research Report

**Section 10: Timeline**

| Milestone | Target Date |
|---|---|
| Work item registered (WK number assigned) | July 2026 |
| ILS Protocol distributed to potential labs | August 2026 |
| Lab recruitment complete (≥6 signed agreements) | October 2026 |
| Dataset distributed to participating labs | November 2026 |
| Execution window opens | November 2026 |
| 30-day check-in | December 2026 |
| 60-day check-in | January 2027 |
| 90-day check-in | February 2027 |
| Execution window closes (results due) | March 2027 |
| Statistical analysis complete | April 2027 |
| Draft Research Report circulated to labs | May 2027 |
| Research Report finalized | June 2027 |
| Draft standard updated with §17 P&B data | June 2027 |
| Subcommittee ballot | September–December 2027 |

**Section 11: Reporting**

Each lab submits:
1. `cho_results.json` for each condition × replicate
2. Completed Results Submission Template (per draft standard §16)
3. Signed ILS Participation Agreement (existing template, updated to reference registered WK number)
4. Computational environment disclosure

The ILS Coordinator aggregates results, computes E691 statistics, and prepares the ASTM Research Report. Labs are identified by alphabetical code only in all published materials.

**Section 12: Governance**

- ILS Coordinator: Christopher D. Cocchiaraley (WKXXXXX Sponsor)
- Upon work item registration, coordination transitions to ASTM staff oversight
- Disputes regarding protocol adherence are resolved by the F04.15 task group chair
- The Research Report is submitted to ASTM's Research Report Program per E691 §19

---

## Attachment: Rev 05 Draft Standard

The existing `ASTM_MAR_Standard.md` (Rev 05, dated 2026-05-29), exported to Word or PDF in ASTM manuscript format. No content changes needed — the document is attached as-is.

---

## Pre-Submission Checklist

1. [ ] Resolve IP question: make `ASTM_MAR_Standard.md` private on GitHub or confirm with ASTM staff that prior open publication is compatible with IP assignment
2. [ ] Export Rev 05 draft standard to Word/PDF in ASTM manuscript format
3. [ ] Write Document 2 (Cover Memorandum) as a standalone PDF
4. [ ] Write Document 3 (ILS Protocol) as a standalone PDF
5. [ ] Complete ASTM online wizard (Document 1 field content)
6. [ ] Attach Documents 2, 3, and the Rev 05 draft to the work item registration
7. [ ] Update ILS Participation Agreement template to reference the assigned WK number once received
