# ASTM MAR Type Test: New Work Item & ILS Submission Package Design

**Date:** 2026-06-27
**Author:** Christopher D. Cocchiaraley
**Status:** Approved

---

## Objective

Prepare a complete submission package for registering a new ASTM work item through Form 01 under F04.15, Material Test Methods, for the MAR Type Test standard, with a concurrent Interlaboratory Study (ILS) under ASTM E691. The single-package approach leverages the maturity of the Rev 05 draft standard, established reference baseline, and existing reference implementation to present a near-complete standard with a concrete plan to populate §17, Precision and Bias.

## Strategic Context

### Regulatory Stack

| Layer | Document | Role | Status |
|---|---|---|---|
| 1 | IEC 60601-2-44 Ed. 4 §203.6.7.101 | Requires MAR availability, user info, DICOM recording | FDIS approved 2026-07-17; publication scheduled 2026-10-23 |
| 2 | ASTM FXXXX, this submission | TYPE TEST measuring ΔAUC | Form 01 not yet submitted |
| 3 | FDA guidance | Acceptance criteria: non-degradation threshold | Conceptual framework drafted |

### IEC Incorporation Path

The ASTM standard is intended for incorporation by normative reference into IEC 60601-2-44 Ed. 4 §203.6.7.101.1 via a post-publication Amendment. The current compliance statement, "Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS", would be replaced with "Compliance is checked by the TYPE TEST specified in ASTM FXXXX." ISO/IEC Directives Part 2 prohibits referencing non-existent standards, so the Amendment is submitted after ASTM FXXXX publishes. The 3-year IEC transition window, ending about 2029, establishes the deadline.

The IEC linkage appears in the Form 01 rationale text only. The Amendment proposal itself is managed separately through USNC TAG SC 62B and is not attached to the ASTM submission.

### DICOM CP-2575 Precedent

DICOM CP-2575, which introduced the Metal Artifact Reduction Macro in DICOM 2026b, was approved and is already incorporated by reference into IEC 60601-2-44 Ed. 4 §203.6.7.101.3. This establishes that §203.6.7.101 already references external standards by compliance statement. The proposed Amendment to §203.6.7.101.1 is syntactically identical, substituting ASTM FXXXX for DICOM PS3.3 as the referenced standard.

### Committee Selection: F04.15

Terry Woods of FDA CDRH and ASTM advised in November 2023 that F04 has no CT experts and no CT-related standards, and recommended IEC TC 62B WG30 instead. The sponsor subsequently joined WG30 as Consumer Advocate and DICOM WG-21 as Consumer Member. F04.15 remains the correct ASTM home for four reasons:

1. **The method evaluates a medical device feature, the MAR algorithm, not CT hardware.** F04 covers medical devices; E07/E17 cover industrial CT.
2. **The test is entirely computational.** The expertise needed to evaluate the standard is metrology and signal detection theory, not CT scanner operation.
3. **F2119 provides the direct procedural precedent.** MR artifact evaluation from passive implants → CT artifact reduction evaluation. Same subcommittee, complementary scope.
4. **The ILS recruits CT expertise into F04.15.** Imaging physics labs and FDA CDRH staff joining the task group bring the domain knowledge the subcommittee currently lacks.

The cover memorandum acknowledges Terry Woods' feedback directly and explains the evolution.

---

## Deliverables

| Document | Purpose | Format | Length |
|---|---|---|---|
| Document 1: Form 01 field content | Online wizard entries | Text to paste into the ASTM portal | N/A |
| Document 2: Cover memorandum | Executive summary for F04.15 subcommittee chair | PDF/Word attachment | 2–3 pages |
| Document 3: ILS Protocol | E691 study design, structured after D7778 Table 4, with the coordinator letter, return form and questionnaire as attachments | PDF/Word attachment | 13 pages |
| Attachment: Rev 05 draft standard | The standard itself | PDF/Word in ASTM manuscript format | ~60 pages |

---

## Document 1: ASTM Form 01, Complete Wizard Field Content

### Page 1: Type

- **Type:** Proposed New ASTM Standard
- **Committee:** F04, Medical and Surgical Materials and Devices
- **Subcommittee:** F04.15, Material Test Methods

### Page 2: Copyright

- **Selection:** "I am submitting original material (i.e. it is not copyrighted, patented, pending patent, or published elsewhere)"
- **Check** the IP policy acknowledgment box

**IP note:** ~~The draft standard text (`ASTM_MAR_Standard.md`) is currently in a public GitHub repository.~~ **RESOLVED 2026-07-14:** Standard text removed from public repo and moved to the gitignored `private/ASTM_MAR_Standard.md`. The reference implementation code remains public under Apache 2.0; ASTM's copyright covers the standard document, not the software.

### Page 3: Target

| Field | Value | Rationale |
|---|---|---|
| Target date for Subcommittee ballot | **June 2028** | Pilot Mar–Apr 2027; full-scale ILS May–Oct 2027; Research Report Jan 2028; ballot Mar–Jun 2028; re-baselined 2026-10-17 after adding the pilot run |
| Authorized at a Subcommittee meeting? | **No** | Submission itself requests authorization |
| Authorization date | **10 / 2026** | Month of submission; adjust if submitted later |
| Emergency / regulatory requirement? | **Yes** | IEC 60601-2-44 Ed. 4 publishes ~Oct 2026 with a 3-year transition window; the ASTM standard must publish within that window for incorporation by normative reference |
| Patented or pending patent? | **No** | |
| Expected target date for approval | **18–24 months** | Select the longest available option in the dropdown; if only "3–6 months" is offered, select it and note the actual 18–24 month timeline in the rationale field. The dropdown constrains presentation, not commitment |
| Technical contact | **I will** | |

### Page 4: Data

**Type of Standard:** Test Method

**Title**, after the dropdown prepends "Test Method for":
> Quantitative Evaluation of Metal Artifact Reduction Performance in Tomographic Imaging Systems Using a Channelized Hotelling Observer

**Proposed Scope:**
> This test method specifies a quantitative procedure for evaluating the performance of Metal Artifact Reduction (MAR) methods implemented in tomographic imaging systems. The measurand is the scalar difference in area under the receiver operating characteristic curve (ΔAUC) derived from a signal-known-exactly, background-known-statistically (SKE/BKS) detection task using a fixed channelized Hotelling observer (CHO) implementation, predefined regions of interest, and specified statistical estimation procedures. The method uses a standardized synthetic digital dataset distributed as fan-beam line-integral sinograms and reference reconstructions, verified by SHA-256 checksums. The method applies to systems producing HU-calibrated reconstructed image data and is structured to support precision and bias evaluation in accordance with ASTM E691 and ISO 5725. It is intended for type testing and conformity assessment and does not establish performance acceptance criteria.

**Keywords:**
> metal artifact reduction; computed tomography; channelized Hotelling observer; model observer; area under the ROC curve; ΔAUC; lesion detectability; interlaboratory study; type test; digital phantom; DICOM; sinogram; fan-beam; signal detection; image quality; ISO 5725

**Rationale:**
> Metal artifact reduction (MAR) algorithms are increasingly incorporated into computed tomography (CT) systems, but no standardized test method exists to objectively quantify their effect on lesion detectability. MAR may improve or degrade diagnostic task performance depending on algorithm design and imaging parameters. IEC 60601-2-44 Ed. 4, scheduled for publication on 23 October 2026, establishes at §203.6.7.101 that CT systems shall have MAR methods available, described to users, and recorded in DICOM metadata as specified by DICOM CP-2575 in DICOM 2026b. However, the current compliance statement verifies documentation only, not algorithmic performance. This test method provides the missing quantitative measurement. It is intended for incorporation by normative reference into IEC 60601-2-44 Ed. 4 §203.6.7.101.1 via a post-publication Amendment, establishing a TYPE TEST that quantifies the signed change in lesion detectability (ΔAUC) attributable to MAR. Users include CT system manufacturers demonstrating MAR performance for regulatory submissions such as 510(k) and CE marking, regulatory authorities evaluating MAR claims, and research laboratories conducting interlaboratory comparison studies. An E691 interlaboratory study is proposed concurrently with standard development to establish the precision and bias statement in §17.

**Existing Standards:**
> ASTM F2119-24, Standard Test Method for Evaluation of MR Image Artifacts from Passive Implants. F2119 characterizes the physical extent of image artifacts produced by passive implants under standardized MR scanning conditions. The proposed standard is complementary, not duplicative: it quantifies the observer-based task-detectability impact of MAR, an algorithmic countermeasure applied within CT imaging systems. The two methods address non-overlapping axes: modality, MR versus CT, and object of measurement, physical artifact extent versus algorithmic task impact. No existing ASTM, ISO, IEC, or NEMA standard defines a task-based, model-observer test method for MAR performance evaluation. IEC 60601-2-44 Ed. 4 requires MAR availability but does not specify a test method for performance verification.

**Notify Other:**
> ASTM E07.11, Computed/Digital Radiography; IEC TC 62/SC 62B WG 30, Computed Tomography; USNC TAG SC 62B

### Page 5: Summary

Review all fields. Collaboration area: **Yes**, to provide a shared workspace for task group members during ILS coordination.

### Page 6: Confirm

Submit.

---

## Document 2: Cover Memorandum

**Title:** Cover Memorandum: New Work Item Proposal: Standard Test Method for Quantitative Evaluation of Metal Artifact Reduction Performance

**Addressee:** F04.15 Subcommittee Chair

**Format:** 2–3 page PDF/Word attachment

### Structure

**Section 1: Purpose**, about 1 paragraph

States that the memorandum accompanies a New Work Item Proposal for a standard test method evaluating MAR performance, and that an interlaboratory study under E691 is proposed concurrently. Identifies the sponsor: Christopher D. Cocchiaraley, Consumer Member, Account 2439061.

**Section 2: Clinical Motivation**, about 2 paragraphs

The problem: MAR algorithms may improve or degrade lesion detectability depending on implementation. No standardized test exists to quantify this effect. Current IEC 60601-2-44 Ed. 4 §203.6.7.101.1 requires only documentation inspection, so a manufacturer can satisfy compliance with a MAR method that has no measurable effect or that degrades diagnostic performance.

The consequence: patients scanned on systems without effective MAR, or with MAR that degrades detectability near implants, face diagnostic risk that is currently invisible to regulatory conformity assessment.

**Section 3: Technical Approach**, about 2 paragraphs

Summarizes the method without requiring the reader to open Rev 05: standardized digital dataset of SHA-256-verified fan-beam sinograms and reconstructions, channelized Hotelling observer scoring a binary lesion detection task, ΔAUC as the scalar measurand. Emphasizes that the test is entirely computational; no CT scanner hardware is required. References the framework of Vaishnav et al. (Medical Physics, 2020) as the scientific basis.

Notes the approved DICOM CP-2575 MAR metadata, now in DICOM 2026b, as complementary infrastructure. The recording mechanism for MAR status already exists; this test method provides the performance measurement.

**Section 4: Maturity of the Draft Standard**, about 1 bulleted paragraph

- Rev 05 draft standard attached, with 18 normative sections and 2 annexes
- Reference baseline established: AUC_noMAR = 0.8294, N=40, fan-beam geometry
- Reference dataset generated: 80 realizations, 35 GB, SHA-256 verified
- Reference CHO implementation: `run_cho_analysis_v7_0.py`, hardcoded normative parameters
- Reference MAR algorithm: LI-MAR v7, parameter-free linear interpolation
- 106 automated tests across 9 suites, all passing
- DICOM 2026b compliant output with the CP-2575 Metal Artifact Reduction Macro

**Section 5: Jurisdictional Fit: F04.15**, about 2 paragraphs

Acknowledges the November 2023 feedback from Terry Woods of FDA CDRH that F04 has no CT experts or CT-related standards. Explains why the method's evolution addresses this:

1. The method evaluates a medical device feature, the MAR algorithm, not CT hardware. Jurisdictionally it is a medical-device matter, not industrial CT.
2. The test is entirely computational; the expertise needed is metrology and signal detection theory, not CT scanner operation.
3. F2119 provides the direct procedural precedent: MR artifact evaluation from passive implants → CT artifact reduction evaluation. Same subcommittee, complementary scope.
4. The ILS will recruit imaging physics laboratories and FDA CDRH staff into the task group, bringing CT domain expertise into F04.15.

Notes that per Dr. Woods' recommendation, the sponsor joined the CT groups IEC SC 62B WG30 and DICOM WG-21. The ASTM standard and IEC incorporation are complementary efforts, not duplicative.

**Section 6: Regulatory Context and Timeline**, about 1 paragraph

Three-layer framework: IEC 60601-2-44 Ed. 4 requires MAR → ASTM FXXXX measures ΔAUC → FDA guidance establishes acceptance criteria. The IEC 3-year transition window, ending about 2029, establishes the deadline for ASTM publication. Target: pilot run March–April 2027, full-scale ILS May–October 2027, Research Report January 2028, ballot March–June 2028, F-designation late 2028.

**Section 7: Requested Actions**, about 3 bullets

1. Register the work item and assign a WK number
2. Constitute a task group, or assign an existing one
3. Authorize the sponsor to begin E691 ILS lab recruitment under the registered work item

---

## Document 3: ILS Protocol, E691 Study Design

**Title:** Interlaboratory Study Protocol, ASTM WKXXXXX: Evaluation of Metal Artifact Reduction Performance

**Format:** 13-page PDF/Word attachment. The editable source is `private/d7778/protocol.md`, built with `private/d7778/build_docx.py` using the previous protocol as the style reference.

### Structure

Restructured 2026-10-17 to follow the ILS Protocol Plan in ASTM D7778-15(2026) Table 4. D7778 is a D02 guide (§1.1); the protocol uses its plan structure and example forms, adapted to a computational method, and not its D02-specific procedures: D6300, Cochran and Hawkins tests, D2PP software, the 30 degrees of freedom target, the 75 % manufacturer-laboratory rule and DOT/IATA shipping. E691-23 governs the statistics. D7778 is copyrighted and was obtained under an ASTM license, so the protocol paraphrases it and cites it by section; the PDF stays out of the repository and the package folder.

| Section | Content | D7778 source |
|---|---|---|
| Introduction | Subcommittee, ILS title, test method version, coordinator | Table 4 header, §6.8.4 |
| Protocol Plan Summary | Activity, plan and status for A1–A12 and B1–B9 | Table 4 |
| A1 Purpose | Measurand, reference baseline with provenance footnote | Table 4 A1 |
| A2 Funding | Self-funded by participants; sponsor provides materials; ASTM ILS Program support requested | Table 4 A2, §6.2.1 |
| A3 Staffing | Coordinator, F04.15 task group, statistician to be appointed | Table 4 A3, §6.2.2–6.2.4 |
| A4 Participants | Tiers A–C; 8 or more, preferably 10; Tier A capped at 3; withdrawal | Table 4 A4, §6.5 |
| A5 Equipment | Computational platform; integration paths A–C | Table 4 A5 |
| A6 Test materials | Dataset table; single-material justification; blinding; familiarization dataset | Table 4 A6, §6.3.1, §6.6.3–6.6.4 |
| A7 Tests | Rev 05, generator v7.0.0, CHO release; two conditions; replicates; four-decimal reporting | Table 4 A7, §6.8.4, §6.8.9 |
| A8 Special instructions | General instructions; Steps 1–7; output specification | Table 4 A8, Fig. X1.2 |
| A9 Reporting form | Two-part return form, Attachment 2 | Table 4 A9, Fig. X1.3 |
| A10 Statistical tools | E691-23 §15–19; digit retention; software verification | Table 4 A10, §10.3 |
| A11 Data acceptance | Committed runs, baseline check, completeness, cross-check, timeliness, h/k handling, 5 % cap | Table 4 A11, §10.7–10.8 |
| A12 Questionnaire | Attachment 3 | Table 4 A12, Fig. X1.4 |
| B1–B9 | Subcommittee notice, solicitation, materials, pilot, full run, review, statistics, presentation, Research Report | Table 4 B1–B9, §8, §10, §11, §12.1 |
| C1–C3 | Timeline, confidentiality, amendments | |
| Attachments 1–3 | Coordinator letter, return form, questionnaire | Figs. X1.1, X1.3, X1.4 |

### Design Decisions, 2026-10-17

- **Pilot run:** separate, before the full-scale run. The first three signed laboratories run Condition 1 only, three replicates, on the blinded dataset; the coordinator scores centrally with the key and returns results. Rev 05 §17.10 requires pilot data from at least 3 laboratories before first ballot showing (a) a control ΔAUC of consistent sign distinguishable from zero, (b) within-laboratory SD of ΔAUC at most 0.05 and (c) ΔAUC bias, resubstitution minus hold-out, at most 0.02; §17.11 designates LI-MAR as the control. The archived LI-MAR record meets all three: ΔAUC −0.230 with p ≈ 3e−11, SD 0 for deterministic replicates, ΔAUC bias 0.004. Pilot data is not used in the precision estimates. Adds two months to the schedule.
- **Blinding:** the distributed dataset is blinded, and the key is released after the manifest deadline so that laboratories score their own runs. See Blinded Distribution below.
- **Number of materials:** one, the canonical 12 HU configuration. E691-23 §10.2.2 recommends at least three; the protocol justifies one normative test point under Rev 05 A2.3–A2.4 and limits the precision statement to it, per E691-23 §10.2.3 and D7778 §6.6.4.
- **Funding and staffing:** self-funded by participants; statistician to be appointed by the task group.
- **Familiarization dataset:** a small labeled dataset with different noise seeds, for trial runs and tuning, so that no laboratory tunes on the ILS dataset, per D7778 Fig. X1.2 item 8 and E691-23 §9.3.

### Blinded Distribution

The master dataset keeps the labeled layout and stays with the coordinator. The distribution copy for Zenodo is derived from it:

- **Layout:** `sinograms/realization_001.h5` to `realization_080.h5` and `noMAR_recon/realization_001/` to `realization_080/`, a random permutation of the 40 LP and 40 LA realizations.
- **Removed from the files:** HDF5 `noise_params` attributes `place_lesion`, `lesion_z_extent`, `seed` and `jitter_deg`; any attributes on `line_integrals` that carry them; DICOM `SeriesDescription` and `ProtocolName`, which hold the condition label; `ContentDate` and `ContentTime`, set to one value; the per-realization jitters in `generator_provenance.json`. Compressed file sizes and leftover HDF5 bytes are not scrubbed.
- **Key:** `blinding_key.json`, mapping each code to its condition and original realization index, with a random salt. Its SHA-256 hash is published in the dataset record at distribution; the key is published after the manifest deadline.
- **Submission sequence:** laboratories submit a SHA-256 manifest of each run directory before key release, then score with the key and submit `cho_results.json` and the 80 scored slices per run. The laboratory's `cho_results.json` is the official result; the coordinator's rescoring is a cross-check.
- **Honesty about strength:** the generator and `BASE_SEED` are public and LP/LA pairs share a seed, so a laboratory could unblind deliberately. The protocol presents blinding as a safeguard against unconscious bias, and the Participation Agreement adds a commitment not to unblind.
- **Tooling, not yet built:** a blinding script that writes the distribution copy, its checksums, its lab instructions PDF and the key; `run_cho_analysis_v7_0.py --blinding-key`, which builds the LP and LA folder lists in original realization order so the fixed-seed bootstrap is unchanged, with the no-key path untouched; a manifest utility; and the coded layout in LI-MAR and the Python, MATLAB and C harnesses, which use the labeled layout when `sinograms/LP` exists and the coded layout otherwise. Regression target with a blinded copy and its key: self-test 0.829375, CI [0.761219, 0.902516]; LI-MAR ΔAUC −0.230, CI [−0.2969, −0.1475]. The release containing this tooling is the one the protocol cites.
- **Zenodo:** only the blinded copy is uploaded, and the record description is updated for the coded layout. Published Zenodo files cannot be changed, so the key is added after the manifest deadline as a new version of the record, with its own DOI; the v1 description carries the key hash. The Drive copy is the coordinator's master archive and is not shared.

### Timeline

| Milestone | Target Date |
|---|---|
| Work item registered and WK number assigned | November 2026 |
| Protocol circulated to candidate laboratories | December 2026 |
| First task group meeting: integration paths, plan approval | January 2027, virtual |
| Recruitment complete: at least 8 signed agreements | February 2027 |
| Pilot run | March–April 2027 |
| Pilot review and any protocol revisions | April 2027 |
| Dataset distributed; execution window opens | May 2027 |
| Check-ins at 30, 60 and 90 days | June–August 2027 |
| Manifest deadline; execution window closes | September 2027 |
| Key released; scored results due two weeks later | September–October 2027 |
| Statistical analysis complete | November 2027 |
| Draft Research Report circulated to laboratories | December 2027 |
| Research Report finalized; §17 updated | January 2028 |
| Subcommittee ballot | March–June 2028 |

_Re-baselined 2026-10-04 for registration in November 2026, and again 2026-10-17 to add the pilot run._

---


## Attachment: Rev 05 Draft Standard

The existing `private/ASTM_MAR_Standard.md` Rev 05, dated 2026-05-29, exported to Word or PDF in ASTM manuscript format. No content changes are needed; the document is attached as-is.

---

## Pre-Submission Checklist

1. [x] ~~Resolve IP question~~ DONE 2026-07-14: standard text moved to the gitignored `private/` folder and removed from the public repo
2. [ ] Export Rev 05 draft standard to Word/PDF in ASTM manuscript format
3. [x] Write Document 2, the Cover Memorandum, as a standalone PDF: drafted 2026-07-27 and re-dated 2026-10-04 in `~/Desktop/ASTM Submission Package/`
4. [x] Write Document 3, the ILS Protocol, as a standalone PDF: drafted 2026-07-27, re-dated 2026-10-04, restructured after D7778 Table 4 on 2026-10-17
5. [ ] Complete the ASTM online wizard using the Document 1 field content
6. [ ] Attach Documents 2, 3, and the Rev 05 draft to the work item registration
7. [x] ILS Participation Agreement template aligned with the protocol 2026-10-17 (v0.2: no-unblinding clause, two-phase submission, algorithm class only, six-laboratory minimum, CC BY 4.0 dataset license); [ ] replace WKXXXXX with the assigned WK number once received
8. [ ] Build the blinding tooling and the familiarization dataset, and verify the regression target, before distribution in March 2027
9. [ ] Upload the blinded copy to Zenodo, publish the key hash in the record, and update the record description
