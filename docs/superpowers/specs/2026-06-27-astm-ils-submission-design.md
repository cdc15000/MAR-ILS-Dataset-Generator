# ASTM MAR Type Test: New Work Item & ILS Submission Package Design

**Date:** 2026-06-27
**Author:** Christopher D. Cocchiaraley
**Status:** Approved

---

## Objective

Prepare a complete submission package for registering a new ASTM work item through Form 01 under F04.15, Material Test Methods, for the MAR Type Test standard, with a concurrent Interlaboratory Study (ILS) under ASTM E691. The single-package approach leverages the Rev 06 draft standard, its pre-registered validation and the existing reference implementation to present a near-complete standard with a concrete plan to populate §17, Precision and Bias. _Update 2026-11-05: WK101378 was registered by the F04.15 chair instead, so the package now supports the task group and the ILS rather than a Form 01 submission._

## Strategic Context

### Regulatory Stack

| Layer | Document | Role | Status |
|---|---|---|---|
| 1 | IEC 60601-2-44 Ed. 4 Clause 203.6.7.101 | Requires MAR availability, user info, DICOM recording | FDIS approved 2026-07-17; approved for publication 2026-10-09; publication scheduled 2026-11-20 |
| 2 | ASTM FXXXX, this submission | TYPE TEST measuring ΔAUC | Work item WK101378 registered 2026-11-05; task group F04.15.24 |
| 3 | FDA guidance | Acceptance criteria: non-degradation threshold | Conceptual framework drafted |

### IEC Incorporation Path

The ASTM standard is intended for incorporation by normative reference into IEC 60601-2-44 Ed. 4 Clause 203.6.7.101.1 via a post-publication Amendment. The current compliance statement, "Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS", would be replaced with "Compliance is checked by the TYPE TEST specified in ASTM FXXXX." ISO/IEC Directives Part 2 prohibits referencing non-existent standards, so the Amendment is submitted after ASTM FXXXX publishes. The 3-year IEC transition window, ending about 2029, establishes the deadline.

The IEC linkage appears in the work item rationale only. The Amendment proposal itself is managed separately through USNC TAG SC 62B and is not attached to the ASTM submission.

### DICOM CP-2575 Precedent

DICOM CP-2575, which introduced the Metal Artifact Reduction Macro in DICOM 2026b, was approved and is already incorporated by reference into IEC 60601-2-44 Ed. 4 Clause 203.6.7.101.3. This establishes that Clause 203.6.7.101 already references external standards by compliance statement. The proposed Amendment to Clause 203.6.7.101.1 is syntactically identical, substituting ASTM FXXXX for DICOM PS3.3 as the referenced standard.

### Committee Selection: F04.15

Terry Woods of FDA CDRH and ASTM advised in November 2023 that F04 has no CT experts and no CT-related standards, and recommended IEC TC 62B WG30 instead. The sponsor subsequently joined WG30 as Consumer Advocate and DICOM WG-21 as Consumer Member. F04.15 remains the correct ASTM home for four reasons:

1. **The method evaluates a medical device feature, the MAR algorithm, not CT hardware.** F04 covers medical devices; E07/E17 cover industrial CT.
2. **The test is entirely computational.** The expertise needed to evaluate the standard is metrology and signal detection theory, not CT scanner operation.
3. **F2119 provides the direct procedural precedent.** MR artifact evaluation from passive implants → CT artifact reduction evaluation. Same subcommittee, complementary scope.
4. **The ILS recruits CT expertise into F04.15.** Imaging physics labs and FDA CDRH staff joining the task group bring the domain knowledge the subcommittee currently lacks.

The superseded cover memorandum acknowledged Terry Woods' feedback; at the 2026-11-04 meeting F04.15 answered it by agreeing to a dedicated CT task group, F04.15.24.

---

## Deliverables

| Document | Purpose | Format | Length |
|---|---|---|---|
| Document 1: Form 01 field content | Superseded 2026-11-05 (WK101378 registered by the F04.15 chair) | N/A | N/A |
| Document 2: Cover memorandum | Superseded 2026-11-05; never sent | N/A | N/A |
| Document 3: ILS Protocol | E691 study design, structured after D7778 Table 4, with the coordinator letter, return form and questionnaire as attachments | PDF/Word attachment | 13 pages |
| Attachment: Rev 06 draft standard | The standard itself | PDF/Word in ASTM manuscript format | ~60 pages |

---

## Document 1: ASTM Form 01 Field Content (superseded)

_Superseded 2026-11-05._ Form 01 was not submitted. After the presentation at F04.15.11 on 2026-11-04, the F04.15 chair registered WK101378 directly. The Technical Contact then entered the scope, keywords, rationale, existing standards and notification fields through the work item edit page; the posted work item page is now the record. The ASTM site does not store characters such as Δ and ∞, so the posted scope spells out "Delta AUC-infinity". The drafted wizard content is in git history.

## Document 2: Cover Memorandum (superseded)

_Superseded 2026-11-05._ The memorandum, addressed to the F04.15 chair, accompanied the planned Form 01 and asked F04.15 to register the work item, constitute a task group and authorize ILS recruitment. The first two were done on 2026-11-05 (WK101378; task group F04.15.24), and the memorandum was never sent. It is retired from the submission package; the drafted content is in git history.

## Document 3: ILS Protocol, E691 Study Design

**Title:** Interlaboratory Study Protocol, ASTM WK101378: Evaluation of Metal Artifact Reduction Performance

**Format:** 13-page PDF/Word attachment. The editable source is `private/d7778/protocol.md`, built with `private/d7778/build_docx.py` using the previous protocol as the style reference.

### Structure

Restructured 2026-10-17 to follow the ILS Protocol Plan in ASTM D7778-15(2026) Table 4. D7778 is a D02 guide (§1.1); the protocol uses its plan structure and example forms, adapted to a computational method, and not its D02-specific procedures: D6300, Cochran and Hawkins tests, D2PP software, the 30 degrees of freedom target, the 75 % manufacturer-laboratory rule and DOT/IATA shipping. E691-23 governs the statistics. D7778 is copyrighted and was obtained under an ASTM license, so the protocol paraphrases it and cites it by section; the PDF stays out of the repository and the package folder.

| Section | Content | D7778 source |
|---|---|---|
| Introduction | Subcommittee, ILS title, test method version, coordinator | Table 4 header, §6.8.4 |
| Protocol Plan Summary | Activity, plan and status for A1–A12 and B1–B9 | Table 4 |
| A1 Purpose | Measurand, reference baseline with provenance footnote | Table 4 A1 |
| A2 Funding | Self-funded by participants; sponsor provides materials; ASTM ILS Program support requested | Table 4 A2, §6.2.1 |
| A3 Staffing | Coordinator, F04.15 task group, ASTM ILS staff as statistician | Table 4 A3, §6.2.2–6.2.4 |
| A4 Participants | Tiers A–C; 10 to 12; Tier A capped at 3; withdrawal | Table 4 A4, §6.5 |
| A5 Equipment | Computational platform; integration paths A–C | Table 4 A5 |
| A6 Test materials | Dataset table; single-material justification; blinding; familiarization dataset | Table 4 A6, §6.3.1, §6.6.3–6.6.4 |
| A7 Tests | Rev 06, generator v9.0.0, CHO release; two conditions; replicates are three assigned independent datasets; four-decimal reporting | Table 4 A7, §6.8.4, §6.8.9 |
| A8 Special instructions | General instructions; Steps 1–7; output specification | Table 4 A8, Fig. X1.2 |
| A9 Reporting form | Two-part return form, Attachment 2 | Table 4 A9, Fig. X1.3 |
| A10 Statistical tools | E691-23 §15–19; digit retention; software verification | Table 4 A10, §10.3 |
| A11 Data acceptance | Committed runs, baseline check, completeness, cross-check, timeliness, h/k handling, 5 % cap | Table 4 A11, §10.7–10.8 |
| A12 Questionnaire | Attachment 3 | Table 4 A12, Fig. X1.4 |
| B1–B9 | Subcommittee notice, solicitation, materials, pilot, full run, review, statistics, presentation, Research Report in the RR:F04-2001 format | Table 4 B1–B9, §8, §10, §11 |
| C1–C3 | Timeline, confidentiality, amendments | |
| Attachments 1–3 | Coordinator letter, return form, questionnaire | Figs. X1.1, X1.3, X1.4 |

### Design Decisions, 2026-10-17

- **Pilot run:** separate, before the full-scale run. The first three laboratories to join run Condition 1 only, once on each of three assigned replicate datasets; the coordinator scores centrally with the key and returns results. Rev 06 §17.6 requires pilot data from at least 3 laboratories before first ballot showing, at each level, (a) the improving and degrading controls give ΔAUC∞ of the expected sign with 95 % intervals excluding zero, (b) within-laboratory SD of ΔAUC∞ at most 0.05 and (c) the null-signal check within 0.02 of the estimator's null value (0.533). The developer's validation meets all three. Pilot data is not used in the precision estimates. Adds two months to the schedule.
- **Blinding:** the distributed dataset is blinded, and the key is released after the manifest deadline so that laboratories score their own runs. See Blinded Distribution below.
- **Replicates (2026-11-03, review item 37):** a test result is ΔAUC∞ at one level on one replicate dataset; the standard ships three; ILS laboratories each receive three of a pool of nine, with balanced assignment; predictions s_r ≈ 0.004 to 0.008 and s_R ≈ s_r.
- **Registration and task group (2026-11-05):** presented at F04.15.11 on 2026-11-04 (WebEx, New Business); F04.15 had MRI rather than CT expertise, so the chairs agreed to a new CT task group. The F04.15 chair registered WK101378 on 2026-11-05 and asked staff to establish F04.15.24 with the sponsor as chair. Recruitment of CT experts goes through IEC SC 62B WG30, the USNC TAG for IEC SC 62B, DICOM WG-21 and the AAPM Computed Tomography Subcommittee.
- **Number of materials:** three, the signal levels 1.020, 1.025 and 1.030 at dose 1e6, as E691-23 §10.2.2 recommends. The range is narrow by design (estimator accuracy below, control ceiling above); the precision statement says so.
- **Funding and staffing:** self-funded by participants. The study is registered with the ASTM ILS Program under WK101378; ILS staff assign the random laboratory codes, issue the data report form, receive the results and act as statistician. The coordinator hosts the dataset, holds the blinding key and runs the rescoring check.
- **Precision basis:** Condition 1 only. Every laboratory runs the same improving control, so it is a common material under E691. In Condition 2 each laboratory measures a different algorithm, mostly research or reference methods outside Tier A, so its between-laboratory spread reflects differences between algorithms, not the method's precision. Condition 2 is reported descriptively by algorithm class, for applicability, discriminating range and manufacturer participation. Decided 2026-10-17.
- **Laboratory identification:** the Research Report names participating laboratories with contacts and addresses and codes every result, following RR:F04-2001, the F2052 study. Decided 2026-10-17.
- **Familiarization dataset:** a small labeled dataset with different noise seeds, for trial runs and tuning, so that no laboratory tunes on the ILS dataset, per D7778 Fig. X1.2 item 8 and E691-23 §9.3.

### Blinded Distribution

The master dataset keeps the labeled layout and stays with the coordinator. The distribution copy for Zenodo is derived from it:

- **Layout:** `sinograms/realization_001.h5` to `realization_080.h5` and `noMAR_recon/realization_001/` to `realization_080/`, a random permutation of the 40 LP and 40 LA realizations.
- **Removed from the files:** HDF5 `noise_params` attributes `place_lesion`, `lesion_z_extent`, `seed` and `jitter_deg`; any attributes on `line_integrals` that carry them; DICOM `SeriesDescription` and `ProtocolName`, which hold the condition label; `ContentDate` and `ContentTime`, set to one value; the per-realization jitters in `generator_provenance.json`. Compressed file sizes and leftover HDF5 bytes are not scrubbed.
- **Key:** `blinding_key.json`, mapping each code to its condition and original realization index, with a random salt. Its SHA-256 hash is published in the dataset record at distribution; the key is published after the manifest deadline.
- **Submission sequence:** laboratories submit a SHA-256 manifest of each run directory before key release, then score with the key and submit `cho_results.json` and the 80 scored slices per run. The laboratory's `cho_results.json` is the official result; the coordinator's rescoring is a cross-check.
- **Honesty about strength:** the generator and `BASE_SEED` are public and LP/LA pairs share a seed, so a laboratory could unblind deliberately. The protocol presents blinding as a safeguard against unconscious bias, and each laboratory commits in writing not to unblind.
- **Tooling, not yet built:** the v9 dataset writer (sinograms and DICOM), a blinding script that writes the distribution copy, its checksums, its lab instructions PDF and the key; `run_cho_analysis_v9_0.py --blinding-key`, which rebuilds the class, level and pool lists from the key; a manifest utility; and the coded layout in the control methods and the Python, MATLAB and C harnesses. Regression targets are the reference AUC∞ values recorded when the master dataset is generated. The release containing this tooling is the one the protocol cites.
- **Zenodo:** only the blinded copy is uploaded, and the record description is updated for the coded layout. Published Zenodo files cannot be changed, so the key is added after the manifest deadline as a new version of the record, with its own DOI; the v1 description carries the key hash. The Drive copy is the coordinator's master archive and is not shared.

### Timeline

| Milestone | Target Date |
|---|---|
| WK101378 registered; task group F04.15.24 formed | 2026-11-05 (done) |
| Protocol circulated to candidate laboratories | December 2026 |
| First task group meeting: integration paths, plan approval | January 2027, virtual |
| Recruitment complete: at least 10 committed laboratories | February 2027 |
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


## Attachment: Rev 06 Draft Standard

`private/rev06/ASTM_MAR_Standard_Rev06.md`, dated 2026-11-03, exported to Word beside Rev 05 in the package folder. Rev 05 is superseded.

---

## Pre-Submission Checklist

1. [x] ~~Resolve IP question~~ DONE 2026-07-14: standard text moved to the gitignored `private/` folder and removed from the public repo
2. [x] Export Rev 06 draft standard to Word (PDF pending) in ASTM manuscript format
3. [x] Write Document 2, the Cover Memorandum, as a standalone PDF: drafted 2026-07-27 and re-dated 2026-10-04 in `~/Desktop/ASTM Submission Package/`; superseded 2026-11-05 and retired from the package
4. [x] Write Document 3, the ILS Protocol, as a standalone PDF: drafted 2026-07-27, re-dated 2026-10-04, restructured after D7778 Table 4 on 2026-10-17
5. [x] Superseded 2026-11-05: WK101378 registered by the F04.15 chair; fields edited by the Technical Contact
6. [x] Superseded 2026-11-05: the chair registered WK101378 directly; Documents 2 and 3 and the Rev 06 draft go to the F04.15.24 collaboration area instead
7. [x] ILS Participation Agreement template aligned with the protocol 2026-10-17, and with Rev 06 and the ASTM ILS Program process 2026-11-03; set aside 2026-11-03 pending ASTM ILS staff advice on whether ASTM's own laboratory terms suffice, with the protocol relying on a written commitment instead (v0.2: no-unblinding clause, two-phase submission, algorithm class only, six-laboratory minimum, CC BY 4.0 dataset license); [ ] replace WK101378 with the assigned WK number once received
8. [ ] Build the blinding tooling and the familiarization dataset, and verify the regression target, before distribution in March 2027
9. [ ] Upload the blinded copy to Zenodo, publish the key hash in the record, and update the record description
