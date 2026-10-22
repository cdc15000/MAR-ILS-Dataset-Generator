# IEC 60601-2-44 Ed. 4: Post-Publication Compliance-Statement Amendment for §203.6.7.101.1

## Amendment to be submitted after ASTM FXXXX publishes

**Status:** Draft Amendment proposal for IEC SC 62B WG 30 and USNC TAG SC 62B
**Date:** 2026-10-22 (first drafted 2026-04-19)
**Author:** Christopher D. Cocchiaraley (Consumer Advocate, IEC USNC TAG SC 62B; Consumer Advocate Member, IEC TC 62 / SC 62B WG 30; Consumer Member, ASTM F04 / F04.15)
**Target document:** IEC 60601-2-44 Ed. 4. Project leader Andreas Paul Schaller, SC 62B WG 30.
**Project status, from the IEC SC 62B dashboard on 2026-10-21:**

| Stage | Document | Date |
|---|---|---|
| CDV approved | 62B/1400/CDV | 2025-11-21 |
| FDIS circulated | 62B/1413/FDIS | 2026-07-17 |
| FDIS vote closed, approved by all voting P-members with no votes against | 62B/1413/FDIS, 62B/1413(F)/FDIS | 2026-08-28 |
| Approved for publication (BPUB) | 62B/1423/RVD | 2026-10-09 |
| Publication (PPUB), forecast | | 2026-11; current project plan 2026-11-20 |
| Stability date | | 2029 |

**Target stage:** Amendment to Ed. 4, submitted after ASTM FXXXX publishes (F designation targeted for late 2028)
**Gating prerequisite:** Publication of ASTM FXXXX, now Work Item WKXXXXX at ASTM F04.15

---

## Strategic Timing

The substantive goal is a one-sentence amendment that binds the compliance statement of §203.6.7.101.1 to the TYPE TEST in ASTM FXXXX. ASTM FXXXX does not yet exist as a published standard. ISO/IEC Directives Part 2 allows normative reference only to published documents, so WG 30 could not have accepted the reference at the FDIS stage, and none was proposed. The compliance statement published in Ed. 4, "Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS," stands until the amendment.

Two dates bound the window:

- **The transition period.** The 3-year transition NOTE, reinstated by the co-convenors on 2026-04-16, runs from publication in November 2026 to about late 2029. An amendment adopted within that period reaches national adoption before manufacturers must comply with Ed. 4.
- **The stability date, 2029.** SC 62B will review Ed. 4 at its stability date in any case. If ASTM FXXXX publishes on its late-2028 target, the amendment can be raised at that review, or ahead of it as a separate amendment project if SC 62B agrees.

The ASTM schedule sets the earliest submission date: pilot ILS March to April 2027, full ILS May to October 2027, Research Report January 2028, ballot March to June 2028, F designation late 2028.

---

## Proposed Amendment

The amendment changes two places in Ed. 4: the compliance statement of §203.6.7.101.1, and Clause 2.

### 1. Clause 203.6.7.101.1 Method(s)

**Normative requirement (unchanged):**

> The CT SCANNER shall have a metal artifact reduction (MAR) method available.

**Compliance statement as published in Ed. 4 (to be replaced):**

> *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.*

**Proposed compliance statement:**

> *Compliance is checked by the TYPE TEST specified in ASTM FXXXX-YY.*

`FXXXX-YY` is the dated designation of the published ASTM standard, for example F9999-28.

### 2. Clause 2 Normative references

Clause 2 of IEC 60601-1 applies, with the additions listed in Clause 2 of Ed. 4. The amendment adds:

> *Addition:*
>
> ASTM FXXXX-YY, *Standard Test Method for Evaluation of Metal Artifact Reduction Performance in Tomographic Imaging Systems Using a Channelized Hotelling Observer*

The title is the WKXXXXX working title. Replace it with the published title.

### 3. Dated reference

The reference is dated. Ed. 4 uses both forms in Clause 2: IEC 62985:2019, the SSDE calculation method, is dated, while IEC 61223-3-5 and ISO 12052 are undated. The compliance statement depends on specific numerical content of the ASTM method: the reference dataset, the observer parameters and the ±0.005 AUC tolerance. A later ASTM revision could change that content without review by WG 30. A dated reference fixes the edition used to show compliance. Adopting a later edition then takes a further amendment, which WG 30 controls.

The undated reference to ISO 12052 suits DICOM, a data dictionary that grows by additions such as the MAR attribute. A test method with numerical criteria is closer to the SSDE calculation method, which Ed. 4 cites dated.

If WG 30 prefers an undated reference, the Directives call for a statement that all future changes to the referenced document are acceptable for the purposes of this standard. That is a weaker position for a TYPE TEST with numerical criteria.

### 4. Acceptability of an ASTM document as a normative reference

ISO/IEC Directives Part 2 permits normative reference to documents of bodies other than ISO and IEC, provided the document is published and publicly available. ASTM FXXXX will meet these conditions on publication: it is sold publicly through ASTM, and ASTM is a recognized standards development organization. The proposal should state this when it is submitted, with the relevant clause of the then-current Directives cited. Check the clause numbers against the current edition of Directives Part 2 at submission.

The reference dataset and the reference CHO implementation are published openly: the dataset under CC BY 4.0, with a reserved Zenodo DOI, and the software under Apache 2.0. A laboratory can therefore run the TYPE TEST with nothing beyond the ASTM standard itself.

---

## Rationale

### 1. "Inspection of accompanying documents" verifies only documentation

The current compliance statement verifies that the manufacturer has described a MAR method. It does not verify that the method has any measurable effect on diagnostic task performance. A MAR method with no effect, or one that degrades lesion detectability, satisfies §203.6.7.101.1 as published.

### 2. The clause already incorporates an external standard by reference

§203.6.7.101.3 requires that, if MAR is used, it "shall be recorded in the corresponding standard DICOM attribute of the image header." Clause 2 of Ed. 4 lists ISO 12052, the ISO publication of DICOM, as a normative reference. The attribute in question was added to DICOM by CP-2575 in DICOM 2026b. §203.6.7.101 therefore already relies on a standard maintained outside IEC to define what manufacturers must do. The amendment applies the same approach to §203.6.7.101.1, with ASTM FXXXX defining the test. Confirm before submission how ISO 12052 ties to the current DICOM edition, since that link carries the CP-2575 attribute into Ed. 4.

The DICOM liaison report to SC 62B for the November 2026 plenary (62B/1420/INF) also draws attention to the new MAR attribute and encourages vendors to use it.

The compliance statements show the same pattern of departing from document inspection where a more specific check exists:

| Subclause | Compliance statement in Ed. 4 |
|---|---|
| §203.6.7.101.1 | *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.* |
| §203.6.7.101.2 | *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.* |
| **§203.6.7.101.3** | **Compliance is checked by inspection of representative DICOM image headers.** |

§203.6.7.101.3 names the specific item to be inspected. The amendment applies the same principle to §203.6.7.101.1 by naming the specific test.

The requirement itself has broad support. The FDIS was approved without a vote against, and the abstract of Ed. 4 lists MAR in imaging performance among the new requirements of the edition.

### 3. References to tests in other standards are usual in IEC 60601

Compliance statements in IEC 60601 particular standards routinely refer to tests defined in other standards instead of reproducing them. In the FDIS text of Ed. 4, §203.6.7.2.1 (Quality Assurance) and §203.6.7.2.2 (Performance information) both rely on the constancy and acceptance tests specified in IEC 61223-3-5. The amendment follows the same pattern with ASTM FXXXX as the source of the test. Confirm the subclause numbers against the published Ed. 4 before submission.

### 4. The scope is that of an amendment, not a new edition

The change is technical, because it changes how compliance is shown, so the vehicle is an amendment, not a corrigendum. IEC corrigenda are limited to correcting errors. The amendment is small:

- one compliance-statement sentence replaced
- one entry added to Clause 2
- no new normative requirement
- no new terms, definitions or symbols
- no new documentation obligations

It does not require a new edition of IEC 60601-2-44.

---

## Effect on the Regulatory Framework

**From publication of Ed. 4 to the amendment:** §203.6.7.101.1 keeps the published compliance statement. Manufacturers comply through the ACCOMPANYING DOCUMENTS.

**After the amendment:** the three-layer framework is complete.

| Layer | Document | Role |
|---|---|---|
| 1. MAR method and metadata | IEC 60601-2-44 Ed. 4 §203.6.7.101 (.1 method, .2 user information, .3 DICOM recording) | Requires MAR to be available, described and recorded |
| 2. Performance verification | ASTM FXXXX, cited by the amendment in §203.6.7.101.1 and Clause 2 | TYPE TEST that measures ΔAUC |
| 3. Acceptance | FDA guidance and its counterparts at other national regulators | ΔAUC acceptance thresholds for non-degradation and superiority |

IEC and ASTM do not set acceptance thresholds. The TYPE TEST reports the measured ΔAUC, and regulators decide what value is acceptable.

---

## Submission Vehicle

Two routes are available, and they can be used together.

**National body.** Submit through USNC TAG SC 62B as a national-body proposal for an amendment to IEC 60601-2-44 Ed. 4. Under USNC MOP §9.9.1, a national-body position needs a two-thirds TAG vote per §9.6(d). The proposal then goes to SC 62B for a decision on an amendment project assigned to WG 30, or for consideration at the 2029 stability-date review.

**ASTM liaison.** ASTM is a Category A liaison to IEC SC 62B. Its liaison representatives are listed in the liaison reports to SC 62B (62B/1420/INF). Through this liaison ASTM can present the published standard to SC 62B and support the amendment. Category A liaisons take part in the work of the committee, and may be able to propose work directly. Confirm what the current ISO/IEC Directives Part 1 allow before relying on that. A liaison report from ASTM at an SC 62B plenary is the natural way to announce FXXXX when it publishes.

Attach to the proposal:

- the published ASTM FXXXX
- the ASTM Research Report with the precision statement
- the reference dataset DOI and the software repository

---

## Withdrawn Proposals

This proposal supersedes:

**1. `IEC_203_6_7_102_draft.md` (2026-04-05, superseded 2026-04-14):** Proposed a new §203.6.7.102, "Quantitative evaluation of METAL ARTIFACT REDUCTION performance." Withdrawn because a new normative subclause at the PRVC stage was much less feasible procedurally than a compliance-statement edit, which achieves the same outcome.

**2. Single-phase amendment before approval (2026-04-18):** Proposed binding §203.6.7.101.1 to ASTM FXXXX before FDIS approval. Withdrawn because ASTM FXXXX was not yet published and ISO/IEC Directives Part 2 does not allow normative reference to an unpublished document.

**3. Two-phase proposal with an informative NOTE before approval (2026-04-19, superseded the same day):** Considered adding an informative NOTE to §203.6.7.101.1 during the FDIS window to point to the ASTM work. Withdrawn because the benefit did not justify a formal WG 30 or TAG submission and a possible two-thirds vote. Contact with WG 30 leadership continues informally.

---

## Revision History

- **2026-04-19:** First draft.
- **2026-10-21:**
  - Project dates updated from the IEC dashboard: FDIS approved 2026-07-17, approved for publication 2026-10-09, publication forecast 2026-11, stability date 2029.
  - Clause 2 addition added.
  - Dated reference specified, with the reasoning.
  - Acceptability of an ASTM reference addressed.
  - Vehicle changed from "corrigendum or amendment" to amendment.
  - Timing tied to the ASTM schedule and the 2029 stability date.
- **2026-10-22:**
  - FDIS vote result added.
  - §203.6.7.101.3 and ISO 12052 cited as the existing incorporation by reference of the DICOM MAR attribute.
  - IEC 61223-3-5 references corrected to §203.6.7.2.1 and §203.6.7.2.2; §203.7.6 does not cite it.
  - Dated and undated references in Ed. 4 Clause 2 compared.
  - ASTM Category A liaison added as a second submission route.
