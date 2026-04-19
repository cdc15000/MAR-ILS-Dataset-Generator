# IEC 60601-2-44 Ed. 4 — Post-Publication Compliance-Statement Amendment for §203.6.7.101.1

## Binding Amendment to be submitted after ASTM FXXXX publishes

**Status:** Draft Corrigendum/Amendment proposal for IEC SC 62B WG 30 and USNC TAG SC 62B
**Date:** 2026-04-19
**Author:** Christopher D. Cocchiaraley (Consumer Advocate, IEC USNC TAG SC 62B; Consumer Advocate Member, IEC TC 62 / SC 62B WG 30; Consumer Member, ASTM F04 / F04.15)
**Target document:** IEC 60601-2-44 Ed. 4 (approved CDV 62B/1400/CDV 2025-11-21; Approval 2026-08-21; Publication 2027-02-12)
**Target stage:** Corrigendum or minor Amendment to be submitted after ASTM FXXXX publishes (expected F-designation 2027–2028)
**Gating prerequisite:** Publication of ASTM FXXXX (currently WKXXXXX under development at F04.15)

---

## Strategic Timing

The substantively correct target is a single-sentence compliance-statement amendment binding §203.6.7.101.1 to *"the TYPE TEST specified in ASTM FXXXX."* However, ASTM FXXXX does not yet exist as a published standard — ASTM WKXXXXX is under development at F04.15 with an expected F-designation in 2027–2028. Under ISO/IEC Directives Part 2, a compliance-statement reference to a non-existent external standard is a drafting-rule violation, and WG 30 is unlikely to accept such a reference at FDIS Approval.

Rather than force the binding reference into a procedurally hostile window, this proposal is **held in abeyance until ASTM FXXXX publishes.** The 3-year transition NOTE (reinstated 2026-04-16 by the co-convenors) gives this strategy natural cover: ASTM FXXXX is expected to publish during the 2027–2030 transition window, well before national implementation of Ed. 4. An Amendment can be carried into national adoption without disrupting manufacturers already in transition.

No action is proposed against the current FDIS. The existing compliance statement (*"Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS"*) is retained as-published in Ed. 4. The binding amendment follows post-publication, on WG 30's normal work cycle, once the drafting-rule obstacle has cleared.

---

## Proposed Amendment

**Clause:** 203.6.7.101.1 Method(s)

**Normative requirement (unchanged):**

> The CT SCANNER shall have a metal artifact reduction (MAR) method available.

**Compliance statement — current Ed. 4 published text (to be replaced):**

> *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.*

**Compliance statement — proposed Amendment text:**

> *Compliance is checked by the TYPE TEST specified in ASTM FXXXX.*

---

## Rationale

### 1. "Inspection of accompanying documents" is procedurally weak

The current compliance statement verifies only that the manufacturer has written about a MAR method. It does not verify that the MAR method has any measurable effect on diagnostic task performance. A manufacturer satisfying this requirement today could ship a MAR method that has no effect, or that degrades lesion detectability, without triggering any non-compliance finding under §203.6.7.101.1.

### 2. Referencing an external TYPE TEST is already the clause's internal pattern

§203.6.7.101 itself establishes the precedent for a targeted, non-boilerplate compliance statement:

| Subclause | Compliance statement |
|---|---|
| §203.6.7.101.1 (current) | *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.* |
| §203.6.7.101.2 | *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.* |
| **§203.6.7.101.3** | **Compliance is checked by inspection of representative DICOM image headers.** |

§203.6.7.101.3 has already moved beyond boilerplate by specifying the exact artifact to be inspected (DICOM headers). The proposed Amendment applies the same principle to §203.6.7.101.1 by specifying the exact test procedure (ASTM FXXXX TYPE TEST).

### 3. Cross-reference to external standards is IEC 60601 house style

Compliance statements in IEC 60601 particular standards routinely reference tests defined in collateral or external standards without inlining the procedure. Representative examples already present in IEC 60601-2-44 Ed. 4:

- §203.6.7.2.1 (Quality Assurance) — references tests specified in IEC 61223-3-5.
- §203.7.6 (Test for half-value layer) — references measurement at the system isocentre per IEC 61223-3-5.

The proposed Amendment is syntactically identical to this pattern, substituting ASTM FXXXX for IEC 61223-3-5 as the host standard of the test procedure.

### 4. Procedurally viable as a Corrigendum or Amendment

By the time this proposal is submitted, ASTM FXXXX exists as a published standard with an F-designation. The ISO/IEC Directives Part 2 objection to referencing a not-yet-published standard disappears. The amendment becomes:

- a single-sentence compliance-statement edit;
- no new normative requirement;
- no new terminology, definitions, or symbols;
- no new documentation obligations;
- substitution of a named external-standard reference for a document-inspection boilerplate.

This is within the scope of a technical Corrigendum or minor Amendment under IEC procedures — it does not require a new edition of IEC 60601-2-44.

---

## Effect on the Regulatory Stack

**During the Ed. 4 transition period (2027–2030):** §203.6.7.101.1 retains the current compliance statement as published. Manufacturers continue to satisfy §203.6.7.101.1 via accompanying-documents inspection.

**After the Amendment (2028+):** The three-layer regulatory framework is established:

| Layer | Document | Role |
|---|---|---|
| 1 (MAR method + metadata) | **IEC 60601-2-44 Ed. 4 §203.6.7.101** (.1 method, .2 user info, .3 DICOM recording) | Requirement to *have*, *describe*, and *record* MAR |
| 2 (performance verification) | **ASTM FXXXX** (incorporated by Amendment to §203.6.7.101.1) | TYPE TEST that quantifies ΔAUC |
| 3 (acceptance) | FDA guidance (and analogous national regulatory bodies) | ΔAUC acceptance thresholds (non-degradation; superiority) |

The 3-year transition window means the Amendment is in force before national implementation of Ed. 4. Manufacturers preparing Ed. 4 conformity during 2027–2030 will have ASTM FXXXX available as the compliance vehicle.

---

## Submission Vehicle

Submission via USNC TAG SC 62B as a formal Corrigendum or Amendment proposal under IEC procedures, transmitted per USNC MOP §9.9.1 (requiring §9.6(d) 2/3 TAG vote for national-body position). No time pressure analogous to the FDIS window; the Amendment can be carried on WG 30's normal work cycle.

---

## Withdrawn Proposals

This proposal supersedes:

**1. `IEC_203_6_7_102_draft.md` (2026-04-05, superseded 2026-04-14):** Proposed a new §203.6.7.102 subclause titled "Quantitative evaluation of METAL ARTIFACT REDUCTION performance." Withdrawn because a new normative subclause at PRVC stage had materially lower procedural feasibility than a compliance-statement edit, and the same substantive outcome is achievable without a new subclause.

**2. Single-phase binding compliance-statement amendment (2026-04-18):** Proposed a one-step binding reference from §203.6.7.101.1 compliance statement to ASTM FXXXX, submitted pre-Approval. Withdrawn because ASTM FXXXX does not yet exist as a published standard, and ISO/IEC Directives Part 2 prohibits compliance-statement references to non-existent external standards. The current proposal preserves the substantive outcome by deferring submission until the drafting-rule obstacle clears.

**3. Two-phase proposal with pre-Approval informative NOTE (2026-04-19, superseded same day):** Briefly considered adding an informative NOTE at §203.6.7.101.1 during the FDIS Approval window to anchor the forthcoming ASTM work in the published Ed. 4 text. Withdrawn because the NOTE's payoff (conditional signposting to an audience already tracking F04.15) was judged insufficient to warrant the procedural effort (formal WG 30 or TAG submission, potential §9.6(d) 2/3 vote). The post-publication Amendment alone carries the substantive outcome; relationship-maintenance with WG 30 leadership is handled through ongoing informal correspondence rather than a formal Phase 1 vehicle.
