# IEC 60601-2-44 Ed. 4 — Two-Phase Proposal for §203.6.7.101.1

## Phase 1: Informative NOTE (current FDIS window) → Phase 2: Binding Amendment (post-ASTM-publication)

**Status:** Draft comment for IEC SC 62B WG 30 and USNC TAG SC 62B
**Date:** 2026-04-19
**Author:** Christopher D. Cocchiaraley (Consumer Advocate Member, USNC TAG SC 62B; Consumer Member, IEC TC 62 / SC 62B WG 30; Member, ASTM F04 / F04.15)
**Target document:** IEC 60601-2-44 Ed. 4 CDV (62B/1400/CDV, approved 2025-11-21)
**Target stage (Phase 1):** Pre-Approval ballot window (Approval 2026-08-21; Publication 2027-02-12)
**Target stage (Phase 2):** Corrigendum or Amendment, post-publication of ASTM FXXXX
**References:** ASTM WKXXXXX (MAR Type Test work item, ASTM F04/F04.15, expected F-designation 2027–2028)

---

## Strategic Framing: Why Two Phases

A single-sentence compliance-statement amendment binding §203.6.7.101.1 to *"the TYPE TEST specified in ASTM FXXXX"* is the substantively correct target. However, **ASTM FXXXX does not yet exist as a published standard.** ASTM WKXXXXX is under development at F04.15 with an expected F-designation timeline of 2027–2028. Under ISO/IEC Directives Part 2, a normative or compliance-statement reference to a non-existent external standard is a drafting-rule violation. WG 30 is unlikely to accept such a reference at FDIS Approval.

Rather than force the binding reference into a procedurally hostile window, this proposal is structured in **two phases**:

| Phase | Window | Vehicle | Effect |
|---|---|---|---|
| **Phase 1** | Pre-Approval (by 2026-08-21) | Informative NOTE added to §203.6.7.101.1 | Anchors forthcoming ASTM work in Ed. 4 published text; does not bind compliance |
| **Phase 2** | After ASTM FXXXX publishes | Formal Corrigendum or Amendment to Ed. 4 | Replaces compliance-statement boilerplate with binding reference to ASTM FXXXX |

Phase 1 is procedurally cheap: an informative NOTE does not collide with ISO/IEC Directives Part 2, does not retrigger CD or CDV, and does not impose new normative obligations. Phase 2 is procedurally clean: by the time it is submitted, ASTM FXXXX exists as a published standard with an F-designation, and the drafting-rule objection evaporates.

The 3-year transition NOTE (reinstated 2026-04-16 by the convener) gives this two-phase strategy natural cover: ASTM FXXXX is expected to publish during the 2027–2030 transition window, well before national implementation of Ed. 4. A Phase 2 Amendment can be carried into national adoption without disrupting manufacturers already in transition.

---

## Phase 1 — Informative NOTE (submit now)

### Proposed Insertion

**Clause:** 203.6.7.101.1 Method(s)

**Normative requirement (unchanged):**

> The CT SCANNER shall have a metal artifact reduction (MAR) method available.

**Compliance statement (unchanged):**

> *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.*

**NOTE (proposed new informative text):**

> *NOTE: A standardized test method for quantitative evaluation of MAR performance is under development at ASTM International (Committee F04, Subcommittee F04.15) as Work Item WKXXXXX (expected F-designation 2027–2028). This test method operationalizes the model-observer framework of Vaishnav et al. (Medical Physics 47(8):3344–3355, 2020) and provides a reproducible, interlaboratory-comparable measurement of the signed change in area under the ROC curve (ΔAUC) between MAR-enabled and MAR-disabled reconstructions of a standardized digital CT dataset. Upon publication, this test method is expected to serve as the normative TYPE TEST demonstrating compliance with 203.6.7.101.1. National regulatory authorities and manufacturers are encouraged to consider this forthcoming standard in their conformity-assessment frameworks.*

### Rationale for Phase 1

**1. Anchors the ASTM work in Ed. 4 published text.** Every manufacturer and national regulator reading IEC 60601-2-44 Ed. 4 will see the NOTE. Without it, the ASTM work is known only to the subset of stakeholders who track F04.15 activity directly.

**2. Respects ISO/IEC Directives Part 2.** An informative NOTE does not require the referenced work to exist as a published standard. NOTES signaling forthcoming related work are a recognized drafting pattern.

**3. Does not retrigger CDV or CD.** A NOTE addition is editorial, not normative. It can be dispositioned as an editorial comment without opening a new substantive review cycle.

**4. Respects the 3-year transition NOTE.** The Phase 1 NOTE and the convener's 2026-04-16 transition NOTE are complementary: the transition period gives ASTM time to publish; the NOTE tells manufacturers and regulators what is coming.

**5. Creates forward-path optionality.** Phase 2 (binding Amendment) is easier to argue when Ed. 4 already names the forthcoming work in its own text.

---

## Phase 2 — Binding Compliance-Statement Amendment (submit post-ASTM-publication)

### Proposed Amendment

**Clause:** 203.6.7.101.1 Method(s)

**Normative requirement (unchanged):**

> The CT SCANNER shall have a metal artifact reduction (MAR) method available.

**Compliance statement — current Ed. 4 published text (to be replaced):**

> *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.*

**Compliance statement — proposed Amendment text:**

> *Compliance is checked by the TYPE TEST specified in ASTM FXXXX.*

### Rationale for Phase 2

#### 1. "Inspection of accompanying documents" is procedurally weak

The current compliance statement verifies only that the manufacturer has written about a MAR method. It does not verify that the MAR method has any measurable effect on diagnostic task performance. A manufacturer satisfying this requirement today could ship a MAR method that has no effect, or that degrades lesion detectability, without triggering any non-compliance finding under §203.6.7.101.1.

#### 2. Referencing an external TYPE TEST is already the clause's internal pattern

§203.6.7.101 itself establishes the precedent for a targeted, non-boilerplate compliance statement:

| Subclause | Compliance statement |
|---|---|
| §203.6.7.101.1 (current) | *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.* |
| §203.6.7.101.2 | *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.* |
| **§203.6.7.101.3** | **Compliance is checked by inspection of representative DICOM image headers.** |

§203.6.7.101.3 has already moved beyond boilerplate by specifying the exact artifact to be inspected (DICOM headers). Phase 2 applies the same principle to §203.6.7.101.1 by specifying the exact test procedure (ASTM FXXXX TYPE TEST).

#### 3. Cross-reference to external standards is IEC 60601 house style

Compliance statements in IEC 60601 particular standards routinely reference tests defined in collateral or external standards without inlining the procedure. Representative examples already present in IEC 60601-2-44 Ed. 4:

- §203.6.7.2.1 (Quality Assurance) — references tests specified in IEC 61223-3-5.
- §203.7.6 (Test for half-value layer) — references measurement at the system isocentre per IEC 61223-3-5.

The proposed Phase 2 statement is syntactically identical to this pattern, substituting ASTM FXXXX for IEC 61223-3-5 as the host standard of the test procedure.

#### 4. Procedurally viable as a Corrigendum or Amendment

By the time Phase 2 is submitted, ASTM FXXXX exists as a published standard with an F-designation. The ISO/IEC Directives Part 2 objection that blocked Phase 1 disappears. The amendment becomes:

- a single-sentence compliance-statement edit;
- no new normative requirement;
- no new terminology, definitions, or symbols;
- no new documentation obligations;
- substitution of a named external-standard reference for a document-inspection boilerplate.

This is within the scope of a technical Corrigendum or minor Amendment under IEC procedures — it does not require a new edition of IEC 60601-2-44.

---

## Effect on the Regulatory Stack

**During Phase 1 (2027–2028):** §203.6.7.101.1 retains the current compliance statement; the NOTE signals forthcoming ASTM work. Manufacturers continue to satisfy §203.6.7.101.1 via accompanying-documents inspection, but are aware that a quantitative Type Test is being finalized.

**After Phase 2 (2028 onward):** The three-layer regulatory framework is established:

| Layer | Document | Role |
|---|---|---|
| 1 (MAR method + metadata) | **IEC 60601-2-44 Ed. 4 §203.6.7.101** (.1 method, .2 user info, .3 DICOM recording) | Requirement to *have*, *describe*, and *record* MAR |
| 2 (performance verification) | **ASTM FXXXX** (incorporated by Phase 2 Amendment to §203.6.7.101.1) | TYPE TEST that quantifies ΔAUC |
| 3 (acceptance) | FDA guidance (and analogous national regulatory bodies) | ΔAUC acceptance thresholds (non-degradation; superiority) |

The 3-year transition window means Phase 2 Amendment is in force before national implementation of Ed. 4. Manufacturers preparing Ed. 4 conformity during 2027–2030 will have ASTM FXXXX available as the compliance vehicle.

---

## Submission Vehicles

### Phase 1 — two complementary channels

1. **WG 30 informal channel (fastest):** email to convener Andreas Schaller with the NOTE text, personal-capacity expert comment per IEC MOP §4(i). Seeks direct adoption into FDIS editorial text before Approval 2026-08-21.
2. **USNC TAG SC 62B formal channel (backup):** TAG comment transmitted via TA/DTA/Secretary per USNC MOP §9.9.1, requiring 2/3 TAG vote per §9.6(d). US Consumer Advocate seat (SC 62B) supports submission authority.

Both channels may be pursued in parallel. If Schaller accepts the NOTE editorially via the informal channel, the TAG comment is unnecessary; if he declines, the TAG comment formalizes the US position.

### Phase 2 — standard Corrigendum/Amendment procedure

Submitted once ASTM FXXXX is published (expected 2027–2028). Submission via USNC TAG SC 62B as a formal Amendment proposal under IEC procedures. No time pressure analogous to the FDIS window; the Amendment can be carried on WG 30's normal work cycle.

---

## Withdrawn Proposals

This two-phase proposal supersedes:

**1. `IEC_203_6_7_102_draft.md` (2026-04-05, superseded 2026-04-14):** Proposed a new §203.6.7.102 subclause titled "Quantitative evaluation of METAL ARTIFACT REDUCTION performance." Withdrawn because a new normative subclause at PRVC stage had materially lower procedural feasibility than a compliance-statement edit, and the same substantive outcome is achievable without a new subclause.

**2. Single-phase binding compliance-statement amendment (2026-04-18, superseded 2026-04-19):** Proposed a one-step binding reference from §203.6.7.101.1 compliance statement to ASTM FXXXX. Withdrawn because ASTM FXXXX does not yet exist as a published standard, and ISO/IEC Directives Part 2 prohibits compliance-statement references to non-existent external standards. The two-phase approach preserves the substantive binding outcome while respecting IEC drafting conventions.

The two-phase proposal retains the substantive goal of the single-phase version — binding compliance with §203.6.7.101.1 to the quantitative ASTM Type Test — but partitions the work across the windows in which each part is procedurally feasible.
