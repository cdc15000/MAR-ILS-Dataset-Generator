# IEC 60601-2-44 Ed. 4 — Proposed Compliance-Statement Amendment

## §203.6.7.101.1 — Compliance-Statement Upgrade

**Status:** Draft PRVC comment for IEC SC 62B WG 30
**Date:** 2026-04-18
**Author:** Christopher D. Cocchiaraley (Consumer Member, ASTM F04; IEC SC 62B WG 30)
**Target document:** IEC 60601-2-44 Ed. 4 CDV (62B/1400/CDV, approved 2025-11-21)
**Target stage:** PRVC (milestone 2026-02-13; Approval 2026-08-21; Publication 2027-02-12)
**References:** ASTM FXXXX (the ASTM MAR Type Test standard, formerly WKXXXXX)

---

## Proposed Change

**Clause:** 203.6.7.101.1 Method(s)

**Normative requirement (unchanged):**

> The CT SCANNER shall have a metal artifact reduction (MAR) method available.

**Compliance statement — current FDIS text (to be replaced):**

> *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.*

**Compliance statement — proposed:**

> *Compliance is checked by the TYPE TEST specified in ASTM FXXXX.*

---

## Rationale

### 1. "Inspection of accompanying documents" is procedurally weak

The current compliance statement verifies only that the manufacturer has written about a MAR method. It does not verify that the MAR method has any measurable effect on diagnostic task performance. A manufacturer satisfying this requirement today could ship a MAR method that has no effect, or that degrades lesion detectability, without triggering any non-compliance finding under §203.6.7.101.1.

### 2. Referencing an external TYPE TEST is already the clause's internal pattern

§203.6.7.101 itself establishes the precedent for a targeted, non-boilerplate compliance statement:

| Subclause | Compliance statement |
|---|---|
| §203.6.7.101.1 (current FDIS) | *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.* |
| §203.6.7.101.2 | *Compliance is checked by inspection of the ACCOMPANYING DOCUMENTS.* |
| **§203.6.7.101.3** | **Compliance is checked by inspection of representative DICOM image headers.** |

§203.6.7.101.3 has already moved beyond boilerplate by specifying the exact artifact to be inspected (DICOM headers). The proposed change applies the same principle to §203.6.7.101.1 by specifying the exact test procedure to be performed (ASTM FXXXX TYPE TEST).

### 3. Cross-reference to external standards is IEC 60601 house style

Compliance statements in IEC 60601 particular standards routinely reference tests defined in collateral or external standards without inlining the procedure. Representative examples already present in IEC 60601-2-44 Ed. 4:

- §203.6.7.2.1 (Quality Assurance) — references tests specified in IEC 61223-3-5.
- §203.7.6 (Test for half-value layer) — references measurement at the system isocentre per IEC 61223-3-5.

The proposed statement is syntactically identical to this pattern, substituting ASTM FXXXX for IEC 61223-3-5 as the host standard of the test procedure.

### 4. Procedurally feasible at PRVC

This is a **single-sentence editorial amendment to a compliance statement**, not a new normative requirement and not a new subclause. It can be dispositioned as a PRVC comment without retriggering CD or CDV, because it:

- does not alter the normative requirement (MAR method shall be available);
- does not introduce new terminology, definitions, or symbols;
- does not impose new documentation obligations;
- substitutes an external-standard reference for a document-inspection boilerplate.

If WG 30 declines the change at PRVC, the fallback vehicle is a post-publication amendment to Ed. 4 or incorporation into Ed. 5.

### 5. Dependency on ASTM FXXXX approval

This amendment is conditional on ASTM FXXXX being approved and assigned a designation prior to IEC Ed. 4 publication (target 2027-02-12). If ASTM FXXXX is not yet approved at that date, the amendment is not viable in Ed. 4 and should be carried to the first amendment or Ed. 5.

ASTM FXXXX is the former ASTM WKXXXXX work item. Under ASTM practice, the designation shifts from `WK` (work item) to `F` (balloted standard) upon subcommittee approval and assignment of a permanent designation by ASTM International. The "XXXX" placeholder is replaced with the assigned number at that time.

---

## Effect on the Regulatory Stack

With this amendment, the three-layer regulatory framework collapses to two layers plus an external acceptance layer:

| Layer | Document | Role |
|---|---|---|
| 1 (MAR method + metadata) | **IEC 60601-2-44 Ed. 4 §203.6.7.101** (.1 method, .2 user info, .3 DICOM recording) | Requirement to *have*, *describe*, and *record* MAR |
| 2 (performance verification) | **ASTM FXXXX** (incorporated by compliance-statement reference from §203.6.7.101.1) | TYPE TEST that quantifies ΔAUC |
| 3 (acceptance) | FDA guidance (national regulatory bodies) | ΔAUC acceptance thresholds |

The previously-proposed §203.6.7.102 (quantitative evaluation as a separate subclause) is withdrawn in favour of this compliance-statement framing, which achieves the same substantive outcome with a one-sentence change.

---

## Withdrawn Proposal

This proposal supersedes the previously-drafted `IEC_203_6_7_102_draft.md` (2026-04-05), which proposed adding a new §203.6.7.102 subclause titled "Quantitative evaluation of METAL ARTIFACT REDUCTION performance." That approach is withdrawn because:

- a new subclause at PRVC stage has materially lower procedural feasibility than a compliance-statement edit;
- the same technical outcome (binding ASTM FXXXX to §203.6.7.101) is achievable through the compliance statement alone;
- collapsing to a single MAR subclause simplifies the regulatory narrative and avoids a split between qualitative (.101) and quantitative (.102) requirements.
