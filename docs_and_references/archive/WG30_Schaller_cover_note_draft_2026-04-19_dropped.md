> **ARCHIVED / WITHDRAWN 2026-04-19.** This draft was written as the Phase 1 companion to the two-phase IEC proposal (informative NOTE at §203.6.7.101.1 pre-FDIS Approval). **Phase 1 was dropped the same day** as producing too little benefit for the effort involved. The operative regulatory path is the post-publication Corrigendum/Amendment documented in `../IEC_203_6_7_101_compliance_statement_proposal.md`. Retained here for historical reference only — do not send.

---

# WG 30 Status Update — ASTM MAR Type Test Progress

**Status:** Draft status-update email to IEC SC 62B WG 30 co-convenor (Andi Schaller)
**Date:** 2026-04-19
**Author:** Christopher D. Cocchiaraley
**Channel:** Ongoing WG 30 working correspondence (not a formal comment)
**Purpose:** Keep ASTM WKXXXXX visible on WG 30 agenda through FDIS Approval window; offer optional §203.6.7.101.1 NOTE text if Schaller finds it useful
**Companion document:** `IEC_203_6_7_101_compliance_statement_proposal.md` (two-phase proposal — Phase 2 binding Amendment is the operative goal; Phase 1 NOTE is optional)

---

## Draft Email

**To:** Andreas Schaller (Co-Convenor, IEC SC 62B WG 30) — direct, no cc
**Subject:** ASTM MAR Type Test — spring 2026 progress update

---

Hi Andi,

Quick spring update on the ASTM MAR Type Test — figured I'd bring you up to speed before the Approval window gets busy.

### Where the ASTM work stands

ASTM WKXXXXX has moved onto a stable footing since our last correspondence:

- **Metrology baseline locked** (2026-04-07): AUC_noMAR = 0.8294, CI [0.7612, 0.9025] on N=40 LP + 40 LA realizations, fan-beam geometry (SID=570 mm, SDD=1040 mm), internal noise σ=15, lesion contrast 12 HU. Sinogram-domain only — no post-FBP hard-sets.
- **Editorial pass complete** (2026-04-18, Rev 04): title and structure aligned to the F2119-24 pattern; §5 consolidated 7→5 clauses; §2 expanded (F2119, IEC 60601-2-44, DICOM PS3.3, Vaishnav 2020, Wunderlich & Noo, Kak & Slaney); new §1A.5 on scope and precedent; new §1A.6 on metal-material rationale; new informational Annex A2 reconciling Vaishnav's full-curve recommendation with the single-point ΔAUC normative metric.
- **DICOM 2026b compliance** (CP-2575): reference implementation is the first to emit the Metal Artifact Reduction Macro at `(0018,9390)` / `(0018,9391)`, complementing §203.6.7.101.3.
- **Engineering hardening**: 79 metrology guards in the test suite, all green against the locked constants; SHA-256–verified N=40 dataset (35 GB) available.
- **Sponsorship target confirmed**: ASTM F04.15 (same subcommittee that developed F2119-24 for MR passive-implant artifact evaluation), which gives the work a natural home and a procedural precedent.

The remaining pre-ballot milestone is the §17.1.6 pilot precision study across ≥3 independent laboratories (ASTM E691). Lab recruitment is underway; I expect S_r and S_R values by late 2026, with ballot targeted for 2027 and F-designation expected 2027–2028.

### Relationship to §203.6.7.101.1

Thanks again for putting the 3-year transition NOTE back in on 16 April — that gives the ASTM work real runway against Ed. 4 national implementation.

The layered intent remains unchanged: §203.6.7.101 establishes that a MAR method shall be *available*, *described*, and *recorded*; the ASTM Type Test will establish *how much it measurably affects lesion detectability*. These complement rather than overlap, following the IEC 62220-1 / FDA DQE arrangement for digital detectors.

The plan for binding the two is a post-publication Corrigendum or minor Amendment to §203.6.7.101.1 once ASTM FXXXX is actually published — Directives Part 2 pretty clearly rules out a compliance-statement reference to a not-yet-published standard, so the binding step naturally waits for the post-publication window. The transition period carries it.

### Optional: a signposting NOTE, if useful

Not a formal proposal — just floating it. If you think WG 30 would find value in anchoring the ASTM work in the Ed. 4 published text so national regulators and manufacturers see it during the transition, a single informative NOTE at §203.6.7.101.1 would do it without touching the compliance statement or adding any normative obligation:

> **NOTE:** *A standardized test method for quantitative evaluation of MAR performance is under development at ASTM International (Committee F04, Subcommittee F04.15) as Work Item WKXXXXX (expected F-designation 2027–2028). This test method operationalizes the model-observer framework of Vaishnav et al. (Medical Physics 47(8):3344–3355, 2020) and provides a reproducible, interlaboratory-comparable measurement of the signed change in area under the ROC curve (ΔAUC) between MAR-enabled and MAR-disabled reconstructions of a standardized digital CT dataset. Upon publication, this test method is expected to serve as the normative TYPE TEST demonstrating compliance with 203.6.7.101.1.*

If you'd rather let ASTM publish on its own timeline and carry the binding Amendment in one step post-publication, that works equally well from my side. I only flag the NOTE because the transition window has natural room for it. No need to decide now — whichever path you prefer.

### Reference materials

The reference implementation, the locked metrology baseline, the 79-test metrology guard suite, the Rev 04 draft ASTM standard, and the draft FDA acceptance-criteria framework are all public at:

> https://github.com/[your-github]/mar-ils

Happy to do a short walkthrough call if it'd be useful — otherwise I'll send the next update once the ≥3-lab precision data land.

Thanks for everything, as always.

Best,

Christopher D. Cocchiaraley
Executor, Estate of Veronica M. Cocchiaraley
Consumer Advocate, IEC USNC TAG SC 62B
Consumer Member, ASTM Int'l (Acct. No.: 2439061)
Consumer Member, DICOM WG-21
Mobile:  +1 (914) 980-9367 | +1 (347) 772-8160
cc15000@gmail.com

---

## Notes to self (not to be sent)

- **Tone.** Casual / collegial — established working relationship since November 2024. Open with "Hi Andi" (his preferred nickname). Not a formal comment; let the register match the existing correspondence.
- **Primary purpose is visibility, not the NOTE.** The email keeps the ASTM work on Schaller's agenda through the Approval window; the NOTE is optional upside. If he declines, we lose nothing — the Phase 2 post-publication Amendment is the operative path.
- **No TAG fallback mentioned.** No procedural threat, no §9.6(d) signalling. That framing fit a formal-comment channel; it doesn't fit a status update.
- **No MOP §4(i) framing.** Personal-capacity-expert-comment language is procedural freight this email doesn't need.
- **GitHub URL placeholder.** Replace `[your-github]` with the actual repository path before sending.
- **Send timing.** Reasonable any time in spring/early summer 2026 before the Approval ballot. No tight deadline — this is a progress update, not a procedural ask.
- **No cc list.** Direct to Schaller only, confirmed 2026-04-19. Do not cc Jaeckle or any other WG 30 correspondents on this thread. If a formal comment ever becomes necessary, that's a separate email with a different structure.
