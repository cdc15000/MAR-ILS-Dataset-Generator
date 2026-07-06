# ASTM WKXXXXX MAR Type Test — Interlaboratory Study Participation Agreement

**Template version:** v0.1 (draft 2026-04-19)
**Governing procedure:** ASTM E691-23 (Standard Practice for Conducting an Interlaboratory Study to Determine the Precision of a Test Method)
**Study subject:** ASTM Work Item WKXXXXX Revision 05 — Standard Test Method for Evaluation of Metal Artifact Reduction Performance in Tomographic Imaging Systems Using a Channelized Hotelling Observer
**ILS Coordinator:** Christopher D. Cocchiaraley, WKXXXXX Sponsor, ASTM F04 / F04.15 (Consumer Member)

---

## 1. Parties

This Agreement is between:

**(a) The ILS Coordinator** — Christopher D. Cocchiaraley, acting as Sponsor of ASTM Work Item WKXXXXX, responsible for coordinating the pilot interlaboratory precision study leading to ASTM F04.15 ballot and ASTM Research Report publication.

**(b) The Participating Laboratory** — __________________________________________________________

**Principal Investigator:** _____________________________________________________________________

**Institution:** ________________________________________________________________________________

**Contact email:** ____________________________________________________________________________

---

## 2. Purpose

The Participating Laboratory agrees to execute the test method specified in WKXXXXX Revision 05 against the reference dataset provided by the Coordinator, under ASTM E691-23 protocol, and to return results for statistical aggregation by the Coordinator and the ASTM F04.15 committee.

---

## 3. Obligations of the Participating Laboratory

The Participating Laboratory agrees to:

**3.1 Protocol adherence.** Execute the test method strictly as specified in WKXXXXX Revision 05, without substitution, modification, or omission of any normative parameter (fan-beam geometry, N=40 LP + 40 LA realizations, internal noise σ=15, 2D CHO on slice 128 only, 121×121 ROI, 10 Laguerre-Gauss channels with a=7.5, AUC tolerance ±0.005). Any deviation, however minor, shall be reported to the Coordinator in writing with a proposed correction before results are returned.

**3.2 MAR algorithm disclosure.** Specify in the submission form which MAR algorithm(s) the lab applies to the dataset — either (i) the lab's own in-house MAR implementation, disclosed by name and version, or (ii) one or more of the reference implementations provided under `/algorithms/` in the WKXXXXX repository.

**3.3 Computational environment disclosure.** Report the computational environment used (OS, Python version, NumPy/SciPy/pydicom/numba versions, CPU or accelerator model) in the submission form per WKXXXXX §16.1(i).

**3.4 Deliverables.** Return the following to the Coordinator within the timeline of §7:
- `cho_results.json` from `run_cho_analysis_v7_0.py`
- Completed MAR ILS Results Submission Template with all §16 fields populated
- Signed attestation (§10 of this Agreement) that protocol was followed without modification

**3.5 Raw data retention.** Retain the reconstructed DICOM volumes (`mar_recon/`) for a minimum of 18 months from submission, for possible central CHO re-run or audit by the Coordinator or ASTM staff.

**3.6 Confidentiality of interim results.** Not publicly disclose ΔAUC values or any lab's results, including the Laboratory's own, prior to ASTM publication of the Research Report and the standard, except to the Coordinator and ASTM F04.15 committee members under ASTM confidentiality.

---

## 4. Obligations of the ILS Coordinator

The ILS Coordinator agrees to:

**4.1 Dataset provision.** Provide the SHA-256 verified reference dataset (80 realizations × 2 conditions, fan-beam sinograms + noMAR reconstructions, ~35 GB) via a verifiable distribution channel (Zenodo, figshare, or ASTM-hosted) with integrity manifest.

**4.2 Technical support.** Respond to protocol and implementation questions from the Participating Laboratory within 5 business days during the execution window.

**4.3 Central aggregation.** Aggregate all participating laboratories' results under ASTM E691 methodology, compute within-laboratory (S_r) and between-laboratory (S_R) precision statistics, and circulate the draft Research Report to all participating laboratories for review and approval prior to ASTM submission.

**4.4 Anonymization.** Identify laboratories only by alphabetical code ("Lab A," "Lab B," ...) in all public-facing artifacts (Research Report, final standard). Lab-identity → code mapping retained only by the Coordinator and ASTM F04.15 staff.

**4.5 ASTM registration.** Register the ILS with the ASTM Research Report Program and transition coordination responsibility to ASTM staff upon registration. At that point, ASTM procedures supersede the administrative provisions of this Agreement.

---

## 5. Data and Intellectual Property

**5.1 Dataset ownership.** The reference dataset is distributed under the Apache License 2.0 and remains under that license for all derivative use.

**5.2 Laboratory data ownership.** Each Participating Laboratory retains ownership of its reconstructions and results. Submission of results to the Coordinator grants a non-exclusive license to use those results for (i) aggregation under E691, (ii) publication in the ASTM Research Report, and (iii) publication of a co-authored peer-reviewed article describing the ILS.

**5.3 MAR algorithm IP.** Participating Laboratories that apply proprietary MAR algorithms retain all rights to those algorithms. The Agreement does not require disclosure of MAR algorithm internals beyond the name-and-version disclosure of §3.2.

**5.4 No commercial endorsement.** Participation does not constitute an endorsement by the Coordinator, ASTM, or the F04.15 committee of any MAR algorithm. The ILS is a precision study, not a comparative effectiveness study.

---

## 6. Attribution

**6.1 ASTM Research Report.** All Participating Laboratories completing the protocol receive co-authorship on the ASTM Research Report, with authors listed in alphabetical order of institution after the Coordinator.

**6.2 Peer-reviewed publication.** A companion peer-reviewed manuscript is planned for submission to *Medical Physics* or equivalent. Participating Laboratories completing the protocol are eligible for co-authorship under ICMJE criteria, with the Principal Investigator of each lab invited as co-author.

**6.3 ASTM F04.15 ballot package.** Participation is acknowledged in the WKXXXXX ballot package circulated to F04.15 members.

---

## 7. Timeline and Withdrawal

**7.1 Execution window.** 90 to 120 days from dataset receipt, with specific dates set by mutual agreement at the time of dataset distribution.

**7.2 Status check-ins.** The Coordinator will conduct informal status check-ins at the 30-day, 60-day, and 90-day milestones. Missed check-ins do not constitute withdrawal.

**7.3 Withdrawal.** The Participating Laboratory may withdraw at any time by written notice to the Coordinator. Withdrawal before results submission forfeits co-authorship rights under §6 but does not trigger any penalty or liability. Withdrawal after submission but before Research Report publication requires Coordinator approval and does not delete already-submitted data from the aggregate analysis unless the Laboratory specifically requests deletion in writing.

**7.4 Good-faith participation.** A minimum of three (3) laboratories completing the protocol is required for E691 precision estimates. If the aggregate falls below three after withdrawals, the Coordinator will notify remaining participants and either extend the window for new recruits or, with participant agreement, reschedule the study.

---

## 8. Governing Standards

This Agreement is executed under and governed by:

- **ASTM E691-23** — Standard Practice for Conducting an Interlaboratory Study to Determine the Precision of a Test Method
- **ASTM E177-20** — Standard Practice for Use of the Terms Precision and Bias in ASTM Test Methods
- **ICMJE authorship guidelines** — for peer-reviewed publication co-authorship

---

## 9. Limitation of Liability

The Coordinator provides the reference dataset and reference CHO implementation "as is," without warranty of any kind, consistent with the Apache License 2.0. The Participating Laboratory agrees that ILS participation is a scientific collaboration and not a fee-for-service engagement; no monetary consideration is exchanged in either direction. Each party bears its own costs.

---

## 10. Attestation and Signatures

By signing below, the Principal Investigator of the Participating Laboratory attests that (a) the Laboratory will execute the WKXXXXX Revision 05 protocol without modification, (b) results will be returned in good faith within the timeline of §7, and (c) the terms of this Agreement are understood and accepted.

| Party | Name | Title | Signature | Date |
|---|---|---|---|---|
| Participating Laboratory | _____________ | _____________ | _____________ | _____________ |
| ILS Coordinator | Christopher D. Cocchiaraley | WKXXXXX Sponsor, ASTM F04.15 | _____________ | _____________ |

---

## Notes to self (not to be sent — strip before distribution)

- **Template status:** draft v0.1. Review with ASTM F04 staff contact once registration is initiated; they may have a preferred template form.
- **Legal review:** not independently reviewed by counsel. For Apache-2.0 + E691 academic ILS with no monetary consideration, this is likely sufficient, but a pro-bono legal review through any participating university's OGC may be worth soliciting.
- **Institutional variation:** universities may require institutional signature in addition to PI signature (sponsored research office, technology transfer). Add a second signature block if so.
- **Foreign-lab considerations:** if EU/UK/Japan labs participate, verify GDPR/equivalent compliance language is not needed. The dataset is fully synthetic with no PHI, so most privacy regimes are a non-issue — but the Agreement should say so explicitly if asked.
- **FDA / government labs:** federal labs may require FAR/DFARS language, cooperative research agreement (CRADA), or similar. Flag Vaishnav / FDA CDRH group as a potential special-form case.
- **Commercial MAR vendor participation:** industry Tier C labs may want additional IP protection, non-disclosure, or clean-room attestation. Those conversations happen lab-by-lab, not via this template.
