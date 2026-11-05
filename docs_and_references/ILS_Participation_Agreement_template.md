# ASTM WK101378 MAR Type Test: Interlaboratory Study Participation Agreement

**Template version:** v0.3 (draft 2026-11-03)
**Governing procedure:** ASTM E691-23, Standard Practice for Conducting an Interlaboratory Study to Determine the Precision of a Test Method
**Study subject:** ASTM Work Item WK101378 Revision 06 (draft dated 2026-11-03), Standard Test Method for Evaluation of Metal Artifact Reduction (MAR) Performance in Tomographic Imaging Systems Using a Model Observer
**Study protocol:** ASTM WK101378 Interlaboratory Study Protocol, November 2026
**Study administration:** ASTM Interlaboratory Study (ILS) Program, ILS@astm.org
**ILS Coordinator:** Christopher D. Cocchiaraley, WK101378 Sponsor, ASTM F04 / F04.15 (Consumer Member)

---

## 1. Parties

This Agreement is between:

**(a) The ILS Coordinator:** Christopher D. Cocchiaraley, acting as Sponsor of ASTM Work Item WK101378, responsible for coordinating the interlaboratory precision study leading to ASTM F04.15 ballot and ASTM Research Report publication.

**(b) The Participating Laboratory:** __________________________________________________________

**Principal Investigator:** _____________________________________________________________________

**Institution:** ________________________________________________________________________________

**Contact email:** ____________________________________________________________________________

---

## 2. Purpose

The Participating Laboratory agrees to execute the test method specified in WK101378 Revision 06 against the reference dataset provided by the Coordinator, following the Study Protocol and ASTM E691-23. The Laboratory returns its results to the ASTM ILS Program, which performs the statistical analysis, and its run files to the Coordinator, who checks them. The results support the precision statement of the test method and the ASTM Research Report.

---

## 3. Obligations of the Participating Laboratory

The Participating Laboratory agrees to:

**3.1 Protocol adherence.** Execute the test method strictly as specified in WK101378 Revision 06 and the Study Protocol, without substitution, modification, or omission of any normative parameter, including the reference reconstruction of Annex A1.1, the 121 × 121 region of interest centered on pixel (256, 256) counted from zero, the ten dense difference-of-Gaussians channels, the antler-plot AUC∞ of Annex A1.6, and the ±0.005 equivalence tolerance on AUC∞(noMAR). Any deviation, however minor, shall be reported to the Coordinator in writing and recorded on the return form.

**3.2 Blinding and familiarization.** Not attempt to identify which coded realizations are lesion-present, or which form pairs, before the Coordinator releases the blinding key. Use the familiarization dataset, not the study dataset, for all trial runs and any tuning of MAR parameters.

**3.3 MAR algorithm disclosure.** Apply the provided improving control in Condition 1. State on the return form which MAR algorithm the Laboratory applies in Condition 2: either (i) the Laboratory's own MAR implementation, identified by algorithm class, such as "projection-domain interpolation" or "iterative reconstruction," or (ii) a published method or one of the reference implementations provided in the WK101378 repository. Disclosure of the product name or version is at the Laboratory's discretion.

**3.4 Computational environment disclosure.** Report the computational environment used (OS, Python version, NumPy/SciPy/pydicom/numba versions, CPU or accelerator model) in the questionnaire, per WK101378 §16.1(i).

**3.5 Deliverables.** Return the following within the timeline of §7, in two submissions:
- Before the key is released, to the Coordinator: a SHA-256 manifest of each run directory, and Part I of the return form
- After the key is released, to the Coordinator: for each run, `cho_results.json` from `run_cho_analysis_v9_0.py` and the 121 × 121 scored regions of every realization; the questionnaire; and the event log
- After the key is released, to the ASTM ILS Program at ILS@astm.org: the results of Part II of the return form, on the ASTM data report form issued for this study, which is the only form on which results are accepted
- Signed attestation (§10 of this Agreement) that the protocol was followed without modification

**3.6 Raw data retention.** Retain all run directories for a minimum of 18 months from submission, for possible re-analysis or audit by the Coordinator, the F04.15 task group or ASTM ILS staff.

**3.7 Confidentiality of interim results.** Not publicly disclose ΔAUC values or any laboratory's results, including the Laboratory's own, prior to ASTM publication of the Research Report and the standard, except to the Coordinator, ASTM ILS staff and the F04.15 task group.

---

## 4. Obligations of the ILS Coordinator

The ILS Coordinator agrees to:

**4.1 Dataset provision.** Provide the SHA-256 verified reference dataset (three of a pool of nine independent replicate datasets of 8,000 realizations each under coded identifiers, covering three signal levels, about 96 GB, plus a control dataset for pilot laboratories, as two-dimensional fan-beam sinograms in HDF5 and DICOM reference reconstructions) and a labeled familiarization dataset through Zenodo, with an integrity manifest. Publish the SHA-256 hash of the blinding key when the dataset is distributed, and the key itself after the manifest deadline.

**4.2 Technical support.** Respond to protocol and implementation questions from the Participating Laboratory within 5 business days during the execution window.

**4.3 Review and analysis.** Review each file submission on receipt and rescore the submitted regions with the key to confirm the Laboratory's `cho_results.json`. The ASTM ILS Program computes the repeatability (s_r) and reproducibility (s_R) standard deviations from the Condition 1 results under ASTM E691-23. The Coordinator summarizes the Condition 2 results by algorithm class and circulates the draft Research Report to all participating laboratories for review and approval prior to ASTM submission.

**4.4 Identification of laboratories.** List the Participating Laboratory by name, contact and address among the participants in the ASTM Research Report, as F04.15 Research Reports do, but identify every result only by the random laboratory code assigned by ASTM ILS staff. The standard itself identifies no laboratory. The mapping from laboratory to code is held only by ASTM ILS staff and the Coordinator.

**4.5 ASTM registration.** Register the study with the ASTM ILS Program once the work item is registered. ILS staff then assign the laboratory codes, issue the data report form, receive the results and perform the statistical analysis. Where ASTM procedures differ from the administrative provisions of this Agreement, ASTM procedures govern.

---

## 5. Data and Intellectual Property

**5.1 Dataset and software licenses.** The reference dataset is distributed under the Creative Commons Attribution 4.0 International license (CC BY 4.0). The reference software, including the CHO implementation and the harness templates, is distributed under the Apache License 2.0.

**5.2 Laboratory data ownership.** Each Participating Laboratory retains ownership of its reconstructions and results. Submission of results to the Coordinator and to ASTM grants a non-exclusive license to use those results for (i) aggregation under E691, (ii) publication in the ASTM Research Report, and (iii) publication of a co-authored peer-reviewed article describing the ILS.

**5.3 MAR algorithm IP.** Participating Laboratories that apply proprietary MAR algorithms retain all rights to those algorithms. The Agreement does not require disclosure of MAR algorithm internals beyond the algorithm class of §3.3.

**5.4 No commercial endorsement.** Participation does not constitute an endorsement by the Coordinator, ASTM, or the F04.15 committee of any MAR algorithm. The ILS is a precision study, not a comparative effectiveness study.

---

## 6. Attribution

**6.1 ASTM Research Report.** All Participating Laboratories completing the protocol are listed by name as participants in the ASTM Research Report (§4.4), in alphabetical order of institution.

**6.2 Peer-reviewed publication.** A companion peer-reviewed manuscript is planned for submission to *Medical Physics* or equivalent. Participating Laboratories completing the protocol are eligible for co-authorship under ICMJE criteria, with the Principal Investigator of each lab invited as co-author.

**6.3 ASTM F04.15 ballot package.** Participation is acknowledged in the WK101378 ballot package circulated to F04.15 members.

---

## 7. Timeline and Withdrawal

**7.1 Execution window.** As set out in the Study Protocol: about four months from dataset distribution to the manifest deadline, followed by two weeks after the key is released for scoring and the second submission. Specific dates are confirmed at the time of dataset distribution.

**7.2 Status check-ins.** The Coordinator will conduct informal status check-ins at the 30-day, 60-day, and 90-day milestones. Missed check-ins do not constitute withdrawal.

**7.3 Withdrawal.** The Participating Laboratory may withdraw at any time by written notice to the Coordinator. Withdrawal before results submission forfeits co-authorship rights under §6 but does not trigger any penalty or liability. Withdrawal after submission but before Research Report publication requires Coordinator approval and does not delete already-submitted data from the aggregate analysis unless the Laboratory specifically requests deletion in writing. The Research Report notes any withdrawal without identifying the laboratory.

**7.4 Minimum participation.** The precision statement requires acceptable results from at least six (6) laboratories, per E691-23 §9.1.2. If the number of participating laboratories falls below that after withdrawals, the Coordinator will notify remaining participants and either extend recruitment or, with participant agreement, reschedule the study.

**7.5 Pilot run.** The first three laboratories to sign this Agreement may be asked to take part in the pilot run described in the Study Protocol before the full-scale run. Pilot results are not used in the precision estimates.

---

## 8. Governing Standards

This Agreement is executed under and governed by:

- **ASTM E691-23:** Standard Practice for Conducting an Interlaboratory Study to Determine the Precision of a Test Method
- **ASTM E177-20:** Standard Practice for Use of the Terms Precision and Bias in ASTM Test Methods
- **ICMJE authorship guidelines:** for peer-reviewed publication co-authorship

---

## 9. Limitation of Liability

The Coordinator provides the reference dataset and reference CHO implementation "as is," without warranty of any kind, consistent with their licenses under §5.1. The Participating Laboratory agrees that ILS participation is a scientific collaboration and not a fee-for-service engagement; no monetary consideration is exchanged in either direction. Each party bears its own costs.

---

## 10. Attestation and Signatures

By signing below, the Principal Investigator of the Participating Laboratory attests that (a) the Laboratory will execute the WK101378 Revision 06 test method and the Study Protocol without modification, (b) the Laboratory will not attempt to unblind the study dataset before the key is released, (c) results will be returned in good faith within the timeline of §7, and (d) the terms of this Agreement are understood and accepted.

| Party | Name | Title | Signature | Date |
|---|---|---|---|---|
| Participating Laboratory | _____________ | _____________ | _____________ | _____________ |
| ILS Coordinator | Christopher D. Cocchiaraley | WK101378 Sponsor, ASTM F04.15 | _____________ | _____________ |

---

## Notes to self (not to be sent; strip before distribution)

- **Template status:** draft v0.3. Review with ASTM ILS staff (ILS@astm.org) at the initial call after ILS Program registration; they may have a preferred template form or their own laboratory terms.
- **Legal review:** not independently reviewed by counsel. For a CC BY 4.0 dataset, Apache-2.0 software and an E691 academic ILS with no monetary consideration, this is likely sufficient, but a pro-bono legal review through any participating university's OGC may be worth soliciting.
- **Institutional variation:** universities may require institutional signature in addition to PI signature (sponsored research office, technology transfer). Add a second signature block if so.
- **Foreign-lab considerations:** if EU/UK/Japan labs participate, verify GDPR/equivalent compliance language is not needed. The dataset is fully synthetic with no PHI, so most privacy regimes are a non-issue, but the Agreement should say so explicitly if asked.
- **FDA / government labs:** federal labs may require FAR/DFARS language, cooperative research agreement (CRADA), or similar. Flag Vaishnav / FDA CDRH group as a potential special-form case.
- **Commercial MAR vendor participation:** Tier A manufacturer labs may want additional IP protection, non-disclosure, or clean-room attestation. Those conversations happen lab-by-lab, not via this template.
- **Before sending:** replace WK101378 with the registered WK number, and confirm the dates against the Study Protocol.
