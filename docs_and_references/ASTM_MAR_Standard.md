# ASTM WKXXXXX — Revision 04

**Designation:** WKXXXXX
**Work Item Number:** WKXXXXX
**Date:** 2026-04-05
**Supersedes:** Revision 03 (03/14/2026)

> **Change summary (Rev 03 → Rev 04):** Acquisition geometry changed from parallel-beam (360 angles over 180°) to fan-beam (SID=570 mm, SDD=1040 mm, equi-angular curved detector, 720 angles over 360° full rotation). FBP reconstruction changed from parallel-beam rotation-sum to fan-beam cosine-weighted distance-weighted backprojection. CHO implementation equivalence tolerance relaxed from ±0.001 to ±0.005 AUC (§6.4 and §11.2 updated for consistency). Screening mode (20 realizations) added for pilot evaluation; 40 remains the minimum for formal reporting. Acceptance criteria cross-reference added (§5.5) to IEC 60601-2-44 Ed. 4 §203.6.7.101.1 (via proposed compliance-statement amendment binding this TYPE TEST) and FDA guidance. Baseline AUC_noMAR established at 0.8294 under fan-beam geometry (N=40, σ=15, 2026-04-07). Scope-boundary clause §1.9 added clarifying type-test vs. clinical-validation distinction. §1A.5 (Scope and Precedent) added citing F2119, IEC 62220-1, AAPM TG-233, ACR CT Accreditation precedents and F2119 complementarity. §1A.6 (Metal-Material Rationale) added. §2 expanded to cite F2119, IEC 60601-2-44, DICOM PS3.3, Wunderlich & Noo, Kak & Slaney. §14.12 preset-reporting clause added. §16(o) DICOM 2026b MAR Macro verification item added. §A1.5.3 clarified that 2D constraint applies to observer not to MAR algorithm. Subcommittee of jurisdiction: F04.15 Material Test Methods. Annex A2 (Informational) added specifying optional multi-point signal-amplitude and dose sweeps per Vaishnav et al. (2020). Reference scripts updated to v7.0.0.
>
> **Change summary (Rev 02 → Rev 03):** Observer dimensionality changed from 3D to 2D (Slice 128 only); lesion geometry changed from full z-extent cylinder to single-slice disc; lesion HU implementation changed from 120 HU post-FBP hard-set to ~12 HU sinogram-domain physics contrast (no hard-set); minimum realizations increased from 20 to 40 per condition; Vaishnav internal observer noise regularisation (σ = 15) added as normative requirement; Vaishnav Transition AUC baseline (0.7063) recorded in §1A.2.

---

Metal artifact reduction (MAR) methods are increasingly incorporated into computed tomography and other tomographic imaging systems. These methods may either improve or degrade lesion detectability and quantitative accuracy depending on imaging parameters and algorithm design. Current standards do not define an objective, task-based, reproducible, interlaboratory test method for quantifying this effect.

This test method establishes a procedure for measuring MAR performance using a standardized digital volumetric dataset, a single canonical metal and lesion configuration, and a channelized Hotelling observer (CHO) model. The figure of merit is ΔAUC: the signed difference in area under the ROC curve between MAR-enabled and MAR-disabled conditions. The digital, checksum-verified dataset eliminates physical phantom logistics, permitting reproducible interlaboratory comparison. The method applies to systems producing HU-calibrated reconstructed image data and is structured to support precision and bias evaluation in accordance with ISO 5725. It is intended for type testing and does not establish performance acceptance criteria.

## Standard Test Method for Evaluation of Metal Artifact Reduction Performance in Tomographic Imaging Systems Using a Channelized Hotelling Observer

*This test method is under the jurisdiction of ASTM Committee F04 on Medical and Surgical Materials and Devices and is the direct responsibility of Subcommittee F04.15 on Material Test Methods. Current edition (Revision 04) approved 2026-04-05; editorial pass 2026-04-18. DOI:10.1520/XXXXX-XX*

---

## 1. Scope

**1.1** This test method specifies a quantitative procedure for evaluating the performance of Metal Artifact Reduction (MAR) methods implemented in tomographic imaging systems.

**1.2** The method applies to tomographic imaging systems that produce reconstructed volumetric image data.

**1.3** The measurand is the scalar difference in area under the receiver operating characteristic curve (ΔAUC) derived from a signal-known-exactly, background-known-statistically (SKE/BKS) detection task using a fixed channelized Hotelling observer (CHO) implementation, predefined regions of interest, and specified statistical estimation procedures as defined in Annex A1.

**1.4** This method is modality-independent in the sense that the measurand (ΔAUC) is computed exclusively from reconstructed volumetric image data and does not depend on acquisition hardware, raw projection data, or reconstruction physics. The canonical dataset (§10.1.1) is defined in Hounsfield Unit (HU) space and is therefore directly applicable to systems that produce HU-calibrated reconstructed images, including but not limited to computed tomography (CT). Application to modalities that do not produce HU-calibrated output requires a modality-specific canonical dataset established by a separate work item; results from such datasets shall not be reported as compliant with this standard unless the relevant modality annex has been approved.

**1.5** This standard defines a single canonical test configuration intended for type testing and conformity assessment. No additional configurations are permitted under this standard.

**1.6** *Units* — The values stated in SI units are to be regarded as the standard. No other units of measurement are included in this standard.

**1.7** *This standard does not purport to address all of the safety concerns, if any, associated with its use. It is the responsibility of the user of this standard to establish appropriate safety, health, and environmental practices and determine the applicability of regulatory limitations prior to use.*

**1.8** *This international standard was developed in accordance with internationally recognized principles on standardization established in the Decision on Principles for the Development of International Standards, Guides and Recommendations issued by the World Trade Organization Technical Barriers to Trade (TBT) Committee.*

**1.9** *Scope boundary* — This test method provides a controlled, reproducible measurement of MAR algorithmic behavior under a single canonical test configuration. It is a type test for method comparison and conformity assessment. Clinical validity of a MAR algorithm for any specific patient anatomy, implant geometry, or acquisition protocol is not established by this test alone and shall be supported by additional evidence as required by the incorporating authority or regulatory pathway. Labeling claims derived from this test method shall be scoped to the canonical test configuration (e.g., "In a Type Test per this standard, ΔAUC = X ± CI").

---

## 1A. Background and Technical Basis (Informational)

**1A.1** Metal artifacts in CT arise primarily from beam hardening, photon starvation, and partial volume effects in the presence of highly attenuating objects. MAR algorithms mitigate these artifacts but may alter image statistics in ways that affect clinically relevant tasks, including low-contrast lesion detection. The net effect of MAR on diagnostic utility is not captured by artifact severity metrics alone.

**1A.2** The framework underlying this standard was first described in *Vaishnav, et al.* (Medical Physics, 47(8), 2020), which demonstrated that a model observer-based approach could objectively measure MAR effects on low-contrast detectability. The channelized Hotelling observer (CHO) is a linear model observer that has been extensively validated against human performance in CT detection tasks. The area under the ROC curve (AUC), estimated via the Mann-Whitney statistic, provides a scalar figure of merit that is interpretable, reproducible, and independent of decision threshold. **[Rev 03]** The Vaishnav Transition (2026-03-14) fully operationalized this framework by removing post-FBP hard-set overrides, establishing sinogram-domain physics contrast (~12 HU), and expanding to 40 realizations per condition. **[Rev 04]** The acquisition geometry was updated from parallel-beam to fan-beam (SID=570 mm, SDD=1040 mm) to align with clinical CT geometry. The baseline AUC_noMAR under the canonical fan-beam geometry (§A1.1(f,g)) was established at **0.8294** (N=40, σ=15, CI [0.7612, 0.9025], 2026-04-07). The prior parallel-beam baseline (AUC_noMAR = 0.7063) is retained for historical reference only and shall not be used for fan-beam dataset validation.

**1A.3** The use of a deterministic digital dataset eliminates physical phantom fabrication, scanner time, and shipping logistics from interlaboratory studies. SHA-256 checksums ensure bitwise identity of the dataset across laboratories. The only source of interlaboratory variability under this standard is therefore the CHO implementation and the MAR algorithm under test, which is the intended behavior.

**1A.4** The canonical dataset provides background variability through per-realization rotation of the artifact template (§A1.7). This mechanism prevents the CHO from exploiting static artifact patterns and ensures that the observer's performance reflects genuine signal detectability rather than artifact fingerprint memorization.

**1A.5** *Scope and precedent* — The single canonical configuration approach follows established precedent in medical imaging type testing, including ASTM F2119 (Evaluation of MR Image Artifacts from Passive Implants), IEC 62220-1 (detector DQE), AAPM TG-233 (CT image quality), and the ACR CT Accreditation Program. A standardized, narrow measurement enables interlaboratory precision and bias characterization per ASTM E691 and ISO 5725; clinical-practice claims are supported by separate, device-specific evidence as required by the relevant regulatory authority. This test method is complementary to ASTM F2119 (evaluation of MR image artifacts from passive implants): F2119 characterizes the physical extent of image artifacts produced by a passive implant under standardized MR scanning conditions; the present method quantifies the observer-based task-detectability impact of an algorithmic countermeasure (MAR) applied within the imaging system. The two methods address non-overlapping axes — modality (MR / CT) × object of measurement (physical artifact extent / algorithmic task impact) — and together cover the passive-implant / active-algorithm quadrants of artifact assessment in F04's jurisdiction.

**1A.6** *Metal-material rationale* — The canonical metal material (iron, μ = 2.408 cm⁻¹ at 60 keV) is selected as a representative high-Z attenuator sitting within the range of clinical implant materials (titanium 1.41, stainless steel 2.3, cobalt-chromium 2.7 cm⁻¹ at 60 keV). The 10 mm diameter and centered-on-axis geometry are chosen to produce a defined, reproducible beam-hardening and photon-starvation artifact signature, not to simulate any specific clinical implant. The canonical metal functions as a controlled acoustic test tone for MAR algorithmic response — every algorithm under test faces the same attenuation challenge.

---

## 2. Referenced Documents

**2.1** *ASTM Standards:*

- ASTM E177 – Practice for Use of the Terms Precision and Bias in ASTM Test Methods
- ASTM E691 – Practice for Conducting an Interlaboratory Study to Determine the Precision of a Test Method
- ASTM F2119 – Standard Test Method for Evaluation of MR Image Artifacts from Passive Implants

**2.2** *ISO Standards:*

- ISO 5725-1 – Accuracy (trueness and precision) of measurement methods and results – Part 1
- ISO 5725-2 – Basic method for determination of repeatability and reproducibility of a standard measurement method

**2.3** *IEC Standards:*

- IEC 60601-2-44 – Medical electrical equipment – Part 2-44: Particular requirements for the basic safety and essential performance of X-ray equipment for computed tomography

**2.4** *DICOM Standards:*

- DICOM PS3.3 – Information Object Definitions (including the Metal Artifact Reduction Macro, CP-2575, 2026b)

**2.5** *Other References:*

- Vaishnav JY et al. CT metal artifact reduction algorithms: Toward a framework for objective performance assessment. Medical Physics 47(8):3344–3355, 2020. DOI: 10.1002/mp.14231
- Barrett HH, Myers KJ. Foundations of Image Science. Wiley, 2004.
- Wunderlich A, Noo F. On the efficiency of two-sample tests based on the Mann-Whitney U statistic for imaging tasks. IEEE Trans Med Imaging 34(2):522–533, 2015.
- Kak AC, Slaney M. Principles of Computerized Tomographic Imaging. IEEE Press, 1988.
- FIPS PUB 180-4 — Secure Hash Standard (SHA-256)

---

## 3. Terminology

**3.1** *Definitions:*

**3.1.1** metal artifact — image distortion in reconstructed tomographic data arising from the presence of a high-attenuation object, manifesting as streak, shadow, or halo patterns.

**3.1.2** metal artifact reduction (MAR) — algorithmic processing applied within the imaging system pipeline to mitigate metal artifacts in reconstructed image data.

**3.1.3** channelized Hotelling observer (CHO) — a linear model observer that evaluates signal detectability by projecting image data onto predefined spatial-frequency channel templates and applying the Hotelling discriminant to the resulting channel output vectors.

**3.1.4** area under the ROC curve (AUC) — a scalar measure of binary detection performance ranging from 0.5 (chance-level discrimination) to 1.0 (perfect discrimination), estimated in this standard via the Mann–Whitney statistic.

**3.1.5** ΔAUC — the signed difference AUC_MAR − AUC_noMAR, where positive values indicate that MAR improved lesion detectability and negative values indicate degradation.

**3.1.6** test result — the mean ΔAUC calculated across all replicate paired image sets under specified conditions, reported to three decimal places.

**3.1.7** signal-known-exactly (SKE) — a detection task paradigm in which the signal (lesion) location, shape, size, and intensity are fixed and known to the observer.

**3.1.8** background-known-statistically (BKS) — a detection task paradigm in which the background statistics are known but individual background realizations vary. In this standard, variation arises from per-realization artifact jitter and independent Gaussian noise.

**3.1.9** canonical test configuration — the single, fully specified combination of phantom geometry, lesion parameters, dataset generation rules, and observer implementation defined in this standard. No other configuration is permitted.

**3.1.10** realization — one independent volumetric image set (lesion-present or lesion-absent) generated with a unique random seed, constituting an independent sample for CHO training and testing.

**3.1.11** artifact jitter — per-realization rotation of the artifact template by a specified random angle, providing background variability that prevents CHO overtraining on static artifact patterns.

---

## 4. Summary of Test Method

**4.1** A standardized reconstructed digital volumetric dataset is provided to participating laboratories. The dataset represents a simplified anthropomorphic torso cross-section containing a single cylindrical metallic rod and background tissue. The dataset is fully synthetic, generated from defined geometric primitives, and does not require clinical scanner acquisition.

**4.2** The standardized volumetric dataset shall be processed: (a) with MAR disabled, and (b) with MAR enabled, using identical reconstruction, preprocessing, postprocessing, and display parameters. The MAR enable/disable state shall be the only permitted difference between conditions.

**4.3** A specified lesion disc is present in a defined spatial relationship to the metallic object in the lesion-present series, at Slice 128 only. **[Rev 03]** The lesion is a single-slice disc (not a full z-extent cylinder); its contrast (~12 HU above background) is established in the sinogram domain via physics-based attenuation. The lesion-absent series contains no lesion.

**4.4** **[Rev 03]** A two-dimensional channelized Hotelling observer (2D CHO) analysis is performed on Slice 128 of paired lesion-present and lesion-absent image volumes for each condition (MAR and noMAR). Three-dimensional volumetric integration across z is prohibited (§A1.5.3).

**4.5** AUC values are computed for each condition via the Mann–Whitney statistic.

**4.6** ΔAUC = AUC_MAR − AUC_noMAR.

**4.7** The mean ΔAUC across replicate image sets constitutes the test result.

---

## 5. Significance and Use

**5.1** This test method provides an objective, reproducible, task-based measurement of MAR algorithmic impact on lesion detectability. Unlike methods that rely on physical dimensions of image artifacts, this method evaluates image quality by the ability of a fully specified deterministic model observer (CHO) to perform a binary detection task, removing human-reader variability.

**5.2** The method supports comparison of MAR implementations across systems and laboratories by measuring the preservation of lesion detectability in a standardized SKE/BKS detection task with bit-identical input data across all participating sites.

**5.3** The test method is intended for type testing, research, and performance characterization. It is not a substitute for clinical validation; see §1.9.

**5.4** This test method does not establish performance acceptance criteria. A positive ΔAUC indicates that MAR improves lesion detectability relative to the no-MAR condition; a negative ΔAUC indicates degradation. Both outcomes are scientifically valid results and shall be reported without suppression or sign correction.

**5.5** This test method is structured to support incorporation by normative reference into external performance standards. **[Rev 04]** Acceptance criteria based on ΔAUC values are established by the incorporating authority (e.g., IEC 60601-2-44 §203.6.7.101.1 via its proposed compliance-statement amendment referencing this TYPE TEST; national regulatory guidance), not by this standard. This standard defines only the measurement method.

---

## 6. Interferences

**6.1** Variability in image processing parameters between MAR-enabled and MAR-disabled conditions will inflate or deflate ΔAUC and invalidate the test result.

**6.2** Deviations from the specified Laguerre–Gauss channel templates, channel width parameter, Tikhonov regularization value, internal noise parameter (§A1.5.2(d)), or covariance estimation procedure will reduce interlaboratory reproducibility.

**6.3** Alteration of the standardized dataset, including resampling, interpolation, windowing before CHO input, or truncation of the HU range, invalidates the test.

**6.4** Use of floating-point arithmetic below 64-bit precision may introduce numerical bias exceeding the ±0.005 AUC equivalence tolerance specified in §8.3.

**6.5** Failure to verify the SHA-256 checksum prior to analysis invalidates the test result.

---

## 7. Apparatus

**7.1** Imaging system or processing platform capable of applying MAR processing to the supplied volumetric dataset in DICOM CT format.

**7.2** **[Rev 04]** Computational platform capable of executing the reference 2D CHO analysis software (`run_cho_analysis_v7_0.py`).

**7.3** Statistical software capable of Mann–Whitney AUC estimation and bootstrap confidence interval computation.

**7.4** SHA-256 checksum verification utility.

**7.5** The computational environment (software version, operating system, numerical libraries, and hardware platform) used for CHO analysis shall be documented in the test report.

---

## 8. Reagents and Materials

**8.1** **[Rev 03]** Standardized reconstructed digital volumetric dataset distributed with this standard, SHA-256 checksum verified per §11.1. The dataset consists of 160 DICOM CT series: 40 lesion-present / noMAR, 40 lesion-absent / noMAR, 40 lesion-present / MAR-ready, and 40 lesion-absent / MAR-ready, each comprising 256 axial slices at 512 × 512 × 0.5 mm isotropic. The reference dataset is `./astm_reference_dataset/`.

**8.2** Laguerre–Gauss channel template definitions as specified in §A1.5.1 and provided in machine-readable form with the dataset distribution.

**8.3** **[Rev 04]** Reference CHO implementation distributed with this standard (`run_cho_analysis_v7_0.py`). The reference implementation is the normative arbiter of correct CHO output. Alternative implementations shall demonstrate numerical equivalence within ±0.005 AUC on the supplied checksum-verified validation dataset before use.

**8.4** Checksum manifest file (SHA-256) distributed with the dataset.

---

## 9. Hazards

**9.1** No ionizing radiation exposure is required when using the supplied digital dataset.

---

## 10. Sampling, Test Specimens, and Test Units

**10.1** The test unit consists of paired volumetric image datasets (MAR enabled and MAR disabled) derived from identical input data.

**10.1.1** *Canonical Test Configuration* — This test method defines a single canonical configuration. All parameters are normative and fully specified in this section and in Annex A1. No additional configurations, parameter substitutions, or alternative geometries are permitted. The canonical configuration consists of:

| Parameter | Value | Notes |
|---|---|---|
| Matrix (x, y, z) | 512 × 512 × 256 voxels | Per §A1.1 |
| Voxel size | 0.5 mm isotropic | FOV = 256 mm |
| Background HU | 40 HU (soft tissue) | Uniform; noise added per realization |
| Metal rod diameter | 10 mm | Full z-extent, centered at (256, 256) |
| Metal HU | 3000 HU (fixed) | Not user-modifiable |
| **Lesion geometry [Rev 03]** | **Single disc, Slice 128 only** | **Not full z-extent; single representative slice** |
| Lesion disc diameter | 5 mm | Radius = 2.5 mm = 5 voxels |
| Lesion offset | 5 mm beyond metal boundary, +x | Center at (x_metal + r_metal + 5 mm + r_lesion, 256) = (281, 256) |
| **Lesion contrast [Rev 03]** | **~12 HU physics contrast** | **MU_LESION_CM = MU_TISSUE_CM × (1 + 12/1000); sinogram-domain only; no post-FBP hard-set** |
| Background noise | 30 HU Gaussian, IID | Applied only outside lesion and metal masks |
| **Acquisition geometry [Rev 04]** | **Fan-beam, SID=570 mm, SDD=1040 mm** | **Equi-angular curved detector; §A1.1(f)** |
| **Projection angles [Rev 04]** | **720 over [0°, 360°)** | **Full rotation, 0.5° spacing; §A1.1(f)** |
| **Reconstruction [Rev 04]** | **Fan-beam FBP, Ram-Lak, cosine pre-weight** | **Distance-weighted backprojection; §A1.1(g)** |
| Artifact model | Physics-based Poisson + scatter + Gaussian (Vaishnav noise model) | 60 keV monochromatic; see §A1.7 |
| Artifact jitter | Uniform (−15°, +15°) per realization | Independent per realization; see §A1.7 |
| **Realizations [Rev 03]** | **40 per condition (minimum)** | **4 conditions: LP/LA × MAR/noMAR; 160 total** |
| **Screening mode [Rev 04]** | **20 per condition (optional)** | **Pilot evaluation only; not reportable under §10.2** |

**10.1.2** *Natural Reconstruction Rule* — **[Rev 03]** The lesion signal shall be established exclusively in the sinogram domain via physics-based attenuation. No post-FBP HU replacement shall be applied to lesion voxels. The construction order within each realization slice shall be: (1) forward project phantom with and without lesion at defined monochromatic energy; (2) apply Vaishnav noise model (Poisson + scatter + Gaussian electronic noise); (3) FBP reconstruction with Ram-Lak filter and DC calibration offset; (4) restore metal voxels to 3000 HU as the final step, overriding all other values. The ~12 HU effective lesion contrast emerges from the Radon inversion process. Violation of this order, or any post-FBP pixel override applied to lesion voxels, produces a non-physics signal and invalidates the dataset.

**10.2** **[Rev 03]** For the canonical test configuration, a minimum of 40 statistically independent lesion-present and 40 lesion-absent image volumetric realizations shall be analyzed per condition. The minimum total is therefore 160 volumetric image sets (40 LP-noMAR, 40 LA-noMAR, 40 LP-MAR, 40 LA-MAR).

**10.2.1** **[Rev 04]** *Screening mode* — For pilot evaluation and feasibility assessment, a reduced configuration of 20 realizations per condition (80 total) may be used. Results obtained with fewer than 40 realizations per condition are informative only and shall not be reported as compliant with this standard. Screening-mode results shall be clearly labelled as such in any documentation.

**10.3** Statistical independence shall be achieved using the predefined set of independent volumetric realizations provided with the standardized dataset, each generated with a unique, non-overlapping random seed. Laboratories shall not generate additional noise realizations or alter the provided image volumes. Each full 3D volume (256 slices) constitutes one independent realization; individual slices within a volume are not independent samples and shall not be treated as such in CHO training or testing.

---

## 11. Preparation of Apparatus

**11.1** Verify SHA-256 checksum of every file in the distributed dataset against the manifest provided with the standard. Any checksum mismatch disqualifies the dataset and invalidates any results derived from it.

**11.2** Validate the CHO implementation against the supplied reference validation dataset, confirming AUC agreement within ±0.005 per §8.3.

**11.3** Configure the imaging system MAR parameters as specified in the test plan. Document all configuration settings.

---

## 12. Calibration and Standardization

**12.1** Validate CHO implementation using the supplied reference checksum-verified validation dataset prior to each test campaign. Validation shall be repeated whenever the CHO software or computational environment is updated.

**12.2** No additional calibration of the imaging system is required or permitted under this standard.

---

## 13. Conditioning

**13.1** No specimen conditioning is required. The dataset is digital and requires no physical preparation.

---

## 14. Procedure

**14.1** Verify dataset checksums per §11.1.

**14.2** **[Rev 03]** Process the standardized noMAR volumetric dataset (40 LP and 40 LA volumes) through the imaging system with MAR disabled.

**14.3** Process the identical standardized dataset through the imaging system with MAR enabled, using identical reconstruction and postprocessing parameters. The MAR enable/disable state is the only permitted difference.

**14.4** Verify that the processed output volumes match the expected matrix dimensions, voxel size, and bit depth specified in §A1.1 before proceeding.

**14.5** **[Rev 03]** Apply the 2D CHO (§A1.5) to Slice 128 of the lesion-present and lesion-absent volume pairs for each condition (MAR and noMAR), using the fixed ROI specified in §A1.5.4. Compute AUC values for each condition. Three-dimensional volumetric integration across z is prohibited.

**14.6** **[Rev 03]** Estimate covariance from lesion-absent volumes only, using the pooled sample covariance across all 40 LA realizations for the relevant condition.

**14.7** Compute AUC for each condition via the Mann–Whitney statistic (§A1.6).

**14.8** Calculate ΔAUC = AUC_MAR − AUC_noMAR.

**14.9** Compute bootstrap 95% confidence intervals per §A1.6(c).

**14.10** Document all system settings, software versions, and any deviations from this procedure.

**14.11** All preprocessing, reconstruction, and postprocessing parameters other than the MAR enable / disable state shall remain identical between conditions. Any parameter change, including window / level settings applied before CHO input, invalidates the test result.

**14.12** *Multi-preset algorithms* — Where the MAR algorithm under test provides user-selectable strength or operating-point presets (e.g., "low / medium / high"), each preset shall be evaluated independently as a distinct test run. The preset identifier shall be recorded in the test report per §16(b) and ΔAUC reported per preset per §16(h). A single test campaign may include results for multiple presets of the same algorithm.

---

## 15. Calculation or Interpretation of Results

**15.1** ΔAUC = AUC_MAR − AUC_noMAR

**15.2** The reported test result shall be the mean ΔAUC across all replicates. ΔAUC shall be reported to three decimal places with sign.

**15.3** Standard deviation of ΔAUC across replicates shall be reported.

**15.4** Bootstrap 95% confidence intervals shall be reported.

**15.5** The AUC estimation bias (difference between resubstitution and hold-out estimates) shall be reported per §A1.6(d).

**15.6** Negative ΔAUC values shall be reported without modification. A negative result indicates that the MAR algorithm degraded lesion detectability relative to the no-MAR condition and is a scientifically valid and reportable outcome.

---

## 16. Report

**16.1** The report shall include:

- (a) System identification (manufacturer, model, software version)
- (b) MAR algorithm name and version
- (c) All image processing parameters applied during MAR-enabled and MAR-disabled conditions, with explicit confirmation that only the MAR state differed
- (d) Dataset version identifier and SHA-256 manifest verification result
- (e) CHO software version and validation AUC result (§12.1)
- (f) Computational environment (OS, CPU/GPU, numerical library versions)
- (g) Number of LP and LA replicates analyzed per condition
- (h) Mean ΔAUC (three decimal places, with sign)
- (i) Standard deviation of ΔAUC
- (j) Bootstrap 95% confidence interval
- (k) AUC estimation bias (resubstitution minus hold-out)
- (l) Individual AUC_MAR and AUC_noMAR values
- (m) Any deviations from this test method
- **(n) [Rev 03]** Internal noise parameter σ_internal used (normative value: 15; see §A1.5.2(d))
- **(o) [Rev 04]** Verification of the DICOM Metal Artifact Reduction Macro (PS3.3 C.8.15.3.15, CP-2575, 2026b). The report shall confirm that the MAR enable/disable state is recorded in the Metal Artifact Reduction Applied attribute (0018,9391) of the Metal Artifact Reduction Sequence (0018,9390) on every output DICOM series, and that the recorded value is consistent with the algorithm configuration under test

---

## 17. Precision and Bias

**17.1** *Precision* — The precision of this test method shall be determined by an interlaboratory study conducted in accordance with ASTM E691 and ISO 5725-2.

**17.1.1** The following precision statistics shall be determined:

- (a) Repeatability standard deviation (S_r)
- (b) Reproducibility standard deviation (S_R)
- (c) Repeatability limit (r = 2.8 × S_r)
- (d) Reproducibility limit (R = 2.8 × S_R)

**17.1.2** The precision statement shall be incorporated into this standard following completion of the full interlaboratory study.

**17.1.3** The full interlaboratory study shall include a minimum of 6 laboratories in accordance with ASTM E691 recommendations, each processing the identical checksum-verified dataset.

**17.1.4** The precision statement shall report S_r, S_R, r, and R values expressed in units of ΔAUC.

**17.1.5** The digital, checksum-verified nature of this test method minimizes the resource barriers traditionally associated with interlaboratory studies by eliminating physical phantoms, shipping logistics, and clinical scanner time requirements.

**17.1.6** Pilot precision data requirement: Before this standard proceeds to first ASTM ballot, the sponsoring subcommittee shall provide pilot precision data from a minimum of 3 laboratories demonstrating that: (a) mean ΔAUC is statistically distinguishable from zero for a MAR algorithm known to improve detectability; (b) the within-laboratory standard deviation of ΔAUC does not exceed 0.05 AUC units; and (c) the AUC estimation bias (resubstitution minus hold-out) does not exceed 0.02 AUC units at N=40 realizations.

**17.2** *Bias* — Bias in the absolute sense cannot be determined because no accepted reference value for true lesion detectability exists independently of the measurement method. Systematic effects identified during interlaboratory evaluation shall be reported. Sources of systematic effect include: CHO implementation differences, floating-point accumulation errors, and covariance regularization choices.

---

## 18. Keywords

metal artifact; metal artifact reduction; MAR; channelized Hotelling observer; CHO; model observer; ROC curve; AUC; ΔAUC; lesion detectability; interlaboratory study; type test; precision and bias; tomographic imaging; computed tomography; fan-beam; signal-known-exactly; Laguerre–Gauss channels; reproducibility; 2D observer; Vaishnav framework; sinogram-domain contrast; DICOM; Metal Artifact Reduction Macro; CP-2575; IEC 60601-2-44

---

## ANNEX A1 (Normative)

### Dataset Geometry, Lesion Specification, Observer Definition, and Statistical Procedures

---

### A1.1 Volume Geometry

- (a) Matrix: 512 × 512 × 256 voxels (x, y, z)
- (b) Voxel size: 0.5 mm isotropic in all three dimensions
- (c) Field of view: 256 mm (512 voxels × 0.5 mm / voxel)
- (d) HU range encoded as signed 16-bit integer (INT16) in DICOM pixel data
- (e) Rescale slope: 1.0; Rescale intercept: 0.0 (HU = stored value)
- **(f) [Rev 04] Acquisition geometry: 2D fan-beam.** Source-to-isocenter distance (SID) = 570 mm. Source-to-detector distance (SDD) = 1040 mm. Equi-angular curved detector array, 512 elements, angular pitch Δγ = 2·arcsin(FOV/(2·SID))/N_det ≈ 0.0507°. Full 360° rotation, 720 equi-spaced projection angles (0.5° angular spacing). Maximum fan half-angle γ_max = arcsin(128/570) ≈ 12.97°.
- **(g) [Rev 04] Reconstruction: fan-beam filtered backprojection (FBP)** with cosine pre-weighting, Ram-Lak (ramp) filter, and (SID/L)² distance-weighted backprojection, where L is the source-to-pixel distance. Scaling factor: π / N_angles / (SID × Δγ). This follows the equi-angular fan-beam FBP formulation of Kak & Slaney (1988), §3.4.

---

### A1.2 Phantom Cross-Section Geometry

The phantom cross-section represents a simplified torso geometry consisting of an elliptical body region, a cylindrical metal rod, and (when present) a lesion disc at Slice 128. **[Rev 03]** All structures except the lesion are uniform cylinders spanning the full z-extent of the volume. The lesion is a single disc confined to Slice 128 only.

- (a) Body ellipse semi-axis x: 85 mm (170 voxels at 0.5 mm/voxel)
- (b) Body ellipse semi-axis y: 60 mm (120 voxels at 0.5 mm/voxel)
- (c) Body center: (x, y) = (256, 256) voxels (image center)
- (d) Body interior HU: 40 HU (soft tissue equivalent)
- (e) Exterior (outside body ellipse) HU: −1000 HU (air)
- (f) Gaussian noise (σ = 30 HU) applied to body interior only, excluding lesion and metal masks, independently per realization per §A1.7

---

### A1.3 Metal Object Specification

- (a) Geometry: right circular cylinder, full z-extent
- (b) Diameter: 10 mm (radius = 5 mm = 10 voxels at 0.5 mm/voxel)
- (c) Center: (X_m, Y_m) = (256, 256) voxels
- (d) Fixed HU value: 3000 HU
- (e) Metal boundary x-coordinate (positive x face): X_m + r = 256 + 10 = 266 voxels
- (f) Metal voxels shall be restored to 3000 HU as the final step in slice construction, overriding noise and artifact

---

### A1.4 Lesion Specification

**[Rev 03]** The lesion is a single-slice disc (not a full z-extent cylinder). Its contrast is established in the sinogram domain via physics-based attenuation. No post-FBP HU replacement shall be applied.

- (a) **[Rev 03]** Geometry: right circular disc, **Slice 128 only** (zero-indexed central slice; not full z-extent)
- (b) Diameter: 5 mm (radius = 2.5 mm = 5 voxels at 0.5 mm/voxel)
- (c) Lesion center x-coordinate: X_0 = X_m + r_m + 5 mm + r_lesion = 256 + 10 + 10 + 5 = 281 voxels
- (d) Lesion center y-coordinate: Y_0 = 256 voxels
- (e) **[Rev 03]** Physics-based contrast: MU_LESION_CM = MU_TISSUE_CM × (1 + 12/1000) ≈ 0.20837 cm⁻¹; effective FBP contrast ~12 HU above background; no post-FBP hard-set override
- (f) **[Rev 03]** Lesion contrast is established in the sinogram domain only. No post-FBP HU replacement shall be applied to voxels within the lesion mask. The ~12 HU contrast emerges from the FBP inversion of the physics-based sinogram difference. Noise exclusion from the lesion ROI is a natural consequence of the physics model; noise and artifact are present at the lesion location.
- (g) **[Rev 03]** Signal contrast: ~12 HU above 40 HU soft tissue background (sinogram-domain; realized value varies per realization due to noise)
- (h) **[Rev 03]** Pre-observer per-voxel CNR: ~12/30 ≈ 0.4; noise-limited detection task. The 2D CHO with Vaishnav internal noise regularisation (σ_internal = 15, §A1.5.2(d)) is calibrated to AUC_noMAR = 0.8294 for the canonical fan-beam geometry (was 0.7063 for parallel-beam).

---

### A1.5 Channelized Hotelling Observer (CHO) Specification

#### A1.5.1 Channel Type and Parameters

- (a) Channel type: Laguerre–Gauss (LG) radial channels
- (b) Number of channels: 10
- (c) Channel order n: 0 through 9 (one channel per order)
- (d) Channel width parameter a: 1.5 × r_lesion = 1.5 × 5 voxels = 7.5 voxels (3.75 mm). This value is fixed and not user-modifiable.
- (e) Channel normalization: each channel u_n(r) is L2-normalized over the ROI domain such that ‖u_n‖₂ = 1
- (f) **[Rev 03]** The n-th 2D Laguerre–Gauss channel is defined as: u_n(r) = L_n(2πr²/a²) × exp(−πr²/a²), where L_n is the n-th Laguerre polynomial and r is the radial distance from the lesion center in voxels. The channel operates in 2D on Slice 128 only. Three-dimensional extension along z is prohibited (§A1.5.3).

#### A1.5.2 Covariance Estimation

- (a) The CHO covariance matrix K shall be estimated from lesion-absent (LA) volumes exclusively.
- (b) **[Rev 03]** Covariance shall be estimated by pooling channel output vectors across all 40 LA realizations of the relevant condition (MAR or noMAR). Separate covariance matrices shall be estimated for the MAR and noMAR conditions.
- (c) Regularization: Tikhonov regularization shall be applied as K_reg = K + λI, where λ = 0.01 × trace(K) / p, p = number of channels = 10. This normalization ensures λ scales with the data and is invariant to HU units. λ is fixed by this formula and shall not be user-modified.
- **(d) [Rev 03] Internal Observer Noise** — To prevent infinite SNR on low-variance pixels and to match human observer thresholds consistent with the Vaishnav framework, an internal noise variance shall be added to the diagonal of the covariance matrix before inversion: K_total = K_external + σ_internal² × I, where **σ_internal = 15** (HU, in channel space). The total regularized matrix used for Hotelling template estimation shall be: K_final = K_total + λI = K_external + σ_internal² × I + λI. This parameter is normative and shall not be user-modified. In the reference software, this is implemented as `--internal-noise-sigma 15`. Omission of this parameter or substitution of a different value invalidates interlaboratory comparability.

#### A1.5.3 Observer Dimensionality

**[Rev 03]** The CHO shall operate on a two-dimensional (2D) region of interest from **Slice 128 (LESION_SLICE_INDEX = 128, zero-indexed) only**. Three-dimensional volumetric integration across z is **PROHIBITED** under this standard. The 2D ROI is 121 × 121 voxels centred at (281, 256) as specified in §A1.5.4. Loading additional slices and aggregating channel responses across z constitutes a violation of this standard and will produce artificially inflated AUC values that are not comparable across laboratories.

> **[Rev 04] Scope of the 2D constraint:** The 2D constraint specified in this section applies to the **CHO observer**, not to the **MAR algorithm under test**. The MAR algorithm may operate on the full 3D volume or on any subset of slices appropriate to its implementation (e.g., iterative MBIR methods that use multi-slice context). Only the observer-based evaluation is restricted to Slice 128. This ensures that algorithms relying on 3D context are evaluated on the same basis as algorithms operating slice-by-slice.

> **Rationale:** When the lesion is confined to a single slice and the CHO integrates across all 256 z-slices, the observer accumulates √256 ≈ 16× coherent signal gain from the 255 lesion-absent slices in the LA condition while adding noise-only z-planes in the LP condition, driving d′ → ∞ and AUC → 1.000 for both conditions regardless of MAR algorithm quality. The 2D mandate eliminates this artifact.

> **[Rev 04] Note:** The CHO operates on reconstructed DICOM images and is therefore independent of the acquisition geometry (parallel-beam, fan-beam, or cone-beam). The fan-beam geometry change (§A1.1(f,g)) affects only the sinogram generation and FBP reconstruction, not the CHO mathematics.

#### A1.5.4 Region of Interest (ROI)

- (a) ROI x-extent: lesion center x ± 60 voxels = 221 to 341 (121 voxels)
- (b) ROI y-extent: lesion center y ± 60 voxels = 196 to 316 (121 voxels)
- (c) **[Rev 03]** ROI z-extent: **Slice 128 only** (single 2D slice; not full z-extent of volume)
- (d) ROI center coordinates are fixed per this annex and shall not be adjusted between conditions or realizations.
- (e) The same ROI shall be used for MAR-enabled and MAR-disabled conditions.

---

### A1.6 Statistical Procedures

- (a) AUC shall be estimated via the Mann–Whitney statistic applied to CHO test statistics from LP and LA volumes.
- (b) Ties in the Mann–Whitney statistic shall be handled using mid-rank assignment.
- (c) Bootstrap confidence intervals shall be computed using 1000 resamples (sampling LP and LA test statistics jointly with replacement). The 2.5th and 97.5th percentiles of the bootstrap AUC distribution define the 95% CI.
- (d) AUC estimation bias shall be quantified using resubstitution and hold-out CHO training / testing strategies per Wunderlich and Noo (IEEE Trans Med Imaging, 34(2), 2015). The bias estimate is defined as b = AUC_resubstitution − AUC_hold-out. Both estimates and the bias shall be reported.
- (e) Minimum 64-bit (double-precision) floating-point arithmetic shall be used throughout CHO computation, including channel projection, covariance estimation, matrix inversion, and test statistic computation.

---

### A1.7 Background Variability Mechanism

**A1.7.1** The canonical dataset provides background variability via two independent mechanisms: (i) per-realization artifact jitter, and (ii) independent Gaussian noise realizations.

**A1.7.2** *Artifact jitter* — **[Rev 03]** For each realization i (i = 1 to 40), a jitter angle θ_i shall be drawn from a uniform distribution on [−15°, +15°]. The base artifact template (§A1.7.4) shall be rotated by θ_i about the image center using bilinear interpolation. The rotated template shall be clipped to the body mask and zeroed within the metal mask before addition.

**A1.7.3** *Noise realizations* — Independent Gaussian noise (σ = 30 HU, μ = 0) is added to each realization using a unique random seed. For realization i, the seed shall be BASE_SEED + i where BASE_SEED is a fixed integer specified in the dataset metadata. Noise is applied only within the body mask and outside the lesion and metal masks.

**A1.7.4** *Artifact template* — The base artifact template is a 2D HU field computed once by: (1) **[Rev 04]** forward projecting a phantom containing the body background and metal rod using the canonical fan-beam geometry (§A1.1(f)); (2) simulating photon starvation by replacing sinogram values above the 99th percentile of non-zero values with 2% of the 50th percentile; (3) **[Rev 04]** reconstructing both the original and corrupted sinograms using fan-beam FBP (§A1.1(g)); (4) computing the difference (corrupted minus original); (5) zeroing the template within the metal and outside the body mask; (6) scaling so that the maximum absolute value within the body (excluding metal) is 400 HU.

**A1.7.5** **[Rev 03]** The artifact template, jitter angles, and noise seeds for all 40 realizations are fixed in the reference dataset. Laboratories shall use the provided dataset and shall not regenerate these values.

---

### A1.8 Reproducibility Requirements

The following parameters shall not be modified under any circumstance. Modification of any parameter invalidates the test result under this standard:

| Parameter | Specified Value | Location |
|---|---|---|
| Voxel size | 0.5 mm isotropic | §A1.1 |
| Matrix dimensions | 512 × 512 × 256 | §A1.1 |
| LG channel type | Laguerre–Gauss, n = 0..9 | §A1.5.1(a-c) |
| Channel width parameter a | 7.5 voxels (1.5 × r_lesion) | §A1.5.1(d) |
| Channel normalization | L2-normalized | §A1.5.1(e) |
| Tikhonov λ | 0.01 × trace(K)/p | §A1.5.2(c) |
| **Internal noise σ [Rev 03]** | **15** | **§A1.5.2(d)** |
| Covariance source | LA volumes only, pooled | §A1.5.2(a,b) |
| **Observer dimensionality [Rev 03]** | **2D, Slice 128 only** | **§A1.5.3** |
| ROI dimensions | **121 × 121 voxels (2D)** | §A1.5.4 |
| ROI center | (281, 256) voxels in xy | §A1.4(c,d) |
| **Lesion geometry [Rev 03]** | **Single disc, Slice 128 only** | **§A1.4(a)** |
| **Lesion contrast [Rev 03]** | **~12 HU physics (MU_LESION_CM = MU_TISSUE × (1+12/1000)); no hard-set** | **§A1.4(e,f)** |
| Metal HU | 3000 HU (restored last) | §A1.3(d,f) |
| Artifact peak HU | 400 HU | §A1.7.4 |
| Artifact jitter range | Uniform [−15°, +15°] | §A1.7.2 |
| **Realizations per condition [Rev 03]** | **40 minimum** | **§10.2** |
| Floating-point precision | 64-bit minimum (double) | §A1.6(e) |
| **Acquisition geometry [Rev 04]** | **Fan-beam, SID=570 mm, SDD=1040 mm** | **§A1.1(f)** |
| **Projection angles [Rev 04]** | **720 over [0°, 360°)** | **§A1.1(f)** |
| **Reconstruction [Rev 04]** | **Fan-beam FBP, Ram-Lak, cosine pre-weight, (SID/L)²** | **§A1.1(g)** |
| **CHO equivalence tolerance [Rev 04]** | **±0.005 AUC** | **§8.3** |
| **Baseline AUC_noMAR [Rev 04]** | **0.8294, 95% CI [0.7612, 0.9025] (N=40 LP + 40 LA, σ_internal=15, 2026-04-07)** | **§1A.2** |

---

## ANNEX A2 (Informational)

### Multi-Point Performance Characterization

**[Rev 04]** This annex is informational. The normative deliverable of this test method is the scalar ΔAUC at the canonical lesion contrast (12 HU) and dose setpoint (I₀ = 310,853, σ_noise = 30 HU target). The scalar metric supports interlaboratory precision and bias characterization per ASTM E691 and ISO 5725.

For investigators, manufacturers, and regulatory reviewers who wish to characterize MAR algorithmic behavior across a range of operating conditions, Vaishnav et al. (Medical Physics 47(8), 2020) recommend presenting full detectability curves as functions of signal amplitude and dose. This annex specifies an optional multi-point reporting protocol that preserves the canonical configuration as the normative anchor while extending the characterization.

### A2.1 Signal-Amplitude Sweep

- (a) Signal-amplitude sweep values: 4, 8, 12, 16, 20 HU nominal lesion contrast (five operating points, inclusive of the canonical 12 HU).
- (b) Each operating point shall use the complete sinogram-domain physics chain specified in §10.1.2 with `MU_LESION_CM = MU_TISSUE_CM × (1 + c/1000)` where c is the contrast in HU.
- (c) Each operating point shall be generated with N = 40 independent realizations per condition (160 volumes per point, 800 volumes for the full sweep).
- (d) CHO analysis shall be performed per §14 for each operating point, producing ΔAUC, 95% CI, and standard deviation per operating point.
- (e) The resulting curve ΔAUC(c) shall be reported as a table of five values with sign, CI, and standard deviation.

### A2.2 Dose Sweep

- (a) Dose sweep values: I₀ × {0.5, 0.71, 1.0, 1.41, 2.0} relative to the canonical I₀ = 310,853 (five operating points, inclusive of the canonical value). These ratios correspond to ±√2 in photon flux and therefore approximately ±√2 in per-pixel SNR in the reconstructed image.
- (b) Each operating point shall use the complete forward-projection and FBP chain specified in §A1.1 and §A1.7, with I₀ scaled per the sweep value. Gaussian electronic noise (σ_e) scales as 1/√I₀ per the analytic calibration in §1A.3.
- (c) Each operating point shall be generated with N = 40 independent realizations per condition.
- (d) CHO analysis shall be performed per §14 for each operating point.
- (e) The resulting curve ΔAUC(I₀) shall be reported as a table of five values with sign, CI, and standard deviation.

### A2.3 Reporting

**A2.3.1** The multi-point report shall be clearly labelled as informational and shall not be reported as compliant with the normative scalar deliverable of this standard. Laboratories submitting multi-point results shall also submit the canonical scalar ΔAUC at the reference operating point (§10.1.1) as the primary normative result.

**A2.3.2** Recommended labeling claim language derived from multi-point results follows the pattern of Vaishnav et al. (2020):

> "In a Type Test per ASTM FXXXX signal-amplitude sweep (Annex A2.1), [Device X] improved AUC over the no-MAR baseline by up to [value] across lesion contrasts from 4 to 20 HU."

**A2.3.3** Multi-point characterization is recommended but not required for submissions seeking substantial-equivalence determination based on non-degradation. Multi-point characterization is recommended for submissions seeking scoped quantitative improvement claims per the framework of Vaishnav et al. (2020).

### A2.4 Relationship to Normative Scalar

The scalar ΔAUC at the canonical operating point (§10.1.1) remains the normative result under this standard. Precision and bias statistics per §17 shall be computed from the scalar result only. Multi-point results are informational and their precision characterization is outside the scope of ASTM E691 as implemented in §17.1.

---

*End of ASTM WKXXXXX Revision 04*

*Revision 04 (2026-04-05, editorial pass 2026-04-18) updates the acquisition geometry from parallel-beam to fan-beam (SID=570 mm, SDD=1040 mm), relaxes the CHO equivalence tolerance to ±0.005 AUC, adds a screening mode (20 realizations), establishes a layered acceptance criteria framework with cross-reference to IEC 60601-2-44 Ed. 4 §203.6.7.101.1 (via its proposed compliance-statement amendment binding this TYPE TEST) and FDA guidance, aligns title and structure with ASTM F2119 conventions, documents subcommittee of jurisdiction (F04.15), clarifies type-test vs. clinical-validation scope boundary, adds F2119 complementarity and precedent clauses, adds preset-reporting and DICOM 2026b MAR Macro verification requirements, and introduces Annex A2 (Informational) specifying optional multi-point characterization per Vaishnav et al. (2020). All **[Rev 04]** markers indicate normative changes from Rev 03. Prior **[Rev 03]** markers are retained for traceability.*
