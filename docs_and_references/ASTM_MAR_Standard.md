# ASTM WKXXXXX — Revision 05

**Designation:** XXXXX-XX
**Date:** 2026-05-29

---

## Standard Test Method for Evaluation of Metal Artifact Reduction (MAR) Performance in Tomographic Imaging Systems Using a Model Observer

*This test method is under the jurisdiction of ASTM Committee F04 on Medical and Surgical Materials and Devices and is the direct responsibility of Subcommittee F04.15 on Material Test Methods. Current edition approved XXX XX, XXXX. Published XXX XXXX. DOI:10.1520/XXXXX-XX.*

---

## 1. Scope

**1.1** This test method specifies a quantitative procedure for evaluating the performance of Metal Artifact Reduction (MAR) methods implemented in tomographic imaging systems.

**1.2** Specific dataset geometries, lesion specifications, observer definitions, and statistical procedures are included in Annex A1 to this test method.

**1.3** This test method includes the following sections:

| | Section |
|---|---|
| Scope | 1 |
| Referenced Documents | 2 |
| Terminology | 3 |
| Summary of Test Method | 4 |
| Significance and Use | 5 |
| Interferences | 6 |
| Apparatus | 7 |
| Reagents and Materials | 8 |
| Hazards | 9 |
| Sampling, Test Specimens, and Test Units | 10 |
| Preparation of Apparatus | 11 |
| Calibration and Standardization | 12 |
| Conditioning | 13 |
| Procedure | 14 |
| Calculations | 15 |
| Report | 16 |
| Precision and Bias | 17 |
| Keywords | 18 |
| Dataset Geometries, Lesion Specifications, Observer Definitions, and Statistical Procedures | Annex A1 |
| Multi-Point Characterization | Annex A2 |
| Background and Technical Basis | Appendix X1 |

**1.4** The method applies to tomographic imaging systems that produce reconstructed volumetric image data.

> NOTE — *Scope of the modality-independence claim.* The modality-independence asserted in this section is a property of the measurand (ΔAUC, computed by the CHO from reconstructed HU images), not of the test method as a whole. The MAR algorithm under test typically operates in the projection (sinogram) domain: the canonical dataset distributes line-integral sinograms (§8.2.1, §3.1.12) for that purpose, and the artifact physics and acquisition geometry (§A1.1(f) and (g), §A1.7.4) are CT-specific. Accordingly, §1.4 is to be read as scoped to the measurand, and the apparatus requirement for projection-domain input is stated in §7.1.

**1.5** The measurand is the scalar difference in area under the receiver operating characteristic curve (ΔAUC), derived from a signal-known-exactly, background-known-statistically (SKE/BKS) detection task using a fixed channelized Hotelling observer (CHO). The observer implementation, regions of interest, and statistical estimation procedures are specified in Annex A1.

**1.6** This method is modality-independent in the sense that the measurand (ΔAUC) is computed exclusively from reconstructed volumetric image data and does not depend on acquisition hardware, raw projection data, or reconstruction physics. The canonical dataset (§10.1.1) is defined in Hounsfield Unit (HU) space and is therefore directly applicable to systems that produce HU-calibrated reconstructed images, including but not limited to computed tomography (CT).

> NOTE — Application to modalities that do not produce HU-calibrated output requires a modality-specific canonical dataset established by a separate work item; results from such datasets shall not be reported as compliant with this standard unless the relevant modality annex has been approved.

**1.7** This standard defines a single canonical test configuration intended for type testing and conformity assessment. No additional configurations are permitted under this standard.

**1.8** *Units* — The values stated in SI units are to be regarded as the standard. No other units of measurement are included in this standard.

**1.9** *This standard does not purport to address all of the safety concerns, if any, associated with its use. It is the responsibility of the user of this standard to establish appropriate safety, health, and environmental practices and determine the applicability of regulatory limitations prior to use.*

**1.10** *This international standard was developed in accordance with internationally recognized principles on standardization established in the Decision on Principles for the Development of International Standards, Guides and Recommendations issued by the World Trade Organization Technical Barriers to Trade (TBT) Committee.*

**1.11** *Scope boundary* — This test method provides a controlled, reproducible measurement of MAR algorithmic behavior under a single canonical test configuration. It is a type test for method comparison and conformity assessment. Clinical validity of a MAR algorithm for any specific patient anatomy, implant geometry, or acquisition protocol is not established by this test alone and shall be supported by additional evidence as required by the incorporating authority or regulatory pathway. Labeling claims derived from this test method shall be scoped to the canonical test configuration (e.g., "In a Type Test per this standard, ΔAUC = X ± CI").

---

## 2. Referenced Documents

**2.1** *ASTM Standards:*

- E177 Practice for Use of the Terms Precision and Bias in ASTM Test Methods
- ASTM E691 Practice for Conducting an Interlaboratory Study to Determine the Precision of a Test Method
- F2119 Standard Test Method for Evaluation of MR Image Artifacts from Passive Implants

**2.2** *ISO Standards:*

- 5725-1 Accuracy (trueness and precision) of measurement methods and results – Part 1
- 5725-2 Basic method for determination of repeatability and reproducibility of a standard measurement method

**2.3** *IEC Standards:*

- 60601-2-44 Medical Electrical Equipment – Part 2-44: Particular requirements for the basic safety and essential performance of x-ray equipment for computed tomography

**2.4** *DICOM Standards:*

- PS3.3 Information Object Definitions (including the Metal Artifact Reduction Macro, CP-2575, 2026b)

**2.5** *Other References:*

- FIPS PUB 180-4 — Secure Hash Standard (SHA-256)

---

## 3. Terminology

**3.1** *Definitions:*

**3.1.1** *metal artifact, n* — image distortion in reconstructed tomographic data arising from the presence of a high-attenuation object, manifesting as streak, shadow, or halo patterns.

**3.1.2** *metal artifact reduction (MAR), n* — algorithmic processing applied within the imaging system pipeline to mitigate metal artifacts in reconstructed image data.

**3.1.3** *channelized Hotelling observer (CHO), n* — a linear model observer that evaluates signal detectability by projecting image data onto predefined spatial-frequency channel templates and applying the Hotelling discriminant to the resulting channel output vectors.

**3.1.4** *area under the ROC curve (AUC), n* — a scalar measure of binary detection performance ranging from 0.5 (chance-level discrimination) to 1.0 (perfect discrimination), estimated in this standard via the Mann–Whitney statistic.

**3.1.5** *ΔAUC, n* — the signed difference AUC_MAR – AUC_noMAR, where positive values indicate that MAR improved lesion detectability and negative values indicate degradation.

**3.1.6** *test result, n* — the mean ΔAUC calculated across all replicate paired image sets under specified conditions, reported to three decimal places.

**3.1.7** *signal-known-exactly (SKE), adj* — a detection task paradigm in which the signal (lesion) location, shape, size, and intensity are fixed and known to the observer.

**3.1.8** *background-known-statistically (BKS), adj* — a detection task paradigm in which the background statistics are known but individual background realizations vary.

**3.1.8.1** *Discussion* — In this standard, background variation arises from per-realization artifact jitter and independent Gaussian noise.

**3.1.9** *canonical test configuration, n* — the single, fully specified combination of phantom geometry, lesion parameters, dataset generation rules, and observer implementation defined in this standard. No other configuration is permitted.

**3.1.10** *realization, n* — one independent volumetric image set (lesion-present or lesion-absent) generated with a unique random seed, constituting an independent sample for CHO training and testing.

**3.1.11** *artifact jitter, n* — per-realization rotation of the artifact template by a specified random angle, providing background variability that prevents CHO overtraining on static artifact patterns.

**3.1.12** *line-integral sinogram (MAR-ready series), n* — the projection-domain representation of a realization, stored as fan-beam line integrals (neper) on the canonical acquisition geometry (§A1.1(f)), supplied so that the laboratory may apply its MAR algorithm and reconstruct the result into a DICOM image series for observer analysis. Each sinogram derives from the identical phantom, noise seed, and acquisition geometry as the corresponding noMAR reference reconstruction, so that the MAR enable/disable state is the only difference between the two image series ultimately scored (§4.2, §14).

**3.1.12.1** *Discussion* — The sinograms are distributed with the dataset (§8.2.1) in addition to the reconstructed image series.

---

## 4. Summary of Test Method

**4.1** A standardized, fully synthetic digital volumetric dataset representing a simplified anthropomorphic torso cross-section with a single cylindrical metallic rod is provided to participating laboratories. The dataset comprises, for each realization and condition, (i) a reconstructed reference image series with MAR disabled (the noMAR series), and (ii) a line-integral sinogram (the MAR-ready series, §3.1.12) on the canonical fan-beam geometry (§A1.1(f)).

**4.2** Two reconstructed image series are obtained for observer analysis: (a) the noMAR series, the provided reference reconstruction; and (b) the MAR series, produced by the laboratory applying its MAR algorithm to the provided MAR-ready sinograms and reconstructing the result, using identical parameters except for the MAR processing itself.

**4.3** Laboratories whose MAR operates in the image domain rather than the projection domain shall apply it to the provided noMAR reference series and document this in the report (§16). A two-dimensional CHO is applied to paired lesion-present and lesion-absent image volumes for each condition. AUC is computed for each condition via the Mann–Whitney statistic, and the test result is ΔAUC = AUC_MAR − AUC_noMAR.

---

## 5. Significance and Use

**5.1** This test method provides an objective and reproducible means of quantifying the impact of MAR algorithms on diagnostic task performance. Unlike methods that rely on physical dimensions of image artifacts, this method defines image quality by the ability of a model observer to perform a clinically relevant detection task.

**5.2** The body of the test method specifies the normative measurement: a single scalar ΔAUC computed from a standardized SKE/BKS detection task at one configuration with checksum-verified identical input data across all participating sites. This supports type testing, interlaboratory comparison per ASTM E691, and incorporation by normative reference into external performance standards.

**5.3** Annex A1 (normative) specifies the digital phantom, acquisition geometry, CHO parameters, and statistical procedures required to produce the ΔAUC measurement. Annex A2 (informational) describes optional multi-point characterization across signal amplitudes and dose levels for research and extended performance evaluation.

**5.4** This test method does not establish performance acceptance criteria. A positive ΔAUC indicates that MAR improves lesion detectability relative to the no-MAR condition; a negative ΔAUC indicates degradation. Both outcomes are scientifically valid results and shall be reported without suppression or sign correction.

**5.5** This test method is structured to support incorporation by normative reference into external performance standards. Acceptance criteria based on ΔAUC values are established by the incorporating authority, not by this standard.

---

## 6. Interferences

**6.1** Variability in image processing parameters between MAR-enabled and MAR-disabled conditions will inflate or deflate ΔAUC and invalidate the test result.

**6.2** Deviations from the specified Laguerre–Gauss channel templates, channel width parameter, Tikhonov regularization value, internal noise parameter (§A1.5.2(d)), or covariance estimation procedure will reduce interlaboratory reproducibility.

**6.3** Alteration of the standardized dataset, including resampling, interpolation, windowing before CHO input, or truncation of the HU range, invalidates the test.

**6.4** Use of floating-point arithmetic below 64-bit precision may introduce numerical bias exceeding the ±0.005 AUC equivalence tolerance specified in §8.4.

**6.5** Failure to verify the SHA-256 checksum prior to analysis invalidates the test result.

---

## 7. Apparatus

**7.1** Imaging system or processing platform capable of applying the MAR algorithm under test to the supplied line-integral sinograms (§8.2.1) and reconstructing the corrected result into a DICOM CT image series. (Image-domain MAR algorithms instead operate on the supplied reconstructed noMAR reference series; see §4.1.)

**7.2** Computational platform capable of executing the reference 2D CHO analysis software.

**7.3** Statistical software capable of Mann–Whitney AUC estimation and bootstrap confidence interval computation.

**7.4** SHA-256 checksum verification utility.

---

## 8. Reagents and Materials

**8.1** No reagents are required for this test method.

**8.2** Standardized synthetic dataset distributed with this standard, SHA-256 checksum verified per §11.1, consisting of two co-registered parts derived from the identical phantom, noise seeds, and acquisition geometry:

- (a) *Reconstructed reference image series (noMAR)* — 80 DICOM CT series (40 lesion-present, 40 lesion-absent), each 256 axial slices at 512 × 512 × 0.5 mm isotropic, reconstructed with MAR disabled per §A1.1(g).
- (b) *MAR-ready line-integral sinograms (§3.1.12)* — 80 sinograms (40 lesion-present, 40 lesion-absent) on the canonical fan-beam geometry (§A1.1(f)), supplied for the laboratory to apply MAR and reconstruct per §4.1.

**8.2.1** *Sinogram format* — Each MAR-ready sinogram is a single-precision (float32) array of fan-beam line integrals in neper, shape (256 slices × 720 projection angles × 512 detector elements), accompanied by the acquisition-geometry parameters of §A1.1(f) and the noise parameters of §A1.7 as metadata. The reference distribution stores each in HDF5 format at "sinograms/{LP,LA}/realization\_NNN.h5" (dataset line integrals, geometry and noise attributes per the distribution manifest).

**8.3** Laguerre–Gauss channel template definitions as specified in §A1.5.1 and provided in machine-readable form with the dataset distribution.

**8.4** Reference CHO implementation distributed with this standard. The reference implementation is the normative arbiter of correct CHO output. Alternative implementations shall demonstrate numerical equivalence within ±0.005 AUC on the supplied checksum-verified validation dataset before use.

**8.5** Checksum manifest file (SHA-256) distributed with the dataset.

**TABLE 1 Canonical Test Configuration**

| Parameter | Value | Notes |
|---|---|---|
| Matrix (x, y, z) | 512 × 512 × 256 voxels | Per §A1.1 |
| Voxel size | 0.5 mm isotropic | FOV = 256 mm |
| Background HU | 40 HU (soft tissue) | Uniform; noise added per realization |
| Metal rod diameter | 10 mm | Full z-extent, centered at (256, 256) |
| Metal HU | 3000 HU (fixed) | Not user-modifiable |
| Lesion disc diameter | 5 mm | Slice 128 only; single representative slice |
| Lesion offset | 5 mm beyond metal boundary, +x | Center at (281, 256) |
| Lesion contrast | ~12 HU physics contrast | Established in sinogram-domain via μ\_lesion; no post-FBP hard-set |
| Background noise | 30 HU Gaussian, IID | Applied only outside lesion and metal masks |
| Artifact model | Physics-based Poisson | 60 keV monochromatic; see A1.7 |
| Artifact jitter | Uniform (–15°, +15°) per realization | Independent per realization; see A1.7 |
| Realizations | 40 per condition | LP/LA × MAR/noMAR; 160 total |
| Acquisition geometry | Fan-beam, SID=570 mm, SDD=1040 mm | Equiangular curved detector; §A1.1(f) |
| Projection angles | 720 over (0°, 360°) | Full rotation, 0.5° spacing; §A1.1(f) |
| Reconstruction | Fan-beam FBP, Ram-Lak, cosine pre-weight | Distance-weighted backprojection; §A1.1(g) |
| Screening mode | 20 per condition (optional) | Pilot evaluation only; not reportable under §10.2 |

---

## 9. Hazards

**9.1** No ionizing radiation exposure is required when using the supplied digital dataset.

---

## 10. Sampling, Test Specimens, and Test Units

**10.1** The test unit consists of paired MAR enabled and MAR disabled volumetric image datasets derived from identical input data.

**10.1.1** *Canonical Test Configuration* — This test method defines a single canonical configuration. All parameters are normative and fully specified in this section and in Annex A1. No additional configurations, parameter substitutions, or alternative geometries are permitted.

**10.1.2** *Natural Reconstruction Rule* — The lesion signal shall be established exclusively in the sinogram domain via physics-based attenuation. No post-filtered backprojection (post-FBP) Hounsfield Unit (HU) replacement or "hard-set" overrides shall be applied to lesion voxels. The construction order within each realization slice shall be:

(1) *Forward Projection*: Generate fan-beam projections (SID=570mm, SDD=1040mm, 720 angles, 512 equiangular detectors) of the phantom background and, for lesion-present (LP) realizations, the lesion disc using the specified monochromatic attenuation coefficients at 60 keV;

(2) *Noise Modeling*: Apply the Poisson photon statistics noise model (scatter and Gaussian electronic noise) to the raw projection data;

(3) *Reconstruction*: Perform fan-beam FBP reconstruction using a Ram-Lak filter with cosine pre-weighting and distance-weighted backprojection, and apply the DC calibration offset to ensure background tissue consistency; and

(4) *Metal Restoration*: Restore voxels within the metallic object mask to 3000 HU as the final step, overriding all previously calculated values. The target effective lesion contrast of ~12 HU emerges as a natural consequence of the Radon inversion process. Violation of this construction order, or the application of any artificial pixel-replacement logic to the lesion ROI, produces a non-physical signal and invalidates the test result under this standard.

**10.1.3** For the canonical test configuration, a minimum of 40 statistically independent lesion-present and 40 lesion-absent image volumetric realizations shall be analyzed per condition. The minimum total is therefore 160 volumetric image sets (40 LP-noMAR, 40 LA-noMAR, 40 LP-MAR, 40 LA-MAR).

> NOTE — The statistical basis for N = 40 is provisional. On the reference characterization (N = 40, σ\_internal = 15), the sampling standard deviation of ΔAUC (≈ 0.038) and ΔAUC bias (≈ 0.004) meet the §17.11 targets. The pilot precision study (§17.11) remains definitive.

**TABLE 2 ΔAUC Test Results Meet §17.11 Targets**

| Quantity | Value | §17.11 target |
|---|---|---|
| Sampling SD (ΔAUC) | ≈ 0.038 | ≤ 0.05 ✓ |
| ΔAUC Bias | ≈ 0.004 | ≤ 0.02 ✓ |

**10.2** *Screening mode* — For pilot evaluation and feasibility assessment, a reduced configuration of 20 realizations per condition (80 total) may be used. Results obtained with fewer than 40 realizations per condition are informative only and shall not be reported as compliant with this standard. Screening-mode results shall be clearly labelled as such in any documentation.

**10.3** Statistical independence shall be achieved using the predefined set of independent volumetric realizations provided with the standardized dataset, each generated with a unique, non-overlapping random seed. Laboratories shall not generate additional noise realizations or alter the provided image volumes. Each full 3D volume (256 slices) constitutes one independent realization. Individual slices within a volume, or multiple slices extracted from the same volume, are not independent samples and shall not be treated as such in CHO training or testing.

---

## 11. Preparation of Apparatus

**11.1** Verify SHA-256 checksum of every file in the distributed dataset against the manifest provided with the standard. Any checksum mismatch disqualifies the dataset and invalidates any results derived from it.

**11.2** Validate the CHO implementation against the supplied reference validation dataset, confirming AUC agreement within ±0.005 per §8.4.

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

**14.1** *Dataset Processing*

**14.2** Verify dataset checksums per §11.1.

**14.3** Use the provided reconstructed noMAR reference series (40 LP and 40 LA volumes; §8.2(a)) as the MAR-disabled condition. Confirm its checksums per §11.1; do not re-reconstruct it.

**14.4** Produce the MAR-enabled condition by applying the MAR algorithm under test to the provided MAR-ready sinograms (40 LP and 40 LA; §8.2(b)) and reconstructing the corrected result, using reconstruction and postprocessing parameters identical to those of the provided noMAR reference series (§A1.1(g)). The MAR processing is the only permitted difference. (Image-domain MAR algorithms instead operate on the noMAR reference series per §4.1.)

**14.5** Verify that the processed output volumes match the expected matrix dimensions, voxel size, and bit depth specified in §A1.1 before proceeding.

**14.6** *CHO Analysis*

**14.7** Apply the 2D CHO (§A1.5) to Slice 128 only of the paired lesion-present and lesion-absent image volumes for MAR and noMAR conditions, using the fixed ROI specified in §A1.5.4.

**14.8** Estimate covariance from lesion-absent volumes only, using the pooled sample covariance across all 40 LA realizations for the relevant condition.

**14.9** Compute AUC for each condition via the Mann–Whitney statistic (§A1.6(b)).

**14.10** Calculate ΔAUC = AUC_MAR – AUC_noMAR.

**14.11** Compute bootstrap 95% confidence intervals per §A1.6(c).

**14.12** Document all system settings, software versions, and any deviations from this procedure.

**14.13** *Multi-preset algorithms* — Where the MAR algorithm under test provides user-selectable strength or operating-point presets (e.g., "low / medium / high"), each preset shall be evaluated independently as a distinct test run. The preset identifier shall be recorded in the test report per §16.1 and ΔAUC reported per preset per §16.2. A single test campaign may include results for multiple presets of the same algorithm.

---

## 15. Calculations

**15.1** The following calculations shall be performed and reported:

**15.2** ΔAUC = AUC_MAR – AUC_noMAR

**15.3** The reported test result shall be the mean ΔAUC across all replicates. ΔAUC shall be reported to three decimal places with sign.

**15.4** Standard deviation of ΔAUC across replicates shall be reported.

**15.5** Bootstrap 95% confidence intervals shall be reported.

**15.6** The AUC estimation bias (difference between resubstitution and hold-out estimates) shall be reported per §A1.6(d).

**15.7** Negative ΔAUC values shall be reported without modification. A negative result indicates that the MAR algorithm degraded lesion detectability relative to the no-MAR condition and is a scientifically valid and reportable outcome.

---

## 16. Report

**16.1** A complete report shall include the following general information:

- (a) Test performed per ASTM FXXXX-XX (designation and revision)
- (b) Date(s) of test performance
- (c) Name of person accepting technical responsibility for the test report
- (d) System identification (manufacturer, model, software version)
- (e) MAR algorithm name and version
- (f) All image processing parameters applied during MAR-enabled and MAR-disabled conditions, with explicit confirmation that only the MAR state differed
- (g) Dataset version identifier and SHA-256 manifest verification result
- (h) CHO software version and validation AUC result (§12.1)
- (i) Computational environment (hardware platform, OS, CPU/GPU, numerical library versions)
- (j) Number of LP and LA replicates analyzed per condition
- (k) Internal noise parameter σ\_internal used (normative value: 15; see §A1.5.2(d))

**16.2** A complete report shall include the following test results for the tests performed:

- (a) Mean ΔAUC (three decimal places, with sign)
- (b) Standard deviation of ΔAUC
- (c) Bootstrap 95% confidence interval
- (d) AUC estimation bias (resubstitution minus hold-out)
- (e) Individual AUC\_MAR and AUC\_noMAR values
- (f) Per-realization held-out test statistics *t* for MAR and noMAR conditions as well as LP and LA classes, reported as four vectors of length N
- (g) Verification of the DICOM Metal Artifact Reduction Macro (PS3.3 C.8.15.3.15, CP-2575, 2026b). The report shall confirm that the MAR enable/disable state is recorded in the Metal Artifact Reduction Applied attribute (0018,9391) of the Metal Artifact Reduction Sequence (0018,9390) on every output DICOM series, and that the recorded value is consistent with the algorithm configuration under test.
- (h) Any deviations from this test method

---

## 17. Precision and Bias

**17.1** *Precision* — The precision of this test method shall be determined by an interlaboratory study conducted in accordance with ASTM E691 and ISO 5725-2.

**17.2** The following precision statistics shall be determined:

**17.3** Repeatability standard deviation (s\_r)

**17.4** Reproducibility standard deviation (s\_R)

**17.5** Repeatability limit (r = 2.8 × s\_r)

**17.6** Reproducibility limit (R = 2.8 × s\_R)

**17.7** The precision statement shall be incorporated into this standard following completion of the full interlaboratory study.

**17.8** The full interlaboratory study shall include a minimum of 6 laboratories in accordance with ASTM E691 recommendations, each processing the identical checksum-verified dataset.

**17.9** The precision statement shall report s\_r, s\_R, r, and R values expressed in units of ΔAUC.

**17.10** Pilot precision data requirement: Before this standard proceeds to first ASTM ballot, the sponsoring subcommittee shall provide pilot precision data from a minimum of 3 laboratories demonstrating that:

(a) the test pipeline resolves a known, signed ΔAUC of the expected sign and magnitude (i.e., across laboratories the measured ΔAUC for a designated control algorithm agrees in sign and is statistically distinguishable from zero) (a negative control such as the LI-MAR reference designated in the note below satisfies this; a positive control may be used additionally but is not required);

(b) the within-laboratory standard deviation of ΔAUC does not exceed 0.05 AUC units; and

(c) the bias of the ΔAUC test result (resubstitution minus hold-out) does not exceed 0.02 AUC units at N = 40 realizations.

> NOTE — Items (a)-(c) gate ΔAUC, not the per-condition AUC. The per-condition resubstitution−hold-out optimism of a 10-channel plug-in Hotelling observer is intrinsically large (≈ 0.12 at N = 40, reported as a diagnostic per §16.2.4) but is common-mode and cancels in ΔAUC. Gating the per-condition AUC bias at 0.02 would be unachievable for reasons unrelated to the accuracy of the reported test result.

**17.11** Control algorithms for the pilot:

(a) *Designated control* — the LI-MAR negative control distributed with the dataset tooling. On the canonical configuration (N = 40, σ\_internal = 15) it yields ΔAUC ≈ −0.23 (95 % CI excluding zero). The pilot shall use this control to verify that each laboratory's pipeline reproduces the known signed, significant ΔAUC within tolerance.

(b) *Positive control* — optional. A MAR algorithm known to improve detectability (ΔAUC > 0) under the canonical configuration has not yet been validated. Identifying a positive control is left to the subcommittee and participating laboratories.

**17.12** *Bias* — Bias in the absolute sense cannot be determined because no accepted reference value for true lesion detectability exists independently of the measurement method. Systematic effects identified during interlaboratory evaluation shall be reported. Sources of systematic effect include: CHO implementation differences, floating-point accumulation errors, and covariance regularization choices.

---

## 18. Keywords

**18.1** metal artifact; metal artifact reduction; MAR; channelized Hotelling observer; model observer; ROC curve; AUC; ΔAUC; lesion detectability; interlaboratory study; precision and bias; tomographic imaging; computed tomography; signal-known-exactly; Laguerre–Gauss channels; reproducibility.

---

## ANNEX

### (Normative)

## A1. Dataset Geometries, Lesion Specifications, Observer Definitions, and Statistical Procedures

**A1.1** *Volume Geometry:*

- (a) Matrix: 512 × 512 × 256 voxels (x, y, z)
- (b) Voxel size: 0.5 mm isotropic in all three dimensions
- (c) Field of view: 256 mm (512 voxels × 0.5 mm / voxel)
- (d) HU range encoded as signed 16-bit integer (INT16) in DICOM pixel data
- (e) Rescale slope: 1.0; Rescale intercept: 0.0 (HU = stored value)
- (f) Acquisition geometry: 2D fan-beam. Source-to-isocenter distance (SID) = 570 mm. Source-to-detector distance (SDD) = 1040 mm. Equiangular curved detector array, 512 elements, angular pitch Δγ = 2·arcsin(FOV/(2·SID))/N\_det ≈ 0.0507°. Full 360° rotation, 720 equispaced projection angles (0.5° angular spacing). Maximum fan half-angle γ\_max = arcsin(128/570) ≈ 12.97°.
- (g) Reconstruction: fan-beam filtered backprojection (FBP) with cosine pre-weighting, Ram-Lak (ramp) filter, and (SID/L)² distance-weighted backprojection, where L is the source-to-pixel distance. Scaling factor: π / N\_angles / (SID × Δγ). This follows the equiangular fan-beam FBP formulation in Ref (4), §3.4.

**A1.2** *Phantom Cross-Section Geometry:*

The phantom cross-section represents a simplified torso geometry consisting of an elliptical body region, a cylindrical metal rod, and (when present) a lesion disc. The metal rod is a uniform cylinder spanning the full z-extent of the volume to ensure consistent artifact generation. The lesion is a single disc confined to Slice 128 only.

- (a) Body ellipse semi-axis x: 85 mm (170 voxels at 0.5 mm/voxel)
- (b) Body ellipse semi-axis y: 60 mm (120 voxels at 0.5 mm/voxel)
- (c) Body center: (x, y) = (256, 256) voxels (image center)
- (d) Body interior HU: 40 HU (soft tissue equivalent)
- (e) Exterior (outside body ellipse) HU: −1000 HU (air)
- (f) Gaussian noise (σ = 30 HU) applied to body interior only, excluding lesion and metal masks, independently per realization per §A1.7

**A1.3** *Metal Object Specification:*

- (a) Geometry: right circular cylinder, full z-extent
- (b) Diameter: 10 mm (radius = 5 mm = 10 voxels at 0.5 mm/voxel)
- (c) Center: (x\_m, y\_m) = (256, 256) voxels
- (d) Fixed HU value: 3000 HU
- (e) Metal boundary x-coordinate (positive x face): x\_m + r = 256 + 10 = 266 voxels
- (f) Metal voxels shall be restored to 3000 HU as the final step in slice construction, overriding noise and artifact

**A1.4** *Lesion Specification:*

- (a) Geometry: right circular disc, slice 128 only
- (b) Diameter: 5 mm (radius = 2.5 mm = 5 voxels at 0.5 mm/voxel)
- (c) Lesion center x-coordinate: x\_0 = x\_m + r\_m + 5 mm + r\_lesion = 256 + 10 + 10 + 5 = 281 voxels
- (d) Lesion center y-coordinate: y\_0 = 256 voxels
- (e) Fixed HU value (SKE): ~12 HU effective FBP contrast (via MU\_LESION\_CM)
- (f) Implementation: Lesion contrast is established in the sinogram domain only. No post-FBP HU replacement shall be applied to voxels within the lesion mask.
- (g) Signal contrast: ~12 HU effective FBP contrast above soft tissue background
- (h) Pre-observer per-voxel CNR: ~0.4 (noise-limited task)

**A1.5** *Channelized Hotelling Observer (CHO) Specification:*

**A1.5.1** *Channel Type and Parameters:*

- (a) Channel type: Laguerre–Gauss (LG) radial channels
- (b) Number of channels: 10
- (c) Channel order n: 0 through 9 (one channel per order)
- (d) Channel width parameter a: 1.5 × r\_lesion = 1.5 × 5 voxels = 7.5 voxels (3.75 mm). This value is fixed and not user-modifiable.
- (e) Channel normalization: each channel u\_n(r) is L₂-normalized over the two-dimensional ROI domain such that ‖u\_n‖₂ = 1.
- (f) The n-th 2D Laguerre–Gauss channel is defined as: u\_n(r) = L\_n(2πr²/a²) × exp(−πr²/a²), where L\_n is the n-th Laguerre polynomial and r is the radial distance from the lesion center in voxels. The channel templates are strictly two-dimensional and shall be applied to Slice 128 only. Three-dimensional extension shall not be performed.

**A1.5.2** *Covariance Estimation:*

- (a) The CHO covariance matrix K shall be estimated from lesion-absent (LA) volumes exclusively.
- (b) Covariance shall be estimated by pooling channel output vectors across all 40 LA realizations of the relevant condition (MAR or noMAR). Separate covariance matrices shall be estimated for the MAR and noMAR conditions.
- (c) Regularization: Tikhonov regularization shall be applied as K\_reg = K + λI, where λ = 0.01 × trace(K) / p, p = number of channels = 10. This normalization ensures λ scales with the data and is invariant to HU units. λ is fixed by this formula and shall not be user-modified.
- (d) *Internal Observer Noise*: An internal noise variance shall be added to the diagonal of the covariance matrix before inversion: K\_total = K\_external + σ\_internal² × I, where σ\_internal = 15 HU. The total regularized matrix used for Hotelling template estimation shall be: K\_final = K\_external + σ\_internal² · I + λI. This parameter is normative and shall not be modified.

> NOTE — The channel outputs are in HU and K\_external is in HU² because the channel templates are dimensionless and L₂-normalized (§A1.5.1(e)); σ\_internal = 15 is therefore in the same HU units, and its numerical meaning is fixed only because the channel definition, normalization, and ROI (§A1.5.1, §A1.5.4) are fixed. Any deviation in channel scaling convention changes the meaning of "15" and shall not be made.

**A1.5.3** *Observer Dimensionality:*

The CHO shall operate on a two-dimensional (2D) region of interest from Slice 128 only. Three-dimensional volumetric integration across z shall not be performed.

**A1.5.4** *Region of Interest (ROI):*

- (a) ROI x-extent: lesion center x ± 60 voxels = 221 to 341 (121 voxels)
- (b) ROI y-extent: lesion center y ± 60 voxels = 196 to 316 (121 voxels)
- (c) ROI z-extent: Slice 128 only
- (d) ROI center coordinates are fixed per this annex and shall not be adjusted between conditions or realizations.
- (e) The same ROI shall be used for MAR-enabled and MAR-disabled conditions.

**A1.5.5** *Channel Features, Hotelling Template, and Test Statistic:* This subsection specifies the central CHO computation normatively. Let U denote the 10 × P matrix whose rows are the L₂-normalized Laguerre–Gauss channel templates of §A1.5.1, P = ROI\_SIZE² = 121² = 14,641.

- (a) *Channel feature vector* — For a single ROI extracted from Slice 128 (§A1.5.4) and vectorized as v ∈ ℝ^P in Hounsfield Units, the channel feature vector is g = U v ∈ ℝ¹⁰. No other channel scaling convention is permitted.

> NOTE — Because the channel templates are dimensionless and L₂-normalized (§A1.5.1(e)), the channel outputs g are in HU; consequently, the channel covariance K (§A1.5.2) is in HU² and the internal-noise term σ\_internal²·I (§A1.5.2(d)) is added in HU² with σ\_internal in HU. With the fixed channel definition, normalization, and ROI of §A1.5.1 and §A1.5.4, the numerical scale of g, and, therefore the meaning of σ\_internal = 15, is fully determined.

- (b) *Hotelling template* — The CHO template is w = K\_final⁻¹ (ḡ\_LP − ḡ\_LA), where ḡ\_LP and ḡ\_LA are the sample means of the channel feature vectors over the training lesion-present and lesion-absent realizations respectively, and K\_final is the LA-only regularized covariance of §A1.5.2 (K\_final = K\_external + σ\_internal²·I + λI). The signal template is the estimated difference of class sample means; the known-exactly signal shall not be substituted. The inverse shall be realized by solving the linear system K\_final w = (ḡ\_LP − ḡ\_LA) rather than forming K\_final⁻¹ explicitly.

- (c) *Test statistic* — For any feature vector g, the scalar CHO decision variable (test statistic) is t = wᵀ g. Larger t indicates greater evidence for lesion presence. AUC (§A1.6) is computed from the t values of lesion-present versus lesion-absent test realizations under the training/scoring protocol of §A1.6(a).

---

### A1.6 Statistical Procedures

(a) *Primary protocol — leave-one-out (LOO) hold-out.* The unit of cross-validation and resampling throughout this section is the realization index i = 1 … N. Lesion-present realization i and lesion-absent realization i constitute a single fold i (N folds total, not 2N); they are always withheld, resampled, and scored together by index.

(a.1) For each fold i, estimate the Hotelling template w⁽ⁱ⁾ per §A1.5.5(b) from the N−1 LP and N−1 LA realizations with index ≠ i.

(a.2) Record the held-out test statistics t\_LP,i = w⁽ⁱ⁾ᵀ g\_LP,i and t\_LA,i = w⁽ⁱ⁾ᵀ g\_LA,i for the withheld realization.

(a.3) The reported AUC for a condition is the Mann–Whitney statistic (§A1.6(b)) computed over the N held-out LP versus N held-out LA test statistics. This LOO hold-out AUC is the normative test result.

(a.4) A resubstitution AUC (template estimated from, and scored on, all N realizations) shall be computed only for the bias estimate of §A1.6(d) and shall not be reported as the test result.

> NOTE — An implementation that treats the 40 LP and 40 LA realizations as 80 independent samples, rather than pairing them by index, does not conform to this standard.

(b) Ties in the Mann–Whitney statistic shall be handled using mid-rank assignment.

(c) *Bootstrap confidence intervals.* All bootstrap procedures resample the fixed per-realization held-out test statistics produced in §A1.6(a); the Hotelling template shall not be re-estimated within bootstrap replicates. Use 1000 resamples in all cases.

(c.1) *Single-condition CI.* For each replicate, draw one set of N realization indices uniformly with replacement and apply the same index set to both the held-out LP and held-out LA test-statistic vectors of the condition (resampling by realization index, preserving the LP/LA fold pairing); recompute the Mann–Whitney AUC. The 2.5th and 97.5th percentiles of the 1000 bootstrap AUCs define the 95% CI for that condition's AUC.

(c.2) *Paired ΔAUC CI.* ΔAUC is a paired quantity: AUC\_MAR and AUC\_noMAR derive from the same underlying realizations and are correlated. For each replicate, draw one set of N realization indices and apply it jointly to both conditions (i.e., use the identical resampled indices for the noMAR and MAR held-out test statistics) then compute ΔAUC\* = AUC\_MAR\* − AUC\_noMAR\* on that common resample. The 2.5th and 97.5th percentiles of the 1000 ΔAUC\* values define the 95% CI for ΔAUC. Independent (unpaired) resampling of the two conditions shall not be used, as it ignores the cross-condition correlation and inflates the interval.

(d) AUC estimation bias shall be quantified using resubstitution and hold-out CHO training / testing strategies per Ref (3). The bias estimate is defined as b = AUC\_resubstitution – AUC\_hold-out. Both estimates and the bias shall be reported.

(e) Minimum 64-bit (double-precision) floating-point arithmetic shall be used throughout CHO computation, including channel projection, covariance estimation, matrix inversion, and test statistic computation.

---

### A1.7 Background Variability Mechanism

**A1.7.1** The canonical dataset provides background variability via two independent mechanisms: (i) per-realization artifact jitter, and (ii) independent Gaussian noise realizations.

**A1.7.2** *Artifact jitter* — For each realization i (i = 1 to 40), a jitter angle θ\_i shall be drawn from a uniform distribution on [−15°, +15°]. The base artifact template (§A1.7.4) shall be rotated by θ\_i about the image center using bilinear interpolation. The rotated template shall be clipped to the body mask and zeroed within the metal mask before addition.

**A1.7.3** *Noise realizations* — Independent Gaussian noise (σ = 30 HU, µ = 0) is added to each realization using a unique random seed. For realization i, the seed shall be BASE\_SEED + i where BASE\_SEED is a fixed integer specified in the dataset metadata. Noise is applied only within the body mask and outside the lesion and metal masks.

**A1.7.4** *Artifact template* — The base artifact template is a 2D HU field provided as part of the reference dataset. The template was constructed by: (1) forward projecting a phantom containing the body background and metal rod using the fan-beam geometry specified in §A1.2; (2) simulating photon starvation by replacing sinogram values above the 99th percentile of non-zero values with 2% of the 50th percentile; (3) reconstructing both the original and corrupted sinograms using fan-beam FBP; (4) computing the difference (corrupted minus original); (5) zeroing the template within the metal and outside the body mask; (6) scaling so that the maximum absolute value within the body (excluding metal) is 400 HU. This construction procedure is documented for transparency; laboratories shall use the pre-computed template provided in the dataset (§A1.7.5).

**A1.7.5** The artifact template, jitter angles, and noise seeds for all 40 realizations are fixed in the reference dataset. Laboratories shall use the provided dataset and shall not regenerate these values.

---

### A1.8 Reproducibility Requirements

The five physics-generation constants (energy, μ values, I₀, scatter/σ\_e, DC offset) are fixed in the distributed dataset and are listed here for completeness and for any party reproducing the dataset. Laboratories use the provided checksum-verified dataset and do not regenerate it (§A1.7.5). These values cannot be altered in normal use.

The result shall not be reported as compliant with this standard if any of the listed parameters have been modified.

**TABLE 3 Reproducibility Requirements^A**

| Parameter | Specified Value | Location |
|---|---|---|
| Voxel size | 0.5 mm isotropic | §A1.1 |
| Matrix dimensions | 512 × 512 × 256 | §A1.1 |
| LG channel type | Laguerre–Gauss, n = 0..9 | §A1.5.1(a-c) |
| Channel width parameter a | 7.5 voxels (1.5 × r\_lesion) | §A1.5.1(d) |
| Channel normalization | L₂-normalized | §A1.5.1(e) |
| Internal noise σ | 15 (Normative) | §A1.5.2(d) |
| Tikhonov λ | 0.01 × trace(K)/p | §A1.5.2(c) |
| Covariance source | LA volumes only, pooled | §A1.5.2(a,b) |
| Observer dimensionality | 2D, Slice 128 only | §A1.5.3 |
| ROI dimensions | 121 × 121 voxels (2D) | §A1.5.4 |
| ROI center | (281, 256) voxels in xy | §A1.4(c,d) |
| Lesion contrast | ~12 HU (Sinogram-domain) | §A1.4(e,f) |
| Lesion geometry | Single disc, Slice 128 only | §A1.4(a) |
| Metal HU | 3000 HU (restored last) | §A1.3(d,f) |
| Artifact peak HU | 400 HU | §A1.7.4 |
| Artifact jitter range | Uniform [–15°, +15°] | §A1.7.2 |
| Realizations per condition | 40 LP + 40 LA (80 per dataset; lab produces 80 MAR + 80 noMAR = 160 processed volumes) | §10.2 |
| Floating-point precision | 64-bit minimum (double) | §A1.6(e) |
| Baseline AUC\_noMAR | 0.8294, CI [0.7612, 0.9025] (N=40, σ=15, fan beam, 2026-04-07) | §X1.3 |
| SHA-256 checksum | Must match manifest | §8.5, 11.1 |
| Acquisition geometry | Fan-beam, SID=570 mm, SDD=1040 mm | §A1.1(f) |
| Projection angles | 720 over (0°, 360°) | §A1.1(f) |
| Reconstruction | Fan-beam FBP, Ram-Lak, cosine pre-weight, (SID/L)² | §A1.1(g) |
| CHO equivalence tolerance | ±0.005 AUC | §8.4 |
| Monochromatic Energy | 60 keV | §10.1.1, §A1.7.4 |
| μ\_soft tissue / μ\_iron | 0.2059 cm⁻¹ / 2.408 cm⁻¹ (at 60 keV) | §X1.7, §A1.4(e) |
| Calibrated photon flux I₀ | 310,853 (calibrated to 30 HU soft-tissue noise) | §X1.3, §A2.2 |
| Scatter fraction / Electronic noise σ\_e | 0.05 / 5.0 counts | §10.1.1, §A1.7 |
| DC calibration offset | −0.029 cm⁻¹ (≈ −141 HU) | §10.1.2, §A1.1(g) |

^A Modification of any parameter invalidates the test result under this standard.

---

## ANNEX

### (Informational)

## A2. Multi-Point Characterization

**A2.1** *Signal-amplitude sweep* — To characterize MAR performance across a range of lesion contrasts, laboratories may repeat the normative procedure at signal amplitudes of 4, 8, 12, 16, and 20 HU by substituting the appropriate MU\_LESION\_CM value. The canonical 12 HU configuration (§A1.4) remains the normative test point.

**A2.2** *Dose-level sweep* — To characterize MAR performance across dose levels, laboratories may repeat the normative procedure at I₀ scaling factors of {0.5, 0.71, 1.0, 1.41, 2.0} relative to the calibrated I₀. The canonical I₀ = 310,853 (scaling factor 1.0) remains the normative test point.

**A2.3** Multi-point results shall be reported separately from the normative ΔAUC and clearly labeled as informational. They do not replace the normative scalar test result.

**A2.4** Multi-point characterization does not affect the precision statement (§17). Each operating point would require independent precision evaluation per ASTM E691 if normative status were sought.

---

## APPENDIX

### (Nonmandatory Information)

## X1. BACKGROUND AND TECHNICAL BASIS

**X1.1** Metal artifact reduction (MAR) methods are increasingly incorporated into computed tomography and other tomographic imaging systems. These methods may either improve or degrade lesion detectability and quantitative accuracy depending on imaging parameters and algorithm design. Current standards do not define an objective, task-based, reproducible, interlaboratory test method for quantifying this effect.

**X1.2** Metal artifacts in CT arise primarily from beam hardening, photon starvation, and partial volume effects in the presence of highly attenuating objects. MAR algorithms mitigate these artifacts but may alter image statistics in ways that affect clinically relevant tasks, including low-contrast lesion detection. The net effect of MAR on diagnostic utility is not captured by artifact severity metrics alone.

**X1.3** The framework underlying this standard was first described in Ref (2), which demonstrated that a model observer-based approach could objectively measure MAR effects on low-contrast detectability. The channelized Hotelling observer (CHO) is a linear model observer that has been extensively validated against human performance in CT detection tasks. The area under the ROC curve (AUC), estimated via the Mann-Whitney statistic, provides a scalar figure of merit that is interpretable, reproducible, and independent of decision threshold. Building upon this framework, the reference baseline AUC\_noMAR = 0.8294 (LOO hold-out; 95% CI [0.7612, 0.9025], N=40, σ=15) was computed using fan-beam geometry (SID=570mm, SDD=1040mm) with sinogram-domain physics contrast (~12 HU). This result confirms that the task is noise-limited and metrologically sensitive to MAR quality, consistent with human-correlated task-based assessment paradigms.

**X1.4** The use of a deterministic digital dataset eliminates physical phantom fabrication, scanner time, and shipping logistics from interlaboratory studies. SHA-256 checksums ensure bitwise identity of the dataset across laboratories. The only source of interlaboratory variability under this standard is therefore the CHO implementation and the MAR algorithm under test, which is the intended behavior.

**X1.5** The canonical dataset provides background variability through per-realization rotation of the artifact template (§A1.7). This mechanism prevents the CHO from exploiting static artifact patterns and ensures that the observer's performance reflects genuine signal detectability.

**X1.6** *Scope and precedent* — The single canonical configuration approach follows established precedent in medical imaging type testing, including ASTM F2119 (Evaluation of MR Image Artifacts from Passive Implants), IEC 62220-1 (detector DQE), AAPM TG-233 (CT image quality), and the ACR CT Accreditation Program. A standardized, narrow measurement enables interlaboratory precision and bias characterization per ASTM E691 and ISO 5725; clinical-practice claims are supported by separate, device-specific evidence as required by the relevant regulatory authority. This test method is complementary to ASTM F2119, which characterizes the physical extent of image artifacts produced by a passive implant under standardized MR scanning conditions. The present method quantifies the observer-based task-detectability impact of an algorithmic countermeasure (MAR) applied within the imaging system. The two methods address non-overlapping axes: modality (MR / CT) × object of measurement (physical artifact extent / algorithmic task impact). Together, they cover the passive-implant / active-algorithm quadrants of artifact assessment in F04's jurisdiction.

**X1.7** *Metal-material rationale* — The canonical metal material (iron, μ = 2.408 cm⁻¹ at 60 keV) is selected as a representative high-Z attenuator sitting within the range of clinical implant materials (titanium 1.41, stainless steel 2.3, cobalt-chromium 2.7 cm⁻¹ at 60 keV). The 10 mm diameter and centered-on-axis geometry are chosen to produce a defined, reproducible beam-hardening and photon-starvation artifact signature, not to simulate any specific clinical implant. The canonical metal provides a controlled, reproducible attenuation challenge against which every algorithm under test is evaluated.

---

## SUMMARY OF CHANGES

**Reproducibility formalization and dataset-deliverable corrections. [Rev 05] (May 29, 2026).**

(1) §A1.5.5 added — channel feature vector g = U·v, estimated Hotelling template w = K\_final⁻¹(ḡ\_LP − ḡ\_LA), and test statistic t = wᵀg, with the channel-output (HU) scale pinned.
(2) §A1.6(a) rewritten to designate the LOO hold-out AUC as the normative test result and to fix the realization-index pairing (N folds, not 2N).
(3) §A1.6(c) rewritten to specify the single-condition and paired ΔAUC bootstraps over the fixed held-out statistics (template not re-estimated within replicates).
(4) §A1.5.2(d) clarifies σ\_internal units.
(5) §3.1.12, §4.1, §7.1, §8.2–8.2.1, and §14.1 add the line-integral sinogram ("MAR-ready series") deliverable and align the procedure with the projection-domain workflow.
(6) §A1.8 records the dataset physics-generation constants.
(7) Three previously-open items resolved: §1.4 scoped to the measurand; §10.2 N = 40 statistical basis recorded; §17.11(a) revised to use the LI-MAR negative control, with a positive control optional.

**Clinical geometry alignment and test-method infrastructure expansion. [Rev 04] (Apr. 5, 2026).**

(1) Acquisition geometry changed from parallel-beam to fan-beam (SID=570 mm, SDD=1040 mm, 720 angles over 360°).
(2) FBP changed to fan-beam cosine-weighted distance-weighted backprojection; CHO equivalence tolerance relaxed from ±0.001 to ±0.005 AUC; screening mode (20 realizations) added.
(3) Acceptance-criteria cross-reference added (§5.5); baseline AUC\_noMAR established at 0.8294.
(4) Scope-boundary clause §1.9, §X1.6 (Scope and Precedent), and §X1.7 (Metal-Material Rationale) added.
(5) §14.2.8 preset-reporting and §16.2.7 DICOM 2026b MAR Macro verification added.
(6) §A1.5.3 clarified.
(7) Annex A2 added.

**Sinogram-domain physics, 2D observer, and normative regularization. [Rev 03] (Mar. 14, 2026).**

(1) Observer dimensionality changed from 3D to 2D (Slice 128 only).
(2) Lesion geometry changed to single-slice disc.
(3) Lesion contrast changed to ~12 HU sinogram-domain physics.
(4) minimum realizations increased to 40 per condition; internal observer noise (σ = 15) added as normative.

---

## BIBLIOGRAPHY

(1) Barrett, H.H., and Myers, K.J., *Foundations of Image Science*, first edition, Wiley-Interscience, Hoboken, NJ, 2004.

(2) Vaishnav, J.Y., et al., "CT metal artifact reduction algorithms: Toward a framework for objective performance assessment," *Med. Phys.*, Vol. 47, No. 8, August 2020, pp. 3344–3355.

(3) Wunderlich, A., and Noo, F., "On Efficient Assessment of Image-Quality Metrics Based on Linear Model Observers," *IEEE Trans. Med. Imaging*, Vol. 34, No. 2, February 2015, pp. 508–519.

(4) Kak, A.C., and Slaney, M., *Principles of Computerized Tomographic Imaging*, IEEE Press, New York, 1988.
