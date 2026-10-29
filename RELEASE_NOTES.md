# Release Notes

This file records per-version status, performance work, tooling additions, metadata-standard compliance, and regulatory positioning for the MAR ILS Dataset Generator and Evaluation Framework. For installation, usage, and dataset structure see [`README.md`](README.md).

---

## v9.0.0: October 2026 (in development)

### Redesign to the framework of Vaishnav et al. (2020)

- **Why:** an October 2026 audit of v7.0.0 found that lesion-present and lesion-absent realization *i* shared their noise, the lesion rotated with the artifact jitter, and scatter was added but not corrected. v8 (`generator_v8_0_0.py`, `run_cho_analysis_v8_0.py`, `pilot_gate_v8_0.py`) fixed these in the v7 design; under the fixes the v7 LI-MAR result (ΔAUC ≈ −0.23) does not hold and is withdrawn. Pre-registered pilot gates on the v7/v8 design then failed, and the method was rebuilt around the paper it cites.
- **Phantom and physics:** Vaishnav's head phantom with rotatable lumpy inset, 2D; 120 kV polychromatic fan beam (Mehranian 2013 geometry), SpekPy spectrum, NIST attenuation, Poisson noise, constant scatter (De Man 1999) and electronic noise (La Rivière 2005); analytic projections; FBP with the Ram-Lak kernel.
- **Observer:** dense difference-of-Gaussians CHO (Abbey and Barrett 2001), pooled covariance, no internal noise; antler-plot AUC∞ with separate 1000-image training and test pools.
- **Calibration:** lump scale and dose mapping fitted to the paper's published FBP curves (s = 0.0588, dose in counts per bin).
- **Controls:** the paper's two sinogram-inpainting methods, an improving control and a degrading control at dose 4 × 10⁶.
- **Validation:** pre-registered known-answer checks on five independent replicate datasets all pass (ΔAUC∞ of the improving control +0.215, +0.238, +0.254 at amplitudes 1.020, 1.025, 1.030; degrading control −0.061; replicate SD 0.004 to 0.008). The observer's AUC∞ reads 0.533 at chance; the null check is judged against that value.
- **Pending:** the v9 dataset writer (sinograms and DICOM), blinding tooling and the master dataset; the wiki update.

---

## v7.0.0 — April 2026 (superseded by v9.0.0)

### October 2026: Version Labels

v7.0.0 is the only current version label. The tag `v7.1.0-dicom2026b`, created on 2026-04-07 to mark the first CP-2575 implementation, was removed on 2026-10-24. Despite its higher number, it pointed to an April snapshot (commit `97af94e7`) that predates the v7.0.0 release. That commit remains in the history. The suffix `-dicom2026b` is no longer used, because every v7.0.0 dataset carries the CP-2575 macro.

### July 2026: ASTM Rev 05 Alignment

All code, documentation, and cross-references updated to reflect **ASTM WKXXXXX Revision 05** (2026-05-29). Rev 05 is an editorial-only revision — no numerical parameters changed. Key changes from Rev 04:

- §16 split into §16.1 (general information) and §16.2 (test results)
- §8.3 (AUC tolerance) renumbered to §8.4
- §1.5 (single-configuration constraint) renumbered to §1.7
- LI-MAR reclassified from "positive control" to "negative control" (designated control) per §17.11
- ASTM normative language harmonized: "PROHIBITED" → "shall not"
- E691 reference updated from E691-22 to E691-23

The locked metrology baseline (AUC_noMAR = 0.8294, N=40, σ=15) is unchanged.

### April 2026 Status: Metrology Baseline Locked

The framework has officially transitioned from parallel-beam research tiers to a **Normative Fan-Beam Geometry** (SID=570mm, SDD=1040mm).

#### ILS Reference Baseline ($N=40$)

The following values are established as the normative floor for ASTM WKXXXXX Rev 05:

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Baseline AUC_noMAR** | **0.8294** | **LOCKED** |
| **95% Confidence Interval** | [0.7612, 0.9025] | Verified |
| **Lesion Contrast** | 12.0 HU | Normative |
| **Integrity Check** | SHA-256 | Validated |

### Performance Breakthrough: Numba Acceleration

The v7.0.0 engine features a JIT-compiled **Batch Backprojector** optimized for Apple Silicon (M3) and modern x86_64 architectures, providing a **24x speedup** over standard NumPy implementations.

* **Throughput:** ~61s per realization (256 slices).
* **Efficiency:** Full $N=40$ dataset generation in <30 minutes on standard workstations.

### AI-Integrated Laboratory (MCP)

This repository includes **Model Context Protocol (MCP)** servers, allowing AI assistants to act as "Laboratory Hands" on your local workstation.

* **Data Inspector (`mcp_data_inspector.py`):** Automatically parses CHO JSON results and summarizes statistical significance.
* **Visualization (`mcp_visualization.py`):** Renders HDF5 sinogram slices and ROI comparisons directly in the AI chat interface.

### Validation & Audit Tools

To ensure inter-laboratory consistency, use the following in-memory physics auditors:

* **`plot_spectral_transparency.py`:** Visualizes the "Transparency Jump" between 60 keV and 140 keV.
* **`check_metal_overflow.py`:** Audits the normative **3000 HU metal-ROI hard-set** requirement.

### Deliverables & Repository Structure

| Path | Description |
| :--- | :--- |
| `generator_v7_0_0.py` | **Normative** fan-beam dataset generator (Rev 05). |
| `run_cho_analysis_v7_0.py` | **Normative** 2D CHO scoring tool (Rev 05). |
| `private/ASTM_MAR_Standard.md` | Draft standard text (Rev 05) — removed from public repo per ASTM IP policy |
| `docs_and_references/IEC_203_6_7_101_compliance_statement_proposal.md` | Post-publication Amendment proposal for §203.6.7.101.1 (deferred until ASTM FXXXX publishes) |
| `docs_and_references/FDA_guidance_framework.md` | Draft FDA guidance for acceptance criteria |
| `/algorithms` | Reference MAR implementations (iMAR, MBIR, Spectral). |
| `/legacy` | Archived v6.0.0 parallel-beam research framework. |

### Metadata Standard: DICOM 2026b Compliant

This is the **first reference implementation** of the **DICOM CP-2575 Metal Artifact Reduction Macro** (PS3.3 C.8.15.3.15), finalized in DICOM 2026b.

All DICOM files produced by the generator include:

| Tag | Keyword | Value |
| :--- | :--- | :--- |
| `(0018,9390)` | Metal Artifact Reduction Sequence | Present (1 item) |
| `(0018,9391)` | Metal Artifact Reduction Applied | `NO` |

The generator has written the macro natively since 2026-04-07. `patch_2026b_metadata.py` is a legacy tool that only retrofits datasets generated before then; it is not needed for v7.0.0 output.

The `noMAR` reconstructions are tagged `NO` by definition. The reference LI-MAR (`algorithms/v7/reference_li_mar_v7.py`) tags its output `YES` via the `mar_applied` argument of `mar_ils_core.dicom_utils.write_dicom_slice()`. Laboratories applying MAR algorithms must set `(0018,9391)` to `YES` and optionally populate `(0018,9392)` Metal Artifact Reduction Algorithm with a value from **CID 10036** (e.g., `MAR_IMAR`, `MAR_SPECTRAL`).

#### Verification (hex-tag method)

Because pydicom's bundled data dictionary does not yet include the 2026b additions, auditing requires hex-tag access:

```python
import pydicom
ds = pydicom.dcmread('astm_reference_dataset/noMAR_recon/LP/realization_001/slice_0129.dcm')

# Access the MAR Macro via hex tags
mar_seq = ds[0x00189390]                          # Metal Artifact Reduction Sequence
mar_applied = mar_seq.value[0][0x00189391].value   # → "NO"
print(f'(0018,9391) Metal Artifact Reduction Applied = {mar_applied}')
```

### Regulatory Framework

| Layer | Document | Role |
|-------|----------|------|
| **1 — What to record** | DICOM CP-2575 (2026b) | MAR metadata tags in DICOM |
| **2 — Must have / describe / record** | IEC 60601-2-44 Ed. 4 §203.6.7.101 (.1 method, .2 user info, .3 DICOM) | Regulatory mandate |
| **3 — How to measure (post-publication binding to §203.6.7.101.1)** | ASTM FXXXX (formerly WKXXXXX Rev 05) — Amendment submitted after ASTM FXXXX publishes | Quantitative ΔAUC TYPE TEST |
| **4 — Acceptance** | FDA guidance (proposed) | Non-degradation / superiority thresholds |
