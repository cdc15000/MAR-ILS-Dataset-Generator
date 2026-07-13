# Sinogram Ingestion and Reconstruction Output Specification

**ASTM WKXXXXX Rev 05 — MAR Interlaboratory Study**\
**Version:** 1.0\
**Date:** 2026-07-13

---

## 1. Purpose

This document specifies the data interface for the MAR ILS. Participating laboratories receive a dataset of HDF5 sinogram files and must return DICOM image volumes produced by their MAR reconstruction pipeline. The interface is **language-agnostic** — implementations in C, C++, CUDA, MATLAB, Python, or any other environment are equally valid provided the input is read and output is written according to this specification.

---

## 2. Dataset Layout

The dataset delivered to each laboratory has this structure:

```
astm_reference_dataset/
    sinograms/
        LP/                          ← Lesion-Present condition
            realization_001.h5
            realization_002.h5
            ...
            realization_040.h5
        LA/                          ← Lesion-Absent condition
            realization_001.h5
            ...
            realization_040.h5
    noMAR_recon/                     ← Reference FBP (provided for validation)
        LP/  realization_001/ ... realization_040/
        LA/  realization_001/ ... realization_040/
    checksums_sha256.txt
    generator_provenance.json
```

Each condition contains **40 realizations** (screening studies may use 20). LP and LA sinograms are paired by realization number and differ only in whether a lesion signal is present in the sinogram domain.

---

## 3. HDF5 Sinogram Format

Each `realization_NNN.h5` file contains three groups:

### 3.1 Dataset: `/line_integrals`

| Property | Value |
|---|---|
| **Shape** | `(256, 720, 512)` — `(n_slices, n_angles, n_det)` |
| **Dtype** | `float32` |
| **Units** | **Neper** (dimensionless) — line integrals of linear attenuation: ∫μ(l) dl |
| **Compression** | gzip level 4 |
| **Chunking** | `(1, 720, 512)` — one chunk per slice |

The data represents noisy line integrals including Poisson photon noise and additive electronic noise. The forward model is: `I_measured = Poisson(I₀ · exp(−p) + S) + N(0, σ_e²)`, where `p` is the clean line integral, `S = scatter_frac × I₀`, and `σ_e` is electronic noise. The stored value is: `p_measured = −ln(max(I_measured, 0.1) / I₀)`. Note that scatter is **not** subtracted before the log transform — this is intentional and matches the reference reconstruction pipeline. The stored line integrals are in neper and are ready to reconstruct as-is.

**Axis convention:**
- Axis 0: slice index (z), 0-indexed. Slice 128 contains the lesion signal in LP realizations.
- Axis 1: projection angle index, corresponding to `geometry/angles_deg`.
- Axis 2: detector element index, corresponding to `geometry/det_fan_angles_deg`.

### 3.2 Group: `/geometry` (attributes)

| Attribute | Type | Value | Description |
|---|---|---|---|
| `type` | string | `"fan-beam"` | Acquisition geometry type |
| `n_slices` | int | `256` | Number of axial slices |
| `n_angles` | int | `720` | Number of projection angles (full 360°) |
| `n_det` | int | `512` | Number of equi-angular detector elements |
| `voxel_mm` | float | `0.5` | Isotropic voxel size (mm) |
| `SID_mm` | float | `570.0` | Source-to-isocenter distance (mm) |
| `SDD_mm` | float | `1040.0` | Source-to-detector distance (mm) |
| `gamma_max_deg` | float | ~12.97° | Half fan-angle: arcsin(FOV_half / SID) |
| `delta_gamma_deg` | float | ~0.0507° | Angular spacing between detector elements |
| `angles_deg` | float[720] | 0.0, 0.5, 1.0, … 359.5 | Source rotation angles (degrees) |
| `det_fan_angles_deg` | float[512] | −12.94 … +12.94 | Fan angle of each detector element (degrees) |

**Geometry notes:**

- **Equi-angular detectors.** Elements are uniformly spaced in fan angle γ, not in linear position. Element *j* subtends angle: `γ_j = (j − N_det/2 + 0.5) × Δγ`, where `Δγ = 2 × γ_max / N_det`.
- **Source rotation convention.** At angle β (from `angles_deg`), the X-ray source position in the reconstruction coordinate frame is: `(cx + SID_vox × cos(β), cy + SID_vox × sin(β))`, where `(cx, cy)` is the isocenter (pixel 256, 256) and `SID_vox = SID_mm / voxel_mm = 1140`. Each ray from the source to detector element *j* travels at angle `β + π + γ_j`. All source/detector distances in the backprojection loop use **voxel units**, not millimetres.
- **Reconstruction grid.** 512 × 512 pixels, 0.5 mm isotropic. The isocenter is at pixel (256, 256).
- **Full 360° acquisition.** 720 angles at 0.5° spacing, starting at 0°.

### 3.3 Group: `/noise_params` (attributes)

| Attribute | Type | Description |
|---|---|---|
| `I0` | float | Calibrated incident photon count (310,853) |
| `scatter_frac` | float | Scatter-to-primary ratio (0.05) |
| `sigma_e_counts` | float | Electronic noise σ in counts (5.0) |
| `seed` | int | Random seed for this realization |
| `jitter_deg` | float | Angular jitter applied to this realization |
| `place_lesion` | int | 1 = LP (lesion present), 0 = LA (lesion absent) |
| `lesion_slice_index` | int | 0-indexed slice containing lesion (128) |
| `lesion_z_extent` | int | Number of slices with lesion signal (0 or 1) |

These attributes are provided for traceability. Labs do not need them for reconstruction — the line integrals are ready to reconstruct as-is.

---

## 4. Reading the Sinograms

### 4.1 Pseudocode

```
file = HDF5_open("sinograms/LP/realization_001.h5", read_only)

# Line integrals: float32 array, shape (256, 720, 512)
sinogram = file["/line_integrals"]           # full volume
sinogram_slice = file["/line_integrals"][128] # single slice (720, 512)

# Geometry
geo = file["/geometry"].attrs
sid_mm   = geo["SID_mm"]            # 570.0
sdd_mm   = geo["SDD_mm"]            # 1040.0
angles   = geo["angles_deg"]        # float array, length 720
det_angles = geo["det_fan_angles_deg"]  # float array, length 512
voxel_mm = geo["voxel_mm"]          # 0.5
n_det    = geo["n_det"]             # 512

file.close()
```

### 4.2 Language-Specific HDF5 Libraries

| Language | Library | Read call |
|---|---|---|
| C/C++ | HDF5 C API (libhdf5) | `H5Dread(dataset, H5T_NATIVE_FLOAT, ...)` |
| CUDA | HDF5 C API (host-side) | Read to host, then `cudaMemcpy` to device |
| MATLAB | Built-in `h5read` | `h5read('file.h5', '/line_integrals')` |
| Python | `h5py` | `f['line_integrals'][:]` |
| Java | HDF5 Java (HDF Group) | `H5.H5Dread(...)` |

---

## 5. Reconstruction Requirements

### 5.1 What the Lab Must Do

1. **Read** each HDF5 sinogram file (§3–4).
2. **Reconstruct** 256 axial slices from the `(256, 720, 512)` line integrals using the fan-beam geometry specified in `/geometry` attributes.
3. **Apply** the lab's MAR algorithm during or after reconstruction.
4. **Calibrate** reconstructed pixel values to **Hounsfield Units** (HU), where water = 0 HU and air = −1000 HU.
5. **Write** 256 DICOM slices per realization (§6).

### 5.2 What the Lab Must NOT Do

- **Do not** alter the sinogram data (no pre-processing of the delivered line integrals beyond what the MAR algorithm itself performs).
- **Do not** change the reconstruction grid size (must remain 512 × 512 × 256 at 0.5 mm isotropic).
- **Do not** apply post-reconstruction spatial filtering, windowing, or cropping that would affect the pixel values in the scoring ROI.

### 5.3 Reconstruction Geometry Reference

For labs implementing their own fan-beam FBP (without MAR), the reference algorithm is:

1. **Pre-weight** each sinogram row by `cos(γ_j)` (cosine weighting for equi-angular detectors).
2. **Filter** each pre-weighted row with a Ram-Lak (ramp) filter in the frequency domain.
3. **Backproject** with distance weighting: for each pixel at position `(x, y)`, each angle β contributes `(SID / L)²` × (interpolated filtered value at γ), where `L` is the source-to-pixel distance and `γ` is the fan angle from source through pixel.
4. **Scale** the result by `π / N_angles / (SID_cm × Δγ_rad)` to obtain reconstructed μ in cm⁻¹.
5. **Convert** from linear attenuation (cm⁻¹) to HU. The reference pipeline uses: `HU = (μ_recon − μ_tissue − dc_offset) / μ_tissue × 1000 + BACKGROUND_HU`, where `μ_tissue = 0.2059 cm⁻¹`, `dc_offset = −0.029 cm⁻¹`, and `BACKGROUND_HU = 40`. This maps soft tissue to exactly 40 HU. Labs using their own HU calibration (e.g., water-referenced) are acceptable provided the resulting soft-tissue HU is consistent.

**Units in grid coordinates:** When computing source positions and distances in the reconstruction loop, use `SID` and `SDD` in **voxel units** (SID_vox = 570 / 0.5 = 1140; SDD_vox = 1040 / 0.5 = 2080), not millimetres. The final μ→HU conversion uses `SID_cm` (57.0 cm) for the analytic scaling factor.

This algorithm is provided for reference and validation only. Labs are expected to use their own reconstruction and MAR pipeline.

---

## 6. DICOM Output Format

### 6.1 Directory Structure

```
mar_recon/
    LP/
        realization_001/
            slice_0001.dcm
            slice_0002.dcm
            ...
            slice_0256.dcm
        realization_002/
            ...
        realization_040/
            ...
    LA/
        realization_001/
            ...
        realization_040/
            ...
```

### 6.2 File Naming

- Slices are **1-indexed**: `slice_0001.dcm` through `slice_0256.dcm`.
- Slice number corresponds to `z + 1` where `z` is the 0-indexed axis-0 index in the HDF5 sinogram.
- The CHO analysis reads **only `slice_0129.dcm`** (z-index 128, the lesion slice).

### 6.3 Required DICOM Attributes

| Tag | Keyword | Required Value |
|---|---|---|
| `(0028,0010)` | Rows | `512` |
| `(0028,0011)` | Columns | `512` |
| `(0028,0030)` | PixelSpacing | `[0.5, 0.5]` |
| `(0028,0100)` | BitsAllocated | `16` |
| `(0028,1052)` | RescaleIntercept | Any valid value (see below) |
| `(0028,1053)` | RescaleSlope | Any valid value (see below) |
| `(0008,0060)` | Modality | `CT` |

**HU calibration:** The CHO scoring pipeline computes display HU as `stored_value × RescaleSlope + RescaleIntercept`. Labs may use any valid combination — for example, `RescaleIntercept = −1024, RescaleSlope = 1` with unsigned pixels (common on commercial scanners), or `RescaleIntercept = 0, RescaleSlope = 1` with signed pixels (as in the reference implementation). What matters is that the resulting HU values are correctly calibrated.

### 6.4 DICOM 2026b Metal Artifact Reduction Macro (CP-2575)

Labs **must** include the MAR Macro in their output DICOM files:

| Tag | VR | Keyword | Value |
|---|---|---|---|
| `(0018,9390)` | SQ | Metal Artifact Reduction Sequence | 1 item |
| `(0018,9391)` | CS | Metal Artifact Reduction Applied | `YES` |

**Optional:** `(0018,9392)` Metal Artifact Reduction Algorithm — a code string from CID 10036 (e.g., `MAR_IMAR`, `MAR_SPECTRAL`, `MAR_MBIR`).

Because pydicom and most DICOM toolkits do not yet include the 2026b tag definitions, these must be written using hex tags:

```
# Python (pydicom)
mar_item = Dataset()
mar_item.add_new(0x00189391, 'CS', 'YES')
ds.add_new(0x00189390, 'SQ', [mar_item])

// C++ (DCMTK)
DcmItem *marItem = new DcmItem();
marItem->putAndInsertString(DcmTag(0x0018, 0x9391), "YES");
DcmSequenceOfItems *marSeq = new DcmSequenceOfItems(DcmTag(0x0018, 0x9390));
marSeq->insert(marItem);
dataset->insert(marSeq);

% MATLAB
dcmInfo.MetalArtifactReductionApplied = 'YES';
% or use dicomwrite with raw tag specification
```

### 6.5 HU Calibration Validation

The reference `noMAR_recon/` is provided so labs can validate their reconstruction pipeline before applying MAR:

- **Soft tissue ROI** (away from metal): should reconstruct to approximately **40 HU**.
- **Metal region**: the reference pipeline hard-sets metal pixels to **3000 HU**. Labs may handle metal pixels differently.
- **Lesion contrast** (LP, slice 129 only): approximately **12 HU** above background in the reference reconstruction.

---

## 7. CHO Scoring Region

The CHO observer evaluates a single region of interest from each realization:

| Parameter | Value |
|---|---|
| Scored slice | `slice_0129.dcm` (z-index 128) |
| ROI center | pixel **(281, 256)** |
| ROI size | **121 × 121** pixels |
| Dimensionality | **2D only** — 3D integration shall not be performed |

Labs do not need to know the CHO algorithm details. The scoring pipeline is run centrally by the ILS coordinator on the submitted DICOM files.

---

## 8. Submission Checklist

Before submitting `mar_recon/` to the ILS coordinator, verify:

- [ ] Directory structure matches §6.1 (LP and LA subdirectories, 40 realizations each)
- [ ] Each realization folder contains 256 DICOM slices named `slice_0001.dcm` through `slice_0256.dcm`
- [ ] DICOM pixel values are in Hounsfield Units (RescaleSlope=1, RescaleIntercept=0)
- [ ] Image matrix is 512 × 512 at 0.5 mm pixel spacing
- [ ] Metal Artifact Reduction Macro is present with `(0018,9391) = "YES"`
- [ ] `slice_0129.dcm` exists and is valid in every realization folder
- [ ] Soft tissue HU in non-metal regions is plausible (~0–100 HU range)

---

## Appendix A: Quick-Start Validation

To verify correct sinogram ingestion, reconstruct a single LP realization without MAR and compare `slice_0129.dcm` against the reference:

```
Reference noMAR_recon/LP/realization_001/slice_0129.dcm:
  - Tissue background: ~40 HU
  - Metal rod:         3000 HU (hard-set)
  - Lesion region:     ~52 HU (40 + 12)
  - Noise σ:           ~30 HU in soft tissue
```

If your reconstruction produces similar values in non-metal regions, your sinogram ingestion is correct and you can proceed to apply MAR.

---

## Appendix B: Coordinate System

```
              0        256       512
          0   ┌─────────┼─────────┐
              │         │         │
              │    ┌────┼────┐    │
              │    │  body   │    │
              │    │ (ellip) │    │
        256   ┼────│──●──────│────┤   ● = isocenter (256, 256)
              │    │  │ ○    │    │   │ = metal rod (center)
              │    │  └─┤    │    │   ○ = lesion at (281, 256)
              │    │    │    │    │
              │    └────┼────┘    │
        512   └─────────┼─────────┘

  Pixel (row, col) = (y, x).  Isocenter at (256, 256).
  Metal rod centered at isocenter.  Lesion at col=281, row=256.
  Body ellipse: semi-axes 170 (x) × 120 (y) voxels.
```
