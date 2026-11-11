# Sinogram Ingestion and Reconstruction Output Specification

**ASTM WK101378, draft Revision 06: MAR Interlaboratory Study**\
**Version:** 2.0 (v9 design; replaces version 1.0 of 2026-07-13, which described the superseded v7 design)\
**Date:** 2026-11-11\
**Status:** draft. The v9 dataset writer and the blinding tooling shall produce files in exactly this format; any difference found when they are built is resolved before distribution, and this document is reissued.

---

## 1. Purpose and Scope

This document describes how the files of the MAR ILS dataset are organized and how to read them, and the format in which laboratories return their reconstructions. It is language-independent: any environment that reads HDF5 and writes DICOM can be used.

In the study's Condition 1 (protocol §A7), each laboratory reads the sinograms from this document, implements the improving control of draft standard §A1.9 and the reconstruction of §A1.1(g) from the text of the standard, and does not consult or reuse the reference software. This document is the only reading aid allowed for Condition 1. For that reason it describes the data and their coordinates only. It does not describe how to reconstruct or how to apply MAR 2; those are defined by the standard alone, and where this document and the standard differ, the standard governs.

---

## 2. Datasets

| Dataset | Content | Labels |
|---|---|---|
| Replicate datasets `REP01` to `REP09` | 8,000 realizations each (draft standard §8.2.1). Each laboratory receives three (protocol §A6.3). | Coded |
| Control dataset `CONTROL` | 4,000 realizations at level 1.030 and dose 4 × 10⁶, for the degrading control in the pilot | Coded |
| Familiarization dataset `FAMILIARIZATION` | A small set of realizations with the same configuration and different seeds, the reconstruction check of draft standard §12.2, and the noMAR images. No improving-control output (protocol §A6.4). | Labeled |

---

## 3. Directory Layout

Each dataset is distributed as one directory:

```
REP01/
    sinograms/
        <code>.h5          one file per realization
    noMAR/
        <code>.dcm         the noMAR reconstruction of the same realization
    dataset_record.json    dataset-level parameters (§3.2)
    checksums_sha256.txt   SHA-256 of every other file in the directory
```

The familiarization dataset has the same layout, plus:

```
FAMILIARIZATION/
    labels.csv             code, class, level, pool, index for each realization
    recon_check/
        recon_check.h5     the reconstruction check sinogram and its reference image (§5)
```

### 3.1 Coded identifiers

- In the replicate and control datasets, each realization is named by a code of eight characters from `0-9` and `A-Z` (for example `K7Q2M9XD`). Codes are unique across all datasets of the study.
- The code says nothing about the realization's class (lesion present or absent), signal level, pool (training or test) or index. Files are listed in no meaningful order.
- `<code>.h5` and `<code>.dcm` with the same code belong to the same realization.
- The key that maps codes to labels is released after the manifest deadline (protocol §A6.3 and §A8.2, Step 5). No information in the distributed files is needed to read, process or reconstruct a realization.

### 3.2 `dataset_record.json`

A JSON object with the dataset identifier, the generator version, the number of realizations, the parameters of draft standard §A1.1 and §A1.2 that are the same for every realization, the spectrum table of §A1.1(d), the MAR 2 threshold of §A1.9(f), and the SHA-256 of the blinding key as published in the study's Zenodo record. It contains no per-realization entries in the coded datasets.

---

## 4. HDF5 Sinogram File (`<code>.h5`)

Each file holds one realization: one 2D fan-beam sinogram of one slice.

### 4.1 Dataset `/line_integrals`

| Property | Value |
|---|---|
| Shape | `(984, 888)`, indexed `[view k, channel j]` |
| Type | 32-bit IEEE floating point, little-endian |
| Storage | contiguous, uncompressed |
| Units | dimensionless (natural logarithm of an intensity ratio) |

Each value is the line integral p = −ln(max(counts, 1)/I₀) of draft standard §A1.1(f), where counts is the noisy detector reading of that ray and I₀ = 1 × 10⁶. Scatter and electronic noise are included in the reading and are not subtracted. No scatter or beam-hardening correction has been applied. The values are ready for reconstruction as stored.

A method that needs detector counts instead of line integrals recovers them as I₀ · exp(−p). Readings below one count were set to one before the logarithm, so the largest possible value of p is ln(10⁶) ≈ 13.8155; rays with this value carry no information beyond "at most one count."

Row order: row k is view k. Within a row, element j is channel j. No views or channels are omitted, reordered or flipped.

### 4.2 Attributes of `/line_integrals`

These attributes repeat the fixed acquisition of draft standard §A1.1 for convenience. They are identical in every file.

| Attribute | Type | Value |
|---|---|---|
| `geometry` | string | `"fan-beam, equi-angular curved detector"` |
| `SID_mm` | float64 | 541.0 |
| `SDD_mm` | float64 | 949.0 |
| `n_views` | int32 | 984 |
| `n_channels` | int32 | 888 |
| `channel_pitch_mm` | float64 | 1.0 (on the detector arc) |
| `delta_gamma_rad` | float64 | 0.1/94.9 = 0.001 053 740 779 768 177 |
| `gamma0_rad` | float64 | −443.5 × `delta_gamma_rad`, the fan angle of channel 0 |
| `betas_rad` | float64[984] | source angle of each view, β~k~ = 2πk/984 |
| `gammas_rad` | float64[888] | fan angle of each channel, γ~j~ = (j − 443.5) Δγ |
| `I0` | float64 | 1.0e6 counts per detector bin per view |
| `scatter_counts` | float64 | 100.0 (10⁻⁴ I₀) |
| `electronic_variance_counts2` | float64 | 160.0 |
| `log_floor_counts` | float64 | 1.0 |

The coded files carry no other attributes. In particular, the lesion flag, signal level, seed and inset angle are not present. In the familiarization dataset they are given in `labels.csv`, not in the HDF5 files.

### 4.3 Coordinates

The coordinates are those of draft standard §A1.1(i), restated here so that the axes of the array can be interpreted. The standard governs.

- x and y lie in the image plane, with the origin at the isocenter and y pointing toward row 0 of the image (§6.2).
- View k has the source at (SID cos β~k~, SID sin β~k~). β increases from the positive x axis toward the positive y axis, which is counterclockwise as the image is displayed with row 0 at the top.
- For a point (x, y) and source angle β, let u = SID − (x cos β + y sin β) and v = x sin β − y cos β. The fan angle of the ray through the point is γ = atan2(v, u), and its distance from the source is L = √(u² + v²).
- Channel j lies at fan angle γ~j~ = (j − 443.5) Δγ. The central ray, through the isocenter, falls midway between channels 443 and 444; no channel lies on it.

**Units.** The reconstructed attenuation must come out in cm⁻¹ (§A1.1(h), §A1.9(a) and (b)). p is dimensionless, so the unit of μ is set by the unit of length used in the reconstruction. Express SID, L and the pixel coordinates in centimeters (SID = 54.1 cm). Using millimeters gives μ in mm⁻¹, a factor of 10 too small, and the MAR 2 threshold in cm⁻¹ would then select no pixels.

### 4.4 Reading example (Python)

```python
import h5py
import numpy as np

with h5py.File("REP01/sinograms/K7Q2M9XD.h5", "r") as f:
    p = f["line_integrals"][()].astype(np.float64)   # shape (984, 888), [view, channel]
    betas = f["line_integrals"].attrs["betas_rad"]
    gammas = f["line_integrals"].attrs["gammas_rad"]
```

The equivalent calls are `h5read` in MATLAB and `H5Dread` with `H5T_NATIVE_FLOAT` in C. MATLAB's `h5read` returns HDF5 arrays with the axes reversed, as (888, 984); transpose before indexing by [view, channel].

---

## 5. Reconstruction Check (`recon_check/recon_check.h5`)

This file serves the check of draft standard §12.2 and protocol §A8.2, Step 2.1.

| Dataset | Shape | Type | Content |
|---|---|---|---|
| `/line_integrals` | (984, 888) | float32 | a sinogram in the format of §4, with the same attributes |
| `/reference_image_hu` | (512, 512) | float64 | its reconstruction by the reference implementation of §A1.1(g), in HU per §A1.1(h), indexed [row, column] as in §6.2 |

The reference image is stored in floating point, not as DICOM, so that the comparison is not limited by integer rounding: a 16-bit DICOM image rounded to whole HU would use up half of the 1 HU tolerance. Reconstruct `/line_integrals` with the implementation to be used, convert to HU per §A1.1(h), and record the largest absolute difference from `/reference_image_hu` over all 512 × 512 pixels. The implementation passes if it is 1 HU or less.

---

## 6. Images

### 6.1 Supplied noMAR images (`noMAR/<code>.dcm`)

One single-frame DICOM CT image per realization: the reference reconstruction of the uncorrected sinogram. These images are the noMAR series of draft standard §14.2.

| Attribute | Value |
|---|---|
| Rows, Columns (0028,0010), (0028,0011) | 512, 512 |
| Pixel Spacing (0028,0030) | 0.5859375\0.5859375 (300/512 mm) |
| Bits Allocated, Bits Stored (0028,0100), (0028,0101) | 16, 16 |
| Pixel Representation (0028,0103) | 1 (signed) |
| Rescale Slope, Rescale Intercept (0028,1053), (0028,1052) | 1, 0 |
| Modality (0008,0060) | CT |
| Metal Artifact Reduction Sequence (0018,9390) | one item, with Metal Artifact Reduction Applied (0018,9391) = NO |

Stored values are HU per §A1.1(h), rounded to the nearest integer; values beyond the signed 16-bit range are clipped. Series Description, Protocol Name and any other attribute that could reveal a label are blank in the coded datasets, and Content Date and Content Time have one value for all files.

Because of the rounding and clipping, the noMAR DICOM images shall not be used for the MAR 2 segmentation (draft standard §A1.9(a)). MAR 2 starts from the sinogram.

### 6.2 Pixel indexing

Pixel (row r, column c) counts from zero. Its center is at x = (c − 256) Δ, y = (256 − r) Δ, with Δ = 300/512 mm, so pixel (256, 256) is at the isocenter, in the 257th row and column, and row 0 is at the largest y. In the DICOM pixel array, row r is the r-th row transmitted.

### 6.3 Laboratory output

Each run directory holds one image per coded realization of the replicate dataset (protocol §A8.3):

```
C1_R1/
    <code>.dcm
```

| Requirement | Value |
|---|---|
| File name | the code of the realization, with extension `.dcm` |
| Matrix and pixel indexing | 512 × 512, pixel spacing 300/512 mm, pixel (256, 256) at the isocenter, indexed as in §6.2 |
| Pixel values | HU per §A1.1(h), stored in 16 bits; any Rescale Slope and Rescale Intercept that represent them correctly |
| Modality | CT |
| MAR Macro | Metal Artifact Reduction Sequence (0018,9390) with one item, Metal Artifact Reduction Applied (0018,9391) = YES |

The scoring tool computes HU as stored value × Rescale Slope + Rescale Intercept and reads the 121 × 121 region centered on pixel (256, 256). Clipping of values outside that region does not affect the score.

The MAR Macro tags are recent additions to DICOM (2026b, CP-2575) and may be missing from a toolkit's dictionary; they can be written by their numeric tags, for example in pydicom:

```python
from pydicom.dataset import Dataset
item = Dataset()
item.add_new(0x00189391, "CS", "YES")
ds.add_new(0x00189390, "SQ", [item])
```

---

## 7. Checklist Before Submission

- [ ] Every file of the assigned datasets matches `checksums_sha256.txt`.
- [ ] The reconstruction check passed, and the largest difference is recorded on the return form.
- [ ] Each run directory holds exactly one `<code>.dcm` for every `<code>.h5` of its replicate dataset, and nothing else.
- [ ] Every image is 512 × 512, indexed as in §6.2, in HU, with (0018,9391) = YES.
- [ ] The SHA-256 manifest of each run directory has been produced in the format of `checksums_sha256.txt`.

---

## Revision History

- **2.0 (2026-11-11):** rewritten for the v9 design and draft Revision 06: single-slice 984 × 888 sinograms (SID 541 mm, SDD 949 mm), coded layout of protocol §A6.3, coordinates of §A1.1(i), reconstruction check file, laboratory output under coded names. The description of the reference reconstruction in version 1.0 is removed, because in Condition 1 the reconstruction is implemented from the standard's text.
- **1.0 (2026-07-13):** v7 design, Revision 05.
