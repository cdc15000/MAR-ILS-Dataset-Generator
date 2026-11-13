# Sinogram Ingestion and Reconstruction Output Specification

**ASTM WK101378, draft Revision 06: MAR Interlaboratory Study**\
**Version:** 2.3 (v9 design; replaces version 1.0 of 2026-07-13, which described the superseded v7 design)\
**Date:** 2026-11-12\
**Status:** draft. The v9 dataset writer and the control runner shall produce files in exactly this format; any difference found when they are built is resolved before distribution, and this document is reissued.

---

## 1. Purpose and Scope

This document describes how the files of the MAR ILS dataset are organized and how to read them, and the format in which laboratories return their reconstructions. It is language-independent: any environment that reads HDF5 and writes DICOM can be used.

In the study, each laboratory processes the sinograms with the supplied software: the improving control of draft standard §A1.9, the reference reconstruction of §A1.1(g) and the reference observer (protocol §A7). The software reads and writes the files described here, so a laboratory in the study needs this document for reference only. It describes the data and their coordinates. It does not describe how to reconstruct or how to apply MAR 2; those are defined by the standard, and where this document and the standard differ, the standard governs.

---

## 2. Datasets

| Dataset | Content |
|---|---|
| Replicate datasets, `REP01` and so on | 8,000 realizations each (draft standard §8.2.1). Each laboratory receives three (protocol §A6.3). |
| Control dataset `CONTROL` | 4,000 realizations at level 1.030 and dose 4 × 10⁶, for the degrading control in the pilot |
| Null-check dataset `NULL` | 4,000 lesion-absent realizations in two independent sets (`absent` and `absent2`), 1,000 training and 1,000 test each, for the null-signal check of draft standard §A1.7. Every laboratory receives it. |
| Familiarization dataset `FAMILIARIZATION` | A small set of realizations with the same configuration and different seeds, the reconstruction check of draft standard §12.2, the noMAR images, and the images of the supplied improving control for every realization, against which a laboratory checks its installation (protocol §A6.4 and §A8.2, Step 2). |

---

## 3. Directory Layout

Each dataset is distributed as one directory:

```
REP01/
    sinograms/
        <name>.h5          one file per realization
    noMAR/
        <name>.dcm         the noMAR reconstruction of the same realization
    dataset_record.json    dataset-level parameters and the label of every realization (§3.2)
    checksums_sha256.txt   SHA-256 of every other file in the directory
```

The null-check and familiarization datasets also have `labels.csv`. The familiarization dataset has the reconstruction check and the expected images of the improving control:

```
FAMILIARIZATION/
    labels.csv             name, class, level, pool, index for each realization
    recon_check/
        recon_check.h5     the reconstruction check sinogram and its reference image (§5)
    expected_C1/
        <name>.dcm         the image the supplied improving control gives for the realization (§6.4)
```

### 3.1 File names

- All datasets are labeled. Each realization is named by its class, signal level, pool and index: `absent_train_0000`, `present1025_test_0412`, and, in the null-check dataset, `absent2_train_0000` for the second lesion-absent set. The level is given in thousandths, so `present1025` is level 1.025.
- `<name>.h5` and `<name>.dcm` with the same name belong to the same realization.
- The study does not conceal labels from the laboratories (protocol §A6.3). The result is computed by fixed supplied software, and the coordinator computes the same result independently.

### 3.2 `dataset_record.json`

A JSON object with the dataset identifier, the generator version, the number of realizations, the parameters of draft standard §A1.1 and §A1.2 that are the same for every realization, the spectrum table of §A1.1(d), the thresholds of the control methods (§A1.9(f), §17.7(c)), and one entry per realization with its name, class, level, pool, index and inset angle. The field `reference_auc_inf_nomar` gives, for each signal level, AUC∞ of the noMAR series as computed by the coordinator with the reference observer; the scoring of protocol §A8.2, Step 4 compares its own value with it. A dataset with fewer than 50 training realizations per class, such as the familiarization dataset, has no such field.

---

## 4. HDF5 Sinogram File (`<name>.h5`)

Each file holds one realization: one 2D fan-beam sinogram of one slice.

### 4.1 Dataset `/line_integrals`

| Property | Value |
|---|---|
| Shape | `(984, 888)`, indexed `[view k, channel j]` |
| Type | 32-bit IEEE floating point, little-endian |
| Storage | contiguous, uncompressed |
| Units | dimensionless (natural logarithm of an intensity ratio) |

Each value is the line integral p = −ln(max(counts, 1)/I₀) of draft standard §A1.1(f), where counts is the noisy detector reading of that ray and I₀ is the dose of the dataset: 1 × 10⁶ for the replicate and familiarization datasets and 4 × 10⁶ for the control dataset (attribute `I0`). Scatter and electronic noise are included in the reading and are not subtracted. No scatter or beam-hardening correction has been applied. The values are ready for reconstruction as stored.

A method that needs detector counts instead of line integrals recovers them as I₀ · exp(−p). Readings below one count were set to one before the logarithm, so the largest possible value of p is ln(I₀), about 13.8155 at 1 × 10⁶; rays with this value carry no information beyond "at most one count."

Row order: row k is view k. Within a row, element j is channel j. No views or channels are omitted, reordered or flipped.

### 4.2 Attributes of `/line_integrals`

These attributes repeat the acquisition of draft standard §A1.1 for convenience. They are identical in every file of a dataset; only `I0` and `scatter_counts` differ between datasets.

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
| `I0` | float64 | 1.0e6 counts per detector bin per view (4.0e6 in the control dataset) |
| `scatter_counts` | float64 | 10⁻⁴ I₀: 100.0 (400.0 in the control dataset) |
| `electronic_variance_counts2` | float64 | 160.0 |
| `log_floor_counts` | float64 | 1.0 |

The files carry no other attributes. The class, signal level, pool and index are given by the file name and the dataset record, not inside the HDF5 files.

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

### 6.1 Supplied noMAR images (`noMAR/<name>.dcm`)

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

Stored values are HU per §A1.1(h), rounded to the nearest integer; values beyond the signed 16-bit range are clipped. Series Description and Protocol Name are absent, and Content Date and Content Time have one value for all files of a dataset.

Because of the rounding and clipping, the noMAR DICOM images shall not be used for the MAR 2 segmentation (draft standard §A1.9(a)). MAR 2 starts from the sinogram.

### 6.2 Pixel indexing

Pixel (row r, column c) counts from zero. Its center is at x = (c − 256) Δ, y = (256 − r) Δ, with Δ = 300/512 mm, so pixel (256, 256) is at the isocenter, in the 257th row and column, and row 0 is at the largest y. In the DICOM pixel array, row r is the r-th row transmitted.

### 6.3 Laboratory output

Each run directory holds one image per realization of its dataset (protocol §A8.3). The control runner, `run_control_v9_0.py`, writes it in this format:

```
C1_R1/
    <name>.dcm
```

| Requirement | Value |
|---|---|
| File name | the name of the realization, with extension `.dcm` |
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

### 6.4 Expected images (`FAMILIARIZATION/expected_C1/<name>.dcm`)

One image per realization of the familiarization dataset, in the format of §6.3: the output of the supplied improving control and reference reconstruction, as run by the coordinator. A laboratory processes the familiarization dataset and compares its images with these, pixel for pixel (`run_control_v9_0.py --compare`; protocol §A8.2, Step 2).

---

## 7. Checklist Before Submission

- [ ] Every file of the assigned datasets matches `checksums_sha256.txt`.
- [ ] The reconstruction check passed, and the largest difference is recorded on the return form.
- [ ] The images of the familiarization run equal the expected images.
- [ ] Each run directory holds exactly one `<name>.dcm` for every `<name>.h5` of its dataset, and nothing else.
- [ ] Every image is 512 × 512, indexed as in §6.2, in HU, with (0018,9391) = YES.

---

## Revision History

- **2.3 (2026-11-12):** all datasets labeled, with files named by label; no codes and no key (protocol §A6.3). Laboratories run the supplied software. `expected_C1/` added to the familiarization dataset. The manifest of each run directory is no longer required. No change to the sinogram or image file format.
- **2.2 (2026-11-11):** null-check dataset `NULL` added (draft standard §8.2.1 and §A1.7). No change to the file format.
- **2.1 (2026-11-11):** `I0` and `scatter_counts` stated per dataset, since the control dataset is at 4 × 10⁶; found when the dataset writer was built. No change to the file format.
- **2.0 (2026-11-11):** rewritten for the v9 design and draft Revision 06: single-slice 984 × 888 sinograms (SID 541 mm, SDD 949 mm), coded layout of protocol §A6.3, coordinates of §A1.1(i), reconstruction check file, laboratory output under coded names. The description of the reference reconstruction in version 1.0 is removed, because in Condition 1 the reconstruction is implemented from the standard's text.
- **1.0 (2026-07-13):** v7 design, Revision 05.
