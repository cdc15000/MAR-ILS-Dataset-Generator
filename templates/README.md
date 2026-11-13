# MAR ILS Laboratory Harness Templates (v9 dataset)

> **Not used in the interlaboratory study (2026-11-12).** The study now has one condition, in which every laboratory runs the supplied software (`run_control_v9_0.py`; see `docs_and_references/ILS_Lab_Instructions.md`). These templates remain for a laboratory that wishes to measure its own MAR method with the test method, outside the study. References below to Condition 1, Condition 2 and coded names describe the earlier design.

Harnesses for laboratories in the ASTM WK101378 interlaboratory study (draft Revision 06). Each one reads the coded sinograms of a replicate dataset in the format of the [sinogram ingestion specification](../docs_and_references/sinogram_ingestion_spec.md) (version 2.0), calls the laboratory's `apply_mar()` for each realization, and writes one DICOM CT image per realization, named by its code, with the CP-2575 Metal Artifact Reduction Macro (protocol §A8.3).

**Use these templates for Condition 2 (the laboratory's own MAR) only.** In Condition 1, each laboratory implements the improving control (draft standard §A1.9) and the reconstruction (§A1.1(g)) from the text of the standard, reading the data with the ingestion specification alone, and does not consult or reuse these templates or the reference reconstruction in them (protocol §A7).

## Languages

| Directory | Language | Dependencies |
|---|---|---|
| `python/` | Python 3.9+ | `numpy`, `h5py`, `pydicom` 3.x; `numba` optional, for speed |
| `matlab/` | MATLAB R2016b+ | none beyond base MATLAB (`h5read`); no toolbox |
| `c/` | C, C++ or CUDA | `libhdf5` |

## Interface

The laboratory implements two functions in the file for its language:

- `apply_mar(p, geo)`: input one sinogram, `p`, 984 views × 888 channels of line integrals indexed [view, channel] (ingestion spec §4.1), and the geometry read from the file's attributes (ingestion spec §4.2). Output one 512 × 512 image in HU (§A1.1(h)), indexed [row, column] with pixel (256, 256) at the isocenter (ingestion spec §6.2).
- `reconstruct(p, geo)`: the reconstruction used under draft standard §14.3, with the same input and output. As supplied it is the reference reconstruction, which §14.3 allows. If you substitute your own, it must pass the reconstruction check.

As supplied, `apply_mar()` applies no MAR and calls `reconstruct()`, so its output matches the supplied noMAR image of the same realization to within DICOM rounding. Run it once that way before plugging in your method. A sinogram-domain method corrects `p` and then calls `reconstruct()`.

The reference reconstruction in each template reproduces the reference implementation to about 1e-6 HU, well inside the 1 HU tolerance of §12.2. Counts, if a method needs them, are `I0 * exp(-p)`.

## Steps

1. Reconstruction check (draft standard §12.2, protocol Step 2.1), on the familiarization dataset:

   ```bash
   python run_mar_harness.py --recon-check FAMILIARIZATION/recon_check/recon_check.h5
   ./mar_harness --recon-check FAMILIARIZATION/recon_check/recon_check.h5
   ```
   ```matlab
   run_mar_harness('--recon-check', 'FAMILIARIZATION/recon_check/recon_check.h5')
   ```

2. Trial runs on the familiarization dataset, then one run directory per assigned replicate dataset (protocol §A8.2, Step 3):

   ```bash
   python run_mar_harness.py REP04 C2_R1
   ./mar_harness REP04 C2_R1
   ```
   ```matlab
   run_mar_harness('REP04', 'C2_R1')
   ```

3. Manifest of each run directory (protocol §A8.2). The Python harness writes `C2_R1_checksums_sha256.txt` beside the run directory itself. For C and MATLAB:

   ```bash
   (cd C2_R1 && shasum -a 256 *.dcm) > C2_R1_checksums_sha256.txt
   ```

## Building the C harness

```bash
cd templates/c
cp apply_mar_example.c my_mar.c      # put your implementation in my_mar.c
make MAR_SRC=my_mar.c                # add OPENMP=1 to parallelize the reference backprojection
```

For CUDA, implement `apply_mar()` and `reconstruct()` as host functions that launch your kernels, compile the `.cu` files separately and link them with the harness.

## Output

```
C2_R1/
    <code>.dcm      one image per coded realization of the replicate dataset
```

Each image is 512 × 512, pixel spacing 300/512 mm, signed 16-bit with Rescale Slope 1 and Rescale Intercept 0, rounded to whole HU, with Metal Artifact Reduction Applied (0018,9391) = YES. The scoring tool reads the 121 × 121 region centered on pixel (256, 256).
