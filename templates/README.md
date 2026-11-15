# MAR Test-Method Harness Templates (v9 dataset)

These templates are for measuring **your own MAR method** with the ASTM WK101378 test method (draft Revision 06). Each one reads the sinograms of a dataset in the format of the [sinogram ingestion specification](../docs_and_references/sinogram_ingestion_spec.md) (version 2.3), calls your `apply_mar()` for each realization, and writes one DICOM CT image per realization, with the sinogram's name and the CP-2575 Metal Artifact Reduction Macro set to YES. The images are then scored with `run_cho_analysis_v9_0.py`.

The interlaboratory study does not use these templates. In the study every laboratory runs the supplied improving control with `run_control_v9_0.py`; see [`docs_and_references/ILS_Lab_Instructions.md`](../docs_and_references/ILS_Lab_Instructions.md).

If your method produces corrected sinograms and you work in Python, you may not need a template: save the corrected sinograms with the dataset's file names and run `python run_control_v9_0.py REP01 MY_R1 --sinograms my_corrected/REP01`, which applies the reference reconstruction and writes the DICOM images.

## Languages

| Directory | Language | Dependencies | Verified |
|---|---|---|---|
| `python/` | Python 3.9+ | `numpy`, `h5py`, `pydicom` 3.x; `numba` optional, for speed | Run on a v9 dataset: images identical to the supplied noMAR images; reconstruction check 0.000000 HU |
| `c/` | C, C++ or CUDA | `libhdf5` | Built with no warnings and run on a v9 dataset: images identical to the supplied noMAR images; reconstruction check 0.000000 HU |
| `matlab/` | MATLAB R2016b+ | none beyond base MATLAB (`h5read`); no toolbox | Reviewed against the Python template line by line; not run, because no MATLAB was available |

## Interface

You implement two functions in the file for your language:

- `apply_mar(p, geo)`: input one sinogram, `p`, 984 views × 888 channels of line integrals indexed [view, channel] (ingestion spec §4.1), and the geometry read from the file's attributes (ingestion spec §4.2). Output one 512 × 512 image in HU (draft standard §A1.1(h)), indexed [row, column] with pixel (256, 256) at the isocenter (ingestion spec §6.2).
- `reconstruct(p, geo)`: the reconstruction used under draft standard §14.3, with the same input and output. As supplied it is the reference reconstruction, which §14.3 allows. If you substitute your own, it must pass the reconstruction check.

As supplied, `apply_mar()` applies no MAR and calls `reconstruct()`, so its output matches the supplied noMAR image of the same realization. Run it once that way before plugging in your method. A sinogram-domain method corrects `p` and then calls `reconstruct()`. Counts, if your method needs them, are `I0 * exp(-p)`.

## Steps

1. Reconstruction check (draft standard §12.2), on the familiarization dataset:

   ```bash
   python run_mar_harness.py --recon-check FAMILIARIZATION/recon_check/recon_check.h5
   ./mar_harness --recon-check FAMILIARIZATION/recon_check/recon_check.h5
   ```
   ```matlab
   run_mar_harness('--recon-check', 'FAMILIARIZATION/recon_check/recon_check.h5')
   ```

2. A trial run on the familiarization dataset, then one run directory per replicate dataset:

   ```bash
   python run_mar_harness.py REP01 MY_R1
   ./mar_harness REP01 MY_R1
   ```
   ```matlab
   run_mar_harness('REP01', 'MY_R1')
   ```

3. Manifest of each run directory. The Python harness writes `MY_R1_checksums_sha256.txt` beside the run directory itself. For C and MATLAB:

   ```bash
   (cd MY_R1 && shasum -a 256 *.dcm) > MY_R1_checksums_sha256.txt
   ```

4. Score each run directory against the dataset it came from:

   ```bash
   python run_cho_analysis_v9_0.py --dataset-dir REP01 --mar-output-dir MY_R1
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
MY_R1/
    <name>.dcm      one image per realization of the dataset, named as its sinogram
```

Each image is 512 × 512, pixel spacing 300/512 mm, signed 16-bit with Rescale Slope 1 and Rescale Intercept 0, rounded to whole HU, with Metal Artifact Reduction Applied (0018,9391) = YES. The scoring tool reads the 121 × 121 region centered on pixel (256, 256).
