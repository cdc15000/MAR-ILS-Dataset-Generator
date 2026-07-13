# MAR ILS Lab Harness Templates

Turnkey harnesses for participating labs. Each template handles all sinogram
I/O, DICOM writing, directory layout, and CP-2575 MAR macro injection.
**The lab implements one function — `apply_mar()` — and the harness does
the rest.**

## Choose Your Language

| Directory | Language | Dependencies |
|-----------|----------|-------------|
| `python/` | Python 3.8+ | `h5py`, `numpy`, `pydicom` |
| `matlab/` | MATLAB R2019b+ | Built-in (`h5read`, `dicomwrite`) |
| `c/`      | C / C++ / CUDA | `libhdf5` |

## Interface Contract

Every `apply_mar()` function has the same signature:

```
Input:   sinogram    — float32 array, shape (256, 720, 512)
                       Line integrals in neper.
         geometry    — Fan-beam parameters (SID, SDD, angles, detector
                       fan angles, voxel size).

Output:  hu_volume   — float32 array, shape (256, 512, 512)
                       Reconstructed image in Hounsfield Units.
```

## Quick Start

### Python
```bash
cd templates/python
# Edit apply_mar.py — replace the example FBP with your MAR pipeline
python run_mar_harness.py --dataset-dir ../../astm_reference_dataset
```

### MATLAB
```matlab
cd templates/matlab
% Edit apply_mar.m — replace the example FBP with your MAR pipeline
run_mar_harness('../../astm_reference_dataset', './mar_recon')
```

### C / C++ / CUDA
```bash
cd templates/c
# Copy apply_mar_example.c to my_mar.c, replace with your implementation
make MAR_SRC=my_mar.c
./mar_harness ../../astm_reference_dataset ./mar_recon
```

For CUDA: implement `apply_mar()` as a host function that launches your
kernels. Compile your `.cu` files separately and link against the harness.

## What the Lab Provides

Each `apply_mar()` file ships with a working FBP example (no MAR) that
reproduces the reference `noMAR_recon/` output. Replace the body with your
reconstruction + MAR pipeline. The example serves as:

1. **Validation** — run it first to verify your sinogram ingestion matches
   the reference reconstruction.
2. **Starting point** — the geometry setup, HU conversion, and loop
   structure are already correct.

## Output

The harness writes to `mar_recon/` with the structure expected by the CHO
scoring pipeline:

```
mar_recon/
    LP/  realization_001/slice_0001.dcm ... slice_0256.dcm
         realization_002/ ...
    LA/  realization_001/ ...
```

Submit this directory to the ILS coordinator.
