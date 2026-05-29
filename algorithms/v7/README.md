# algorithms/v7/ — fan-beam reference MAR

Reference MAR implementations compatible with the **v7.0.0 fan-beam** dataset
(ASTM Rev 04). Unlike the parallel-beam references in `algorithms/`, these
operate on the (256, 720, 512) fan-beam sinograms and reuse the validated
`forward_project_slice` / `fbp_reconstruct_slice` from `generator_v7_0_0.py`.

## reference_li_mar_v7.py — Linear-Interpolation MAR

Parameter-free LI-MAR. **Non-normative**: the ASTM type test measures the
*lab's* algorithm. This reference exists as a CHO-pipeline positive control and
a reproducible delta-AUC anchor (LI-MAR is the universal baseline every
commercial MAR is expected to beat).

It detects metal by thresholding the existing noMAR reconstruction, fills the
metal trace in the sinogram by linear interpolation along the detector axis,
reconstructs with fan-beam FBP, and writes `slice_0129.dcm` only (what the CHO
reads).

### Usage

```bash
python algorithms/v7/reference_li_mar_v7.py \
    --dataset-dir ./astm_reference_dataset \
    --output-dir  ./li_mar_recon
# then score it:
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./li_mar_recon \
    --internal-noise-sigma 15
```

Options: `--realizations N` (default auto-detect), `--metal-hu-thresh 2000`,
`--dc-offset-cm <float>` (default read from `generator_provenance.json`).

### delta-AUC anchor

Run the two commands above on the locked reference dataset (N=40, sigma=15) and
record the result here:

> delta-AUC (LI-MAR vs noMAR) = _TBD — fill in after the manual anchor run_

This is a manual step (a full dataset is ~35 GB / ~10 min), not part of CI.
