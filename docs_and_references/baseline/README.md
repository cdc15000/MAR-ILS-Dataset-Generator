# Reference CHO results

> **Superseded.** This is the archived v7.0.0 record. The LI-MAR result below came from defects in the v7.0.0 generator and is withdrawn; see the v9.0.0 release.

`cho_results_li_mar_v7_n40.json` is the archived output of `run_cho_analysis_v7_0.py` in ILS mode, scoring the reference LI-MAR v7 reconstructions against the noMAR baseline on the N=40 fan-beam dataset with internal noise σ = 15.

| Quantity | Value |
|---|---|
| AUC_noMAR | 0.8294, 95% CI [0.7612, 0.9025] |
| AUC_MAR, LI-MAR v7 | 0.5994 |
| ΔAUC | −0.230 |

History:

- Baseline established 2026-04-07 (commit `3c4af4e2`); LI-MAR anchor added 2026-05-29 (commit `c743313f`).
- On 2026-10-04 a self-test on the scored slices of the copy generated 2026-07-08 reproduced AUC_noMAR = 0.829375 and the same CI exactly, using Python 3.11 and scipy 1.16.3. That copy later proved incomplete: 28 LP sinograms were empty in its Google Drive upload.
- On 2026-10-14 the dataset was regenerated. Its sinogram arrays and scored slices are identical to those of the 2026-07-08 copy wherever both exist, the self-test again gives 0.829375, CI [0.761219, 0.902516], and LI-MAR scoring reproduces this record exactly. The regenerated copy is the master; the ILS distribution copy will be a blinded derivative of it.

The scoring tool writes `cho_results.json` to the working directory on every run, so the repository-root copy is ignored by git and this archived record is the reference.
