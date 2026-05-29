# LI-MAR v7 Reference — Design Spec

**Date:** 2026-05-29
**Status:** Approved (design); pending implementation plan
**Scope:** A single new reference algorithm; non-normative.

## 1. Purpose

A faithful, deterministic linear-interpolation Metal Artifact Reduction (LI-MAR)
implementation operating on the **v7.0.0 fan-beam** dataset. It is **not part of
the normative measurement** (the ASTM type test measures the *lab's* algorithm
via ΔAUC). It exists to:

1. **Positive control** — exercise the full "MAR-applied" path (sinogram → MAR →
   fan-beam FBP → DICOM → CHO) with a known, deterministic algorithm. Today the
   only end-to-end check is `--self-test` (noMAR-vs-noMAR, ΔAUC = 0 by
   definition), a *null* control; nothing currently proves the pipeline detects a
   real, non-zero MAR effect with the correct sign and magnitude.
2. **ΔAUC anchor** — establish a reproducible reference rung. LI-MAR is the
   universal, parameter-free baseline (Kalender 1987) that every commercial MAR
   is expected to beat; a lab scoring below it has a red flag.
3. **Worked I/O example** — a runnable demonstration of the contract a
   participating lab must satisfy.

LI-MAR is chosen over NMAR/iMAR precisely because it is parameter-free,
deterministic, and auditable — the right properties for a *reference*.

## 2. Design Decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Metal detection | **Threshold the noMAR recon** (`HU > METAL_HU_THRESH`) | Textbook LI-MAR; no privileged ground-truth knowledge — mirrors what a real lab algorithm does. |
| Output scope | **Slice 129 only** | What the CHO reads; matches the v6 reference pattern; sufficient for anchor + validation. Avoids 256× wasted compute. |
| Code location | **`algorithms/v7/`** (new subdir) | Separates fan-beam references from the legacy parallel-beam ones; anticipates future v7 ports. |
| Module structure | **Single self-contained script** (Approach A) | YAGNI: clean internal functions make it testable without a speculative shared I/O layer. Extract shared helpers when a 2nd v7 algorithm arrives. |
| Geometry | **Import `forward_project_slice` / `fbp_reconstruct_slice` from `generator_v7_0_0`** | Reuse the validated fan-beam projector/FBP. Deliberately does **not** extract them into `mar_ils_core` — keeps the change out of the locked physics code. |

## 3. Architecture & Data Flow

Per realization × condition (LP/LA):

```
sinograms/{cond}/realization_NNN.h5   ──► line_integrals[128]  (720×512) float64
noMAR_recon/{cond}/.../slice_0129.dcm ──► noMAR HU image (512×512)
generator_provenance.json             ──► dc_offset_cm

1. metal_mask  = noMAR_HU > METAL_HU_THRESH (2000)        # textbook detection
2. metal_trace = forward_project_slice(metal_mask.float)  # fan-beam footprint
   W           = clean-ray mask: 1 where trace < 0.05 × per-view peak, else 0
                 (0.05 matches the v6 references' CLEAN_RAY_FRAC)
3. sino_li     = linear_interp_metal(sino, W)             # fill trace per view
4. hu_corr     = fbp_reconstruct_slice(sino_li, dc_offset_cm)
5. write slice_0129.dcm via dicom_utils.write_dicom_slice(..., metal_mask=metal_mask)
        → <output>/{cond}/realization_NNN/slice_0129.dcm
```

Metal pixels are hard-set to 3000 HU by `write_dicom_slice`, consistent with the
noMAR series. The interpolation in step 3 operates along the detector axis and is
geometry-agnostic (works for fan-beam as for parallel-beam).

## 4. Components (`algorithms/v7/reference_li_mar_v7.py`)

| Function | Responsibility | Depends on |
|---|---|---|
| `detect_metal_mask(hu, thresh) -> ndarray[bool]` | Threshold noMAR HU image | — |
| `metal_trace_weights(metal_mask, clean_frac=0.05) -> ndarray` | Forward-project mask, threshold per-view (`< clean_frac × peak` ⇒ clean) → clean-ray weight `W` | `generator_v7_0_0.forward_project_slice` |
| `linear_interp_metal(sino, W) -> ndarray` | Linear interp along detector axis to fill metal-traced rays | — (ported from `reference_nmar.py`) |
| `li_mar_slice(sino, noMAR_hu, dc_offset_cm, thresh) -> (hu_corr, metal_mask)` | Orchestrate one slice (steps 1–4) | `fbp_reconstruct_slice`, above |
| `load_dc_offset(dataset_dir) -> float` | Read `dc_offset_cm` from `generator_provenance.json` | — |
| `process_realization(dataset_dir, output_dir, cond, idx, dc_offset_cm, thresh)` | Per-realization I/O (read H5 + noMAR DICOM, call `li_mar_slice`, write DICOM) | `h5py`, `pydicom`, `dicom_utils.write_dicom_slice` |
| `main()` | CLI parsing + serial loop with `tqdm` | `argparse` |

Each unit is testable in isolation. `linear_interp_metal` and
`detect_metal_mask` are pure functions; `li_mar_slice` is pure given its inputs.

## 5. CLI

```bash
python algorithms/v7/reference_li_mar_v7.py \
    --dataset-dir ./astm_reference_dataset \
    --output-dir  ./li_mar_recon \
    [--realizations N]        # default: auto-detect from sinograms/LP/
    [--metal-hu-thresh 2000]
    [--dc-offset-cm <float>]  # override; otherwise read from provenance
```

Output layout is a drop-in for the CHO analysis:

```
<output-dir>/{LP,LA}/realization_NNN/slice_0129.dcm
```

```bash
python run_cho_analysis_v7_0.py \
    --dataset-dir ./astm_reference_dataset \
    --mar-output-dir ./li_mar_recon \
    --internal-noise-sigma 15
```

## 6. Error Handling

- Missing sinogram H5, noMAR DICOM, or provenance JSON → raise with the offending path.
- `dc_offset_cm` absent from provenance **and** no `--dc-offset-cm` → hard error
  with a hint. Never silently default to `0.0` (would mis-calibrate HU by ~141 HU).
- Sinogram shape ≠ `(256, 720, 512)` → assertion with the actual shape.
- A projection view with no clean rays (all metal) → leave that view unmodified
  (guarded, matching the v6 reference behaviour).

## 7. Testing

**Unit (CI, fast — no full dataset required):**
- `linear_interp_metal`: synthetic sinogram with known metal columns → filled by
  linear interpolation between clean neighbours; metal-free input (`W` all clean)
  → returned unchanged.
- `detect_metal_mask`: image containing a 3000-HU disc → mask matches the disc.
- End-to-end slice: `build_attenuation_map(place_lesion=True)` →
  `forward_project_slice` → noMAR `fbp_reconstruct_slice` → `li_mar_slice` →
  assert output is finite, tissue ROI ≈ 40 HU, metal region recovered. Single
  slice, small, fast (uses the numba path if present, else NumPy fallback).

**Anchor (manual, documented — NOT in CI):**
- Run on the real reference dataset; feed the output to `run_cho_analysis_v7_0.py`;
  record the resulting ΔAUC (and CI) in `algorithms/v7/README.md`. A full dataset
  is ~35 GB / ~10 min to generate, so this is a documented manual step, not CI.

## 8. Deliverables

- `algorithms/v7/reference_li_mar_v7.py` — the algorithm + CLI.
- `algorithms/v7/README.md` — purpose, usage, and the measured ΔAUC anchor
  (placeholder until the manual run is done).
- `tests/test_li_mar_v7.py` — the unit tests above.
- `algorithms/README.md` — status-table row noting `v7/reference_li_mar_v7.py`
  as the one fan-beam-compatible reference.

## 9. Non-Goals (YAGNI)

- No full 256-slice volume reconstruction.
- No shared `algorithms/v7/_mar_io.py` layer (extract when a 2nd v7 algorithm lands).
- No NMAR / iMAR / other ports.
- No parallelism initially (slice-129-only × 80 realizations is fast serially).
- No metal re-insertion logic beyond the existing `write_dicom_slice` hard-set.
- No changes to `generator_v7_0_0.py`, `mar_ils_core/`, or any locked physics.
```
