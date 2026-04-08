# Engineering Hardening: Package Extraction, Tests, CI

**Date:** 2026-04-08
**Status:** Approved

## Summary

Five improvements to bring the MAR ILS reference implementation to production-grade engineering standards, without changing any physics or metrology behavior.

## 1. Package Extraction (`mar_ils_core/`)

Extract shared constants and utilities into an importable package. Fan-beam projection/FBP/Numba kernels stay in the generator.

```
mar_ils_core/
    __init__.py       # version, convenience imports
    constants.py      # all normative physics/geometry constants
    phantom.py        # mask builders, build_attenuation_map
    noise.py          # apply_noise()
    dicom_utils.py    # DICOM read/write, CP-2575 macro
```

- `generator_v7_0_0.py` imports constants and phantom/noise/DICOM from the package
- `run_cho_analysis_v7_0.py` imports constants from the package
- `algorithms/` import constants from `mar_ils_core.constants`

## 2. Test Suite

pytest in `tests/`, covering:

| File | Coverage |
|------|----------|
| `test_constants.py` | Locked baseline values match, no cross-file drift |
| `test_phantom.py` | Mask shapes, voxel counts, lesion at (281,256) |
| `test_noise.py` | Poisson+Gaussian statistics (mean, variance) |
| `test_roundtrip_fbp.py` | Forward project -> FBP -> tissue ~40 HU |
| `test_cho.py` | LG channel shapes, MW-AUC on synthetic data |
| `test_dicom.py` | CP-2575 tags, RescaleSlope/Intercept |
| `test_integration.py` | --dry-run exits 0 |

## 3. Algorithm Quarantine

- `algorithms/README.md`: parallel-beam only, incompatible with v7.0.0
- Runtime `warnings.warn()` in each algorithm's `main()`
- Update constant imports to `mar_ils_core.constants`

## 4. Requirements Pinning

- Pin to exact installed versions
- Remove `torch>=2.0.0` (unused)
- `numba` stays optional (documented, not required)

## 5. CI Pipeline

`.github/workflows/ci.yml`:
- **Lint**: ruff
- **Test**: pytest (Python 3.10)
- **Dry-run**: `python generator_v7_0_0.py --dry-run`

Triggered on push/PR to main.

## Constraints

- Zero changes to physics, noise model, or metrology behavior
- Locked baselines must remain unchanged
- Versioned filename convention preserved
