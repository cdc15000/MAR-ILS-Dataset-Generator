# research/ — non-normative tooling

These are exploratory / research-era utilities. **None are part of the v7.0.0
normative reference** (ASTM Rev 04, fan-beam). They predate the v7 single-config
framework and mostly target the legacy v6 parallel-beam tiered framework
(`tier_config.py`, `legacy/generator_v6_0_0.py`, `legacy/run_cho_analysis_v6_0.py`).

The normative entry points remain at the repo root: `generator_v7_0_0.py`,
`run_cho_analysis_v7_0.py`, `view_sinograms.py`, `patch_2026b_metadata.py`.

| Script | Purpose | Notes |
|---|---|---|
| `generate_tdp_report.py` | Technical Data Package PDF comparing algorithms across tiers | Imports `tier_config` via a repo-root path shim |
| `generate_detectability_curves.py` | AUC vs. contrast/noise plots | Uses `skimage` |
| `sweep_vaishnav_matrix.py` | Batch sensitivity sweep (contrast × noise) | Subprocess-invokes v6 scripts by bare name; **paths predate the move to `legacy/` and `algorithms/`** |
| `apply_mar_sir.py` | Standalone Selective Inpainting Reconstruction | References the retired `run_cho_analysis_v5_3.py` |
| `plot_spectral_transparency.py` | 60 keV vs. 140 keV "transparency jump" plot | Standalone |
| `plot_t3_head_comparison.py` | T3_HEAD tier comparison plot | Standalone |
| `check_metal_overflow.py` | Audit of the 3000 HU metal-ROI hard-set | Standalone |
| `mcp_data_inspector.py` | MCP server: summarize CHO JSON results | Requires the `mcp` package |
| `mcp_visualization.py` | MCP server: render sinogram slices / ROI comparisons | Requires the `mcp` package |

Runtime correctness against the current layout is not guaranteed — several of
these reference scripts or geometries that have since moved or been superseded.
They are retained for reference and reuse, not as supported entry points.
