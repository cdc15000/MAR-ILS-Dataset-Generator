#!/usr/bin/env python3
"""
MCP server for inspecting MAR ILS CHO result files.

Provides tools to list, read, and summarize JSON result files
from results_archive/ and tdp_output/.
"""

import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

PROJECT_DIR = Path(__file__).parent
DATA_DIRS = [
    PROJECT_DIR / "results_archive",
    PROJECT_DIR / "tdp_output",
]

mcp = FastMCP("mar-ils-data-inspector")


@mcp.tool()
def list_result_files() -> str:
    """List all available result files in results_archive/ and tdp_output/."""
    lines = []
    for d in DATA_DIRS:
        if not d.exists():
            continue
        lines.append(f"\n## {d.name}/")
        for f in sorted(d.iterdir()):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                lines.append(f"  {f.name}  ({size_kb:.1f} KB)")
    return "\n".join(lines) if lines else "No result directories found."


@mcp.tool()
def read_result_file(filename: str) -> str:
    """Read and return the contents of a result file by name.

    Args:
        filename: Name of the file (e.g. 'results_t2_sb.json'). Searched
                  in results_archive/ then tdp_output/.
    """
    for d in DATA_DIRS:
        path = d / filename
        if path.exists():
            return path.read_text()
    return f"File '{filename}' not found in {[d.name for d in DATA_DIRS]}."


@mcp.tool()
def summarize_cho_results(filename: str) -> str:
    """Parse a CHO result JSON file and return a human-readable summary.

    Extracts AUC values, confidence intervals, and ΔAUC if present.

    Args:
        filename: Name of the JSON result file.
    """
    data = None
    for d in DATA_DIRS:
        path = d / filename
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                return f"Failed to parse {filename}: {e}"
            break
    if data is None:
        return f"File '{filename}' not found."

    lines = [f"# CHO Summary: {filename}\n"]

    # Handle both flat and nested structures
    def fmt_auc(label, obj, key_auc, key_ci=None):
        auc = obj.get(key_auc)
        if auc is None:
            return
        ci = obj.get(key_ci) if key_ci else None
        if ci and isinstance(ci, (list, tuple)) and len(ci) == 2:
            lines.append(f"  {label}: {auc:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
        elif auc is not None:
            lines.append(f"  {label}: {auc:.4f}")

    # Try common key patterns
    for section_key in [None, "cho_results", "results"]:
        obj = data.get(section_key, data) if section_key else data

        fmt_auc("AUC (noMAR)", obj, "auc_noMAR", "auc_noMAR_ci")
        fmt_auc("AUC (noMAR)", obj, "auc_nomar", "auc_nomar_ci")
        fmt_auc("AUC (MAR)", obj, "auc_MAR", "auc_MAR_ci")
        fmt_auc("AUC (MAR)", obj, "auc_mar", "auc_mar_ci")
        fmt_auc("ΔAUC", obj, "delta_auc", "delta_auc_ci")
        fmt_auc("ΔAUC", obj, "deltaAUC", "deltaAUC_ci")

        pipeline = obj.get("reconstruction_pipeline") or obj.get("pipeline")
        if pipeline:
            lines.append(f"  Pipeline: {pipeline}")
        tier = obj.get("tier")
        if tier:
            lines.append(f"  Tier: {tier}")

    if len(lines) == 1:
        lines.append("  (no recognized CHO fields found — use read_result_file for raw JSON)")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
