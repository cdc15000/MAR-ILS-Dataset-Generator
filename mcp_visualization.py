#!/usr/bin/env python3
"""
MCP server for MAR ILS sinogram and reconstruction visualization.

Provides tools to inspect HDF5 sinogram metadata and generate
static PNG plots (saved to a temp directory).
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
from mcp.server.fastmcp import FastMCP

PROJECT_DIR = Path(__file__).parent
PLOT_DIR = Path(tempfile.mkdtemp(prefix="mar_ils_plots_"))

mcp = FastMCP("mar-ils-visualization")


@mcp.tool()
def list_sinograms(dataset_dir: str = "astm_mar_ils_dataset_v5") -> str:
    """List available sinogram HDF5 files in a dataset directory.

    Args:
        dataset_dir: Dataset directory name relative to the project root.
    """
    base = PROJECT_DIR / dataset_dir / "sinograms"
    if not base.exists():
        return f"No sinograms directory at {base}"

    lines = [f"# Sinograms in {dataset_dir}/sinograms/\n"]
    for condition in ["LP", "LA"]:
        cond_dir = base / condition
        if not cond_dir.exists():
            continue
        files = sorted(cond_dir.glob("*.h5"))
        lines.append(f"## {condition}/ ({len(files)} files)")
        for f in files[:5]:
            lines.append(f"  {f.name}")
        if len(files) > 5:
            lines.append(f"  ... and {len(files) - 5} more")
    return "\n".join(lines)


@mcp.tool()
def sinogram_metadata(h5_path: str) -> str:
    """Read metadata from a sinogram HDF5 file without loading the full array.

    Args:
        h5_path: Path to the .h5 file (absolute, or relative to project root).
    """
    path = Path(h5_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.exists():
        return f"File not found: {path}"

    with h5py.File(path, "r") as f:
        shape = f["line_integrals"].shape
        dtype = str(f["line_integrals"].dtype)
        geo = {k: _serialize(f["geometry"].attrs[k]) for k in f["geometry"].attrs}
        noise = {k: _serialize(f["noise_params"].attrs[k]) for k in f["noise_params"].attrs}

    lines = [
        f"# {path.name}",
        f"  Shape: {shape}  dtype: {dtype}",
        f"  Geometry: {geo}",
        f"  Noise params: {noise}",
    ]
    return "\n".join(lines)


@mcp.tool()
def plot_sinogram_slice(h5_path: str, slice_idx: int = 128) -> str:
    """Render a sinogram slice to a PNG file and return the file path.

    Args:
        h5_path: Path to the .h5 file.
        slice_idx: Slice index to render (default: 128, the lesion slice).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(h5_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.exists():
        return f"File not found: {path}"

    with h5py.File(path, "r") as f:
        sino = f["line_integrals"][slice_idx]  # (N_ANGLES, N_DET)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(sino, aspect="auto", cmap="gray", extent=[0, sino.shape[1], 180, 0])
    ax.set_xlabel("Detector Bin")
    ax.set_ylabel("Projection Angle (deg)")
    ax.set_title(f"{path.name} — slice {slice_idx}")
    plt.colorbar(im, ax=ax, label="Line Integral (neper)")
    plt.tight_layout()

    out = PLOT_DIR / f"{path.stem}_slice{slice_idx}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return f"Plot saved: {out}"


@mcp.tool()
def plot_sinogram_comparison(
    lp_path: str, la_path: str, slice_idx: int = 128
) -> str:
    """Render LP vs LA sinogram comparison to a PNG file.

    Args:
        lp_path: Path to the LP sinogram .h5 file.
        la_path: Path to the LA sinogram .h5 file.
        slice_idx: Slice index to compare (default: 128).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lp = Path(lp_path)
    la = Path(la_path)
    if not lp.is_absolute():
        lp = PROJECT_DIR / lp
    if not la.is_absolute():
        la = PROJECT_DIR / la
    for p in [lp, la]:
        if not p.exists():
            return f"File not found: {p}"

    with h5py.File(lp, "r") as f:
        sino_lp = f["line_integrals"][slice_idx]
    with h5py.File(la, "r") as f:
        sino_la = f["line_integrals"][slice_idx]

    diff = sino_lp - sino_la

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title, cmap in [
        (axes[0], sino_lp, f"LP — slice {slice_idx}", "gray"),
        (axes[1], sino_la, f"LA — slice {slice_idx}", "gray"),
        (axes[2], diff, "LP − LA", "RdBu_r"),
    ]:
        kw = {}
        if cmap == "RdBu_r":
            lim = np.abs(diff).max() or 1
            kw = {"vmin": -lim, "vmax": lim}
        im = ax.imshow(data, aspect="auto", cmap=cmap,
                       extent=[0, data.shape[1], 180, 0], **kw)
        ax.set_xlabel("Detector Bin")
        ax.set_ylabel("Angle (deg)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()

    out = PLOT_DIR / f"compare_slice{slice_idx}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return f"Comparison saved: {out}"


def _serialize(val):
    """Convert numpy types to Python builtins for display."""
    if isinstance(val, np.ndarray):
        if val.size <= 5:
            return val.tolist()
        return f"array({val.shape}, {val.dtype})"
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


if __name__ == "__main__":
    mcp.run()
