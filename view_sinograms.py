#!/usr/bin/env python3
"""
view_sinograms.py
=================
Interactive viewer for MAR ILS sinogram HDF5 files

Usage:
    python view_sinograms.py <path_to_sinogram.h5>
    python view_sinograms.py astm_mar_ils_dataset_v5_optimized/sinograms/LP/realization_001.h5
    
    # View specific slice
    python view_sinograms.py <path> --slice 128
    
    # Compare LP vs LA
    python view_sinograms.py <LP_path> <LA_path> --compare
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def load_sinogram(h5_path: Path):
    """Load sinogram and metadata from HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        sinogram = f['line_integrals'][:]  # (Z_DIM, N_ANGLES, N_DET)
        
        # Load metadata
        geo = f['geometry']
        noise = f['noise_params']
        
        metadata = {
            'n_slices': geo.attrs['n_slices'],
            'n_angles': geo.attrs['n_angles'],
            'n_det': geo.attrs['n_det'],
            'voxel_mm': geo.attrs['voxel_mm'],
            'angles_deg': np.array(geo.attrs['angles_deg']),
            'I0': noise.attrs['I0'],
            'scatter_frac': noise.attrs['scatter_frac'],
            'sigma_e_counts': noise.attrs['sigma_e_counts'],
            'seed': noise.attrs['seed'],
            'jitter_deg': noise.attrs['jitter_deg'],
            'place_lesion': bool(noise.attrs['place_lesion']),
            'lesion_slice_index': noise.attrs['lesion_slice_index'],
        }
    
    return sinogram, metadata


def view_single_sinogram(h5_path: Path, slice_idx: int = None):
    """View a single sinogram with interactive slice selector."""
    print(f"Loading sinogram: {h5_path}")
    sinogram, meta = load_sinogram(h5_path)
    
    print(f"\nMetadata:")
    print(f"  Shape: {sinogram.shape} (slices, angles, detectors)")
    print(f"  Lesion present: {meta['place_lesion']}")
    print(f"  Lesion slice: {meta['lesion_slice_index']}")
    print(f"  Jitter: {meta['jitter_deg']:.2f}°")
    print(f"  I₀: {meta['I0']:.0f}")
    print(f"  Seed: {meta['seed']}")
    
    # Default to lesion slice if available
    if slice_idx is None:
        slice_idx = meta['lesion_slice_index'] if meta['place_lesion'] else meta['n_slices'] // 2
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Sinogram: {h5_path.name}\n"
                 f"{'Lesion Present (LP)' if meta['place_lesion'] else 'Lesion Absent (LA)'} | "
                 f"Jitter: {meta['jitter_deg']:.2f}° | I₀: {meta['I0']:.0f}",
                 fontsize=12, fontweight='bold')
    
    # Initial slice
    sino_slice = sinogram[slice_idx]  # (N_ANGLES, N_DET)
    
    # Plot sinogram
    im1 = axes[0].imshow(sino_slice, aspect='auto', cmap='gray', 
                         extent=[0, meta['n_det'], 180, 0])
    axes[0].set_xlabel('Detector Bin', fontsize=11)
    axes[0].set_ylabel('Projection Angle (°)', fontsize=11)
    axes[0].set_title(f'Sinogram - Slice {slice_idx}/{meta["n_slices"]-1}', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Line Integral (neper)', fontsize=10)
    
    # Plot single projection (angle 0°)
    im2 = axes[1].plot(sino_slice[0], 'b-', linewidth=1.5, label='0°')
    axes[1].plot(sino_slice[90], 'r-', linewidth=1.5, alpha=0.7, label='90°')
    axes[1].set_xlabel('Detector Bin', fontsize=11)
    axes[1].set_ylabel('Line Integral (neper)', fontsize=11)
    axes[1].set_title('Projections at 0° and 90°', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Add metal shadow indicator if present
    metal_center = meta['n_det'] // 2
    metal_radius_det = 10 / meta['voxel_mm']  # 10 voxel radius in detector space
    axes[1].axvspan(metal_center - metal_radius_det, 
                    metal_center + metal_radius_det,
                    alpha=0.2, color='red', label='Metal shadow region')
    axes[1].legend(fontsize=9)
    
    # Add slider for slice selection
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, meta['n_slices'] - 1, 
                    valinit=slice_idx, valstep=1)
    
    # Mark lesion slice if applicable
    if meta['place_lesion']:
        ax_slider.axvline(meta['lesion_slice_index'], color='red', 
                         linestyle='--', linewidth=2, alpha=0.7)
    
    def update(val):
        s = int(slider.val)
        sino_slice = sinogram[s]
        
        # Update sinogram image
        im1.set_data(sino_slice)
        im1.set_clim(vmin=sino_slice.min(), vmax=sino_slice.max())
        axes[0].set_title(f'Sinogram - Slice {s}/{meta["n_slices"]-1}', fontsize=11)
        
        # Update projections
        axes[1].clear()
        axes[1].plot(sino_slice[0], 'b-', linewidth=1.5, label='0°')
        axes[1].plot(sino_slice[90], 'r-', linewidth=1.5, alpha=0.7, label='90°')
        axes[1].set_xlabel('Detector Bin', fontsize=11)
        axes[1].set_ylabel('Line Integral (neper)', fontsize=11)
        axes[1].set_title('Projections at 0° and 90°', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].axvspan(metal_center - metal_radius_det, 
                       metal_center + metal_radius_det,
                       alpha=0.2, color='red', label='Metal shadow region')
        axes[1].legend(fontsize=9)
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.tight_layout()
    plt.show()


def compare_sinograms(lp_path: Path, la_path: Path, slice_idx: int = None):
    """Compare LP and LA sinograms side-by-side."""
    print(f"Loading LP: {lp_path}")
    sino_lp, meta_lp = load_sinogram(lp_path)
    
    print(f"Loading LA: {la_path}")
    sino_la, meta_la = load_sinogram(la_path)
    
    # Use lesion slice
    if slice_idx is None:
        slice_idx = meta_lp['lesion_slice_index']
    
    print(f"\nComparing slice {slice_idx}")
    print(f"  LP jitter: {meta_lp['jitter_deg']:.2f}°")
    print(f"  LA jitter: {meta_la['jitter_deg']:.2f}°")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"LP vs LA Comparison - Slice {slice_idx}\n"
                 f"LP Jitter: {meta_lp['jitter_deg']:.2f}° | "
                 f"LA Jitter: {meta_la['jitter_deg']:.2f}°",
                 fontsize=13, fontweight='bold')
    
    # LP sinogram
    im1 = axes[0, 0].imshow(sino_lp[slice_idx], aspect='auto', cmap='gray',
                            extent=[0, meta_lp['n_det'], 180, 0])
    axes[0, 0].set_xlabel('Detector Bin')
    axes[0, 0].set_ylabel('Projection Angle (°)')
    axes[0, 0].set_title('LP (Lesion Present)', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0], label='Line Integral (neper)')
    
    # LA sinogram
    im2 = axes[0, 1].imshow(sino_la[slice_idx], aspect='auto', cmap='gray',
                            extent=[0, meta_la['n_det'], 180, 0])
    axes[0, 1].set_xlabel('Detector Bin')
    axes[0, 1].set_ylabel('Projection Angle (°)')
    axes[0, 1].set_title('LA (Lesion Absent)', fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1], label='Line Integral (neper)')
    
    # Difference (LP - LA)
    diff = sino_lp[slice_idx] - sino_la[slice_idx]
    im3 = axes[1, 0].imshow(diff, aspect='auto', cmap='RdBu_r',
                            extent=[0, meta_lp['n_det'], 180, 0],
                            vmin=-diff.std()*3, vmax=diff.std()*3)
    axes[1, 0].set_xlabel('Detector Bin')
    axes[1, 0].set_ylabel('Projection Angle (°)')
    axes[1, 0].set_title('Difference (LP - LA)', fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0], label='Δ Line Integral (neper)')
    
    # Profile comparison at 0°
    axes[1, 1].plot(sino_lp[slice_idx, 0], 'b-', linewidth=2, label='LP (0°)', alpha=0.8)
    axes[1, 1].plot(sino_la[slice_idx, 0], 'r-', linewidth=2, label='LA (0°)', alpha=0.8)
    axes[1, 1].plot(diff[0], 'g-', linewidth=1.5, label='Difference', alpha=0.7)
    axes[1, 1].set_xlabel('Detector Bin')
    axes[1, 1].set_ylabel('Line Integral (neper)')
    axes[1, 1].set_title('Projection Profile at 0°', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Mark lesion location
    lesion_det = 281  # Lesion at x=281 projects to detector ~281 at 0°
    axes[1, 1].axvline(lesion_det, color='orange', linestyle='--', 
                      linewidth=2, alpha=0.5, label='Expected lesion')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='View MAR ILS sinogram HDF5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View single sinogram
  python view_sinograms.py sinograms/LP/realization_001.h5
  
  # View specific slice
  python view_sinograms.py sinograms/LP/realization_001.h5 --slice 128
  
  # Compare LP vs LA
  python view_sinograms.py sinograms/LP/realization_001.h5 \\
                           sinograms/LA/realization_001.h5 --compare
        """
    )
    
    parser.add_argument('paths', nargs='+', type=Path,
                        help='Path(s) to sinogram HDF5 file(s)')
    parser.add_argument('--slice', type=int, default=None,
                        help='Slice index to view (default: lesion slice or middle)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare two sinograms (LP vs LA)')
    
    args = parser.parse_args()
    
    # Validate paths
    for path in args.paths:
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        if not path.suffix == '.h5':
            print(f"Error: Not an HDF5 file: {path}", file=sys.stderr)
            sys.exit(1)
    
    # View mode
    if args.compare:
        if len(args.paths) != 2:
            print("Error: --compare requires exactly 2 paths", file=sys.stderr)
            sys.exit(1)
        compare_sinograms(args.paths[0], args.paths[1], args.slice)
    else:
        if len(args.paths) != 1:
            print("Error: Single view mode requires exactly 1 path", file=sys.stderr)
            sys.exit(1)
        view_single_sinogram(args.paths[0], args.slice)


if __name__ == '__main__':
    main()
