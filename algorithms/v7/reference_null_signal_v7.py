"""
Null-signal check — LI-MAR with the metal trace widened to cover every ray
through the lesion, on the fan-beam dataset (v7.0.0 or v8.0.0 geometry).

Non-normative. Not a MAR method and not a degrading control: it removes the
lesion signal from the sinogram, so the CHO should score it at chance
(AUC_MAR = 0.5) and its ΔAUC is close to 0.5 − AUC_noMAR by construction.
It checks that the observer goes to chance when the near-metal signal is
removed. It does not show that ΔAUC separates MAR methods.

Pipeline (per realization, slice 128 / slice_0129.dcm only):
  1. metal_mask  = noMAR_HU > METAL_HU_THRESH       (as reference_li_mar_v7)
  2. trace_mask  = disc of radius NULL_RADIUS_VOX about the metal centroid
  3. sino_li     = linear interpolation across the trace of trace_mask
  4. hu_corr     = fbp_reconstruct_slice(sino_li, dc_offset_cm)
  5. write slice_0129.dcm with metal hard-set from metal_mask only, so the
     3000 HU hard-set never touches the lesion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_li_mar_v7 import (
    METAL_HU_THRESH,
    detect_metal_mask,
    dicom_write_kwargs,
    discover_realizations,
    linear_interp_metal,
    load_dc_offset,
    metal_trace_weights,
)

from generator_v7_0_0 import fbp_reconstruct_slice
from mar_ils_core.constants import (
    LESION_CENTER_X,
    LESION_RADIUS_VOX,
    LESION_SLICE_INDEX,
    PHANTOM_CENTER_X,
)
from mar_ils_core.dicom_utils import write_dicom_slice

# Lesion offset from the rod centre + lesion radius + 1 voxel: the smallest
# disc whose trace contains every ray through the lesion (31 voxels).
NULL_RADIUS_VOX: int = (LESION_CENTER_X - PHANTOM_CENTER_X) + LESION_RADIUS_VOX + 1


def null_signal_slice(
    sino: np.ndarray,
    nomar_hu: np.ndarray,
    dc_offset_cm: float,
    radius_vox: float = NULL_RADIUS_VOX,
    thresh: float = METAL_HU_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate across a disc of ``radius_vox`` about the metal.

    Returns (hu_corrected, metal_mask); metal_mask is the detected rod only.
    """
    metal_mask = detect_metal_mask(nomar_hu, thresh)
    cy, cx = np.argwhere(metal_mask).mean(axis=0)
    yy, xx = np.mgrid[0:metal_mask.shape[0], 0:metal_mask.shape[1]]
    trace_mask = metal_mask | ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius_vox ** 2)
    W = metal_trace_weights(trace_mask)
    sino_li = linear_interp_metal(np.asarray(sino, dtype=np.float64), W)
    return fbp_reconstruct_slice(sino_li, dc_offset_cm=dc_offset_cm), metal_mask


def process_realization(dataset_dir, output_dir, cond: str, idx: int,
                        dc_offset_cm: float, radius_vox: float) -> None:
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    tag = f"realization_{idx + 1:03d}"
    with h5py.File(str(dataset_dir / "sinograms" / cond / f"{tag}.h5"), "r") as f:
        sino = f["line_integrals"][LESION_SLICE_INDEX].astype(np.float64)
    ds = pydicom.dcmread(str(dataset_dir / "noMAR_recon" / cond / tag
                             / f"slice_{LESION_SLICE_INDEX + 1:04d}.dcm"))
    nomar_hu = (ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope)
                + float(ds.RescaleIntercept))
    hu, metal_mask = null_signal_slice(sino, nomar_hu, dc_offset_cm, radius_vox)
    write_dicom_slice(
        hu, LESION_SLICE_INDEX,
        output_dir=output_dir / cond / tag,
        realization_idx=idx,
        condition_label=cond,
        study_uid=generate_uid(),
        series_uid=generate_uid(),
        metal_mask=metal_mask,
        mar_applied="YES",
        **dicom_write_kwargs(dataset_dir),
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Null-signal check (non-normative): LI-MAR with the trace "
                    "widened over the lesion; writes slice_0129.dcm."
    )
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--output-dir", default="./null_signal_recon")
    ap.add_argument("--realizations", type=int, default=None,
                    help="count per condition (default: auto-detect)")
    ap.add_argument("--radius-vox", type=float, default=NULL_RADIUS_VOX)
    args = ap.parse_args(argv)

    dc = load_dc_offset(args.dataset_dir)
    for cond in ("LP", "LA"):
        n = (args.realizations if args.realizations is not None
             else discover_realizations(args.dataset_dir, cond))
        for i in tqdm(range(n), desc=f"null-signal {cond}"):
            process_realization(args.dataset_dir, args.output_dir, cond, i, dc,
                                args.radius_vox)


if __name__ == "__main__":
    main()
