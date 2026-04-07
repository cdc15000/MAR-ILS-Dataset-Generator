#!/usr/bin/env python3
"""
patch_2026b_metadata.py — DICOM 2026b CP-2575 Metadata Patcher

One-time utility to inject the Metal Artifact Reduction Macro (C.8.15.3.15)
into existing DICOM files produced by generator_v7_0_0.py.

Tags added:
  (0018,9390) Metal Artifact Reduction Sequence  [SQ]
  (0018,9391) Metal Artifact Reduction Applied   [CS] = "NO"

After patching, checksums_sha256.txt is regenerated.

Usage:
  python patch_2026b_metadata.py --dataset-dir ./astm_reference_dataset
  python patch_2026b_metadata.py --dataset-dir ./astm_reference_dataset --dry-run
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset


TAG_MAR_SEQ = 0x00189390        # Metal Artifact Reduction Sequence
TAG_MAR_APPLIED = 0x00189391    # Metal Artifact Reduction Applied


def patch_dicom(dcm_path: Path, dry_run: bool = False) -> bool:
    """Inject CP-2575 MAR Macro into a single DICOM file.

    Returns True if the file was modified (or would be modified in dry-run).
    """
    ds = pydicom.dcmread(str(dcm_path))

    # Skip if already patched
    if TAG_MAR_SEQ in ds:
        return False

    mar_item = Dataset()
    mar_item.add_new(TAG_MAR_APPLIED, 'CS', 'NO')
    ds.add_new(TAG_MAR_SEQ, 'SQ', [mar_item])

    if not dry_run:
        ds.save_as(str(dcm_path), write_like_original=False)
    return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description='Patch existing DICOM dataset with DICOM 2026b CP-2575 MAR metadata.'
    )
    parser.add_argument('--dataset-dir', type=Path, required=True,
                        help='Path to astm_reference_dataset/')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would change without modifying files')
    args = parser.parse_args()

    recon_dir = args.dataset_dir / 'noMAR_recon'
    if not recon_dir.is_dir():
        print(f"ERROR: {recon_dir} not found.", file=sys.stderr)
        sys.exit(1)

    # Collect all DICOM files
    dcm_files = sorted(recon_dir.rglob('*.dcm'))
    if not dcm_files:
        print(f"ERROR: No .dcm files found under {recon_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(dcm_files)} DICOM files in {recon_dir}")
    if args.dry_run:
        print("DRY RUN — no files will be modified.\n")

    patched = 0
    skipped = 0
    for dcm_path in dcm_files:
        if patch_dicom(dcm_path, dry_run=args.dry_run):
            patched += 1
        else:
            skipped += 1

    print(f"\nPatched: {patched}  Skipped (already patched): {skipped}")

    if args.dry_run:
        print("\nDry run complete. Re-run without --dry-run to apply.")
        return

    # Regenerate checksums
    checksum_path = args.dataset_dir / 'checksums_sha256.txt'
    print(f"\nRegenerating {checksum_path} ...")

    # Collect all checksummable files (DICOMs + provenance JSON)
    all_files = []
    provenance = args.dataset_dir / 'generator_provenance.json'
    if provenance.exists():
        all_files.append(provenance)
    all_files.extend(sorted((args.dataset_dir / 'noMAR_recon').rglob('*.dcm')))

    with open(checksum_path, 'w') as f:
        f.write('# ASTM WKXXXXX v7.0.0 — SHA-256 manifest (DICOM 2026b patched)\n')
        for fpath in all_files:
            rel = fpath.relative_to(args.dataset_dir)
            digest = sha256_file(fpath)
            f.write(f'{digest}  {rel}\n')

    print(f"Written {len(all_files)} checksums to {checksum_path}")
    print("\nDone. Verify with: python -c \"import pydicom; "
          "ds=pydicom.dcmread('<dcm_path>'); print(ds[0x00189390])\"")


if __name__ == '__main__':
    main()
