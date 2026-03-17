#!/usr/bin/env python3
"""
sweep_vaishnav_matrix.py
========================
ASTM WKXXXXX v1.0.0 — Automated Vaishnav Sensitivity Matrix Sweep

For each tier × contrast_factor combination:
  1. Generate sweep-mode dataset (--sweep-mode: only slice_0129.dcm written)
  2. Run tier-adapted iMAR reconstruction
  3. Score with run_cho_analysis_v6_0.py --noise-sigma-sweep (11-point σ sweep)
  4. Patch the result JSON with contrast_factor and sweep_label for TDP reporting

Default sweep:
  Tiers           : T1_AB, T2_SB
  Contrast factors: 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6  (7 points)
  Sigma sweep     : 0, 5, 10, 15, 20, 25, 30, 40, 50, 65, 80  (11 points, auto)

Total data points per tier: 7 × 11 = 77 (the "Vaishnav Sensitivity Matrix")

Output layout
-------------
  <sweep-root>/
    data/
      T1_AB_cf_1.0/   ← generator + iMAR output (sweep mode)
      T1_AB_cf_1.1/
      ...
      T2_SB_cf_1.6/
    results/
      T1_AB_cf_1.0.json   ← CHO results (patched with contrast_factor)
      T1_AB_cf_1.1.json
      ...
      T2_SB_cf_1.6.json

Restartable: if results/<label>.json already exists, that combination is skipped.

Usage
-----
    # Full sweep (default tiers + contrast factors):
    python sweep_vaishnav_matrix.py

    # Custom sweep root:
    python sweep_vaishnav_matrix.py --sweep-root ./my_sweep

    # Subset of tiers / contrast factors:
    python sweep_vaishnav_matrix.py --tiers T1_AB --contrast-factors 1.0 1.2 1.4

    # Dry run (print plan, no execution):
    python sweep_vaishnav_matrix.py --dry-run

    # Include T3_HEAD (slower — larger body):
    python sweep_vaishnav_matrix.py --tiers T1_AB T2_SB T3_HEAD
After CHO results are written and verified, the per-cell data directory
(sinograms + DICOMs ≈ 14 GB per cell) is automatically deleted.  Only the
small JSON results files (~50 KB each) are kept.  This keeps SSD usage stable
at ≈ 1× cell size regardless of sweep length.

Storage profile per cell (T1_AB):
  sinograms/LP + LA  : 80 × ~180 MB HDF5 = ~14.4 GB
  noMAR_recon        : 80 × 1 DICOM      = ~80 MB
  imar_recon         : 80 × 1 DICOM      = ~80 MB
  Total data dir     : ~14.6 GB  ← deleted after JSON verified
  JSON result        : ~50 KB    ← kept forever
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ─── Sweep parameters ─────────────────────────────────────────────────────────

DEFAULT_TIERS            = ["T1_AB", "T2_SB"]
DEFAULT_CONTRAST_FACTORS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]


# ─── Disk helpers ─────────────────────────────────────────────────────────────

def _disk_free_gb(path: Path) -> float:
    """Return free space in GB on the filesystem containing path."""
    return shutil.disk_usage(str(path)).free / (1024 ** 3)


def _cleanup_dataset(dataset_path: Path, sweep_root: Path) -> None:
    """
    Delete the per-cell data directory after results JSON is safely written.

    Called only when result_path.exists() has been verified.  Prints the
    amount of space freed and the new free total.
    """
    if not dataset_path.exists():
        return
    free_before = _disk_free_gb(sweep_root)
    try:
        shutil.rmtree(dataset_path)
        free_after = _disk_free_gb(sweep_root)
        print(
            f"  CLEANUP → {dataset_path.name} removed  "
            f"(freed {free_after - free_before:.1f} GB, "
            f"{free_after:.1f} GB free)",
            flush=True,
        )
    except Exception as exc:
        print(f"  WARNING: cleanup failed for {dataset_path}: {exc}")


# ─── Step runner ──────────────────────────────────────────────────────────────

def _run(cmd: list, label: str, dry_run: bool = False) -> None:
    """Run a subprocess step.  Raises CalledProcessError on non-zero exit."""
    cmd_strs = [str(c) for c in cmd]
    short = " ".join(cmd_strs[1:3]) + " ..."   # script + first two args
    print(f"    [{label}]  {short}", flush=True)
    if dry_run:
        print(f"    [DRY-RUN]  {' '.join(cmd_strs)}")
        return
    result = subprocess.run(cmd_strs, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Step '{label}' failed with exit code {result.returncode}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ASTM WKXXXXX v1.0.0 — Vaishnav Sensitivity Matrix sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1] if "Usage" in __doc__ else "",
    )
    ap.add_argument("--tiers", nargs="+", default=DEFAULT_TIERS,
                    metavar="TIER",
                    help="Tiers to sweep (default: T1_AB T2_SB)")
    ap.add_argument("--contrast-factors", nargs="+", type=float,
                    default=DEFAULT_CONTRAST_FACTORS,
                    metavar="CF",
                    help="Contrast factors to sweep (default: 1.0 1.1 1.2 1.3 1.4 1.5 1.6)")
    ap.add_argument("--sweep-root", default="./sweep_workspace",
                    help="Root directory for all sweep data (default: ./sweep_workspace)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Workers for generator/iMAR parallel steps (0 = cpu_count)")
    ap.add_argument("--internal-noise-sigma", type=float, default=15.0,
                    help="Normative CHO noise sigma (default: 15.0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the execution plan without running anything")
    args = ap.parse_args()

    sweep_root  = Path(args.sweep_root).resolve()
    data_root   = sweep_root / "data"
    results_dir = sweep_root / "results"
    data_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    total    = len(args.tiers) * len(args.contrast_factors)
    n_done   = 0
    n_skip   = 0
    n_err    = 0
    errors   = []

    print("ASTM WKXXXXX v1.0.0 — Vaishnav Sensitivity Matrix Sweep")
    print("=" * 64)
    print(f"  Tiers           : {', '.join(args.tiers)}")
    print(f"  Contrast factors: {args.contrast_factors}")
    print(f"  Total cells     : {total}  ({total * 11} data points with 11-pt σ sweep)")
    print(f"  Sweep root      : {sweep_root}")
    print(f"  Workers         : {args.workers if args.workers > 0 else 'cpu_count()'}")
    if args.dry_run:
        print("  MODE            : DRY-RUN (no execution)")
    print("=" * 64)

    aborted = False
    for tier_id in args.tiers:
        if aborted:
            break
        for cf in args.contrast_factors:
            label        = f"{tier_id}_cf_{cf:.1f}"
            dataset_path = data_root   / label
            result_path  = results_dir / f"{label}.json"

            free_gb = _disk_free_gb(sweep_root)
            print(f"\n[{n_done + n_skip + 1}/{total}]  {label}  "
                  f"[disk free: {free_gb:.1f} GB]")
            print("-" * 48)

            if free_gb < 20.0 and not args.dry_run:
                print(f"  ABORT — disk space critically low ({free_gb:.1f} GB). "
                      f"Sweep is restartable; free space and re-run.")
                aborted = True
                break

            if result_path.exists() and not args.dry_run:
                print(f"  SKIP — results exist: {result_path}")
                n_skip += 1
                continue

            try:
                # ── Step 1: Generate sweep-mode dataset ───────────────────
                _run([
                    py, "generator_v6_0_0.py",
                    "--tier",            tier_id,
                    "--contrast-factor", f"{cf:.2f}",
                    "--sweep-mode",
                    "--output-dir",      str(dataset_path),
                    "--workers",         str(args.workers),
                    "--no-pdf",
                ], f"GEN {label}", dry_run=args.dry_run)

                # ── Step 2: iMAR reconstruction ───────────────────────────
                _run([
                    py, "reference_imar.py",
                    "--tier",       tier_id,
                    "--input-dir",  str(dataset_path),
                    "--workers",    str(args.workers),
                ], f"iMAR {label}", dry_run=args.dry_run)

                # ── Step 3: CHO analysis with noise sigma sweep ───────────
                imar_dir = dataset_path / "imar_recon"
                _run([
                    py, "run_cho_analysis_v6_0.py",
                    "--dataset-dir",          str(dataset_path),
                    "--mar-output-dir",       str(imar_dir),
                    "--tier",                 tier_id,
                    "--internal-noise-sigma", str(args.internal_noise_sigma),
                    "--noise-sigma-sweep",
                    "--results-file",         str(result_path),
                ], f"CHO {label}", dry_run=args.dry_run)

                # ── Step 4: Patch JSON with contrast_factor ───────────────
                if not args.dry_run and result_path.exists():
                    with open(result_path) as fh:
                        data = json.load(fh)   # also validates JSON integrity
                    data["contrast_factor"] = float(cf)
                    data["sweep_label"]     = label
                    with open(result_path, "w") as fh:
                        json.dump(data, fh, indent=2)
                    print(f"  JSON  → {result_path}")

                    # ── Step 5: Cleanup data dir (JSON verified above) ─────
                    _cleanup_dataset(dataset_path, sweep_root)
                    print(f"  DONE  [{label}]")

                n_done += 1

            except Exception as exc:
                n_err += 1
                errors.append((label, str(exc)))
                print(f"  ERROR: {exc}")
                # Continue with remaining cells (do not abort the sweep)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(f"Sweep complete:  {n_done} run  |  {n_skip} skipped  |  {n_err} errors")
    if errors:
        print("\nError summary:")
        for lbl, msg in errors:
            print(f"  {lbl}: {msg}")
    print(f"\nResults directory: {results_dir}")
    print(f"  {len(list(results_dir.glob('*.json')))} JSON files")

    print()
    print("Next step — generate consolidated TDP with sensitivity surface:")
    print("  python generate_tdp_report.py \\")
    print("      --results results_t1_ab.json results_t2_sb.json results_t3_head.json \\")
    print("      --mar-name 'iMAR v1.0 (Seamless Prior + SG)' \\")
    print(f"      --sweep-dir {results_dir}")


if __name__ == "__main__":
    main()
