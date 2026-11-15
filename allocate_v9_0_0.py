#!/usr/bin/env python3
"""
Allocation of the full study's replicate datasets to laboratories (ILS protocol
§A6.3; issue #18).

The pool holds three replicate datasets per laboratory, ILS001 to ILSnnn,
generated with independent seeds (write_dataset_v9_0_0.py --dataset ILSnnn).
After recruitment closes, the pool is put in random order with seed 20261103,
divided into consecutive groups of three, and the groups are assigned to the
laboratories in the order of their laboratory codes. No dataset goes to more
than one laboratory.

The random order is the order of SHA-256("<seed>:<dataset name>"), so anyone
can reproduce the allocation from the seed, the dataset names and the codes,
in any language and with any library version. Laboratory codes are put in
natural order ("2" before "10").

Usage:
  python allocate_v9_0_0.py --labs lab_codes.txt --out allocation.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from mar_ils_core import release

ALLOCATION_SEED = 20261103
PER_LAB = 3
TOOL_VERSION = "v9.0.0"


def full_study_names(n_labs: int) -> list[str]:
    """Names of the full study's pool for n_labs laboratories: ILS001 to ILS(3·n_labs)."""
    if n_labs < 1:
        raise ValueError("at least one laboratory is needed")
    return [f"ILS{m:03d}" for m in range(1, PER_LAB * n_labs + 1)]


def order_key(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()


def natural_key(code: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", code)]


def allocate(lab_codes: list[str], datasets: list[str] | None = None,
             seed: int = ALLOCATION_SEED) -> dict[str, list[str]]:
    """Laboratory code → its three datasets, in drawn order."""
    codes = [c.strip() for c in lab_codes]
    if not codes or any(not c for c in codes):
        raise ValueError("laboratory codes must be non-empty")
    if len(set(codes)) != len(codes):
        raise ValueError("laboratory codes must be unique")
    names = list(datasets) if datasets is not None else full_study_names(len(codes))
    if len(set(names)) != len(names):
        raise ValueError("dataset names must be unique")
    if len(names) != PER_LAB * len(codes):
        raise ValueError(f"{len(codes)} laboratories need {PER_LAB * len(codes)} datasets, got {len(names)}")
    drawn = sorted(names, key=lambda n: order_key(seed, n))
    labs = sorted(codes, key=natural_key)
    return {lab: drawn[PER_LAB * i: PER_LAB * (i + 1)] for i, lab in enumerate(labs)}


def check_allocation(allocation: dict[str, list[str]], datasets: list[str]) -> list[str]:
    """Problems with an allocation (empty when it meets §A6.3)."""
    problems = []
    used = [d for group in allocation.values() for d in group]
    for lab, group in allocation.items():
        if len(group) != PER_LAB:
            problems.append(f"{lab}: {len(group)} datasets, expected {PER_LAB}")
    for d in sorted(set(used)):
        if used.count(d) > 1:
            problems.append(f"{d} assigned {used.count(d)} times")
    missing = sorted(set(datasets) - set(used))
    if missing:
        problems.append(f"not assigned: {', '.join(missing)}")
    unknown = sorted(set(used) - set(datasets))
    if unknown:
        problems.append(f"not in the pool: {', '.join(unknown)}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Allocate the full study's replicate datasets (ILS protocol §A6.3)")
    ap.add_argument("--labs", type=Path, required=True, help="laboratory codes, one per line")
    ap.add_argument("--out", type=Path, required=True, help="allocation record to write (JSON)")
    ap.add_argument("--seed", type=int, default=ALLOCATION_SEED)
    a = ap.parse_args()

    codes = [ln.strip() for ln in a.labs.read_text().splitlines() if ln.strip()]
    try:
        allocation = allocate(codes, seed=a.seed)
    except ValueError as e:
        sys.exit(str(e))
    datasets = full_study_names(len(codes))
    problems = check_allocation(allocation, datasets)
    if problems:
        sys.exit("allocation check failed: " + "; ".join(problems))
    record = {
        "tool": "allocate_v9_0_0.py", "tool_version": TOOL_VERSION,
        "source_commit": release.source_commit(),
        "protocol": "ILS protocol A6.3",
        "seed": a.seed,
        "method": "pool ordered by SHA-256('<seed>:<dataset name>'); consecutive groups of three; "
                  "laboratories in natural order of their codes",
        "laboratories": sorted(codes, key=natural_key),
        "datasets": datasets,
        "allocation": allocation,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    a.out.write_text(json.dumps(record, indent=1))
    for lab, group in allocation.items():
        print(f"{lab:>12}  {'  '.join(group)}")
    print(f"Allocation record: {a.out}")


if __name__ == "__main__":
    main()
