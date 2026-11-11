#!/usr/bin/env python3
"""
MAR ILS blinding tool v9.0.0: distribution copies, blinding key and the
laboratory assignment (ILS protocol §A6.3; design spec, Blinded Distribution).

Input: the master datasets written by write_dataset_v9_0_0.py (labeled names,
label-free files). Output:

  <out>/REP01 ... REP09, CONTROL     coded copies: sinograms/<code>.h5 and
                                     noMAR/<code>.dcm, byte-identical to the
                                     master files; dataset_record.json without
                                     per-realization entries; new checksums
  <out>/FAMILIARIZATION              labeled copy, unchanged
  <key>                              blinding_key.json: random salt, code ->
                                     class, level, split, pool, index for
                                     every coded realization, and the
                                     laboratory assignment
  <key>.sha256                       its SHA-256, for the Zenodo record

The key is written outside the distribution directory and is never placed in
it. Codes are eight characters from 0-9 and A-Z, drawn from the operating
system's random source, unique across all datasets, so neither the codes nor
their order carry information. Every distributed file gets one modification
time, because the master times follow the order of generation, which
interleaves the classes.

Assignment (review item 37): each laboratory receives three different
replicate datasets from the nine; across laboratories the number receiving
each dataset differs by at most one, and no two laboratories receive the same
three while the number of laboratories allows (84 sets exist). The design is
drawn at random with seed 20261103, after recruitment closes. ILS staff may
supply their own design instead (--assignment), which is checked against the
same conditions.

Usage:
  python blind_dataset_v9_0_0.py --masters /Volumes/ILS/master --out /Volumes/ILS/dist \\
      --key /Volumes/ILS/key/blinding_key.json --labs labs.txt
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import secrets
import shutil
import sys
import time
from pathlib import Path

import numpy as np

TOOL_VERSION = "v9.0.0"
ASSIGNMENT_SEED = 20261103
REPLICATES = tuple(f"REP{m:02d}" for m in range(1, 10))
CODED = REPLICATES + ("CONTROL",)
LABELED = ("FAMILIARIZATION",)
CODE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CODE_LEN = 8
PER_LAB = 3
SKIP = {".writer_state.json", "checksums_sha256.txt"}


# ── Assignment ───────────────────────────────────────────────────────────────

def check_assignment(assignment: dict[str, list[str]], datasets=REPLICATES, k: int = PER_LAB) -> list[str]:
    """Problems with an assignment under the conditions of review item 37 (empty if none)."""
    problems = []
    for lab, sets in assignment.items():
        if len(sets) != k or len(set(sets)) != k or not set(sets) <= set(datasets):
            problems.append(f"{lab}: needs {k} different datasets from {datasets[0]}..{datasets[-1]}, has {sets}")
    counts = [sum(d in s for s in assignment.values()) for d in datasets]
    if max(counts) - min(counts) > 1:
        problems.append(f"dataset counts differ by more than one: {dict(zip(datasets, counts))}")
    triples = [frozenset(s) for s in assignment.values()]
    n_possible = math.comb(len(datasets), k)
    if len(set(triples)) < min(len(triples), n_possible):
        problems.append("two laboratories receive the same three datasets although distinct sets remain")
    return problems


def draw_assignment(labs: list[str], seed: int = ASSIGNMENT_SEED, datasets=REPLICATES,
                    k: int = PER_LAB, max_tries: int = 100_000) -> dict[str, list[str]]:
    """A random balanced design: random order of all k-subsets, taken greedily under a cap
    on each dataset's count, retried until the counts differ by at most one.
    Laboratories are taken in sorted order, so the result depends only on the seed and the codes."""
    labs = sorted(labs)
    if len(set(labs)) != len(labs):
        raise ValueError("laboratory codes must be unique")
    n = len(labs)
    triples = [tuple(t) for t in itertools.combinations(datasets, k)]
    cap = math.ceil(n * k / len(datasets))
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        order = [triples[i] for i in rng.permutation(len(triples))]
        if n > len(triples):                         # more labs than distinct sets: allow repeats
            order = order * math.ceil(n / len(triples))
        counts = dict.fromkeys(datasets, 0)
        chosen = []
        for t in order:
            if all(counts[d] < cap for d in t):
                chosen.append(t)
                for d in t:
                    counts[d] += 1
                if len(chosen) == n:
                    break
        if len(chosen) == n and max(counts.values()) - min(counts.values()) <= 1:
            design = {lab: list(t) for lab, t in zip(labs, chosen)}
            assert not check_assignment(design, datasets, k)
            return design
    raise RuntimeError(f"no balanced design found for {n} laboratories")


# ── Distribution copies ──────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_master(root: Path) -> None:
    """Every file listed in the master checksums is present and matches."""
    bad = []
    for line in (root / "checksums_sha256.txt").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        p = root / rel
        if not p.is_file() or sha256_file(p) != digest:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"{root.name}: {len(bad)} files fail the master checksums, first {bad[0]}")


def new_codes(n: int, taken: set[str]) -> list[str]:
    out = []
    while len(out) < n:
        c = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))
        if c not in taken:
            taken.add(c)
            out.append(c)
    return out


def place(src: Path, dst: Path, copy: bool, mtime: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copyfile(src, dst)
    else:
        try:
            os.link(src, dst)             # same bytes, no extra space; files are never modified
        except OSError as e:
            raise RuntimeError(f"cannot hard-link {src} to {dst} ({e.strerror}); "
                               "use --copy when the masters and the output are on different volumes") from e
    os.utime(dst, (mtime, mtime))


def write_checksums(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "checksums_sha256.txt")
    with open(root / "checksums_sha256.txt", "w") as f:
        f.writelines(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files)


def blind_dataset(master: Path, dist: Path, codes_taken: set[str], copy: bool, mtime: float) -> dict:
    """Coded copy of one master dataset; returns the key entries for it."""
    record = json.loads((master / "dataset_record.json").read_text())
    if record.get("dataset") != master.name:
        raise RuntimeError(f"{master}: the record names dataset {record.get('dataset')!r}, not {master.name!r}")
    reals = record["realizations"]
    codes = new_codes(len(reals), codes_taken)
    entries = {}
    for code, r in zip(codes, reals):
        name = r["name"]
        place(master / "sinograms" / f"{name}.h5", dist / "sinograms" / f"{code}.h5", copy, mtime)
        place(master / "noMAR" / f"{name}.dcm", dist / "noMAR" / f"{code}.dcm", copy, mtime)
        entries[code] = {"class": "present" if r["cls"] else "absent", "level": r["level"],
                         "split": r["split"], "pool": r["pool"], "index": r["index"], "master_name": name}
    stripped = {k: v for k, v in record.items() if k not in ("realizations", "seconds")}
    stripped["blinded"] = True
    stripped["master_record_sha256"] = sha256_file(master / "dataset_record.json")
    return {"entries": entries, "stripped_record": stripped}


def copy_labeled(master: Path, dist: Path, copy: bool, mtime: float) -> None:
    for p in sorted(master.rglob("*")):
        if p.is_file() and p.name not in SKIP and not p.name.endswith(".partial"):
            place(p, dist / p.relative_to(master), copy, mtime)


def blind(masters: Path, out: Path, key_path: Path, assignment: dict[str, list[str]],
          assignment_source: str, copy: bool = False, mtime: float | None = None,
          require_all: bool = True) -> str:
    present = [d for d in CODED + LABELED if (masters / d).is_dir()]
    missing = [d for d in CODED + LABELED if d not in present]
    if require_all and missing:
        raise RuntimeError(f"master datasets missing: {', '.join(missing)}")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} is not empty")
    if key_path.resolve().is_relative_to(out.resolve()):
        raise RuntimeError("the key must be written outside the distribution directory")
    problems = check_assignment(assignment)
    if problems:
        raise RuntimeError("assignment: " + "; ".join(problems))
    if mtime is None:
        mtime = float(math.floor(time.time() / 86400) * 86400)       # midnight UTC today

    for d in present:
        verify_master(masters / d)

    taken: set[str] = set()
    key = {"tool_version": TOOL_VERSION, "salt": secrets.token_hex(32),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "assignment_seed": ASSIGNMENT_SEED if assignment_source == "drawn" else None,
           "assignment_source": assignment_source, "assignment": assignment, "datasets": {}}
    stripped = {}
    for d in present:
        dst = out / d
        if d in LABELED:
            copy_labeled(masters / d, dst, copy, mtime)
            continue
        res = blind_dataset(masters / d, dst, taken, copy, mtime)
        key["datasets"][d] = res["entries"]
        stripped[d] = res["stripped_record"]
        print(f"{d}: {len(res['entries'])} realizations coded", flush=True)

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_bytes = json.dumps(key, indent=1, sort_keys=True).encode()
    key_path.write_bytes(key_bytes)
    key_hash = hashlib.sha256(key_bytes).hexdigest()
    Path(str(key_path) + ".sha256").write_text(f"{key_hash}  {key_path.name}\n")

    for d in present:
        dst = out / d
        if d in stripped:
            rec = {**stripped[d], "blinding_key_sha256": key_hash}
            (dst / "dataset_record.json").write_text(json.dumps(rec, indent=1))
            os.utime(dst / "dataset_record.json", (mtime, mtime))
        write_checksums(dst)
        os.utime(dst / "checksums_sha256.txt", (mtime, mtime))
    return key_hash


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--masters", type=Path, required=True, help="directory holding REP01..REP09, CONTROL, FAMILIARIZATION")
    ap.add_argument("--out", type=Path, required=True, help="new, empty directory for the distribution copies")
    ap.add_argument("--key", type=Path, required=True, help="path of blinding_key.json, outside --out")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--labs", type=Path, help="laboratory codes, one per line; the design is drawn with seed 20261103")
    src.add_argument("--assignment", type=Path, help="JSON {lab: [three datasets]} built by ILS staff")
    ap.add_argument("--copy", action="store_true", help="copy files instead of hard-linking them")
    ap.add_argument("--allow-partial", action="store_true", help="testing only: blind the masters that exist")
    a = ap.parse_args()

    if a.labs:
        labs = [ln.strip() for ln in a.labs.read_text().splitlines() if ln.strip()]
        assignment, source = draw_assignment(labs), "drawn"
    else:
        assignment, source = json.loads(a.assignment.read_text()), "ILS staff"
    try:
        h = blind(a.masters, a.out, a.key, assignment, source, copy=a.copy, require_all=not a.allow_partial)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"Key: {a.key}\nSHA-256 for the Zenodo record: {h}")


if __name__ == "__main__":
    main()
