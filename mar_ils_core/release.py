"""Identify the software release a run or dataset was produced with (ILS protocol §A6.5).

In a git checkout the commit comes from git, with "-dirty" appended when tracked
files have uncommitted changes. In a release archive, where there is no git
repository, it comes from the file RELEASE_COMMIT: .gitattributes marks that
file export-subst, so `git archive` (and the archives GitHub serves for a tag or
release) write the commit hash into it. If neither is available the result is
"unknown".
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_FILE = "RELEASE_COMMIT"
_HASH = re.compile(r"[0-9a-f]{40}")


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _is_repo_root(root: Path) -> bool:
    """True only if root is itself the top of a git work tree, not a folder inside some other repository."""
    top = _git(root, "rev-parse", "--show-toplevel")
    return bool(top) and Path(top).resolve() == root.resolve()


def source_commit(root: Path = REPO_ROOT) -> str:
    if _is_repo_root(root):
        rev = _git(root, "rev-parse", "HEAD")
        if _HASH.fullmatch(rev):
            dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
            return rev + ("-dirty" if dirty else "")
    try:
        text = (root / RELEASE_FILE).read_text().strip()
    except OSError:
        return "unknown"
    return text if _HASH.fullmatch(text) else "unknown"
