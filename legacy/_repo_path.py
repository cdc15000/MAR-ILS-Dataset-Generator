"""Puts the repository root on sys.path, so the scripts in legacy/ can import
the mar_ils_core package and the algorithms/v7 references when run directly."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
