"""
Integration tests for the MAR ILS framework.
"""

import subprocess
import sys



class TestGeneratorDryRun:
    def test_dry_run_exits_zero(self):
        """generator_v7_0_0.py --dry-run should validate config and exit 0."""
        result = subprocess.run(
            [sys.executable, "generator_v7_0_0.py", "--dry-run"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_dry_run_prints_geometry(self):
        result = subprocess.run(
            [sys.executable, "generator_v7_0_0.py", "--dry-run"],
            capture_output=True, text=True, timeout=60,
        )
        assert "fan-beam" in result.stdout
        assert "570.0" in result.stdout
        assert "720 angles" in result.stdout


class TestImports:
    def test_generator_imports(self):
        """Verify generator_v7_0_0.py can be imported without errors."""
        result = subprocess.run(
            [sys.executable, "-c", "import generator_v7_0_0"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_cho_imports(self):
        """Verify run_cho_analysis_v7_0.py can be imported without errors."""
        result = subprocess.run(
            [sys.executable, "-c", "import run_cho_analysis_v7_0"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_mar_ils_core_imports(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "from mar_ils_core.constants import *; "
             "from mar_ils_core.phantom import *; "
             "from mar_ils_core.noise import *; "
             "from mar_ils_core.dicom_utils import *"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
