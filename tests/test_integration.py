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
            capture_output=True, text=True, timeout=60, check=False,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_dry_run_prints_geometry(self):
        result = subprocess.run(
            [sys.executable, "generator_v7_0_0.py", "--dry-run"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        assert "fan-beam" in result.stdout
        assert "570.0" in result.stdout
        assert "720 angles" in result.stdout

    def test_dry_run_warns_when_workers_exceed_memory(self):
        """Far more workers than RAM supports should print a memory warning."""
        result = subprocess.run(
            [sys.executable, "generator_v7_0_0.py", "--dry-run", "--workers", "512"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        assert result.returncode == 0
        assert "Memory" in result.stdout
        assert "WARNING: 512 workers" in result.stdout

    def test_dry_run_no_memory_warning_for_one_worker(self):
        result = subprocess.run(
            [sys.executable, "generator_v7_0_0.py", "--dry-run", "--workers", "1"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        assert "workers need about" not in result.stdout


class TestRecommendedWorkers:
    def test_8gb_machine_gets_2_workers(self):
        """CLAUDE.md: --workers 2 keeps N=40 in RAM on an 8 GB Mac."""
        from generator_v7_0_0 import recommended_workers
        assert recommended_workers(8.0) == 2

    def test_never_below_one(self):
        from generator_v7_0_0 import recommended_workers
        assert recommended_workers(2.0) == 1

    def test_scales_with_ram(self):
        from generator_v7_0_0 import recommended_workers
        assert recommended_workers(32.0) > recommended_workers(16.0) > 2


class TestChecksumManifest:
    def test_os_metadata_files_excluded(self, tmp_path):
        from generator_v7_0_0 import write_checksums
        (tmp_path / "sinograms").mkdir()
        (tmp_path / "sinograms" / "realization_001.h5").write_bytes(b"data")
        for junk in (".DS_Store", "sinograms/.DS_Store", "sinograms/._x.h5",
                     "Thumbs.db"):
            (tmp_path / junk).write_bytes(b"junk")
        write_checksums(tmp_path)
        lines = (tmp_path / "checksums_sha256.txt").read_text().splitlines()
        entries = [ln.split("  ", 1)[1] for ln in lines if not ln.startswith("#")]
        assert entries == ["sinograms/realization_001.h5"]


class TestImports:
    def test_generator_imports(self):
        """Verify generator_v7_0_0.py can be imported without errors."""
        result = subprocess.run(
            [sys.executable, "-c", "import generator_v7_0_0"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        assert result.returncode == 0

    def test_cho_imports(self):
        """Verify run_cho_analysis_v7_0.py can be imported without errors."""
        result = subprocess.run(
            [sys.executable, "-c", "import run_cho_analysis_v7_0"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        assert result.returncode == 0

    def test_mar_ils_core_imports(self):
        result = subprocess.run(
            [sys.executable, "-c",
             ("from mar_ils_core.constants import *; "
              "from mar_ils_core.phantom import *; "
              "from mar_ils_core.noise import *; "
              "from mar_ils_core.dicom_utils import *")],
            capture_output=True, text=True, timeout=30, check=False,
        )
        assert result.returncode == 0
