from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "recon" / "bart" / "run_cartesian_recon.sh"


def _touch_cfl(base: Path) -> None:
    """Create one minimal valid BART CFL pair for wrapper tests."""

    base.with_suffix(".hdr").write_text("# Dimensions\n1\n", encoding="utf-8")
    base.with_suffix(".cfl").write_bytes(b"\x00" * 8)


def _read_calls(path: Path) -> list[list[str]]:
    """Read argument-preserving fake BART invocations from a test log."""

    calls: list[list[str]] = []
    for block in path.read_text(encoding="utf-8").split("CALL\n")[1:]:
        calls.append([line.removeprefix("ARG=") for line in block.splitlines()])
    return calls


class RunCartesianReconTests(unittest.TestCase):
    """Verify the no-wave Cartesian BART wrapper contract."""

    def setUp(self) -> None:
        """Create isolated inputs and an argument-logging fake BART command."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.inputs = self.root / "inputs"
        self.outputs = self.root / "outputs"
        self.inputs.mkdir()
        self.twix = self.root / "meas.dat"
        self.sequence = self.root / "sequence.seq"
        self.twix.touch()
        self.sequence.touch()
        (self.inputs / "manifest.json").write_text(
            json.dumps(
                {
                    "reconstruction": {
                        "mode": "cartesian",
                        "tool": "pics",
                        "readout_oversampling_removed": True,
                    },
                    "echoes": [
                        {"echo": 1, "cartesian_kspace": "cartesian_kspace"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        for name in ("kspace_calib", "cartesian_kspace"):
            _touch_cfl(self.inputs / name)

        self.bart_log = self.root / "bart.log"
        self.fake_bart = self.root / "bart"
        self.fake_bart.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
{
    printf 'CALL\n'
    printf 'ARG=%s\n' "$@"
} >> "$BART_TEST_LOG"
output="${!#}"
printf '# Dimensions\n1\n' > "$output.hdr"
printf '\0\0\0\0\0\0\0\0' > "$output.cfl"
""",
            encoding="utf-8",
        )
        self.fake_bart.chmod(0o755)
        self.environment = {
            **os.environ,
            "BART_BIN": str(self.fake_bart),
            "BART_TEST_LOG": str(self.bart_log),
        }

    def tearDown(self) -> None:
        """Remove the isolated wrapper-test directory."""

        self.temporary_directory.cleanup()

    def test_default_pics_reconstruction_enables_gpu(self) -> None:
        """The no-wave wrapper should use GPU PICS and restore its scaling."""

        command = [
            "bash",
            str(SCRIPT),
            "--bart-input",
            str(self.inputs),
            "--bart-output",
            str(self.outputs),
            "--maps-source",
            "bart",
            "--twix",
            str(self.twix),
            "--seq",
            str(self.sequence),
            "--skip-nifti",
        ]
        result = subprocess.run(
            command,
            check=True,
            env=self.environment,
            capture_output=True,
            text=True,
        )

        calls = _read_calls(self.bart_log)
        self.assertEqual(calls[0][0:3], ["ecalib", "-m", "1"])
        self.assertEqual(
            calls[1],
            [
                "pics",
                "-l1",
                "-r",
                "1e-5",
                "--fista",
                "-i",
                "300",
                "-g",
                "-S",
                str(self.inputs / "cartesian_kspace"),
                str(self.outputs / "coil_sens_bart"),
                str(self.outputs / "image_nowave"),
            ],
        )
        self.assertIn("BART GPU defaults", result.stdout)

    def test_resume_rejects_truncated_pics_output(self) -> None:
        """A nonempty but undersized image must be reconstructed again."""

        command = [
            "bash",
            str(SCRIPT),
            "--bart-input",
            str(self.inputs),
            "--bart-output",
            str(self.outputs),
            "--maps-source",
            "bart",
            "--twix",
            str(self.twix),
            "--seq",
            str(self.sequence),
            "--skip-nifti",
        ]
        subprocess.run(command, check=True, env=self.environment)
        (self.outputs / "image_nowave.cfl").write_bytes(b"\x00")

        subprocess.run(command + ["--resume"], check=True, env=self.environment)

        calls = _read_calls(self.bart_log)
        self.assertEqual([call[0] for call in calls].count("pics"), 2)
        self.assertEqual(calls[-1][0], "pics")


if __name__ == "__main__":
    unittest.main()
