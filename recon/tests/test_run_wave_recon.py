from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "recon" / "bart" / "run_wave_recon.sh"


def _touch_cfl(base: Path) -> None:
    base.with_suffix(".hdr").write_text("# Dimensions\n1\n", encoding="utf-8")
    base.with_suffix(".cfl").write_bytes(b"\x00" * 8)


def _read_calls(path: Path) -> list[list[str]]:
    calls: list[list[str]] = []
    for block in path.read_text(encoding="utf-8").split("CALL\n")[1:]:
        calls.append([line.removeprefix("ARG=") for line in block.splitlines()])
    return calls


class RunWaveReconTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.bart_input = self.root / "bart inputs"
        self.bart_output = self.root / "bart output"
        self.nifti_output = self.root / "nifti output"
        self.bart_input.mkdir()
        self.twix = self.root / "meas.dat"
        self.sequence = self.root / "sequence.seq"
        self.twix.touch()
        self.sequence.touch()

        (self.bart_input / "manifest.json").write_text(
            json.dumps({"echoes": [{"echo": 1, "wave_kspace": "wave_kspace"}]}),
            encoding="utf-8",
        )
        for name in ("kspace_calib", "coil_sens", "wave_kspace", "psf"):
            _touch_cfl(self.bart_input / name)

        self.bart_log = self.root / "bart.log"
        self.python_log = self.root / "python.log"
        self.fake_bart = self.root / "bart"
        self.fake_python = self.root / "python"
        self.fake_bart.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
{
    printf 'CALL\\n'
    printf 'ARG=%s\\n' "$@"
} >> "$BART_TEST_LOG"
output="${!#}"
printf '# Dimensions\\n1\\n' > "$output.hdr"
printf '\\0\\0\\0\\0\\0\\0\\0\\0' > "$output.cfl"
""",
            encoding="utf-8",
        )
        self.fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
{
    printf 'CALL\\n'
    printf 'ARG=%s\\n' "$@"
} >> "$PYTHON_TEST_LOG"
""",
            encoding="utf-8",
        )
        self.fake_bart.chmod(0o755)
        self.fake_python.chmod(0o755)
        self.environment = {
            **os.environ,
            "BART_BIN": str(self.fake_bart),
            "PYTHON_BIN": str(self.fake_python),
            "BART_TEST_LOG": str(self.bart_log),
            "PYTHON_TEST_LOG": str(self.python_log),
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _base_command(self, maps_source: str) -> list[str]:
        return [
            "bash",
            str(SCRIPT),
            "--bart-input",
            str(self.bart_input),
            "--bart-output",
            str(self.bart_output),
            "--maps-source",
            maps_source,
            "--twix",
            str(self.twix),
            "--seq",
            str(self.sequence),
        ]

    def test_forwards_original_bart_options_and_runs_conversion(self) -> None:
        command = self._base_command("bart") + [
            "--save-phase",
            "--ecalib-options",
            "-c",
            "0.85",
            "--end-ecalib-options",
            "--wave-options",
            "-w",
            "-r",
            "0.001",
            "-f",
            "-i",
            "100",
            "-t",
            "1e-6",
            "--end-wave-options",
            "--nifti-options",
            "--nifti-suffix",
            "BARTGRE",
            "--end-nifti-options",
        ]
        subprocess.run(command, check=True, env=self.environment, capture_output=True, text=True)

        bart_calls = _read_calls(self.bart_log)
        self.assertEqual(
            bart_calls[0],
            [
                "ecalib",
                "-m",
                "1",
                "-c",
                "0.85",
                str(self.bart_input / "kspace_calib"),
                str(self.bart_output / "coil_sens_bart"),
            ],
        )
        self.assertEqual(
            bart_calls[1],
            [
                "wave",
                "-w",
                "-r",
                "0.001",
                "-f",
                "-i",
                "100",
                "-t",
                "1e-6",
                str(self.bart_output / "coil_sens_bart"),
                str(self.bart_input / "psf"),
                str(self.bart_input / "wave_kspace"),
                str(self.bart_output / "image_wave"),
            ],
        )
        python_call = _read_calls(self.python_log)[0]
        self.assertTrue(python_call[0].endswith("wave_to_nifti.py"))
        self.assertIn("--save-phase", python_call)
        output_index = python_call.index("--out")
        self.assertEqual(
            python_call[output_index + 1], str(self.bart_output / "nifti")
        )
        self.assertTrue((self.bart_output / "nifti").is_dir())
        self.assertEqual(python_call[-2:], ["--nifti-suffix", "BARTGRE"])

    def test_existing_maps_skip_ecalib(self) -> None:
        command = self._base_command("existing") + [
            "--nifti-output",
            str(self.nifti_output),
            "--existing-maps",
            str(self.bart_input / "coil_sens.hdr"),
            "--wave-options",
            "-l",
            "-r",
            "0.002",
            "-b",
            "8",
            "-f",
            "--end-wave-options",
        ]
        subprocess.run(command, check=True, env=self.environment, capture_output=True, text=True)
        bart_calls = _read_calls(self.bart_log)
        self.assertEqual(len(bart_calls), 1)
        self.assertEqual(bart_calls[0][0:7], ["wave", "-l", "-r", "0.002", "-b", "8", "-f"])
        self.assertEqual(bart_calls[0][7], str(self.bart_input / "coil_sens"))
        python_call = _read_calls(self.python_log)[0]
        output_index = python_call.index("--out")
        self.assertEqual(python_call[output_index + 1], str(self.nifti_output))

    def test_rejects_wavelet_and_llr_together(self) -> None:
        command = self._base_command("existing") + [
            "--wave-options",
            "-w",
            "-l",
            "--end-wave-options",
        ]
        result = subprocess.run(
            command,
            check=False,
            env=self.environment,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("-w and -l are mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()
