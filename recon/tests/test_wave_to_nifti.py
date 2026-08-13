from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from recon.bart.bart_utils.bart_io import write_cfl
from recon.bart.wave_to_nifti import discover_bart_echoes, restore_bart_intensity


class BartWaveToNiftiTests(unittest.TestCase):
    def test_discovers_matching_multi_echo_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            entries = []
            for echo in (1, 2):
                suffix = f"_echo-{echo:02d}"
                kspace_name = f"wave_kspace{suffix}"
                write_cfl(inputs / kspace_name, np.ones((4, 2, 2, 1, 1), np.complex64))
                write_cfl(outputs / f"image_wave{suffix}", np.ones((2, 2, 2, 1, 1), np.complex64))
                entries.append({"echo": echo, "wave_kspace": kspace_name})
            (inputs / "manifest.json").write_text(
                json.dumps({"echoes": entries}), encoding="utf-8"
            )
            resolved = discover_bart_echoes(inputs, outputs)
            self.assertEqual([item["echo"] for item in resolved], [1, 2])
            self.assertEqual(resolved[1]["image"].name, "image_wave_echo-02")

    def test_restores_bart_kspace_norm(self) -> None:
        image = np.full((2, 2, 2), 2 + 1j, np.complex64)
        kspace = np.ones((3, 4), np.complex64)
        restored, scale = restore_bart_intensity(image, kspace)
        self.assertAlmostEqual(scale, np.sqrt(12.0), places=6)
        np.testing.assert_allclose(restored, image * np.sqrt(12.0))


if __name__ == "__main__":
    unittest.main()
