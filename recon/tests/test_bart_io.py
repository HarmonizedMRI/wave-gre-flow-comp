from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from recon.bart.bart_utils.bart_io import export_wave_inputs, read_cfl, write_cfl


def _read_cfl(base: Path) -> np.ndarray:
    return read_cfl(base, trim_trailing_singletons=False)


class BartIoTests(unittest.TestCase):
    def test_write_cfl_round_trip_preserves_column_major_order(self) -> None:
        expected = (
            np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            + 1j * np.arange(24, dtype=np.float32).reshape(2, 3, 4)[::-1]
        ).astype(np.complex64)
        with tempfile.TemporaryDirectory() as folder:
            base = write_cfl(Path(folder) / "array.cfl", expected)
            np.testing.assert_array_equal(read_cfl(base), expected)

    def test_read_cfl_rejects_truncated_data(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            base = write_cfl(Path(folder) / "array", np.ones((2, 3), np.complex64))
            base.with_suffix(".cfl").write_bytes(b"\x00" * 8)
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                read_cfl(base)

    def test_single_echo_exports_required_bart_dimensions(self) -> None:
        wx, sx, sy, sz, nc = 8, 4, 3, 2, 2
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = export_wave_inputs(
                folder,
                wave_kspace=np.ones((wx, sy, sz, 1, nc), np.complex64),
                calibrated_psf=np.ones((1, wx, sy, sz), np.complex64),
                coil_sens=np.ones((nc, sx, sy, sz), np.complex64),
                kspace_calib=np.ones((sx, sy, sz, nc), np.complex64),
            )
            self.assertEqual(_read_cfl(Path(folder) / "wave_kspace").shape, (8, 3, 2, 2, 1))
            self.assertEqual(_read_cfl(Path(folder) / "psf").shape, (8, 3, 2, 1, 1))
            self.assertEqual(_read_cfl(Path(folder) / "coil_sens").shape, (4, 3, 2, 2, 1))
            self.assertEqual(_read_cfl(Path(folder) / "kspace_calib").shape, (4, 3, 2, 2))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["echoes"][0]["wave_kspace"], "wave_kspace")

    def test_multi_echo_uses_matching_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            export_wave_inputs(
                folder,
                wave_kspace=np.ones((8, 3, 2, 2, 2), np.complex64),
                calibrated_psf=np.ones((2, 8, 3, 2), np.complex64),
                coil_sens=np.ones((2, 4, 3, 2), np.complex64),
                kspace_calib=np.ones((4, 3, 2, 2), np.complex64),
            )
            for suffix in ("_echo-01", "_echo-02"):
                self.assertTrue((Path(folder) / f"wave_kspace{suffix}.hdr").is_file())
                self.assertTrue((Path(folder) / f"psf{suffix}.hdr").is_file())


if __name__ == "__main__":
    unittest.main()
