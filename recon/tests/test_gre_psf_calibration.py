from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


RECON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECON_ROOT))

import recon_wave_gre_from_twix_integrated_nifti as native  # noqa: E402


def _quality(length: int) -> dict[str, dict[str, np.ndarray]]:
    """Return uniformly reliable projection evidence for one readout.

    Args:
        length: Oversampled readout length.

    Returns:
        Sin/cos quality vectors accepted by automatic range selection.
    """

    projection = {
        "skipped": np.zeros(length, dtype=bool),
        "valid_pixels": np.full(length, 60, dtype=np.int64),
        "masked_ratio": np.full(length, 0.2, dtype=np.float64),
        "wrapped_rms": np.full(length, 0.08, dtype=np.float64),
    }
    return {"sin": projection, "cos": {**projection}}


class GrePsfCalibrationTests(unittest.TestCase):
    def test_cli_accepts_automatic_or_complete_manual_bounds(self) -> None:
        """Sine-line should use automatic selection when both bounds are absent."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sequence = root / "input.seq"
            sequence.write_text("mock sequence\n", encoding="utf-8")
            default = native._collect_runtime_config(
                [
                    "--twix",
                    str(root / "input.dat"),
                    "--seq",
                    str(sequence),
                    "--out",
                    str(root / "default"),
                    "--validate-only",
                ]
            )
            self.assertEqual(default["psf_coefficient_processing"], "smooth")
            self.assertIsNone(default["psf_fit_kx_min"])
            self.assertIsNone(default["psf_fit_kx_max"])

            automatic = native._collect_runtime_config(
                [
                    "--twix",
                    str(root / "input.dat"),
                    "--seq",
                    str(sequence),
                    "--out",
                    str(root / "automatic"),
                    "--validate-only",
                    "--psf-coefficient-processing",
                    "sine-line",
                ]
            )
            self.assertIsNone(automatic["psf_fit_kx_min"])
            self.assertIsNone(automatic["psf_fit_kx_max"])

            manual = native._collect_runtime_config(
                [
                    "--twix",
                    str(root / "input.dat"),
                    "--seq",
                    str(sequence),
                    "--out",
                    str(root / "manual"),
                    "--validate-only",
                    "--psf-coefficient-processing",
                    "sine-line",
                    "--psf-fit-kx-min",
                    "12",
                    "--psf-fit-kx-max",
                    "180",
                ]
            )
            self.assertEqual(manual["psf_fit_kx_min"], 12)
            self.assertEqual(manual["psf_fit_kx_max"], 180)

            with self.assertRaisesRegex(ValueError, "omit both"):
                native._collect_runtime_config(
                    [
                        "--twix",
                        str(root / "input.dat"),
                        "--seq",
                        str(sequence),
                        "--out",
                        str(root / "invalid"),
                        "--validate-only",
                        "--psf-coefficient-processing",
                        "sine-line",
                        "--psf-fit-kx-min",
                        "12",
                    ]
                )

    def test_automatic_sine_line_uses_quality_and_reports_range(self) -> None:
        """Automatic processing should fit one validated near-global interval."""

        length = 1024
        kx = np.arange(length, dtype=np.float64)
        raw = (
            0.5 * np.sin(2.0 * np.pi * kx / 100.0 + 0.2)
            + 0.0002 * kx
            - 0.1,
            0.8 * np.sin(2.0 * np.pi * kx / 100.0 - 0.4)
            - 0.0001 * kx
            + 0.2,
            0.1 * np.sin(2.0 * np.pi * kx / 100.0 + 0.7)
            + 0.00005 * kx,
        )

        *processed, diagnostics = native._process_psf_coefficients(
            *raw,
            nx_os=length,
            processing="sine-line",
            fit_quality=_quality(length),
            return_diagnostics=True,
        )

        self.assertEqual(len(processed), 3)
        self.assertEqual(diagnostics["fit_range_selection"], "automatic")
        self.assertEqual(diagnostics["kx_range"], [21, 1003])
        self.assertTrue(diagnostics["validation_passed"])
        self.assertEqual(
            diagnostics["range_selection_diagnostics"]["version"], 9
        )

    def test_one_shared_coefficient_fit_builds_all_echo_psfs(self) -> None:
        """All echoes should reuse one coefficient fit and retain trajectories."""

        nx_os = 8
        raw = tuple(torch.zeros(nx_os, dtype=torch.float64) for _ in range(3))
        processed = (
            torch.linspace(0.0, 0.2, nx_os, dtype=torch.float64),
            torch.linspace(0.1, -0.1, nx_os, dtype=torch.float64),
            torch.linspace(-0.05, 0.05, nx_os, dtype=torch.float64),
        )
        evidence = {
            "projection_quality": _quality(nx_os),
            "shared_across_echoes": True,
            "source": "integrated_refscan_projection_calibration",
        }
        diagnostics = {
            "coefficient_processing": "smooth",
            "fit_range_selection": None,
            "kx_range": None,
            "kx_range_convention": "half-open [min, max)",
        }
        trajectories = [
            (np.zeros(nx_os), np.zeros(nx_os)),
            (np.linspace(-0.2, 0.2, nx_os), np.linspace(0.1, -0.1, nx_os)),
        ]
        cfg = {
            "Nx_os": nx_os,
            "Ny": 3,
            "Nz": 2,
            "Necho": 2,
            "yflip": 1,
            "zflip": 1,
        }

        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.object(
                    native,
                    "fit_wave_psf_deviation_from_projection",
                    return_value=(*raw, evidence),
                ) as coefficient_fit,
                patch.object(
                    native,
                    "_process_psf_coefficients",
                    return_value=(*processed, diagnostics.copy()),
                ) as coefficient_processing,
                patch.object(
                    native,
                    "_echo_theoretical_wave_trajectories",
                    return_value=trajectories,
                ),
            ):
                calibrated, theoretical, result = native.generate_calibrated_psfs(
                    twix_file=Path(folder) / "input.dat",
                    image_lines=np.empty((3, 0, nx_os)),
                    calib_lines=np.empty((3, 0, nx_os)),
                    cfg=cfg,
                    out_folder=Path(folder),
                    file_tag="",
                    psf_plot=False,
                    return_diagnostics=True,
                )

        coefficient_fit.assert_called_once()
        coefficient_processing.assert_called_once()
        self.assertEqual(tuple(calibrated.shape), (2, nx_os, 3, 2))
        self.assertEqual(tuple(theoretical.shape), (2, nx_os, 3, 2))
        self.assertFalse(torch.allclose(theoretical[0], theoretical[1]))
        np.testing.assert_allclose(
            (calibrated[0] / theoretical[0]).numpy(),
            (calibrated[1] / theoretical[1]).numpy(),
            atol=1e-6,
        )
        self.assertEqual(result["calibration_scope"]["coefficient_fit_count"], 1)
        self.assertTrue(result["calibration_scope"]["shared_across_echoes"])
        self.assertEqual(
            result["calibration_scope"]["echo_specific_component"],
            "sequence_theoretical_trajectory",
        )


if __name__ == "__main__":
    unittest.main()
