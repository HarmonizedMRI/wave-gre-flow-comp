from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
    def test_config_separates_measured_counts_from_global_label_extents(self) -> None:
        """Sparse global LIN/PAR labels must retain their mapVBVD extents."""

        sequence = self._r4x3_sequence()

        cfg = native._derive_gre_config(sequence)

        self.assertEqual((cfg["Ny_meas"], cfg["Nz_meas"]), (63, 24))
        self.assertEqual(
            (cfg["Ny_label_extent"], cfg["Nz_label_extent"]), (250, 70)
        )

    def test_geometry_diagnostic_accepts_global_label_extents(self) -> None:
        """R4x3 TWIX extents must not be compared with 63x24 sample counts."""

        cfg = native._derive_gre_config(self._r4x3_sequence())
        twix_info = {
            "FOV": {"readout": 224.0, "phase": 220.0, "slice": 180.0},
            "NormalRAS": [0.0, 0.0, 1.0],
            "ReadoutDirectionRAS": [1.0, 0.0, 0.0],
            "PhaseDirectionRAS": [0.0, 1.0, 0.0],
            "SliceDirectionRAS": [0.0, 0.0, 1.0],
        }
        with patch(
            "utils.nifti_export_twix.make_nifti_affine_from_twix",
            return_value=(None, None, twix_info),
        ):
            result = native._report_seq_twix_geometry(
                twix_file=Path("input.dat"),
                cfg=cfg,
                received_image_shape=(1680, 250, 70, 2, 32),
                voxel_size_mm=(224 / 420, 220 / 250, 180 / 72),
                twix_array_axis_roles=("readout", "phase", "slice"),
                twix_array_axis_flips=(False, False, False),
                twix_coord_system="LPS",
                twix_inplane_rot_sign=-1.0,
            )

        self.assertTrue(result["Passed"])
        checks = result["MatrixChecks"]
        self.assertEqual(checks["ExpectedLINMeasured"], 63)
        self.assertEqual(checks["ExpectedPARMeasured"], 24)
        self.assertEqual(checks["ExpectedLINLabelExtent"], 250)
        self.assertEqual(checks["ExpectedPARLabelExtent"], 70)

    def test_config_rejects_inconsistent_measured_counts(self) -> None:
        """Definitions must match the centered accelerated label pattern."""

        with self.assertRaisesRegex(ValueError, "Measured PE counts disagree"):
            native._derive_gre_config(
                SimpleNamespace(
                    definitions={
                        "OrientationMapping": "TRA",
                        "Nx": 8,
                        "Ny": 10,
                        "Nz": 8,
                        "ReadoutOversamplingFactor": 4,
                        "FOV": [0.2, 0.2, 0.2],
                        "Nechoes": 1,
                        "TE": [0.01],
                        "Averages": 1,
                        "Ry": 3,
                        "Rz": 2,
                        "Ny_meas": 4,
                        "Nz_meas": 3,
                    }
                )
            )

    @staticmethod
    def _r4x3_sequence() -> SimpleNamespace:
        """Return the generated R4x3 GRE definition contract."""

        return SimpleNamespace(
            definitions={
                "OrientationMapping": "TRA",
                "Nx": 420,
                "Ny": 250,
                "Nz": 72,
                "Nx_os": 1680,
                "ReadoutOversamplingFactor": 4,
                "FOV": [0.224, 0.220, 0.180],
                "Nechoes": 2,
                "TE": [0.00709, 0.01597],
                "Averages": 1,
                "Ry": 4,
                "Rz": 3,
                "Ny_meas": 63,
                "Nz_meas": 24,
                "CalibrationNcalib1": 72,
                "CalibrationNcalib2": 1,
                "CalibrationNacs": 32,
                "CalibrationNSets": 5,
                "CalibrationACSSetID": 4,
                "WaveSinChannel": "y",
                "WaveCosChannel": "z",
                "KspaceOrdering": "negative_to_positive",
                "UseFlowComp": 1,
            }
        )

    @staticmethod
    def _coil_cfg() -> dict[str, object]:
        """Return a compact GRE coil-calibration configuration fixture.

        Returns:
            Configuration with 4x readout oversampling and set-4 ACS.
        """

        return {
            "Nx": 4,
            "Nx_os": 16,
            "Ny": 6,
            "Nz": 6,
            "os_factor": 4,
            "Ncalib1": 2,
            "Nacs": 2,
            "Nsets": 5,
            "ACSSetID": 4,
        }

    @staticmethod
    def _centered_fft(array: np.ndarray) -> np.ndarray:
        """Return a centered orthonormal readout FFT.

        Args:
            array: Complex image with readout first.

        Returns:
            Centered readout k-space.
        """

        return np.fft.fftshift(
            np.fft.fft(
                np.fft.ifftshift(array, axes=(0,)),
                axis=0,
                norm="ortho",
            ),
            axes=(0,),
        )

    def _refscan_with_outside_fov_signal(self) -> tuple[torch.Tensor, np.ndarray]:
        """Build one mock refscan with removable outside-FOV anatomy.

        Returns:
            Refscan tensor and expected logical ACS k-space.
        """

        logical_image = np.zeros((4, 2, 2, 2), dtype=np.complex64)
        logical_image[1:3] = 1.0 + 0.25j
        oversampled_image = np.zeros((16, 2, 2, 2), dtype=np.complex64)
        oversampled_image[6:10] = logical_image
        oversampled_image[1:3] = 25.0
        raw = self._centered_fft(oversampled_image).astype(np.complex64)
        ref = torch.zeros((16, 2, 2, 5, 2), dtype=torch.complex64)
        ref[:, :, :, 4, :] = torch.from_numpy(raw)
        expected = self._centered_fft(logical_image).astype(np.complex64)
        return ref, expected

    def test_logical_integrated_acs_crops_in_image_domain(self) -> None:
        """GRE set-4 ACS must remove outside-FOV signal without wrapping it."""

        ref, expected = self._refscan_with_outside_fov_signal()

        actual = native._logical_integrated_acs(ref, self._coil_cfg())

        self.assertEqual(tuple(actual.shape), (4, 2, 2, 2))
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=2e-6)

    def test_bart_calibration_uses_alias_free_logical_acs(self) -> None:
        """The BART exporter must embed corrected ACS without RO striding."""

        ref, expected = self._refscan_with_outside_fov_signal()
        with patch.object(native, "load_ref", return_value=ref):
            actual = native._build_bart_calibration_kspace(
                twix_file=Path("input.dat"),
                cfg=self._coil_cfg(),
                wcc=np.eye(2, dtype=np.complex64),
            )

        self.assertEqual(actual.shape, (4, 6, 6, 2))
        np.testing.assert_allclose(actual[:, 2:4, 2:4, :], expected, rtol=2e-6, atol=2e-6)
        outside = actual.copy()
        outside[:, 2:4, 2:4, :] = 0
        self.assertEqual(np.count_nonzero(outside), 0)

    def test_coil_cache_is_versioned_and_hash_bound(self) -> None:
        """Corrected caches should reject changed sources or artifacts."""

        ref, _ = self._refscan_with_outside_fov_signal()
        logical = native._logical_integrated_acs(ref, self._coil_cfg())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            twix = root / "input.dat"
            twix.write_bytes(b"mock twix")
            paths = native._coil_cache_paths(root, "", 2, "3d")
            self.assertIn("roimgcrop-v1", paths["wcc"].name)
            np.save(paths["wcc"], np.eye(2, dtype=np.complex64))
            np.save(paths["csm_full"], np.ones((2, 4, 6, 6), dtype=np.complex64))
            contract = native._coil_calibration_contract(
                twix_file=twix,
                cfg=self._coil_cfg(),
                logical_acs=logical,
                ncc=2,
                espirit_calib_mode="3d",
                espirit_crop=0.8,
            )
            native._write_coil_cache_manifest(
                paths["manifest"],
                contract=contract,
                wcc_path=paths["wcc"],
                csm_full_path=paths["csm_full"],
            )
            native._validate_coil_cache_manifest(
                paths["manifest"],
                expected_contract=contract,
                wcc_path=paths["wcc"],
                csm_full_path=paths["csm_full"],
            )

            np.save(paths["wcc"], np.zeros((2, 2), dtype=np.complex64))
            with self.assertRaisesRegex(ValueError, "hashes"):
                native._validate_coil_cache_manifest(
                    paths["manifest"],
                    expected_contract=contract,
                    wcc_path=paths["wcc"],
                    csm_full_path=paths["csm_full"],
                )

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
            self.assertEqual(default["reconstruction_backend"], "bart")
            self.assertTrue(default["save_bart_inputs"])
            self.assertTrue(default["save_nifti"])

            no_nifti = native._collect_runtime_config(
                [
                    "--twix",
                    str(root / "input.dat"),
                    "--seq",
                    str(sequence),
                    "--out",
                    str(root / "no-nifti"),
                    "--validate-only",
                    "--no-save-nifti",
                ]
            )
            self.assertFalse(no_nifti["save_nifti"])

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
