from __future__ import annotations

import unittest

import numpy as np

from recon.utils.psf_coefficient_processing import (
    AUTO_RANGE_ALGORITHM,
    AUTO_RANGE_VERSION,
    fit_sine_plus_line,
    select_automatic_kx_range,
    sine_line_model,
)


def _quality(length: int, start: int, stop: int) -> dict[str, np.ndarray]:
    """Build per-readout projection quality with one reliable interval.

    Args:
        length: Readout vector length.
        start: Inclusive reliable interval start.
        stop: Exclusive reliable interval stop.

    Returns:
        Projection-quality vectors accepted only inside ``[start, stop)``.
    """

    skipped = np.ones(length, dtype=bool)
    skipped[start:stop] = False
    valid_pixels = np.zeros(length, dtype=np.int64)
    valid_pixels[start:stop] = 60
    masked_ratio = np.ones(length, dtype=np.float64)
    masked_ratio[start:stop] = 0.20
    wrapped_rms = np.full(length, np.nan, dtype=np.float64)
    wrapped_rms[start:stop] = 0.08
    return {
        "skipped": skipped,
        "valid_pixels": valid_pixels,
        "masked_ratio": masked_ratio,
        "wrapped_rms": wrapped_rms,
    }


class AutomaticRangeTests(unittest.TestCase):
    def test_selects_common_sin_cos_interval(self) -> None:
        """The selector should use the intersection of both projections."""

        length = 160
        kx = np.arange(length, dtype=np.float64)
        coefficients = (
            np.sin(2.0 * np.pi * kx / 40.0),
            np.cos(2.0 * np.pi * kx / 40.0),
            0.01 * kx,
        )
        selected, diagnostics = select_automatic_kx_range(
            coefficients,
            {
                "sin": _quality(length, 20, 140),
                "cos": _quality(length, 30, 132),
            },
        )

        self.assertEqual(selected, (5, 155))
        self.assertEqual(diagnostics["name"], AUTO_RANGE_ALGORITHM)
        self.assertEqual(diagnostics["version"], AUTO_RANGE_VERSION)

    def test_nonfinite_central_coefficient_is_excluded(self) -> None:
        """A non-finite central sample should be excluded from the regression."""

        length = 160
        coefficients = [np.zeros(length, dtype=np.float64) for _ in range(3)]
        coefficients[1][90] = np.nan
        quality = _quality(length, 20, 145)

        selected, diagnostics = select_automatic_kx_range(
            coefficients, {"sin": quality, "cos": quality}
        )

        self.assertLessEqual(selected[0], 40)
        self.assertGreaterEqual(selected[1], 120)
        self.assertIn(90, diagnostics["excluded_sample_indices_within_interval"])

    def test_isolated_central_coefficient_transient_is_excluded(self) -> None:
        """A central spike should be excluded without moving the coordinate interval."""

        length = 160
        kx = np.arange(length, dtype=np.float64)
        coefficients = [np.sin(2.0 * np.pi * kx / 40.0) for _ in range(3)]
        coefficients[0][92] = 25.0
        quality = _quality(length, 20, 145)

        selected, diagnostics = select_automatic_kx_range(
            coefficients, {"sin": quality, "cos": quality}
        )

        self.assertLessEqual(selected[0], 40)
        self.assertGreaterEqual(selected[1], 120)
        self.assertIn(92, diagnostics["excluded_sample_indices_within_interval"])

    def test_center_run_wins_over_longer_edge_run(self) -> None:
        """An edge run must not outrank a shorter run containing the center."""

        length = 512
        kx = np.arange(length, dtype=np.float64)
        coefficients = [np.sin(2.0 * np.pi * kx / 40.0) for _ in range(3)]
        quality = _quality(length, 11, 330)
        quality["skipped"][170:180] = True
        quality["valid_pixels"][170:180] = 0
        quality["masked_ratio"][170:180] = 1.0
        quality["wrapped_rms"][170:180] = np.nan

        selected, diagnostics = select_automatic_kx_range(
            coefficients, {"sin": quality, "cos": quality}
        )

        self.assertEqual(selected, (11, 501))
        self.assertEqual(diagnostics["required_central_core_interval"], [216, 296])
        self.assertEqual(
            diagnostics["selection_strategy"],
            "near-global-no-sustained-coefficient-corruption",
        )

    def test_fragmented_center_support_uses_fixed_window(self) -> None:
        """Quality holes should be masked without splitting the centered interval."""

        length = 1024
        kx = np.arange(length, dtype=np.float64)
        coefficients = [np.sin(2.0 * np.pi * kx / 100.0) for _ in range(3)]
        quality = _quality(length, 200, 824)
        quality["skipped"][500:519] = True
        quality["valid_pixels"][500:519] = 0
        quality["masked_ratio"][500:519] = 1.0
        quality["wrapped_rms"][500:519] = np.nan

        selected, diagnostics = select_automatic_kx_range(
            coefficients, {"sin": quality, "cos": quality}
        )

        self.assertEqual(selected, (21, 1003))
        self.assertEqual(diagnostics["central_core_valid_samples"], 61)
        self.assertEqual(
            diagnostics["selection_strategy"],
            "near-global-no-sustained-coefficient-corruption",
        )
        self.assertTrue(
            set(range(500, 519)).issubset(
                diagnostics["excluded_sample_indices_within_interval"]
            )
        )

    def test_sustained_coefficient_corruption_selects_center_region(self) -> None:
        """A long coefficient blow-up should trigger regional fitting."""

        length = 1024
        kx = np.arange(length, dtype=np.float64)
        coefficients = [np.sin(2.0 * np.pi * kx / 100.0) for _ in range(3)]
        coefficients[2] = coefficients[2].copy()
        coefficients[2][700:950] = 20.0 * np.sign(
            np.sin(2.0 * np.pi * kx[700:950] / 7.0)
        )
        quality = _quality(length, 21, 1003)

        selected, diagnostics = select_automatic_kx_range(
            coefficients, {"sin": quality, "cos": quality}
        )

        self.assertLessEqual(selected[0], 472)
        self.assertGreaterEqual(selected[1], 552)
        self.assertLessEqual(selected[1], 700)
        self.assertEqual(
            diagnostics["selection_strategy"],
            "centered-region-after-sustained-coefficient-corruption",
        )

    def test_post_smoothing_slope_variance_triggers_region(self) -> None:
        """Moderate high-variance data should be rejected by center references."""

        length = 1024
        kx = np.arange(length, dtype=np.float64)
        coefficients = [np.sin(2.0 * np.pi * kx / 100.0) for _ in range(3)]
        coefficients[2] = coefficients[2].copy()
        coefficients[2][700:950] = 2.0 * np.sin(
            2.0 * np.pi * kx[700:950] / 20.0
        )
        quality = _quality(length, 21, 1003)

        selected, diagnostics = select_automatic_kx_range(
            coefficients, {"sin": quality, "cos": quality}
        )

        coefficient_diagnostics = diagnostics["coefficient_stability"]["c"]
        self.assertEqual(coefficient_diagnostics["rejected_gross_amplitude_samples"], 0)
        self.assertGreater(
            coefficient_diagnostics["rejected_center_slope_or_variance_samples"], 0
        )
        self.assertLessEqual(selected[1], 700)
        self.assertEqual(
            diagnostics["selection_strategy"],
            "centered-region-after-sustained-coefficient-corruption",
        )

    def test_rejects_inadequate_contiguous_support(self) -> None:
        """Short reliable runs should fail rather than trigger a fallback."""

        length = 160
        coefficients = [np.zeros(length, dtype=np.float64) for _ in range(3)]
        quality = _quality(length, 60, 80)

        with self.assertRaisesRegex(ValueError, "at least 75% reliable support"):
            select_automatic_kx_range(
                coefficients, {"sin": quality, "cos": quality}
            )


class SineLineFitTests(unittest.TestCase):
    def test_recovers_clean_model_and_reports_conditioning(self) -> None:
        """A clean model should fit accurately with finite diagnostics."""

        kx = np.arange(32, 144, dtype=np.float64)
        expected = sine_line_model(kx, 0.7, 2.0 * np.pi / 28.0, 0.4, 0.002, -0.3)

        fitted = fit_sine_plus_line(kx, expected)
        predicted = sine_line_model(
            kx,
            fitted["A"],
            fitted["w"],
            fitted["phi"],
            fitted["C1"],
            fitted["C2"],
        )

        np.testing.assert_allclose(predicted, expected, atol=1e-8, rtol=1e-8)
        self.assertTrue(fitted["success"])
        self.assertTrue(np.isfinite(fitted["standardized_jacobian_condition_number"]))
        self.assertLess(fitted["residual_rmse"], 1e-8)


if __name__ == "__main__":
    unittest.main()

