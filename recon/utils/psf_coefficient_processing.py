"""Select reliable readout support and fit Wave PSF coefficient models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from scipy.signal import lombscargle


AUTO_RANGE_ALGORITHM = "psf-sine-line-auto-range"
AUTO_RANGE_VERSION = 9
AUTO_FIT_PREFILTER_WINDOW = 9
CENTRAL_CORE_SAMPLES = 80
MINIMUM_CORE_SUPPORT_FRACTION = 0.75
QUALITY_DENSITY_WINDOW_SAMPLES = 21
MINIMUM_LOCAL_SUPPORT_FRACTION = 0.75
REGIONAL_MINIMUM_READOUT_FRACTION = 0.20
SUSTAINED_CORRUPTION_READOUT_FRACTION = 0.05
GLOBAL_MINIMUM_SUPPORT_FRACTION = 0.30
REGIONAL_MINIMUM_SUPPORT_FRACTION = 0.60
CENTER_SLOPE_LIMIT_MULTIPLIER = 2.5
CENTER_VARIANCE_LIMIT_MULTIPLIER = 6.0
CENTER_REFERENCE_READOUT_FRACTION = 0.20
NONLINEAR_FIT_MAX_NFEV = 500


def _nan_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered moving average that ignores non-finite samples.

    Args:
        values: One-dimensional input values.
        window: Positive odd moving-average width.

    Returns:
        Smoothed values, with NaN where a window has no finite support.

    Raises:
        ValueError: If ``window`` is not a positive odd integer.
    """

    if window < 1 or window % 2 == 0:
        raise ValueError("Moving-average window must be a positive odd integer.")
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    kernel = np.ones(window, dtype=np.float64)
    numerator = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    denominator = np.convolve(finite.astype(np.float64), kernel, mode="same")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _smoothed_shape_metrics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Measure local slope and variance after coefficient smoothing.

    Args:
        values: One-dimensional raw coefficient vector.

    Returns:
        Per-sample maximum adjacent smoothed slope and local smoothed variance.
    """

    smoothed = _nan_moving_average(values, window=AUTO_FIT_PREFILTER_WINDOW)
    adjacent_slope = np.abs(np.diff(smoothed))
    slope = np.full(smoothed.shape, np.nan, dtype=np.float64)
    if adjacent_slope.size:
        slope[0] = adjacent_slope[0]
        slope[-1] = adjacent_slope[-1]
        if slope.size > 2:
            slope[1:-1] = np.maximum(adjacent_slope[:-1], adjacent_slope[1:])
    local_mean = _nan_moving_average(smoothed, window=QUALITY_DENSITY_WINDOW_SAMPLES)
    local_variance = _nan_moving_average(
        (smoothed - local_mean) ** 2,
        window=QUALITY_DENSITY_WINDOW_SAMPLES,
    )
    return slope, local_variance


def sine_line_model(t, amplitude, angular_frequency, phase, slope, intercept):
    """Evaluate a sine plus linear trend.

    Args:
        t: Readout sample coordinates.
        amplitude: Nonnegative sine amplitude.
        angular_frequency: Angular frequency in radians per sample.
        phase: Sine phase at readout index zero.
        slope: Linear slope per readout sample.
        intercept: Linear intercept at readout index zero.

    Returns:
        Model values at ``t`` as a NumPy array.
    """

    return amplitude * np.sin(angular_frequency * t + phase) + slope * t + intercept


def fit_sine_plus_line(t, values) -> dict[str, Any]:
    """Fit a sine plus line and return optimizer and stability diagnostics.

    Args:
        t: One-dimensional readout sample coordinates.
        values: One-dimensional coefficient samples corresponding to ``t``.

    Returns:
        JSON-compatible fitted parameters and numerical diagnostics.

    Raises:
        ValueError: If the inputs do not contain enough finite, distinct samples
            or the optimizer fails to return finite fitted parameters.
    """

    t = np.asarray(t, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    valid = np.isfinite(t) & np.isfinite(values)
    t = t[valid]
    values = values[valid]
    if t.size < 6:
        raise ValueError("At least 6 finite coefficient samples are required for sine-line fitting.")
    order = np.argsort(t)
    t = t[order]
    values = values[order]
    unique_t = np.unique(t)
    if unique_t.size < 2 or np.ptp(unique_t) == 0:
        raise ValueError("PSF fit coordinates must contain more than one distinct value.")

    t_ref = float(np.mean(t))
    x = t - t_ref
    span = float(np.ptp(x))
    median_dt = float(np.median(np.diff(unique_t)))
    if not np.isfinite(median_dt) or median_dt <= 0:
        raise ValueError("PSF fit coordinates must have a positive finite spacing.")
    w_min = 2.0 * np.pi / span
    w_max = np.pi / median_dt
    if not w_min < w_max:
        raise ValueError("PSF fit interval is too narrow to identify a sine frequency.")

    slope_initial, intercept_ref_initial = np.polyfit(x, values, 1)
    detrended = values - (slope_initial * x + intercept_ref_initial)
    detrended -= np.mean(detrended)
    frequency_grid = np.linspace(w_min, w_max, 10000)
    power = lombscargle(x, detrended, frequency_grid, normalize=True)
    w_initial = float(frequency_grid[int(np.argmax(power))])

    design = np.column_stack(
        [np.sin(w_initial * x), np.cos(w_initial * x), x, np.ones_like(x)]
    )
    sine_coef, cosine_coef, slope_initial, intercept_ref_initial = np.linalg.lstsq(
        design, values, rcond=None
    )[0]
    amplitude_initial = float(np.hypot(sine_coef, cosine_coef))
    phase_ref_initial = float(np.arctan2(cosine_coef, sine_coef))
    initial = np.array(
        [
            max(amplitude_initial, np.finfo(float).eps),
            w_initial,
            phase_ref_initial,
            slope_initial,
            intercept_ref_initial,
        ]
    )

    def residuals(parameters):
        """Return centered-coordinate model residuals for least squares."""

        amplitude, angular_frequency, phase_ref, slope, intercept_ref = parameters
        return (
            amplitude * np.sin(angular_frequency * x + phase_ref)
            + slope * x
            + intercept_ref
            - values
        )

    result = least_squares(
        residuals,
        initial,
        bounds=(
            [0.0, w_min, -np.inf, -np.inf, -np.inf],
            [np.inf, w_max, np.inf, np.inf, np.inf],
        ),
        method="trf",
        x_scale="jac",
        loss="linear",
        max_nfev=NONLINEAR_FIT_MAX_NFEV,
    )
    amplitude, angular_frequency, phase_ref, slope, intercept_ref = result.x
    phase = (phase_ref - angular_frequency * t_ref + np.pi) % (2.0 * np.pi) - np.pi
    intercept = intercept_ref - slope * t_ref

    residual = np.asarray(result.fun, dtype=float)
    jacobian = np.asarray(result.jac, dtype=float)
    column_norm = np.linalg.norm(jacobian, axis=0)
    if np.any(~np.isfinite(column_norm)) or np.any(column_norm <= np.finfo(float).eps):
        condition_number = float("inf")
    else:
        condition_number = float(np.linalg.cond(jacobian / column_norm[None, :]))
    frequency_span = w_max - w_min
    frequency_boundary_fraction = float(
        min(angular_frequency - w_min, w_max - angular_frequency) / frequency_span
    )
    diagnostics = {
        "A": float(amplitude),
        "w": float(angular_frequency),
        "phi": float(phase),
        "C1": float(slope),
        "C2": float(intercept),
        "period_samples": float(2.0 * np.pi / angular_frequency),
        "cycles_in_fit_interval": float(angular_frequency * span / (2.0 * np.pi)),
        "frequency_search_bounds": [float(w_min), float(w_max)],
        "frequency_boundary_fraction": frequency_boundary_fraction,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "n_samples": int(t.size),
        "n_function_evaluations": int(result.nfev),
        "maximum_function_evaluations": NONLINEAR_FIT_MAX_NFEV,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "residual_rmse_relative_to_range": float(
            np.sqrt(np.mean(residual**2)) / max(float(np.ptp(values)), np.finfo(float).eps)
        ),
        "residual_median_absolute": float(np.median(np.abs(residual))),
        "residual_max_absolute": float(np.max(np.abs(residual))),
        "standardized_jacobian_condition_number": condition_number,
    }
    finite_parameters = np.all(
        np.isfinite([amplitude, angular_frequency, phase, slope, intercept])
    )
    if not result.success or not finite_parameters:
        raise ValueError(f"Sine-line optimizer failed: {result.message}")
    return diagnostics


def _quality_vector(
    quality: Mapping[str, Any], name: str, length: int, *, dtype: Any
) -> np.ndarray:
    """Return one validated per-readout quality vector.

    Args:
        quality: Projection-quality mapping.
        name: Required vector key.
        length: Expected readout length.
        dtype: NumPy dtype used for conversion.

    Returns:
        A one-dimensional NumPy quality vector.

    Raises:
        ValueError: If the vector is absent or has an unexpected length.
    """

    if name not in quality:
        raise ValueError(f"Projection quality is missing {name!r}.")
    vector = np.asarray(quality[name], dtype=dtype).reshape(-1)
    if vector.size != length:
        raise ValueError(
            f"Projection quality {name!r} has length {vector.size}; expected {length}."
        )
    return vector


def _robust_upper_limit(values: np.ndarray, absolute_limit: float) -> float:
    """Return a robust upper limit capped by a scientific absolute limit.

    Args:
        values: Finite quality values from provisionally valid samples.
        absolute_limit: Maximum allowed upper limit.

    Returns:
        Robust median/MAD upper limit no larger than ``absolute_limit``.
    """

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust = max(median + 4.0 * 1.4826 * mad, 1.5 * median + 0.02)
    return float(min(absolute_limit, robust))


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals for contiguous true samples.

    Args:
        mask: One-dimensional boolean mask.

    Returns:
        Half-open ``(start, stop)`` intervals in ascending order.
    """

    padded = np.pad(np.asarray(mask, dtype=bool), (1, 1), constant_values=False)
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _coefficient_stability_mask(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject isolated raw-coefficient transients using a local median residual.

    Args:
        values: One-dimensional raw coefficient vector.

    Returns:
        Boolean stability mask and JSON-compatible robust-threshold diagnostics.
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    stable = finite.copy()
    if np.count_nonzero(finite) < 9:
        return stable, {
            "window_samples": 9,
            "deviation_limit": None,
            "median_absolute_local_deviation": None,
            "median_absolute_adjacent_step": None,
            "adjacent_step_percentile_90": None,
            "central_reference_interval": None,
            "central_reference_median": None,
            "gross_amplitude_deviation_limit": None,
            "rejected_gross_amplitude_samples": 0,
            "central_smoothed_max_absolute_slope": None,
            "smoothed_slope_limit": None,
            "central_smoothed_max_local_variance": None,
            "smoothed_local_variance_limit": None,
            "rejected_center_slope_or_variance_samples": 0,
            "rejected_transient_samples": 0,
            "rejected_total_samples": 0,
        }
    indices = np.arange(values.size, dtype=float)
    filled = np.interp(indices, indices[finite], values[finite])
    local_median = median_filter(filled, size=9, mode="reflect")
    deviation = np.abs(filled - local_median)
    finite_deviation = deviation[finite]
    median = float(np.median(finite_deviation))
    mad = float(np.median(np.abs(finite_deviation - median)))
    percentile_90 = float(np.percentile(finite_deviation, 90.0))
    finite_pairs = finite[:-1] & finite[1:]
    finite_steps = np.abs(np.diff(values)[finite_pairs])
    median_step = float(np.median(finite_steps)) if finite_steps.size else 0.0
    step_percentile_90 = (
        float(np.percentile(finite_steps, 90.0)) if finite_steps.size else 0.0
    )
    limit = max(
        median + 8.0 * 1.4826 * mad,
        3.0 * percentile_90,
        8.0 * median_step,
        4.0 * step_percentile_90,
        100.0 * np.finfo(float).eps,
    )
    transient = finite & (deviation > limit)
    stable &= ~transient
    center_width = min(
        values.size,
        max(
            CENTRAL_CORE_SAMPLES,
            int(np.ceil(CENTER_REFERENCE_READOUT_FRACTION * values.size)),
        ),
    )
    center_start = (values.size - center_width) // 2
    center_values = values[center_start : center_start + center_width]
    center_values = center_values[np.isfinite(center_values)]
    if not center_values.size:
        center_values = values[finite]
    center_median = float(np.median(center_values))
    center_mad = float(np.median(np.abs(center_values - center_median)))
    amplitude_limit = max(
        5.0 * 1.4826 * center_mad,
        20.0 * median_step,
        0.25,
    )
    gross_amplitude = finite & (np.abs(values - center_median) > amplitude_limit)
    stable &= ~gross_amplitude
    smoothed_slope, smoothed_local_variance = _smoothed_shape_metrics(values)
    center_slice = slice(center_start, center_start + center_width)
    central_slope = smoothed_slope[center_slice]
    central_slope = central_slope[np.isfinite(central_slope)]
    central_variance = smoothed_local_variance[center_slice]
    central_variance = central_variance[np.isfinite(central_variance)]
    central_max_slope = float(np.max(central_slope)) if central_slope.size else 0.0
    central_max_variance = (
        float(np.max(central_variance)) if central_variance.size else 0.0
    )
    slope_limit = max(
        CENTER_SLOPE_LIMIT_MULTIPLIER * central_max_slope,
        100.0 * np.finfo(float).eps,
    )
    variance_limit = max(
        CENTER_VARIANCE_LIMIT_MULTIPLIER * central_max_variance,
        100.0 * np.finfo(float).eps,
    )
    excessive_shape_variation = finite & (
        (smoothed_slope > slope_limit)
        | (smoothed_local_variance > variance_limit)
    )
    stable &= ~excessive_shape_variation
    return stable, {
        "window_samples": 9,
        "deviation_limit": float(limit),
        "median_absolute_local_deviation": median,
        "median_absolute_adjacent_step": median_step,
        "adjacent_step_percentile_90": step_percentile_90,
        "central_reference_interval": [center_start, center_start + center_width],
        "central_reference_median": center_median,
        "gross_amplitude_deviation_limit": float(amplitude_limit),
        "rejected_gross_amplitude_samples": int(np.count_nonzero(gross_amplitude)),
        "central_smoothed_max_absolute_slope": central_max_slope,
        "smoothed_slope_limit": float(slope_limit),
        "central_smoothed_max_local_variance": central_max_variance,
        "smoothed_local_variance_limit": float(variance_limit),
        "rejected_center_slope_or_variance_samples": int(
            np.count_nonzero(excessive_shape_variation)
        ),
        "rejected_transient_samples": int(np.count_nonzero(transient)),
        "rejected_total_samples": int(np.count_nonzero(finite & ~stable)),
    }


def select_automatic_kx_range(
    raw_coefficients: Sequence[Any],
    projection_quality: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[int, int], dict[str, Any]]:
    """Select one reliable contiguous readout interval for a, b, and c.

    Args:
        raw_coefficients: Three equally sized raw coefficient vectors ``a,b,c``.
        projection_quality: Per-readout quality mappings for ``sin`` and ``cos``
            projection fits. Each must contain ``skipped``, ``valid_pixels``,
            ``masked_ratio``, and ``wrapped_rms``.

    Returns:
        Selected half-open interval and JSON-compatible detection diagnostics.

    Raises:
        ValueError: If the evidence is malformed or no adequate common interval
            remains after finite, support, residual, transient, and edge gates.
    """

    coefficients = [np.asarray(value, dtype=float).reshape(-1) for value in raw_coefficients]
    if len(coefficients) != 3 or not coefficients[0].size:
        raise ValueError("Automatic PSF range selection requires non-empty a, b, and c vectors.")
    nx = int(coefficients[0].size)
    if any(value.size != nx for value in coefficients):
        raise ValueError("Raw PSF coefficient vectors must have one common length.")

    coefficient_valid = np.logical_and.reduce([np.isfinite(value) for value in coefficients])
    coefficient_stability = {}
    for name, values in zip(("a", "b", "c"), coefficients, strict=True):
        stable, stability_diagnostics = _coefficient_stability_mask(values)
        coefficient_valid &= stable
        coefficient_stability[name] = stability_diagnostics
    common_valid = coefficient_valid.copy()
    projection_diagnostics: dict[str, Any] = {}
    for projection in ("sin", "cos"):
        if projection not in projection_quality:
            raise ValueError(f"Automatic PSF range selection requires {projection!r} quality.")
        quality = projection_quality[projection]
        skipped = _quality_vector(quality, "skipped", nx, dtype=bool)
        valid_pixels = _quality_vector(quality, "valid_pixels", nx, dtype=float)
        masked_ratio = _quality_vector(quality, "masked_ratio", nx, dtype=float)
        wrapped_rms = _quality_vector(quality, "wrapped_rms", nx, dtype=float)
        support_fraction = 1.0 - masked_ratio
        provisional = (
            ~skipped
            & np.isfinite(valid_pixels)
            & np.isfinite(support_fraction)
            & np.isfinite(wrapped_rms)
            & (valid_pixels >= 10)
            & (support_fraction > 0.0)
        )
        if not np.any(provisional):
            raise ValueError(f"The {projection} projection has no provisionally reliable samples.")
        median_support = float(np.median(support_fraction[provisional]))
        support_floor = float(max(0.15, 0.5 * median_support))
        rms_ceiling = _robust_upper_limit(wrapped_rms[provisional], absolute_limit=0.7)
        accepted = (
            provisional
            & (support_fraction >= support_floor)
            & (wrapped_rms <= rms_ceiling)
        )
        common_valid &= accepted
        projection_diagnostics[projection] = {
            "provisional_samples": int(np.count_nonzero(provisional)),
            "accepted_samples": int(np.count_nonzero(accepted)),
            "minimum_valid_pixels": 10,
            "support_fraction_floor": support_floor,
            "wrapped_rms_ceiling_rad": rms_ceiling,
            "median_support_fraction": median_support,
            "median_wrapped_rms_rad": float(np.median(wrapped_rms[provisional])),
        }

    if nx < CENTRAL_CORE_SAMPLES:
        raise ValueError(
            "Automatic PSF range selection requires at least "
            f"{CENTRAL_CORE_SAMPLES} readout samples; received {nx}."
        )

    edge_guard = max(5, int(np.ceil(0.02 * nx)))
    common_valid[:edge_guard] = False
    common_valid[nx - edge_guard :] = False
    regional_minimum_width = max(
        CENTRAL_CORE_SAMPLES,
        24,
        int(np.ceil(REGIONAL_MINIMUM_READOUT_FRACTION * nx)),
    )
    core_start = (nx - CENTRAL_CORE_SAMPLES) // 2
    core_stop = core_start + CENTRAL_CORE_SAMPLES
    core_valid_samples = int(np.count_nonzero(common_valid[core_start:core_stop]))
    core_support_fraction = core_valid_samples / CENTRAL_CORE_SAMPLES
    if core_support_fraction < MINIMUM_CORE_SUPPORT_FRACTION:
        raise ValueError(
            f"Automatic PSF range selection requires at least "
            f"{MINIMUM_CORE_SUPPORT_FRACTION:.0%} reliable support in the central "
            f"{CENTRAL_CORE_SAMPLES}-sample span [{core_start}, {core_stop}); "
            f"received {core_valid_samples}/{CENTRAL_CORE_SAMPLES} "
            f"({core_support_fraction:.1%})."
        )

    density_kernel = np.ones(QUALITY_DENSITY_WINDOW_SAMPLES, dtype=np.float64)
    local_total = np.convolve(np.ones(nx, dtype=np.float64), density_kernel, mode="same")
    local_valid = np.convolve(common_valid.astype(np.float64), density_kernel, mode="same")
    local_support_fraction = local_valid / local_total
    dense_support = local_support_fraction >= MINIMUM_LOCAL_SUPPORT_FRACTION
    dense_support[:edge_guard] = False
    dense_support[nx - edge_guard :] = False

    coefficient_local_valid = np.convolve(
        coefficient_valid.astype(np.float64), density_kernel, mode="same"
    )
    coefficient_local_support = coefficient_local_valid / local_total
    coefficient_dense_support = (
        coefficient_local_support >= MINIMUM_LOCAL_SUPPORT_FRACTION
    )
    coefficient_dense_support[:edge_guard] = False
    coefficient_dense_support[nx - edge_guard :] = False

    strict_runs = _contiguous_true_runs(common_valid)
    density_runs = _contiguous_true_runs(dense_support)
    coefficient_density_runs = _contiguous_true_runs(coefficient_dense_support)
    low_coefficient_density_runs = _contiguous_true_runs(~coefficient_dense_support)
    corruption_minimum_width = max(
        16, int(np.ceil(SUSTAINED_CORRUPTION_READOUT_FRACTION * nx))
    )
    interior_start = max(edge_guard, int(np.floor(0.15 * nx)))
    interior_stop = min(nx - edge_guard, int(np.ceil(0.85 * nx)))
    sustained_corruption_runs = []
    for run in low_coefficient_density_runs:
        overlap = max(0, min(run[1], interior_stop) - max(run[0], interior_start))
        if overlap >= corruption_minimum_width:
            sustained_corruption_runs.append(run)

    near_global_interval = (edge_guard, nx - edge_guard)
    if sustained_corruption_runs:
        regional_candidates = [
            run
            for run in coefficient_density_runs
            if run[0] <= core_start
            and run[1] >= core_stop
            and run[1] - run[0] >= regional_minimum_width
        ]
        if not regional_candidates:
            raise ValueError(
                "Automatic PSF range selection detected sustained coefficient "
                "corruption but found no adequate center-containing stable region."
            )
        selected_start, selected_stop = max(
            regional_candidates,
            key=lambda run: (run[1] - run[0], -run[0]),
        )
        while selected_start > edge_guard and coefficient_valid[selected_start - 1]:
            selected_start -= 1
        while selected_stop < nx - edge_guard and coefficient_valid[selected_stop]:
            selected_stop += 1
        selected = (int(selected_start), int(selected_stop))
        selection_strategy = "centered-region-after-sustained-coefficient-corruption"
        minimum_interval_support_fraction = REGIONAL_MINIMUM_SUPPORT_FRACTION
    else:
        selected = near_global_interval
        selection_strategy = "near-global-no-sustained-coefficient-corruption"
        minimum_interval_support_fraction = GLOBAL_MINIMUM_SUPPORT_FRACTION

    fit_sample_mask = np.zeros(nx, dtype=bool)
    fit_sample_mask[selected[0] : selected[1]] = common_valid[selected[0] : selected[1]]
    fit_sample_count = int(np.count_nonzero(fit_sample_mask))
    fit_support_fraction = fit_sample_count / (selected[1] - selected[0])
    minimum_fit_samples = max(80, int(np.ceil(0.10 * nx)))
    if (
        fit_sample_count < minimum_fit_samples
        or fit_support_fraction < minimum_interval_support_fraction
    ):
        raise ValueError(
            f"Automatic PSF range selection found only {fit_sample_count} reliable "
            f"samples ({fit_support_fraction:.1%}) in the centered interval "
            f"[{selected[0]}, {selected[1]}); at least {minimum_fit_samples} samples "
            f"and {minimum_interval_support_fraction:.0%} support are required."
        )
    diagnostics = {
        "name": AUTO_RANGE_ALGORITHM,
        "version": AUTO_RANGE_VERSION,
        "coordinate": "oversampled_readout_sample_index",
        "range_convention": "half-open [min, max)",
        "readout_length": nx,
        "readout_center_index": nx // 2,
        "edge_guard_samples": edge_guard,
        "near_global_interval": list(near_global_interval),
        "regional_minimum_interval_samples": regional_minimum_width,
        "required_central_core_samples": CENTRAL_CORE_SAMPLES,
        "required_central_core_interval": [core_start, core_stop],
        "minimum_central_core_support_fraction": MINIMUM_CORE_SUPPORT_FRACTION,
        "central_core_valid_samples": core_valid_samples,
        "central_core_support_fraction": core_support_fraction,
        "quality_density_window_samples": QUALITY_DENSITY_WINDOW_SAMPLES,
        "minimum_local_support_fraction": MINIMUM_LOCAL_SUPPORT_FRACTION,
        "selection_strategy": selection_strategy,
        "sustained_corruption_minimum_samples": corruption_minimum_width,
        "sustained_coefficient_corruption_intervals": [
            list(run) for run in sustained_corruption_runs
        ],
        "minimum_fit_samples": minimum_fit_samples,
        "minimum_interval_support_fraction": minimum_interval_support_fraction,
        "common_valid_samples": int(np.count_nonzero(common_valid)),
        "strict_common_valid_intervals": [list(run) for run in strict_runs],
        "quality_density_intervals": [list(run) for run in density_runs],
        "coefficient_quality_density_intervals": [
            list(run) for run in coefficient_density_runs
        ],
        "selected_interval": list(selected),
        "fit_sample_count": fit_sample_count,
        "fit_support_fraction": fit_support_fraction,
        "coefficient_stability": coefficient_stability,
        "projection_quality": projection_diagnostics,
        "_fit_sample_mask": fit_sample_mask.tolist(),
    }
    diagnostics["excluded_sample_indices_within_interval"] = (
        np.flatnonzero(~fit_sample_mask[selected[0] : selected[1]]) + selected[0]
    ).astype(int).tolist()
    return selected, diagnostics
