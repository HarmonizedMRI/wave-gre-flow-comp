"""
psf_wrapped_phase_fit.py

Fit wrapped phase planes

    p(kx, y, z) = a(kx) y + b(kx) z + c(kx)

for wrapped phase data p in [-pi, pi), with optional magnitude weighting,
local circular phase-coherence masking, and optional second-pass residual
coherence refinement.

Typical use
-----------
from wrapped_phase_fit_localcoherence_v2 import fit_wrapped_phase_planes

result = fit_wrapped_phase_planes(
    psf_diff,
    hyb_nowave,
    y_norm,
    z_norm,
    mask_mode="combined",
    use_residual_coherence_refinement=True,
    return_quality_maps=True,
)

Author: Yiyun Dong
Affiliation: Athinoula A. Martinos Center for Biomedical Imaging
License: MIT License
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

__version__ = "2026-06-02-localcoherence-v2-renamed"


TensorLike1D = Union[torch.Tensor, Sequence[float]]


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap radians to [-pi, pi)."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def _mean_over_dims(
    x: torch.Tensor,
    dims: Optional[Union[int, Sequence[int]]],
) -> torch.Tensor:
    """Mean over one or more dims, supporting negative dim indexing."""
    if dims is None:
        return x
    if isinstance(dims, int):
        dims = (dims,)
    else:
        dims = tuple(dims)
    ndim = x.ndim
    dims = tuple(d if d >= 0 else ndim + d for d in dims)
    for d in sorted(dims, reverse=True):
        x = x.mean(dim=d)
    return x


def _weighted_lstsq(
    A: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    ridge: float = 1e-8,
) -> torch.Tensor:
    """Solve min ||sqrt(w) * (A theta - b)||^2 with ridge stabilization."""
    w = torch.clamp(w, min=0.0)
    sqrt_w = torch.sqrt(w)
    Aw = A * sqrt_w[:, None]
    bw = b * sqrt_w

    ATA = Aw.T @ Aw
    ATb = Aw.T @ bw
    eye = torch.eye(ATA.shape[0], dtype=ATA.dtype, device=ATA.device)
    ridge_scaled = ridge * torch.trace(ATA).clamp_min(1.0) / ATA.shape[0]

    try:
        return torch.linalg.solve(ATA + ridge_scaled * eye, ATb)
    except RuntimeError:
        return torch.linalg.lstsq(Aw, bw).solution


def local_circular_coherence_2d(
    phase: torch.Tensor,
    window_size: int = 5,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fast local circular phase coherence using avg_pool2d.

    coherence = abs(mean(exp(1j * phase))) over a local window.

    Returns values in [0, 1]. 1 means locally coherent; 0 means noisy/random.
    """
    if phase.ndim != 2:
        raise ValueError(f"phase must be 2D, got shape {tuple(phase.shape)}")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    dtype = phase.dtype
    device = phase.device
    eps = torch.finfo(dtype).eps

    finite = torch.isfinite(phase)
    if valid_mask is None:
        valid = finite
    else:
        valid = valid_mask.to(device=device, dtype=torch.bool) & finite

    valid_f = valid.to(dtype)
    phase_safe = torch.where(finite, phase, torch.zeros((), dtype=dtype, device=device))

    cos_p = torch.cos(phase_safe) * valid_f
    sin_p = torch.sin(phase_safe) * valid_f

    pad = window_size // 2

    cos_avg = F.avg_pool2d(
        cos_p[None, None], kernel_size=window_size, stride=1,
        padding=pad, count_include_pad=False
    )[0, 0]
    sin_avg = F.avg_pool2d(
        sin_p[None, None], kernel_size=window_size, stride=1,
        padding=pad, count_include_pad=False
    )[0, 0]
    valid_avg = F.avg_pool2d(
        valid_f[None, None], kernel_size=window_size, stride=1,
        padding=pad, count_include_pad=False
    )[0, 0]

    cos_mean = cos_avg / valid_avg.clamp_min(eps)
    sin_mean = sin_avg / valid_avg.clamp_min(eps)
    coherence = torch.sqrt(cos_mean.square() + sin_mean.square())
    coherence = torch.where(valid_avg > 0, coherence, torch.zeros_like(coherence))
    return torch.clamp(coherence, 0.0, 1.0)


def _adaptive_mag_mask(
    mag2_t: torch.Tensor,
    phase_t: torch.Tensor,
    *,
    mag_abs_floor: float,
    mag_lower_quantile: float,
    min_keep_fraction: float,
) -> torch.Tensor:
    """Per-slice adaptive magnitude mask."""
    if not (0.0 <= mag_lower_quantile <= 1.0):
        raise ValueError("mag_lower_quantile must be between 0 and 1")
    if not (0.0 <= min_keep_fraction <= 1.0):
        raise ValueError("min_keep_fraction must be between 0 and 1")

    finite = torch.isfinite(mag2_t) & torch.isfinite(phase_t)
    valid_abs = finite & (mag2_t > mag_abs_floor)
    if int(valid_abs.sum().item()) == 0:
        return valid_abs

    q = torch.quantile(mag2_t[valid_abs], mag_lower_quantile)
    mask = valid_abs & (mag2_t >= q)

    min_count = int(round(min_keep_fraction * mag2_t.numel()))
    if min_count > 0 and int(mask.sum().item()) < min_count:
        flat_valid = valid_abs.flatten()
        valid_idx = torch.nonzero(flat_valid, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            return valid_abs
        k = min(min_count, valid_idx.numel())
        flat_mag = mag2_t.flatten()
        _, order = torch.topk(flat_mag[valid_idx], k=k, largest=True)
        flat_mask = torch.zeros_like(flat_valid, dtype=torch.bool)
        flat_mask[valid_idx[order]] = True
        mask = flat_mask.view_as(mag2_t)

    return mask


def _make_initial_mask_and_weights(
    phase_t: torch.Tensor,
    mag2_t: torch.Tensor,
    *,
    mask_mode: str,
    mag_threshold: float,
    mag_abs_floor: float,
    mag_lower_quantile: float,
    min_keep_fraction: float,
    local_window_size: int,
    coherence_threshold: float,
    use_phase_coherence_weight: bool,
    phase_weight_power: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return initial mask, phase-coherence weight map, and coherence map."""
    finite = torch.isfinite(phase_t) & torch.isfinite(mag2_t)
    phase_coherence = None
    phase_weight = torch.ones_like(phase_t)

    if mask_mode == "absolute":
        mask = finite & (mag2_t > mag_threshold)

    elif mask_mode == "adaptive_quantile":
        mask = _adaptive_mag_mask(
            mag2_t,
            phase_t,
            mag_abs_floor=mag_abs_floor,
            mag_lower_quantile=mag_lower_quantile,
            min_keep_fraction=min_keep_fraction,
        )

    elif mask_mode in ("local_phase_variance", "local_phase_coherence"):
        valid_abs = finite & (mag2_t > mag_abs_floor)
        phase_coherence = local_circular_coherence_2d(
            phase_t,
            window_size=local_window_size,
            valid_mask=valid_abs,
        )
        mask = valid_abs & (phase_coherence >= coherence_threshold)

    elif mask_mode == "combined":
        # Weak absolute floor + raw local phase coherence.
        valid_abs = finite & (mag2_t > mag_abs_floor)
        phase_coherence = local_circular_coherence_2d(
            phase_t,
            window_size=local_window_size,
            valid_mask=valid_abs,
        )
        mask = valid_abs & (phase_coherence >= coherence_threshold)

    elif mask_mode == "none":
        mask = finite

    else:
        raise ValueError(
            "mask_mode must be one of: 'absolute', 'adaptive_quantile', "
            "'local_phase_variance', 'local_phase_coherence', 'combined', 'none'"
        )

    if use_phase_coherence_weight:
        if phase_coherence is None:
            valid_abs = finite & (mag2_t > mag_abs_floor)
            phase_coherence = local_circular_coherence_2d(
                phase_t,
                window_size=local_window_size,
                valid_mask=valid_abs,
            )
        phase_weight = torch.clamp(phase_coherence, 0.0, 1.0).pow(phase_weight_power)

    return mask, phase_weight, phase_coherence


def _run_wrapped_irls(
    A: torch.Tensor,
    p: torch.Tensor,
    w_base: torch.Tensor,
    theta0: torch.Tensor,
    *,
    n_irls: int,
    huber_delta: float,
    ridge: float,
    convergence_tol: float,
) -> torch.Tensor:
    """Robust model-based wrapped IRLS for one slice."""
    eps = torch.finfo(p.dtype).eps
    theta = theta0.clone()

    for _ in range(n_irls):
        pred = A @ theta
        r = wrap_to_pi(p - pred)
        p_model_unwrapped = pred + r

        abs_r = torch.abs(r)
        w_huber = torch.ones_like(abs_r)
        large = abs_r > huber_delta
        w_huber[large] = huber_delta / abs_r[large].clamp_min(eps)

        theta_new = _weighted_lstsq(A, p_model_unwrapped, w_base * w_huber, ridge=ridge)

        denom = torch.norm(theta).clamp_min(1.0)
        if torch.norm(theta_new - theta) < convergence_tol * denom:
            theta = theta_new
            break
        theta = theta_new

    return theta


def fit_wrapped_phase_planes(
    psf_diff: torch.Tensor,
    hyb_nowave: torch.Tensor,
    y_norm: TensorLike1D,
    z_norm: TensorLike1D,
    *,
    # Original behavior controls.
    mag_threshold: float = 8e-10,
    max_masked_threshold: float = 0.94,
    n_irls: int = 10,
    huber_delta: float = 0.7,
    ridge: float = 1e-8,
    min_valid_pixels: int = 10,
    weight_clip: Optional[float] = 100.0,
    mag_average_dims: Optional[Union[int, Sequence[int]]] = -1,
    use_previous_fit_as_init: bool = True,
    initial_coefficients: Optional[torch.Tensor] = None,
    verbose: bool = True,
    return_wrapped_prediction: bool = True,
    # New mask modes.
    mask_mode: str = "absolute",
    mag_abs_floor: Optional[float] = None,
    mag_lower_quantile: float = 0.10,
    min_keep_fraction: float = 0.30,
    use_mag_weight: bool = True,
    # Raw local phase coherence.
    local_window_size: int = 5,
    coherence_threshold: float = 0.75,
    use_phase_coherence_weight: bool = False,
    phase_weight_power: float = 2.0,
    # Second-pass residual coherence refinement.
    use_residual_coherence_refinement: bool = False,
    residual_window_size: Optional[int] = None,
    residual_coherence_threshold: Optional[float] = None,
    use_residual_coherence_weight: bool = True,
    residual_weight_power: float = 2.0,
    # Output control.
    return_quality_maps: bool = False,
    convergence_tol: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Fit wrapped phase planes independently for each first-axis slice.

    Parameters most relevant to the new behavior
    --------------------------------------------
    mask_mode:
        'absolute'             : original fixed magnitude mask.
        'adaptive_quantile'    : per-slice magnitude quantile mask.
        'local_phase_variance' : raw local circular phase coherence mask.
        'local_phase_coherence': same as local_phase_variance.
        'combined'             : weak magnitude floor + raw local coherence.
        'none'                 : all finite pixels.

    use_residual_coherence_refinement:
        If True, performs a second fit after computing local circular coherence
        of the first-pass wrapped residual.
    """
    if psf_diff.ndim != 3:
        raise ValueError(f"psf_diff must be [Nx, Ny, Nz], got {tuple(psf_diff.shape)}")
    if not torch.is_floating_point(psf_diff):
        raise TypeError("psf_diff must be a real floating-point tensor")

    device = psf_diff.device
    dtype = psf_diff.dtype
    Nx, Ny, Nz = psf_diff.shape

    if mag_abs_floor is None:
        mag_abs_floor = mag_threshold if mask_mode == "absolute" else 0.0
    if residual_window_size is None:
        residual_window_size = local_window_size
    if residual_coherence_threshold is None:
        residual_coherence_threshold = coherence_threshold

    y = torch.as_tensor(y_norm, dtype=dtype, device=device)
    z = torch.as_tensor(z_norm, dtype=dtype, device=device)
    if y.numel() != Ny:
        raise ValueError(f"len(y_norm)={y.numel()} must match Ny={Ny}")
    if z.numel() != Nz:
        raise ValueError(f"len(z_norm)={z.numel()} must match Nz={Nz}")

    if hyb_nowave.shape[:3] != psf_diff.shape:
        raise ValueError(
            f"hyb_nowave first 3 dims must match psf_diff. "
            f"Got {tuple(hyb_nowave.shape)} vs {tuple(psf_diff.shape)}"
        )

    Y, Z = torch.meshgrid(y, z, indexing="ij")
    A_full = torch.stack([Y.flatten(), Z.flatten(), torch.ones(Ny * Nz, dtype=dtype, device=device)], dim=1)

    mag2 = _mean_over_dims(hyb_nowave.abs().square(), mag_average_dims).to(dtype=dtype, device=device)
    if mag2.shape != psf_diff.shape:
        raise ValueError(
            f"After averaging, mag2 shape must equal psf_diff. Got {tuple(mag2.shape)}. "
            "Check mag_average_dims."
        )

    a_fit_all = torch.full((Nx,), torch.nan, dtype=dtype, device=device)
    b_fit_all = torch.full((Nx,), torch.nan, dtype=dtype, device=device)
    c_fit_all = torch.full((Nx,), torch.nan, dtype=dtype, device=device)
    psf_diff_pred = torch.full_like(psf_diff, torch.nan)
    wrapped_rms = torch.full((Nx,), torch.nan, dtype=dtype, device=device)
    valid_pixels = torch.zeros((Nx,), dtype=torch.long, device=device)
    masked_ratio = torch.full((Nx,), torch.nan, dtype=dtype, device=device)
    skipped = torch.ones((Nx,), dtype=torch.bool, device=device)

    initial_mask_all = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device=device)
    final_mask_all = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device=device)

    if return_quality_maps:
        phase_coherence_all = torch.full_like(psf_diff, torch.nan)
        residual_coherence_all = torch.full_like(psf_diff, torch.nan)
    else:
        phase_coherence_all = None
        residual_coherence_all = None

    if initial_coefficients is None:
        prev_theta = None
    else:
        prev_theta = torch.as_tensor(initial_coefficients, dtype=dtype, device=device).reshape(3)

    eps = torch.finfo(dtype).eps

    with torch.no_grad():
        for k in range(Nx):
            phase_t = psf_diff[k]
            mag2_t = mag2[k]

            init_mask, phase_weight_t, phase_coherence_t = _make_initial_mask_and_weights(
                phase_t,
                mag2_t,
                mask_mode=mask_mode,
                mag_threshold=mag_threshold,
                mag_abs_floor=float(mag_abs_floor),
                mag_lower_quantile=mag_lower_quantile,
                min_keep_fraction=min_keep_fraction,
                local_window_size=local_window_size,
                coherence_threshold=coherence_threshold,
                use_phase_coherence_weight=use_phase_coherence_weight,
                phase_weight_power=phase_weight_power,
            )

            initial_mask_all[k] = init_mask
            if return_quality_maps and phase_coherence_t is not None:
                phase_coherence_all[k] = phase_coherence_t

            total = init_mask.numel()
            n_valid = int(init_mask.sum().item())
            ratio = 1.0 - n_valid / total

            if n_valid < min_valid_pixels or ratio > max_masked_threshold:
                valid_pixels[k] = n_valid
                masked_ratio[k] = ratio
                if verbose:
                    print(f"Skipped {k}/{Nx}: masked_ratio={ratio:.3f}, valid={n_valid}")
                continue

            p_flat = phase_t.flatten()
            mag2_flat = mag2_t.flatten()
            init_mask_flat = init_mask.flatten()

            A = A_full[init_mask_flat]
            p = p_flat[init_mask_flat]

            if use_mag_weight:
                w_mag = mag2_flat[init_mask_flat]
                w_mag = w_mag / w_mag.median().clamp_min(eps)
                if weight_clip is not None:
                    w_mag = torch.clamp(w_mag, max=float(weight_clip))
            else:
                w_mag = torch.ones_like(p)

            w_phase = phase_weight_t.flatten()[init_mask_flat]
            w_base = w_mag * w_phase

            if use_previous_fit_as_init and prev_theta is not None:
                theta0 = prev_theta.clone()
            elif initial_coefficients is not None:
                theta0 = torch.as_tensor(initial_coefficients, dtype=dtype, device=device).reshape(3).clone()
            else:
                theta0 = _weighted_lstsq(A, p, w_base, ridge=ridge)

            theta = _run_wrapped_irls(
                A,
                p,
                w_base,
                theta0,
                n_irls=n_irls,
                huber_delta=huber_delta,
                ridge=ridge,
                convergence_tol=convergence_tol,
            )

            final_mask = init_mask

            if use_residual_coherence_refinement:
                # --- Second pass starts here. ---
                pred_first = (A_full @ theta).view(Ny, Nz)
                resid_t = wrap_to_pi(phase_t - pred_first)

                residual_coherence_t = local_circular_coherence_2d(
                    resid_t,
                    window_size=residual_window_size,
                    valid_mask=init_mask,
                )
                if return_quality_maps:
                    residual_coherence_all[k] = residual_coherence_t

                refined_mask = init_mask & (residual_coherence_t >= residual_coherence_threshold)
                refined_valid = int(refined_mask.sum().item())
                refined_ratio = 1.0 - refined_valid / total

                # Guardrail: if residual mask is too aggressive, keep the original
                # mask but still allow residual coherence as a soft weight.
                if refined_valid >= min_valid_pixels and refined_ratio <= max_masked_threshold:
                    final_mask = refined_mask
                else:
                    final_mask = init_mask

                final_mask_flat = final_mask.flatten()
                A2 = A_full[final_mask_flat]
                p2 = p_flat[final_mask_flat]

                if use_mag_weight:
                    w_mag2 = mag2_flat[final_mask_flat]
                    w_mag2 = w_mag2 / w_mag2.median().clamp_min(eps)
                    if weight_clip is not None:
                        w_mag2 = torch.clamp(w_mag2, max=float(weight_clip))
                else:
                    w_mag2 = torch.ones_like(p2)

                if use_phase_coherence_weight:
                    w_phase2 = phase_weight_t.flatten()[final_mask_flat]
                else:
                    w_phase2 = torch.ones_like(p2)

                if use_residual_coherence_weight:
                    w_resid2 = residual_coherence_t.flatten()[final_mask_flat]
                    w_resid2 = torch.clamp(w_resid2, 0.0, 1.0).pow(residual_weight_power)
                else:
                    w_resid2 = torch.ones_like(p2)

                w_base2 = w_mag2 * w_phase2 * w_resid2

                theta = _run_wrapped_irls(
                    A2,
                    p2,
                    w_base2,
                    theta,
                    n_irls=n_irls,
                    huber_delta=huber_delta,
                    ridge=ridge,
                    convergence_tol=convergence_tol,
                )

            final_mask_all[k] = final_mask
            final_mask_flat = final_mask.flatten()
            A_final = A_full[final_mask_flat]
            p_final = p_flat[final_mask_flat]

            n_final = int(final_mask.sum().item())
            ratio_final = 1.0 - n_final / total

            a_fit_all[k], b_fit_all[k], c_fit_all[k] = theta
            pred_full = A_full @ theta
            psf_diff_pred[k] = pred_full.view(Ny, Nz)

            r_final = wrap_to_pi(p_final - A_final @ theta)
            wrapped_rms[k] = torch.sqrt(torch.mean(r_final.square()))
            valid_pixels[k] = n_final
            masked_ratio[k] = ratio_final
            skipped[k] = False

            if use_previous_fit_as_init:
                prev_theta = theta.clone()

            if verbose:
                suffix = " + second pass" if use_residual_coherence_refinement else ""
                print(
                    f"Finished {k}/{Nx}{suffix}: "
                    f"a={theta[0].item():.4g}, b={theta[1].item():.4g}, c={theta[2].item():.4g}, "
                    f"wrapped RMS={wrapped_rms[k].item():.4g} rad, valid={n_final}/{total}"
                )

    coefficients = torch.stack([a_fit_all, b_fit_all, c_fit_all], dim=1)

    result: Dict[str, torch.Tensor] = {
        "a_fit_all": a_fit_all,
        "b_fit_all": b_fit_all,
        "c_fit_all": c_fit_all,
        "coefficients": coefficients,
        "psf_diff_pred": psf_diff_pred,
        "mask": final_mask_all,
        "initial_mask": initial_mask_all,
        "wrapped_rms": wrapped_rms,
        "valid_pixels": valid_pixels,
        "masked_ratio": masked_ratio,
        "skipped": skipped,
    }

    if return_wrapped_prediction:
        result["psf_diff_pred_wrapped"] = wrap_to_pi(psf_diff_pred)

    if return_quality_maps:
        result["phase_coherence"] = phase_coherence_all
        result["phase_variance"] = 1.0 - phase_coherence_all
        result["residual_coherence"] = residual_coherence_all
        result["residual_phase_variance"] = 1.0 - residual_coherence_all

    return result


def smooth_1d(x, window=5):
    pad = window // 2
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / window
    x_pad = F.pad(x[None, None, :], (pad, pad), mode="replicate")
    return F.conv1d(x_pad, kernel)[0, 0]


def smooth_1d_nan(
    x: torch.Tensor,
    window: int = 11,
    smooth_nans: bool = True,
) -> torch.Tensor:
    """
    NaN-aware moving-average smoothing for a 1D torch tensor.

    smooth_nans=True:
        NaN positions are replaced by the local smoothed value.

    smooth_nans=False:
        Original NaN positions remain NaN.
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")

    if window % 2 == 0:
        raise ValueError("window should be odd")

    pad = window // 2

    valid = torch.isfinite(x)
    x_filled = torch.where(valid, x, torch.zeros_like(x))

    x_ = x_filled[None, None, :]
    mask_ = valid.to(x.dtype)[None, None, :]

    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype)

    # Pad both values and mask
    x_padded = F.pad(x_, (pad, pad), mode="reflect")
    mask_padded = F.pad(mask_, (pad, pad), mode="reflect")

    # Sum of valid neighboring values
    numerator = F.conv1d(x_padded, kernel)

    # Number of valid neighboring values
    denominator = F.conv1d(mask_padded, kernel)

    # Avoid division by zero
    y = numerator / denominator.clamp_min(1)

    y = y[0, 0]

    # If a window had no valid values, keep NaN
    y = torch.where(denominator[0, 0] > 0, y, torch.full_like(y, torch.nan))

    if not smooth_nans:
        y = torch.where(valid, y, torch.full_like(y, torch.nan))

    return y