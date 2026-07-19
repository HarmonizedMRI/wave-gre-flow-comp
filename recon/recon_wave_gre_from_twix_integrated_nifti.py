#!/usr/bin/env python3
"""Integrated Wave-GRE reconstruction from Siemens TWIX data.

Author: Yiyun Dong
Affiliation: Athinoula A. Martinos Center for Biomedical Imaging
License: MIT License

This script reconstructs single- or multi-echo 3D Wave-GRE data acquired with
an appended FLASH calibration module. The image, calibration projections, and
ACS data are read from the same Siemens TWIX file. Acquisition-dependent
parameters are read from the Pulseq ``.seq`` definitions whenever available.

The script is designed to live beside the unchanged ``utils`` directory from
https://github.com/HarmonizedMRI/wave-mprage/tree/main/recon/utils.

Supported acquisition conventions
---------------------------------
* Transverse geometry only: readout=x, LIN/sine=y, PAR/cosine=z.
* One average only. ``Averages > 1`` is rejected.
* Wave imaging must use both sine and cosine wave gradients. Sine-only and
  cosine-only image acquisitions are rejected. Fully no-wave GRE is supported.
* Integrated calibration SET layout:
    SET 0: no-wave LIN projection
    SET 1: sine-wave LIN projection
    SET 2: no-wave PAR projection
    SET 3: cosine-wave PAR projection
    SET 4: no-wave ACS
* GRE and calibration k-space ordering defaults to negative-to-positive, which
  corresponds to ``yflip=+1`` and ``zflip=+1`` in the PSF model.

Output
------
* Coil-compressed multi-echo k-space as a complex NumPy array with shape
  ``(Nx_os, Ny, Nz, Necho, Ncc)``.
* Reconstructed complex images as one NumPy array with shape
  ``(Nx_os, Ny, Nz, Necho)``.
* Optional per-echo complex NumPy files.
* Optional cropped-readout magnitude and phase NIfTI files, one per echo, with
  JSON sidecars.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
import sigpy as sp
import sigpy.mri as mr
import torch
from scipy.ndimage import zoom

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover - depends on local CUDA setup
    cp = None
    _CUPY_IMPORT_ERROR = exc
else:
    _CUPY_IMPORT_ERROR = None

from utils.coil_compression_kspace import (
    apply_cc_coilfirst_np,
    apply_cc_coillast_torch,
    estimate_cc_matrix_coillast,
)
from utils.plot_coil_sens import plot_csm_magnitude_grid, plot_csm_phase_grid
from utils.psf_wrapped_phase_fit import fit_wrapped_phase_planes, smooth_1d_nan
from utils.twix_import import load_img, load_ref
from utils.wave_cg_sense_precondition import (
    cg_sense_wave,
    fft3call,
    fftc_dim,
    ifft3call,
)


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)


# -----------------------------------------------------------------------------
# CLI and runtime configuration
# -----------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct integrated single- or multi-echo Wave-GRE + FLASH "
            "calibration Siemens TWIX data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--twix", required=True, help="Integrated Siemens TWIX .dat file.")
    parser.add_argument("--seq", required=True, help="Matching Pulseq .seq file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--file-tag",
        default="",
        help="Optional tag appended to cached calibration and reconstruction files.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "wave", "nowave"),
        default="auto",
        help=(
            "Image reconstruction mode. 'auto' determines the mode from the "
            "image trajectory and rejects one-axis wave acquisitions."
        ),
    )
    parser.add_argument(
        "--ncc",
        type=int,
        default=12,
        help="Number of virtual coils retained after coil compression.",
    )
    parser.add_argument(
        "--reuse-coil-calib",
        action="store_true",
        help="Reuse cached coil-compression matrix and CSM files when present.",
    )
    parser.add_argument(
        "--espirit-device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Device used for ESPIRiT calibration.",
    )
    parser.add_argument(
        "--espirit-gpu-index",
        type=int,
        default=0,
        help="CUDA device index used when ESPIRiT runs on GPU.",
    )
    parser.add_argument("--cg-iters", type=int, default=50, help="Maximum CG iterations.")
    parser.add_argument("--cg-tol", type=float, default=1e-6, help="Relative CG tolerance.")
    parser.add_argument(
        "--yflip",
        type=int,
        choices=(-1, 1),
        default=None,
        help="Override the sequence-derived LIN PSF sign.",
    )
    parser.add_argument(
        "--zflip",
        type=int,
        choices=(-1, 1),
        default=None,
        help="Override the sequence-derived PAR PSF sign.",
    )
    parser.add_argument(
        "--save-echo-npy",
        action="store_true",
        help="Also save one complex NumPy reconstruction file per echo.",
    )
    parser.add_argument(
        "--save-nifti",
        action="store_true",
        help="Save one magnitude NIfTI and JSON sidecar per echo.",
    )
    parser.add_argument(
        "--save-nifti-phase",
        action="store_true",
        help="Also save one phase NIfTI per echo; implies --save-nifti.",
    )
    parser.add_argument(
        "--nifti-out",
        default=None,
        help="NIfTI output directory; defaults to <out>/nifti.",
    )
    parser.add_argument(
        "--nifti-sub",
        default=None,
        help="Filename subject token. Defaults to the TWIX filename stem.",
    )
    parser.add_argument(
        "--nifti-suffix",
        default="GRE",
        help="Final NIfTI filename suffix.",
    )
    parser.add_argument(
        "--nifti-axis-roles",
        nargs=3,
        default=("readout", "phase", "slice"),
        metavar=("AXIS0", "AXIS1", "AXIS2"),
        help="Physical roles of reconstructed array axes for Twix affine generation.",
    )
    parser.add_argument(
        "--nifti-axis-flips",
        nargs=3,
        type=_parse_bool,
        default=(False, False, False),
        metavar=("FLIP0", "FLIP1", "FLIP2"),
        help=(
            "Physical array flips applied before NIfTI saving. The GRE default is no "
            "additional flip because image LIN/PAR ordering is negative-to-positive."
        ),
    )
    parser.add_argument(
        "--twix-coord-system",
        choices=("LPS", "RAS"),
        default="LPS",
        help="Coordinate convention assumed for Siemens Sag/Cor/Tra vectors.",
    )
    parser.add_argument(
        "--twix-inplane-rot-sign",
        type=float,
        default=-1.0,
        help="Sign applied to the Twix in-plane rotation angle.",
    )
    parser.add_argument(
        "--twix-use-fov-for-voxel-size",
        action="store_true",
        help="Infer NIfTI voxel sizes from Twix FOV rather than sequence resolution.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the sequence and print derived acquisition parameters without reading TWIX data.",
    )
    return parser


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value_norm = str(value).strip().lower()
    if value_norm in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _collect_runtime_config(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_arg_parser().parse_args(argv)

    twix_file = Path(args.twix).expanduser().resolve()
    seq_file = Path(args.seq).expanduser().resolve()
    out_folder = Path(args.out).expanduser().resolve()

    if not seq_file.is_file():
        raise FileNotFoundError(f"Pulseq sequence file not found: {seq_file}")
    if not args.validate_only and not twix_file.is_file():
        raise FileNotFoundError(f"TWIX file not found: {twix_file}")
    if args.ncc <= 0:
        raise ValueError("--ncc must be positive.")
    if args.cg_iters <= 0:
        raise ValueError("--cg-iters must be positive.")
    if args.cg_tol <= 0:
        raise ValueError("--cg-tol must be positive.")

    out_folder.mkdir(parents=True, exist_ok=True)
    nifti_out = (
        Path(args.nifti_out).expanduser().resolve()
        if args.nifti_out
        else out_folder / "nifti"
    )

    return {
        "twix_file": twix_file,
        "seq_file": seq_file,
        "out_folder": out_folder,
        "file_tag": _sanitize_token(args.file_tag),
        "mode": args.mode,
        "ncc": int(args.ncc),
        "reuse_coil_calib": bool(args.reuse_coil_calib),
        "espirit_device": args.espirit_device,
        "espirit_gpu_index": int(args.espirit_gpu_index),
        "cg_iters": int(args.cg_iters),
        "cg_tol": float(args.cg_tol),
        "yflip_override": args.yflip,
        "zflip_override": args.zflip,
        "save_echo_npy": bool(args.save_echo_npy),
        "save_nifti": bool(args.save_nifti or args.save_nifti_phase),
        "save_nifti_phase": bool(args.save_nifti_phase),
        "nifti_out_folder": nifti_out,
        "nifti_sub": _sanitize_token(args.nifti_sub or twix_file.stem),
        "nifti_suffix": _sanitize_token(args.nifti_suffix),
        "nifti_axis_roles": tuple(args.nifti_axis_roles),
        "nifti_axis_flips": tuple(bool(v) for v in args.nifti_axis_flips),
        "twix_coord_system": args.twix_coord_system,
        "twix_inplane_rot_sign": float(args.twix_inplane_rot_sign),
        "twix_use_fov_for_voxel_size": bool(args.twix_use_fov_for_voxel_size),
        "validate_only": bool(args.validate_only),
    }


# -----------------------------------------------------------------------------
# Sequence definitions and GRE acquisition validation
# -----------------------------------------------------------------------------


def _load_sequence(seq_file: Path) -> pp.Sequence:
    seq = pp.Sequence()
    seq.read(str(seq_file), remove_duplicates=False)
    return seq


def _get_definition(
    defs: Mapping[str, Any],
    names: str | Sequence[str],
    default: Any = None,
    *,
    required: bool = False,
) -> Any:
    candidates = (names,) if isinstance(names, str) else tuple(names)
    for name in candidates:
        if name in defs:
            return defs[name]
    if required:
        joined = ", ".join(repr(name) for name in candidates)
        raise KeyError(f"Required Pulseq definition missing. Expected one of: {joined}")
    return default


def _as_int(value: Any, name: str) -> int:
    try:
        out = int(round(float(value)))
    except Exception as exc:
        raise ValueError(f"Pulseq definition {name!r} must be integer-like, got {value!r}.") from exc
    return out


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(
        f"Pulseq definition {name!r} must be boolean-like, got {value!r}."
    )


def _as_float_array(value: Any, name: str, min_length: int = 1) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError(f"Pulseq definition {name!r} must be numeric, got {value!r}.") from exc
    if arr.size < min_length:
        raise ValueError(
            f"Pulseq definition {name!r} must contain at least {min_length} values, got {arr}."
        )
    return arr


def _derive_gre_config(
    seq: pp.Sequence,
    yflip_override: int | None = None,
    zflip_override: int | None = None,
) -> dict[str, Any]:
    defs = seq.definitions

    orientation = str(
        _get_definition(defs, ("OrientationMapping", "SliceOrientation"), "TRA")
    ).upper()
    if orientation != "TRA":
        raise ValueError(
            "This GRE reconstruction currently supports only transverse geometry "
            f"(OrientationMapping='TRA'); sequence reports {orientation!r}."
        )

    nx = _as_int(_get_definition(defs, "Nx", required=True), "Nx")
    ny = _as_int(_get_definition(defs, "Ny", required=True), "Ny")
    nz = _as_int(_get_definition(defs, "Nz", required=True), "Nz")
    os_factor = _as_int(
        _get_definition(defs, ("ReadoutOversamplingFactor", "ro_os"), 4),
        "ReadoutOversamplingFactor",
    )
    nx_os_def = _get_definition(defs, "Nx_os", None)
    nx_os = _as_int(nx_os_def, "Nx_os") if nx_os_def is not None else nx * os_factor
    if nx_os != nx * os_factor:
        raise ValueError(
            f"Inconsistent readout definitions: Nx_os={nx_os}, but Nx*OS={nx * os_factor}."
        )

    fov = _as_float_array(_get_definition(defs, ("FOV", "TargetFOV"), required=True), "FOV", 3)
    fov_xyz = tuple(float(v) for v in fov[:3])
    res_xyz_m = (fov_xyz[0] / nx, fov_xyz[1] / ny, fov_xyz[2] / nz)

    necho = _as_int(_get_definition(defs, ("Nechoes", "NEchoes"), 1), "Nechoes")
    if necho <= 0:
        raise ValueError(f"Nechoes must be positive, got {necho}.")

    te = _as_float_array(_get_definition(defs, "TE", [np.nan] * necho), "TE")
    if te.size == 1 and necho > 1:
        raise ValueError(f"Sequence reports Nechoes={necho}, but TE contains only one value: {te}.")
    if te.size < necho:
        raise ValueError(f"Sequence reports Nechoes={necho}, but TE contains {te.size} values.")
    te = te[:necho]

    averages = _as_int(_get_definition(defs, ("Averages", "Naverages", "naverage"), 1), "Averages")
    if averages != 1:
        raise ValueError(
            "This reconstruction currently supports exactly one average. "
            f"The sequence reports Averages={averages}."
        )

    ry = _as_int(_get_definition(defs, ("Ry", "R_y"), 1), "Ry")
    rz = _as_int(_get_definition(defs, ("Rz", "R_z"), 1), "Rz")
    ny_meas = _as_int(_get_definition(defs, "Ny_meas", int(np.ceil(ny / ry))), "Ny_meas")
    nz_meas = _as_int(_get_definition(defs, "Nz_meas", int(np.ceil(nz / rz))), "Nz_meas")

    ncalib1 = _as_int(
        _get_definition(defs, ("CalibrationNcalib1", "Calibration_Ncalib1"), 72),
        "CalibrationNcalib1",
    )
    ncalib2 = _as_int(
        _get_definition(defs, ("CalibrationNcalib2", "Calibration_Ncalib2"), 1),
        "CalibrationNcalib2",
    )
    nacs = _as_int(
        _get_definition(defs, ("CalibrationNacs", "Calibration_Nacs"), 32),
        "CalibrationNacs",
    )
    nsets = _as_int(
        _get_definition(defs, ("CalibrationNSets", "Calibration_NSets"), 5),
        "CalibrationNSets",
    )
    acs_set_id = _as_int(
        _get_definition(defs, ("CalibrationACSSetID", "Calibration_ACSSetID"), 4),
        "CalibrationACSSetID",
    )
    if ncalib2 != 1:
        raise ValueError(
            "The projection-based PSF fitter requires CalibrationNcalib2 == 1; "
            f"the sequence reports {ncalib2}."
        )
    if nsets != 5 or acs_set_id != 4:
        raise ValueError(
            "Unexpected integrated calibration layout. Expected CalibrationNSets=5 "
            f"and CalibrationACSSetID=4, got {nsets} and {acs_set_id}."
        )

    sin_channel = str(_get_definition(defs, "WaveSinChannel", "y")).lower()
    cos_channel = str(_get_definition(defs, "WaveCosChannel", "z")).lower()
    if sin_channel != "y" or cos_channel != "z":
        raise ValueError(
            "This GRE reconstruction assumes WaveSinChannel='y' and "
            f"WaveCosChannel='z'; sequence reports {sin_channel!r}/{cos_channel!r}."
        )

    ordering = str(
        _get_definition(defs, ("KspaceOrdering", "KSpaceOrdering"), "negative_to_positive")
    ).strip().lower()
    ordering_map = {
        "negative_to_positive": (1, 1),
        "negative-to-positive": (1, 1),
        "positive_to_negative": (-1, -1),
        "positive-to-negative": (-1, -1),
    }
    if ordering not in ordering_map:
        raise ValueError(
            "Unsupported KspaceOrdering. Expected 'negative_to_positive' or "
            f"'positive_to_negative', got {ordering!r}."
        )
    default_yflip, default_zflip = ordering_map[ordering]
    yflip = int(yflip_override if yflip_override is not None else default_yflip)
    zflip = int(zflip_override if zflip_override is not None else default_zflip)

    use_flow_comp = _as_bool(
        _get_definition(defs, "UseFlowComp", False), "UseFlowComp"
    )
    sequence_name = str(_get_definition(defs, "Name", "gre_3d"))

    return {
        "defs": defs,
        "sequence_name": sequence_name,
        "orientation": orientation,
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "Nx_os": nx_os,
        "os_factor": os_factor,
        "FOVxyz_m": fov_xyz,
        "res_xyz_m": res_xyz_m,
        "Necho": necho,
        "TE_s": te,
        "Averages": averages,
        "Ry": ry,
        "Rz": rz,
        "Ny_meas": ny_meas,
        "Nz_meas": nz_meas,
        "Ncalib1": ncalib1,
        "Ncalib2": ncalib2,
        "Nacs": nacs,
        "Nsets": nsets,
        "ACSSetID": acs_set_id,
        "WaveSinChannel": sin_channel,
        "WaveCosChannel": cos_channel,
        "KspaceOrdering": ordering,
        "yflip": yflip,
        "zflip": zflip,
        "UseFlowComp": use_flow_comp,
    }


def _calibration_readout_count(cfg: Mapping[str, Any]) -> int:
    return 4 * int(cfg["Ncalib1"]) * int(cfg["Ncalib2"]) + int(cfg["Nacs"]) ** 2


def _split_adc_trajectory(
    seq: pp.Sequence,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    ktraj_adc, _, _, _, _ = seq.calculate_kspace()
    ktraj_adc = np.asarray(ktraj_adc, dtype=np.float64)
    if ktraj_adc.ndim != 2 or ktraj_adc.shape[0] != 3:
        raise ValueError(f"Unexpected Pulseq ADC trajectory shape: {ktraj_adc.shape}")

    nx_os = int(cfg["Nx_os"])
    if ktraj_adc.shape[1] % nx_os != 0:
        raise ValueError(
            f"ADC sample count {ktraj_adc.shape[1]} is not divisible by Nx_os={nx_os}."
        )
    all_lines = ktraj_adc.reshape(3, -1, nx_os)
    ncalib_lines = _calibration_readout_count(cfg)
    if all_lines.shape[1] <= ncalib_lines:
        raise ValueError(
            f"Sequence contains {all_lines.shape[1]} ADC lines, not enough for "
            f"{ncalib_lines} integrated calibration lines plus image data."
        )
    image_lines = all_lines[:, :-ncalib_lines, :]
    calib_lines = all_lines[:, -ncalib_lines:, :]

    expected_image_lines = (
        int(cfg["Ny_meas"])
        * int(cfg["Nz_meas"])
        * int(cfg["Necho"])
        * int(cfg["Averages"])
    )
    if image_lines.shape[1] != expected_image_lines:
        raise ValueError(
            "Image ADC line count does not match sequence definitions: "
            f"trajectory has {image_lines.shape[1]}, expected "
            f"Ny_meas*Nz_meas*Necho*Averages={expected_image_lines}."
        )
    return image_lines, calib_lines


def _line_wave_excursion(line: np.ndarray) -> float:
    line = np.asarray(line, dtype=np.float64)
    centered = line - np.mean(line)
    return float(np.max(np.abs(centered)))


def _find_center_line(lines: np.ndarray, axis: int) -> np.ndarray:
    axis_lines = np.asarray(lines[axis], dtype=np.float64)
    means = np.mean(axis_lines, axis=-1)
    return axis_lines[int(np.argmin(np.abs(means)))]


def _detect_image_wave_mode(
    image_lines: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    relative_threshold: float = 1e-4,
) -> str:
    """Return ``wave`` or ``nowave`` and reject sine-only/cosine-only image data."""
    necho = int(cfg["Necho"])
    y_excursions: list[float] = []
    z_excursions: list[float] = []
    for echo_idx in range(necho):
        echo_lines = image_lines[:, echo_idx::necho, :]
        y_excursions.append(_line_wave_excursion(_find_center_line(echo_lines, axis=1)))
        z_excursions.append(_line_wave_excursion(_find_center_line(echo_lines, axis=2)))

    x_scale = max(
        _line_wave_excursion(_find_center_line(image_lines, axis=0)),
        np.finfo(float).eps,
    )
    y_active = max(y_excursions, default=0.0) > relative_threshold * x_scale
    z_active = max(z_excursions, default=0.0) > relative_threshold * x_scale

    print(
        "Image wave detection: "
        f"max ky excursion={max(y_excursions, default=0.0):.6g}, "
        f"max kz excursion={max(z_excursions, default=0.0):.6g}, "
        f"readout scale={x_scale:.6g}"
    )

    if y_active != z_active:
        active = "sine/y only" if y_active else "cosine/z only"
        raise ValueError(
            "One-axis wave imaging is not supported by this public reconstruction. "
            f"Trajectory inspection detected {active}. Use both sine and cosine waves, "
            "or disable both."
        )
    return "wave" if y_active and z_active else "nowave"


def _resolve_reconstruction_mode(requested: str, detected: str) -> str:
    if requested == "auto":
        return detected
    if requested != detected:
        raise ValueError(
            f"Requested --mode={requested!r}, but trajectory inspection detected {detected!r}."
        )
    return requested


def _print_sequence_summary(cfg: Mapping[str, Any], detected_mode: str | None = None) -> None:
    res_mm = tuple(v * 1e3 for v in cfg["res_xyz_m"])
    te_ms = [float(v) * 1e3 for v in cfg["TE_s"]]
    print("Integrated Wave-GRE reconstruction")
    print(f"  Sequence name: {cfg['sequence_name']}")
    print(f"  Orientation: {cfg['orientation']} (RO=x, LIN=y, PAR=z)")
    print(
        f"  Matrix: Nx={cfg['Nx']}, Ny={cfg['Ny']}, Nz={cfg['Nz']}, "
        f"Nx_os={cfg['Nx_os']}"
    )
    print(
        "  Resolution [mm]: "
        f"{res_mm[0]:g} x {res_mm[1]:g} x {res_mm[2]:g}"
    )
    print(f"  Echoes: {cfg['Necho']}, TE [ms]: {te_ms}")
    print(f"  Acceleration: Ry={cfg['Ry']}, Rz={cfg['Rz']}")
    print(
        f"  Calibration: Ncalib1={cfg['Ncalib1']}, "
        f"Ncalib2={cfg['Ncalib2']}, Nacs={cfg['Nacs']}"
    )
    print(
        f"  K-space ordering: {cfg['KspaceOrdering']} -> "
        f"yflip={cfg['yflip']}, zflip={cfg['zflip']}"
    )
    print(f"  Flow compensation: {cfg['UseFlowComp']}")
    if detected_mode is not None:
        print(f"  Detected image mode: {detected_mode}")


# -----------------------------------------------------------------------------
# TWIX image and refscan normalization
# -----------------------------------------------------------------------------


def _to_complex64_tensor(data: Any) -> torch.Tensor:
    tensor = data if torch.is_tensor(data) else torch.as_tensor(data)
    return tensor.to(dtype=torch.complex64).contiguous()


def _normalize_gre_image_data(img: Any, cfg: Mapping[str, Any]) -> torch.Tensor:
    """Normalize TWIX image data to (Nx_os, Ny_acq, Nz_acq, Necho, Ncoil)."""
    data = _to_complex64_tensor(img)
    necho = int(cfg["Necho"])
    nx_os = int(cfg["Nx_os"])

    if data.ndim == 4:
        if necho != 1:
            raise ValueError(
                f"load_img returned 4D data {tuple(data.shape)}, but the sequence reports "
                f"Nechoes={necho}. Expected a retained echo dimension."
            )
        data = data.unsqueeze(3)
    elif data.ndim == 5:
        if data.shape[3] != necho:
            raise ValueError(
                f"Echo dimension mismatch: image data shape is {tuple(data.shape)}, "
                f"but sequence Nechoes={necho}."
            )
    else:
        raise ValueError(
            "Expected load_img to return 4D single-echo or 5D multi-echo data in "
            f"(Nx_os, Ny_acq, Nz_acq[, Necho], Ncoil) order, got {tuple(data.shape)}."
        )

    if data.shape[0] != nx_os:
        raise ValueError(
            f"Readout mismatch: TWIX image Nx_os={data.shape[0]}, sequence Nx_os={nx_os}."
        )
    if data.shape[1] > int(cfg["Ny"]) or data.shape[2] > int(cfg["Nz"]):
        raise ValueError(
            f"Acquired PE shape {tuple(data.shape[1:3])} exceeds full sequence matrix "
            f"({cfg['Ny']}, {cfg['Nz']})."
        )
    if data.shape[4] <= 0:
        raise ValueError("TWIX image contains no coil channels.")
    return data


def _embed_full_kspace(img: torch.Tensor, cfg: Mapping[str, Any]) -> torch.Tensor:
    nx_os = int(cfg["Nx_os"])
    ny = int(cfg["Ny"])
    nz = int(cfg["Nz"])
    necho = int(cfg["Necho"])
    ncoil = int(img.shape[-1])
    full = torch.zeros((nx_os, ny, nz, necho, ncoil), dtype=torch.complex64)
    full[:, : img.shape[1], : img.shape[2], :, :] = img
    return full


def _check_integrated_refscan_shape(
    data_ref: Any,
    *,
    ncalib1: int,
    nacs: int,
    nsets: int = 5,
) -> torch.Tensor:
    ref = _to_complex64_tensor(data_ref)
    if ref.ndim != 5:
        raise ValueError(
            "Expected integrated refscan shape (Nx_os, LIN, PAR, SET, Ncoil), "
            f"got {tuple(ref.shape)}."
        )
    if ref.shape[1] < max(ncalib1, nacs) or ref.shape[2] < max(ncalib1, nacs):
        raise ValueError(
            f"Integrated refscan PE extent {tuple(ref.shape[1:3])} is smaller than "
            f"required calibration/ACS sizes {ncalib1}/{nacs}."
        )
    if ref.shape[3] < nsets:
        raise ValueError(
            f"Integrated refscan contains {ref.shape[3]} SETs; at least {nsets} are required."
        )
    return ref


# -----------------------------------------------------------------------------
# Coil compression and ESPIRiT maps
# -----------------------------------------------------------------------------


def _cache_suffix(file_tag: str) -> str:
    return f"_{file_tag}" if file_tag else ""


def _coil_cache_paths(out_folder: Path, file_tag: str, ncc: int) -> dict[str, Path]:
    suffix = _cache_suffix(file_tag)
    return {
        "wcc": out_folder / f"coil_compression_matrix_ncc{ncc}{suffix}.npy",
        "csm_low": out_folder / f"csm_acs_ncc{ncc}{suffix}.npy",
        "csm_full": out_folder / f"csm_full_ncc{ncc}{suffix}.npy",
        "csm_mag": out_folder / f"csm_full_mag_ncc{ncc}{suffix}.png",
        "csm_phase": out_folder / f"csm_full_phase_ncc{ncc}{suffix}.png",
    }


def _select_espirit_device(mode: str, gpu_index: int) -> tuple[sp.Device, bool]:
    if mode == "cpu":
        print("ESPIRiT device: CPU")
        return sp.Device(-1), False

    gpu_available = False
    gpu_count = 0
    gpu_error: Exception | None = None
    if cp is not None:
        try:
            gpu_count = int(cp.cuda.runtime.getDeviceCount())
            gpu_available = gpu_count > 0
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime
            gpu_error = exc

    if mode == "gpu" and not gpu_available:
        details = gpu_error or _CUPY_IMPORT_ERROR or "no CUDA device detected"
        raise RuntimeError(f"GPU ESPIRiT requested but unavailable: {details}")

    if gpu_available and mode in {"auto", "gpu"}:
        if gpu_index < 0 or gpu_index >= gpu_count:
            raise ValueError(f"GPU index {gpu_index} is outside available range 0..{gpu_count - 1}.")
        props = cp.cuda.runtime.getDeviceProperties(gpu_index)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(f"ESPIRiT device: GPU {gpu_index} ({name})")
        return sp.Device(gpu_index), True

    if mode == "auto":
        details = gpu_error or _CUPY_IMPORT_ERROR
        if details is not None:
            print(f"GPU unavailable; using CPU ESPIRiT ({details}).")
        else:
            print("No CUDA GPU detected; using CPU ESPIRiT.")
    return sp.Device(-1), False


def load_or_generate_coil_sens(
    *,
    twix_file: Path,
    cfg: Mapping[str, Any],
    out_folder: Path,
    file_tag: str,
    ncc: int,
    reuse_coil_calib: bool,
    espirit_device: str,
    espirit_gpu_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    paths = _coil_cache_paths(out_folder, file_tag, ncc)
    if reuse_coil_calib and paths["wcc"].is_file() and paths["csm_full"].is_file():
        print("Loading cached coil-compression matrix and sensitivity maps...")
        wcc = np.load(paths["wcc"])
        csm_full = np.load(paths["csm_full"])
        ref = _check_integrated_refscan_shape(
            load_ref(str(twix_file)),
            ncalib1=int(cfg["Ncalib1"]),
            nacs=int(cfg["Nacs"]),
            nsets=int(cfg["Nsets"]),
        )
        ncoil = int(ref.shape[-1])
        _check_cc_and_csm(wcc, csm_full, ncoil=ncoil, ncc=ncc, cfg=cfg)
        return wcc, csm_full, ncoil

    if reuse_coil_calib:
        print("Cached coil calibration was incomplete; recomputing from integrated ACS.")
    return generate_coil_sens(
        twix_file=twix_file,
        cfg=cfg,
        out_folder=out_folder,
        file_tag=file_tag,
        ncc=ncc,
        espirit_device=espirit_device,
        espirit_gpu_index=espirit_gpu_index,
    )


def _check_cc_and_csm(
    wcc: np.ndarray,
    csm_full: np.ndarray,
    *,
    ncoil: int,
    ncc: int,
    cfg: Mapping[str, Any],
) -> None:
    if wcc.ndim != 2 or wcc.shape != (ncoil, ncc):
        raise ValueError(
            f"Cached coil-compression matrix has shape {wcc.shape}; expected {(ncoil, ncc)}."
        )
    expected_csm = (ncc, int(cfg["Nx"]), int(cfg["Ny"]), int(cfg["Nz"]))
    if tuple(csm_full.shape) != expected_csm:
        raise ValueError(
            f"Cached sensitivity map shape is {csm_full.shape}; expected {expected_csm}."
        )


def generate_coil_sens(
    *,
    twix_file: Path,
    cfg: Mapping[str, Any],
    out_folder: Path,
    file_tag: str,
    ncc: int,
    espirit_device: str,
    espirit_gpu_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    device, using_gpu = _select_espirit_device(espirit_device, espirit_gpu_index)
    ref = _check_integrated_refscan_shape(
        load_ref(str(twix_file)),
        ncalib1=int(cfg["Ncalib1"]),
        nacs=int(cfg["Nacs"]),
        nsets=int(cfg["Nsets"]),
    )

    nacs = int(cfg["Nacs"])
    acs_set_id = int(cfg["ACSSetID"])
    kspace_acs = ref[:, :nacs, :nacs, acs_set_id, :]
    nx_os, ny_acs, nz_acs, ncoil = map(int, kspace_acs.shape)
    if ncc > ncoil:
        raise ValueError(f"Requested ncc={ncc}, but the TWIX refscan has only {ncoil} coils.")
    if nx_os != int(cfg["Nx_os"]):
        raise ValueError(
            f"Refscan readout length {nx_os} does not match sequence Nx_os={cfg['Nx_os']}."
        )
    print(f"Integrated ACS shape: {tuple(kspace_acs.shape)}")

    wcc, _, cc_energy = estimate_cc_matrix_coillast(
        kspace_acs,
        ncc=ncc,
        acs=min(ny_acs, nz_acs),
        x_step=int(cfg["os_factor"]),
    )
    print(f"Coil-compression matrix: {wcc.shape}")
    print(f"Energy retained by {ncc} coils: {float(cc_energy[ncc - 1]):.6f}")

    kspace_np = (
        kspace_acs.permute(3, 0, 1, 2)[:, :: int(cfg["os_factor"])]
        .contiguous()
        .numpy()
        .astype(np.complex64, copy=False)
    )
    nx = int(cfg["Nx"])
    low_y = min(32, ny_acs, int(cfg["Ny"]))
    low_z = min(32, nz_acs, int(cfg["Nz"]))
    low_shape = (ncoil, nx, low_y, low_z)
    kspace_low_np = sp.resize(kspace_np, low_shape).astype(np.complex64, copy=False)
    kspace_low_cc_np = apply_cc_coilfirst_np(kspace_low_np, wcc)
    print(f"Low-resolution compressed ACS: {kspace_low_cc_np.shape}")

    if cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    gc.collect()

    kspace_low_cc_sp = sp.to_device(kspace_low_cc_np, device)
    csm_low_cc = mr.app.EspiritCalib(
        kspace_low_cc_sp,
        calib_width=min(24, low_y, low_z),
        device=device,
        crop=0.8,
        show_pbar=True,
    ).run()
    csm_low_cc_np = np.asarray(sp.to_device(csm_low_cc, sp.Device(-1))).astype(
        np.complex64, copy=False
    )
    print(f"Low-resolution CSM: {csm_low_cc_np.shape}")

    zoom_factors = (
        1,
        int(cfg["Nx"]) / csm_low_cc_np.shape[1],
        int(cfg["Ny"]) / csm_low_cc_np.shape[2],
        int(cfg["Nz"]) / csm_low_cc_np.shape[3],
    )
    csm_full = (
        zoom(csm_low_cc_np.real, zoom_factors, order=1)
        + 1j * zoom(csm_low_cc_np.imag, zoom_factors, order=1)
    ).astype(np.complex64)
    rss = np.sqrt(np.sum(np.abs(csm_full) ** 2, axis=0, keepdims=True))
    csm_full /= np.maximum(rss, 1e-8)

    paths = _coil_cache_paths(out_folder, file_tag, ncc)
    np.save(paths["wcc"], np.asarray(wcc))
    np.save(paths["csm_low"], csm_low_cc_np)
    np.save(paths["csm_full"], csm_full)

    plot_csm_magnitude_grid(csm_full, z=csm_full.shape[-1] // 2)
    plt.savefig(paths["csm_mag"], dpi=150, bbox_inches="tight")
    plt.close("all")
    plot_csm_phase_grid(csm_full, z=csm_full.shape[-1] // 2)
    plt.savefig(paths["csm_phase"], dpi=150, bbox_inches="tight")
    plt.close("all")

    if using_gpu and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    gc.collect()
    return np.asarray(wcc), csm_full, ncoil


# -----------------------------------------------------------------------------
# GRE theoretical trajectory and calibrated PSF
# -----------------------------------------------------------------------------


def _echo_theoretical_wave_trajectories(
    image_lines: np.ndarray,
    cfg: Mapping[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return one (delta_ky_idx, delta_kz_idx) pair per echo."""
    necho = int(cfg["Necho"])
    fov_y = float(cfg["FOVxyz_m"][1])
    fov_z = float(cfg["FOVxyz_m"][2])
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for echo_idx in range(necho):
        echo_lines = image_lines[:, echo_idx::necho, :]
        delta_ky = _find_center_line(echo_lines, axis=1)
        delta_kz = _find_center_line(echo_lines, axis=2)
        result.append((delta_ky * fov_y, delta_kz * fov_z))
    return result


def fit_wave_psf_deviation_from_projection(
    *,
    twix_file: Path,
    calib_lines: np.ndarray,
    cfg: Mapping[str, Any],
    out_folder: Path,
    file_tag: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = _check_integrated_refscan_shape(
        load_ref(str(twix_file)),
        ncalib1=int(cfg["Ncalib1"]),
        nacs=int(cfg["Nacs"]),
        nsets=int(cfg["Nsets"]),
    )
    nx_os = int(cfg["Nx_os"])
    ncalib1 = int(cfg["Ncalib1"])
    ncalib2 = int(cfg["Ncalib2"])
    nproj = ncalib1 * ncalib2
    if calib_lines.shape != (3, _calibration_readout_count(cfg), nx_os):
        raise ValueError(
            f"Unexpected calibration trajectory shape {calib_lines.shape}; expected "
            f"{(3, _calibration_readout_count(cfg), nx_os)}."
        )

    fov_y = float(cfg["FOVxyz_m"][1])
    fov_z = float(cfg["FOVxyz_m"][2])
    yflip = int(cfg["yflip"])
    zflip = int(cfg["zflip"])

    a_fit_all: list[np.ndarray] = []
    b_fit_all: list[np.ndarray] = []
    c_fit_all: list[np.ndarray] = []

    for wave_mode in ("sin", "cos"):
        print(f"Calibrating {wave_mode} projection")
        if wave_mode == "sin":
            kspace_nowave = ref[:, :ncalib1, :1, 0, :]
            kspace_wave = ref[:, :ncalib1, :1, 1, :]
            set_lines = calib_lines[:, nproj : 2 * nproj, :]
            delta_ky = set_lines[1, ncalib1 // 2]
            delta_ky_idx = delta_ky * fov_y
            y_norm_lr = (np.arange(ncalib1) - ncalib1 / 2.0) / ncalib1
            z_norm_lr = np.array([0.0], dtype=float)
            psf_np = np.exp(
                -1j * yflip * 2.0 * np.pi * delta_ky_idx[:, None] * y_norm_lr[None, :]
            ).astype(np.complex64)[..., None]
            tag = "projy"
        else:
            kspace_nowave = ref[:, :1, :ncalib1, 2, :]
            kspace_wave = ref[:, :1, :ncalib1, 3, :]
            set_lines = calib_lines[:, 3 * nproj : 4 * nproj, :]
            delta_kz = set_lines[2, ncalib1 // 2]
            delta_kz_idx = delta_kz * fov_z
            y_norm_lr = np.array([0.0], dtype=float)
            z_norm_lr = (np.arange(ncalib1) - ncalib1 / 2.0) / ncalib1
            psf_np = np.exp(
                -1j
                * zflip
                * 2.0
                * np.pi
                * delta_kz_idx[:, None, None]
                * z_norm_lr[None, None, :]
            ).astype(np.complex64)
            tag = "projz"

        psf_theory = torch.from_numpy(psf_np)
        img_nowave = ifft3call(kspace_nowave)
        img_wave = ifft3call(kspace_wave)
        hyb_nowave = fftc_dim(img_nowave, dim=0)
        hyb_wave = fftc_dim(img_wave, dim=0)
        cross = hyb_wave * torch.conj(hyb_nowave) / (
            1e-8 + hyb_nowave * torch.conj(hyb_nowave)
        )
        psf_real = torch.exp(1j * torch.angle(cross.mean(dim=-1)))
        psf_diff_lr = torch.angle(torch.conj(psf_theory) * psf_real)

        result = fit_wrapped_phase_planes(
            psf_diff=psf_diff_lr,
            hyb_nowave=hyb_nowave.clone(),
            y_norm=y_norm_lr,
            z_norm=z_norm_lr,
            mask_mode="combined",
            mag_abs_floor=0.0,
            local_window_size=5,
            coherence_threshold=0.75,
            use_phase_coherence_weight=True,
            phase_weight_power=2.0,
            use_residual_coherence_refinement=True,
            residual_window_size=5,
            residual_coherence_threshold=0.75,
            use_residual_coherence_weight=True,
            residual_weight_power=2.0,
            n_irls=10,
            huber_delta=0.7,
            return_quality_maps=True,
            verbose=False,
        )
        a_fit_all.append(np.asarray(result["a_fit_all"]))
        b_fit_all.append(np.asarray(result["b_fit_all"]))
        c_fit_all.append(np.asarray(result["c_fit_all"]))

        suffix = _cache_suffix(file_tag)
        np.save(out_folder / f"a_fit_all_{tag}{suffix}.npy", result["a_fit_all"])
        np.save(out_folder / f"b_fit_all_{tag}{suffix}.npy", result["b_fit_all"])
        np.save(out_folder / f"c_fit_all_{tag}{suffix}.npy", result["c_fit_all"])

    a_fit = a_fit_all[0]
    b_fit = b_fit_all[1]
    c_fit = c_fit_all[0] + c_fit_all[1]
    return a_fit, b_fit, c_fit


def _build_phase_correction(
    a_fit: np.ndarray,
    b_fit: np.ndarray,
    c_fit: np.ndarray,
    *,
    ny: int,
    nz: int,
) -> torch.Tensor:
    y_norm = (np.arange(ny) - ny / 2.0) / ny
    z_norm = (np.arange(nz) - nz / 2.0) / nz
    y_grid, z_grid = torch.meshgrid(
        torch.from_numpy(y_norm),
        torch.from_numpy(z_norm),
        indexing="ij",
    )
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()
    design = torch.stack((y_flat, z_flat, torch.ones_like(y_flat)), dim=1)

    nx_os = len(a_fit)
    correction = torch.empty((nx_os, ny, nz), dtype=design.dtype)
    for kx_idx in range(nx_os):
        coeff = torch.tensor(
            (a_fit[kx_idx], b_fit[kx_idx], c_fit[kx_idx]), dtype=design.dtype
        )
        correction[kx_idx] = (design @ coeff).view(ny, nz)
    return torch.nan_to_num(correction, nan=0.0).to(torch.float32)


def generate_calibrated_psfs(
    *,
    twix_file: Path,
    image_lines: np.ndarray,
    calib_lines: np.ndarray,
    cfg: Mapping[str, Any],
    out_folder: Path,
    file_tag: str,
    psf_plot: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_raw, b_raw, c_raw = fit_wave_psf_deviation_from_projection(
        twix_file=twix_file,
        calib_lines=calib_lines,
        cfg=cfg,
        out_folder=out_folder,
        file_tag=file_tag,
    )
    a_fit = smooth_1d_nan(a_raw, window=9)
    b_fit = smooth_1d_nan(b_raw, window=9)
    c_fit = smooth_1d_nan(c_raw, window=9)

    if psf_plot:
        plt.figure(figsize=(7, 4))
        plt.plot(a_fit, label="a(t)")
        plt.plot(b_fit, label="b(t)")
        plt.plot(c_fit, label="c(t)")
        plt.axvline(len(a_fit) // 2, linestyle="--", color="k")
        plt.axhline(0, linestyle="--", color="k")
        plt.legend()
        plt.ylim([-3, 3])
        plt.xlim([0, len(a_fit)])
        plt.tight_layout()
        plt.savefig(
            out_folder / f"psf_integrated_calib_fit{_cache_suffix(file_tag)}.png",
            dpi=150,
        )
        plt.close("all")

    phase_correction = _build_phase_correction(
        a_fit,
        b_fit,
        c_fit,
        ny=int(cfg["Ny"]),
        nz=int(cfg["Nz"]),
    )
    y_norm = (np.arange(int(cfg["Ny"])) - int(cfg["Ny"]) / 2.0) / int(cfg["Ny"])
    z_norm = (np.arange(int(cfg["Nz"])) - int(cfg["Nz"]) / 2.0) / int(cfg["Nz"])

    psf_theory_echoes: list[torch.Tensor] = []
    psf_calib_echoes: list[torch.Tensor] = []
    trajectories = _echo_theoretical_wave_trajectories(image_lines, cfg)
    for echo_idx, (delta_ky_idx, delta_kz_idx) in enumerate(trajectories):
        psf_np = np.exp(
            -1j
            * int(cfg["yflip"])
            * 2.0
            * np.pi
            * delta_ky_idx[:, None]
            * y_norm[None, :]
        ).astype(np.complex64)
        psf_np = psf_np[..., None] * np.exp(
            -1j
            * int(cfg["zflip"])
            * 2.0
            * np.pi
            * delta_kz_idx[:, None, None]
            * z_norm[None, None, :]
        ).astype(np.complex64)
        psf_theory = torch.from_numpy(psf_np)
        psf_calib = psf_theory * torch.exp(1j * phase_correction)
        psf_theory_echoes.append(psf_theory)
        psf_calib_echoes.append(psf_calib)
        print(f"Generated theoretical and calibrated PSF for echo {echo_idx + 1}.")

    return torch.stack(psf_calib_echoes, dim=0), torch.stack(psf_theory_echoes, dim=0)


# -----------------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------------


def _build_sensitivity_tensor(csm_full: np.ndarray, cfg: Mapping[str, Any]) -> torch.Tensor:
    expected = (csm_full.shape[0], int(cfg["Nx"]), int(cfg["Ny"]), int(cfg["Nz"]))
    if tuple(csm_full.shape) != expected:
        raise ValueError(f"Sensitivity map shape {csm_full.shape} does not match {expected}.")
    sens = torch.zeros(
        (csm_full.shape[0], int(cfg["Nx_os"]), int(cfg["Ny"]), int(cfg["Nz"])),
        dtype=torch.complex64,
    )
    x0 = int(cfg["Nx_os"]) // 2 - int(cfg["Nx"]) // 2
    x1 = x0 + int(cfg["Nx"])
    sens[:, x0:x1] = torch.from_numpy(csm_full).to(torch.complex64).contiguous()
    return sens


def _sampling_masks(kspace_cc: torch.Tensor) -> list[torch.Tensor]:
    """Return one broadcastable mask per echo."""
    if kspace_cc.ndim != 5:
        raise ValueError(f"Expected 5D compressed k-space, got {tuple(kspace_cc.shape)}.")
    masks: list[torch.Tensor] = []
    for echo_idx in range(kspace_cc.shape[3]):
        mask_2d = torch.sum(torch.abs(kspace_cc[:, :, :, echo_idx, :]) ** 2, dim=(0, 3)) > 0
        masks.append(mask_2d.to(torch.float32).view(1, 1, *mask_2d.shape))
    if len(masks) > 1:
        identical = all(torch.equal(masks[0], mask) for mask in masks[1:])
        print(f"Echo sampling masks identical: {identical}")
    return masks


def _cg_sense_cartesian(
    y: torch.Tensor,
    sens: torch.Tensor,
    mask_t: torch.Tensor,
    *,
    n_iter: int,
    tol: float,
) -> torch.Tensor:
    """Solve no-wave Cartesian SENSE with conjugate gradients."""

    def forward(x: torch.Tensor) -> torch.Tensor:
        return fft3call(sens * x.unsqueeze(0), dim=(1, 2, 3)) * mask_t

    def adjoint(kspace: torch.Tensor) -> torch.Tensor:
        img_coils = ifft3call(kspace * mask_t, dim=(1, 2, 3))
        return (torch.conj(sens) * img_coils).sum(dim=0)

    x = torch.zeros(sens.shape[1:], dtype=torch.complex64)
    b = adjoint(y)
    r = b.clone()
    p = r.clone()
    rr = torch.vdot(r.reshape(-1), r.reshape(-1)).real
    bb = torch.vdot(b.reshape(-1), b.reshape(-1)).real
    if bb <= 0:
        raise ValueError("No-wave CG right-hand side has zero norm.")

    for iteration in range(n_iter):
        ap = adjoint(forward(p))
        p_ap = torch.vdot(p.reshape(-1), ap.reshape(-1)).real
        if p_ap <= 0:
            raise RuntimeError(f"No-wave CG encountered non-positive p^HAp at iteration {iteration}.")
        alpha = rr / p_ap
        x = x + alpha * p
        r = r - alpha * ap
        rr_new = torch.vdot(r.reshape(-1), r.reshape(-1)).real
        rel = torch.sqrt(rr_new / bb)
        print(f"  CG {iteration + 1}/{n_iter}: relative residual={rel.item():.3e}")
        if rel < tol:
            print(f"  CG converged at iteration {iteration + 1}.")
            return x
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    print(f"  CG reached max iterations; relative residual={torch.sqrt(rr / bb).item():.3e}")
    return x


def reconstruct_echoes(
    *,
    kspace_cc: torch.Tensor,
    sens: torch.Tensor,
    masks: Sequence[torch.Tensor],
    mode: str,
    psf_calib_echoes: torch.Tensor | None,
    cg_iters: int,
    cg_tol: float,
) -> torch.Tensor:
    necho = int(kspace_cc.shape[3])
    images: list[torch.Tensor] = []
    for echo_idx in range(necho):
        print(f"Reconstructing echo {echo_idx + 1}/{necho} ({mode})...")
        y_meas = kspace_cc[:, :, :, echo_idx, :].permute(3, 0, 1, 2).contiguous()
        if mode == "wave":
            if psf_calib_echoes is None:
                raise ValueError("Wave reconstruction requires calibrated PSFs.")
            image = cg_sense_wave(
                y=y_meas,
                sens=sens,
                psf_to_use=psf_calib_echoes[echo_idx].clone(),
                mask_t=masks[echo_idx],
                n_iter=cg_iters,
                tol=cg_tol,
                init="zero",
                use_preconditioner=True,
                use_direct_if_full=True,
            )
        else:
            image = _cg_sense_cartesian(
                y_meas,
                sens,
                masks[echo_idx],
                n_iter=cg_iters,
                tol=cg_tol,
            )
        images.append(image.detach().cpu().to(torch.complex64))
    return torch.stack(images, dim=3)


# -----------------------------------------------------------------------------
# Output naming, NumPy, and NIfTI export
# -----------------------------------------------------------------------------


def _sanitize_token(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(ch if ch in allowed else "-" for ch in str(value).strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-_.")


def _format_num(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _recon_stem(cfg: Mapping[str, Any], mode: str, file_tag: str) -> str:
    res_mm = [float(v) * 1e3 for v in cfg["res_xyz_m"]]
    fc = "FCon" if cfg["UseFlowComp"] else "FCoff"
    parts = [
        f"gre_{mode}",
        f"ME{cfg['Necho']}",
        fc,
        "res" + "x".join(_format_num(v) for v in res_mm),
        f"Ry{cfg['Ry']}",
        f"Rz{cfg['Rz']}",
    ]
    if file_tag:
        parts.append(file_tag)
    return "_".join(parts)


def _save_complex_npy(path: Path, data: torch.Tensor | np.ndarray, label: str) -> Path:
    arr = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    final_path = path if path.suffix == ".npy" else path.with_suffix(".npy")
    print(f"Saved {label}: {final_path} shape={arr.shape}")
    return final_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_gre_metadata(
    *,
    cfg: Mapping[str, Any],
    mode: str,
    twix_file: Path,
    seq_file: Path,
    echo_idx: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "SequenceType": "3D multi-echo Wave-GRE with integrated FLASH calibration",
        "SequenceName": cfg["sequence_name"],
        "SourceTwix": twix_file.name,
        "SourcePulseq": seq_file.name,
        "ReconstructionMode": mode,
        "OrientationMapping": cfg["orientation"],
        "MatrixSize": [cfg["Nx"], cfg["Ny"], cfg["Nz"]],
        "ReadoutOversampledSize": cfg["Nx_os"],
        "ReadoutOversamplingFactor": cfg["os_factor"],
        "FOVMeters": cfg["FOVxyz_m"],
        "VoxelSizeMillimeters": [float(v) * 1e3 for v in cfg["res_xyz_m"]],
        "EchoCount": cfg["Necho"],
        "EchoTimesSeconds": cfg["TE_s"],
        "Averages": cfg["Averages"],
        "AccelerationRy": cfg["Ry"],
        "AccelerationRz": cfg["Rz"],
        "FlowCompensation": cfg["UseFlowComp"],
        "KspaceOrdering": cfg["KspaceOrdering"],
        "PSFYFlip": cfg["yflip"],
        "PSFZFlip": cfg["zflip"],
        "CalibrationNcalib1": cfg["Ncalib1"],
        "CalibrationNcalib2": cfg["Ncalib2"],
        "CalibrationNacs": cfg["Nacs"],
        "CalibrationSetLayout": {
            "0": "no-wave LIN projection",
            "1": "sine-wave LIN projection",
            "2": "no-wave PAR projection",
            "3": "cosine-wave PAR projection",
            "4": "no-wave ACS",
        },
    }
    if echo_idx is not None:
        metadata["EchoNumber"] = int(echo_idx + 1)
        metadata["EchoTimeSeconds"] = float(cfg["TE_s"][echo_idx])
    return _json_safe(metadata)


def save_gre_echo_to_nifti(
    *,
    image: torch.Tensor | np.ndarray,
    twix_file: Path,
    out_folder: Path,
    nifti_sub: str,
    suffix: str,
    mode: str,
    echo_idx: int,
    cfg: Mapping[str, Any],
    save_phase: bool,
    twix_array_axis_roles: Sequence[str],
    twix_array_axis_flips: Sequence[bool],
    twix_coord_system: str,
    twix_inplane_rot_sign: float,
    twix_use_fov_for_voxel_size: bool,
    metadata: Mapping[str, Any],
) -> list[tuple[Path, Path]]:
    from utils.nifti_export_twix import (
        apply_array_axis_flips,
        crop_readout_oversampling,
        make_nifti_affine_from_twix,
        prepare_image_array,
        save_nifti_with_json,
    )

    img_np = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    if img_np.ndim != 3:
        raise ValueError(f"Expected a 3D echo image for NIfTI export, got {img_np.shape}.")
    img_crop = crop_readout_oversampling(img_np, crop_readout_os=int(cfg["os_factor"]))
    outputs: list[tuple[str, np.ndarray]] = [("mag", prepare_image_array(img_crop, part="mag"))]
    if save_phase:
        outputs.append(("phase", prepare_image_array(img_crop, part="phase")))

    flipped = apply_array_axis_flips([arr for _, arr in outputs], twix_array_axis_flips)
    outputs = [(part, arr) for (part, _), arr in zip(outputs, flipped)]

    affine, voxel_size_affine, twix_info = make_nifti_affine_from_twix(
        twix_file=twix_file,
        npy_shape=outputs[0][1].shape,
        twix_array_axis_roles=twix_array_axis_roles,
        twix_array_axis_flips=(False, False, False),
        twix_coord_system=twix_coord_system,
        twix_inplane_rot_sign=twix_inplane_rot_sign,
        twix_use_fov_for_voxel_size=twix_use_fov_for_voxel_size,
        voxel_size_mm=tuple(float(v) * 1e3 for v in cfg["res_xyz_m"]),
    )

    out_folder.mkdir(parents=True, exist_ok=True)
    base = f"sub-{nifti_sub}_echo-{echo_idx + 1:02d}_acq-{mode}"
    saved: list[tuple[Path, Path]] = []
    for part, arr in outputs:
        nii_path = out_folder / f"{base}_part-{part}_{suffix}.nii.gz"
        json_path = out_folder / f"{base}_part-{part}_{suffix}.json"
        sidecar = dict(metadata)
        sidecar.update(
            {
                "ImagePart": part,
                "SavedVoxelSizeMillimeters": list(voxel_size_affine),
                "TwixGeometry": twix_info,
                "TwixArrayAxisRoles": list(twix_array_axis_roles),
                "AppliedArrayAxisFlips": [bool(v) for v in twix_array_axis_flips],
            }
        )
        saved.append(save_nifti_with_json(arr, affine, nii_path, json_path, sidecar))
    return saved


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    runtime = _collect_runtime_config(argv)
    seq = _load_sequence(runtime["seq_file"])
    cfg = _derive_gre_config(
        seq,
        yflip_override=runtime["yflip_override"],
        zflip_override=runtime["zflip_override"],
    )
    image_lines, calib_lines = _split_adc_trajectory(seq, cfg)
    detected_mode = _detect_image_wave_mode(image_lines, cfg)
    mode = _resolve_reconstruction_mode(runtime["mode"], detected_mode)
    _print_sequence_summary(cfg, detected_mode=mode)

    if runtime["validate_only"]:
        print("Sequence validation completed successfully.")
        return 0

    print("Importing GRE image data from integrated TWIX file...")
    img = _normalize_gre_image_data(load_img(str(runtime["twix_file"])), cfg)
    ncoil = int(img.shape[-1])
    print(f"Normalized image shape: {tuple(img.shape)}")

    print("Preparing coil-compression matrix and sensitivity maps...")
    wcc, csm_full, ncoil_ref = load_or_generate_coil_sens(
        twix_file=runtime["twix_file"],
        cfg=cfg,
        out_folder=runtime["out_folder"],
        file_tag=runtime["file_tag"],
        ncc=runtime["ncc"],
        reuse_coil_calib=runtime["reuse_coil_calib"],
        espirit_device=runtime["espirit_device"],
        espirit_gpu_index=runtime["espirit_gpu_index"],
    )
    if ncoil_ref != ncoil:
        raise ValueError(
            f"Image/refscan coil-count mismatch: image has {ncoil}, refscan has {ncoil_ref}."
        )

    kspace_full = _embed_full_kspace(img, cfg)
    kspace_cc = torch.empty(
        (*kspace_full.shape[:-1], runtime["ncc"]), dtype=torch.complex64
    )
    for echo_idx in range(int(cfg["Necho"])):
        print(f"Coil-compressing echo {echo_idx + 1}/{cfg['Necho']}...")
        kspace_cc[:, :, :, echo_idx, :] = apply_cc_coillast_torch(
            kspace_full[:, :, :, echo_idx, :],
            wcc,
            x_chunk=8,
        )

    stem = _recon_stem(cfg, mode, runtime["file_tag"])
    _save_complex_npy(
        runtime["out_folder"] / f"kspace_cc_{stem}.npy",
        kspace_cc,
        "coil-compressed multi-echo k-space",
    )

    sens = _build_sensitivity_tensor(csm_full, cfg)
    masks = _sampling_masks(kspace_cc)

    psf_calib_echoes: torch.Tensor | None = None
    if mode == "wave":
        print("Generating echo-specific calibrated PSFs from integrated calibration...")
        psf_calib_echoes, psf_theory_echoes = generate_calibrated_psfs(
            twix_file=runtime["twix_file"],
            image_lines=image_lines,
            calib_lines=calib_lines,
            cfg=cfg,
            out_folder=runtime["out_folder"],
            file_tag=runtime["file_tag"],
        )
        _save_complex_npy(
            runtime["out_folder"] / f"psf_calib_{stem}.npy",
            psf_calib_echoes,
            "calibrated PSFs",
        )
        _save_complex_npy(
            runtime["out_folder"] / f"psf_theory_{stem}.npy",
            psf_theory_echoes,
            "theoretical PSFs",
        )

    images = reconstruct_echoes(
        kspace_cc=kspace_cc,
        sens=sens,
        masks=masks,
        mode=mode,
        psf_calib_echoes=psf_calib_echoes,
        cg_iters=runtime["cg_iters"],
        cg_tol=runtime["cg_tol"],
    )
    image_path = _save_complex_npy(
        runtime["out_folder"] / f"image_cg_integrated_calib_{stem}.npy",
        images,
        "multi-echo complex reconstruction",
    )

    metadata = _build_gre_metadata(
        cfg=cfg,
        mode=mode,
        twix_file=runtime["twix_file"],
        seq_file=runtime["seq_file"],
    )
    metadata_path = image_path.with_suffix(".json")
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved reconstruction metadata: {metadata_path}")

    for echo_idx in range(int(cfg["Necho"])):
        echo_image = images[:, :, :, echo_idx]
        if runtime["save_echo_npy"]:
            _save_complex_npy(
                runtime["out_folder"] / f"image_cg_integrated_calib_{stem}_echo{echo_idx + 1:02d}.npy",
                echo_image,
                f"echo {echo_idx + 1} complex reconstruction",
            )
        if runtime["save_nifti"]:
            echo_metadata = _build_gre_metadata(
                cfg=cfg,
                mode=mode,
                twix_file=runtime["twix_file"],
                seq_file=runtime["seq_file"],
                echo_idx=echo_idx,
            )
            save_gre_echo_to_nifti(
                image=echo_image,
                twix_file=runtime["twix_file"],
                out_folder=runtime["nifti_out_folder"],
                nifti_sub=runtime["nifti_sub"],
                suffix=runtime["nifti_suffix"],
                mode=mode,
                echo_idx=echo_idx,
                cfg=cfg,
                save_phase=runtime["save_nifti_phase"],
                twix_array_axis_roles=runtime["nifti_axis_roles"],
                twix_array_axis_flips=runtime["nifti_axis_flips"],
                twix_coord_system=runtime["twix_coord_system"],
                twix_inplane_rot_sign=runtime["twix_inplane_rot_sign"],
                twix_use_fov_for_voxel_size=runtime["twix_use_fov_for_voxel_size"],
                metadata=echo_metadata,
            )

    print("Reconstruction completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
