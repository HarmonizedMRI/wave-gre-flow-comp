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
* Optional BART CFL inputs for ESPIRiT calibration and Wave-CAIPI
  reconstruction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
    remove_readout_oversampling_kspace,
)
from bart.bart_utils.bart_io import export_wave_inputs
from utils.espirit_calibration import estimate_espirit_maps
from utils.plot_coil_sens import plot_csm_magnitude_grid, plot_csm_phase_grid
from utils.psf_coefficient_processing import (
    AUTO_FIT_PREFILTER_WINDOW,
    fit_sine_plus_line,
    select_automatic_kx_range,
    sine_line_model,
)
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


COIL_CALIBRATION_READOUT_OVERSAMPLING_REMOVAL = {
    "method": "centered-image-domain-crop",
    "version": 1,
    "fft_normalization": "ortho",
}
COIL_CALIBRATION_CACHE_FORMAT_VERSION = 1
COIL_CALIBRATION_CACHE_TAG = "roimgcrop-v1"


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
        "--wave-mode",
        "--mode",
        dest="mode",
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
    parser.add_argument(
        "--espirit-crop",
        type=float,
        default=0.8,
        help=(
            "ESPIRiT eigenvalue crop threshold. Lower values generally retain "
            "a larger sensitivity-map support region."
        ),
    )
    parser.add_argument(
        "--espirit-calib-mode",
        choices=("3d", "slice2d"),
        default="3d",
        help=(
            "ESPIRiT calibration backend. '3d' is native SigPy 3D calibration; "
            "'slice2d' performs CPU-parallel 2D calibration over logical-RO "
            "hybrid-space slices."
        ),
    )
    parser.add_argument(
        "--espirit-cpu-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "CPU process workers used only by --espirit-calib-mode slice2d. "
            "Omit to select the available physical-core count automatically."
        ),
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
        "--save-bart-inputs",
        action="store_true",
        help=(
            "Export BART CFL inputs under <out>/bart_inputs[_tag]. This is "
            "available for wave acquisitions only."
        ),
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
        default=(False, True, False),
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
    parser.add_argument(
        "--psf-coefficient-processing",
        choices=("smooth", "sine-line"),
        default="smooth",
        help=(
            "Post-process fitted PSF coefficients using the existing NaN-aware "
            "smoothing, or replace smoothing with a sine-plus-line model. "
            "Sine-line selects its range automatically unless both manual "
            "bounds are supplied."
        ),
    )
    parser.add_argument(
        "--psf-fit-kx-min",
        type=int,
        default=None,
        help=(
            "Inclusive manual sine-line readout index; omit both bounds for "
            "automatic range selection."
        ),
    )
    parser.add_argument(
        "--psf-fit-kx-max",
        type=int,
        default=None,
        help=(
            "Exclusive manual sine-line readout index; omit both bounds for "
            "automatic range selection."
        ),
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
    if args.validate_only and args.save_bart_inputs:
        raise ValueError("--save-bart-inputs cannot be used with --validate-only.")

    if not np.isfinite(args.espirit_crop) or not 0.0 <= args.espirit_crop <= 1.0:
        raise ValueError("--espirit-crop must be a finite value between 0 and 1.")
    if args.espirit_cpu_workers is not None and args.espirit_cpu_workers < 1:
        raise ValueError("--espirit-cpu-workers must be a positive integer.")
    if args.espirit_calib_mode == "slice2d" and args.espirit_device == "gpu":
        raise ValueError(
            "--espirit-calib-mode slice2d is CPU-only; use --espirit-device cpu "
            "or auto, or select --espirit-calib-mode 3d for GPU calibration."
        )
    psf_processing = str(args.psf_coefficient_processing).strip().lower()
    fit_kx_min = args.psf_fit_kx_min
    fit_kx_max = args.psf_fit_kx_max
    if psf_processing == "sine-line":
        if (fit_kx_min is None) != (fit_kx_max is None):
            raise ValueError(
                "Manual sine-line fitting requires both --psf-fit-kx-min and "
                "--psf-fit-kx-max; omit both for automatic range selection."
            )
        if fit_kx_min is not None and (fit_kx_min < 0 or fit_kx_max <= fit_kx_min):
            raise ValueError(
                "PSF fit bounds must satisfy 0 <= --psf-fit-kx-min < "
                "--psf-fit-kx-max."
            )
    elif fit_kx_min is not None or fit_kx_max is not None:
        raise ValueError(
            "--psf-fit-kx-min/--psf-fit-kx-max are only valid with "
            "--psf-coefficient-processing sine-line."
        )

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
        "espirit_crop": float(args.espirit_crop),
        "espirit_calib_mode": args.espirit_calib_mode,
        "espirit_cpu_workers": args.espirit_cpu_workers,
        "cg_iters": int(args.cg_iters),
        "cg_tol": float(args.cg_tol),
        "yflip_override": args.yflip,
        "zflip_override": args.zflip,
        "save_echo_npy": bool(args.save_echo_npy),
        "save_bart_inputs": bool(args.save_bart_inputs),
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
        "psf_coefficient_processing": psf_processing,
        "psf_fit_kx_min": None if fit_kx_min is None else int(fit_kx_min),
        "psf_fit_kx_max": None if fit_kx_max is None else int(fit_kx_max),
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


def _centered_accelerated_label_geometry(matrix: int, acceleration: int) -> tuple[int, int]:
    """Return measured count and mapVBVD extent for centered global labels."""
    if matrix <= 0 or acceleration <= 0:
        raise ValueError(
            "Centered acceleration requires positive matrix and acceleration; "
            f"got matrix={matrix}, acceleration={acceleration}."
        )
    center_label = matrix // 2
    labels = np.arange(matrix, dtype=np.int64)
    acquired = labels[(labels - center_label) % acceleration == 0]
    return int(acquired.size), int(acquired[-1] + 1)


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
    ny_count_expected, ny_label_extent = _centered_accelerated_label_geometry(ny, ry)
    nz_count_expected, nz_label_extent = _centered_accelerated_label_geometry(nz, rz)
    ny_meas = _as_int(
        _get_definition(defs, "Ny_meas", ny_count_expected), "Ny_meas"
    )
    nz_meas = _as_int(
        _get_definition(defs, "Nz_meas", nz_count_expected), "Nz_meas"
    )
    if ny_meas != ny_count_expected or nz_meas != nz_count_expected:
        raise ValueError(
            "Measured PE counts disagree with the centered global-label sampling pattern: "
            f"sequence Ny_meas/Nz_meas={ny_meas}/{nz_meas}, expected "
            f"{ny_count_expected}/{nz_count_expected} for matrix/acceleration "
            f"{ny}x{nz} at R{ry}x{rz}."
        )

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
        "Ny_label_extent": ny_label_extent,
        "Nz_label_extent": nz_label_extent,
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


def _logical_integrated_acs(
    ref: torch.Tensor, cfg: Mapping[str, Any]
) -> torch.Tensor:
    """Extract set-4 ACS and remove readout oversampling without aliasing.

    Args:
        ref: Validated integrated refscan in
            ``(RO_os, LIN, PAR, SET, physical_coil)`` order.
        cfg: Validated GRE sequence configuration.

    Returns:
        Complex64 ACS on the logical readout grid in coil-last order.

    Raises:
        ValueError: If the ACS geometry or finite-value contract is violated.
    """

    nacs = int(cfg["Nacs"])
    acs_set_id = int(cfg["ACSSetID"])
    oversampled = ref[:, :nacs, :nacs, acs_set_id, :]
    logical = remove_readout_oversampling_kspace(
        oversampled,
        int(cfg["os_factor"]),
        axis=0,
    )
    expected = (int(cfg["Nx"]), nacs, nacs, int(ref.shape[-1]))
    if tuple(logical.shape) != expected:
        raise ValueError(
            "Unexpected logical integrated ACS shape after readout crop: "
            f"received {tuple(logical.shape)}, expected {expected}."
        )
    if not torch.isfinite(logical).all():
        raise ValueError("Logical integrated ACS contains non-finite values.")
    return logical.contiguous()


def _load_logical_integrated_acs(
    twix_file: Path, cfg: Mapping[str, Any]
) -> torch.Tensor:
    """Load and validate the alias-free logical set-4 ACS.

    Args:
        twix_file: Integrated Wave-GRE TWIX file.
        cfg: Validated GRE sequence configuration.

    Returns:
        Complex64 logical-readout ACS in coil-last order.
    """

    ref = _check_integrated_refscan_shape(
        load_ref(str(twix_file)),
        ncalib1=int(cfg["Ncalib1"]),
        nacs=int(cfg["Nacs"]),
        nsets=int(cfg["Nsets"]),
    )
    return _logical_integrated_acs(ref, cfg)


def _array_sha256(array: np.ndarray | torch.Tensor) -> str:
    """Return a shape- and dtype-bound SHA-256 digest for one array.

    Args:
        array: NumPy array or CPU/GPU PyTorch tensor.

    Returns:
        Hexadecimal SHA-256 digest of metadata followed by C-order payload.
    """

    if torch.is_tensor(array):
        values = array.detach().cpu().contiguous().numpy()
    else:
        values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape)).encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: Existing regular file.

    Returns:
        Hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coil_calibration_contract(
    *,
    twix_file: Path,
    cfg: Mapping[str, Any],
    logical_acs: torch.Tensor,
    ncc: int,
    espirit_calib_mode: str,
    espirit_crop: float,
) -> dict[str, Any]:
    """Build the immutable scientific contract for cached coil calibration.

    Args:
        twix_file: Integrated Wave-GRE TWIX source.
        cfg: Validated GRE sequence configuration.
        logical_acs: Alias-free logical set-4 ACS.
        ncc: Requested number of virtual coils.
        espirit_calib_mode: Selected ESPIRiT backend.
        espirit_crop: ESPIRiT eigenvalue crop threshold.

    Returns:
        JSON-compatible cache provenance contract.
    """

    source = twix_file.stat()
    return {
        "format_version": COIL_CALIBRATION_CACHE_FORMAT_VERSION,
        "status": "alias_free_gre_coil_calibration_ready",
        "source_twix": {
            "path": str(twix_file.resolve()),
            "size_bytes": int(source.st_size),
            "mtime_ns": int(source.st_mtime_ns),
        },
        "geometry": {
            "logical_matrix_ro_lin_par": [
                int(cfg["Nx"]),
                int(cfg["Ny"]),
                int(cfg["Nz"]),
            ],
            "readout_oversampling_factor": int(cfg["os_factor"]),
            "integrated_acs_size": int(cfg["Nacs"]),
            "integrated_acs_set_id": int(cfg["ACSSetID"]),
        },
        "readout_oversampling_removal": {
            **COIL_CALIBRATION_READOUT_OVERSAMPLING_REMOVAL,
            "input_readout": int(cfg["Nx_os"]),
            "output_readout": int(cfg["Nx"]),
            "oversampling_factor": int(cfg["os_factor"]),
        },
        "logical_acs": {
            "shape": list(logical_acs.shape),
            "sha256": _array_sha256(logical_acs),
        },
        "coil_compression": {
            "physical_coils": int(logical_acs.shape[-1]),
            "virtual_coils": int(ncc),
            "readout_stride": 1,
        },
        "espirit": {
            "mode": _normalize_espirit_calib_mode(espirit_calib_mode),
            "crop": float(espirit_crop),
            "calib_width_maximum": 24,
            "threshold": 0.02,
            "kernel_width": 6,
            "maximum_iterations": 100,
        },
    }


def _cache_suffix(file_tag: str) -> str:
    return f"_{file_tag}" if file_tag else ""


def _normalize_espirit_calib_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in {"3d", "slice2d"}:
        raise ValueError("ESPIRiT calibration mode must be '3d' or 'slice2d'.")
    return mode


def _coil_cache_paths(
    out_folder: Path,
    file_tag: str,
    ncc: int,
    espirit_calib_mode: str,
) -> dict[str, Path]:
    suffix = _cache_suffix(file_tag)
    mode = _normalize_espirit_calib_mode(espirit_calib_mode)
    calibration = f"_{COIL_CALIBRATION_CACHE_TAG}"
    csm_mode = "" if mode == "3d" else "_slice2d"
    return {
        "wcc": out_folder
        / f"coil_compression_matrix_ncc{ncc}{calibration}{suffix}.npy",
        "csm_low": out_folder
        / f"csm_acs_ncc{ncc}{csm_mode}{calibration}{suffix}.npy",
        "csm_full": out_folder
        / f"csm_full_ncc{ncc}{csm_mode}{calibration}{suffix}.npy",
        "csm_mag": out_folder
        / f"csm_full_mag_ncc{ncc}{csm_mode}{calibration}{suffix}.png",
        "csm_phase": out_folder
        / f"csm_full_phase_ncc{ncc}{csm_mode}{calibration}{suffix}.png",
        "manifest": out_folder
        / f"coil_calibration_ncc{ncc}{csm_mode}{calibration}{suffix}.json",
    }


def _validate_coil_cache_manifest(
    path: Path,
    *,
    expected_contract: Mapping[str, Any],
    wcc_path: Path,
    csm_full_path: Path,
) -> None:
    """Validate cache provenance and artifact hashes before reuse.

    Args:
        path: Cache-manifest JSON path.
        expected_contract: Contract recomputed from the current source ACS and
            reconstruction settings.
        wcc_path: Cached coil-compression matrix path.
        csm_full_path: Cached full-resolution sensitivity-map path.

    Returns:
        None.

    Raises:
        ValueError: If provenance or artifact hashes differ.
    """

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read coil-calibration cache manifest: {path}") from exc
    if manifest.get("contract") != dict(expected_contract):
        raise ValueError(
            "Cached coil calibration does not match the current source, geometry, "
            "alias-free readout-crop method, or ESPIRiT settings."
        )
    expected_artifacts = {
        "coil_compression_matrix": {
            "path": wcc_path.name,
            "sha256": _file_sha256(wcc_path),
        },
        "full_resolution_csm": {
            "path": csm_full_path.name,
            "sha256": _file_sha256(csm_full_path),
        },
    }
    if manifest.get("artifacts") != expected_artifacts:
        raise ValueError("Cached coil-calibration artifact hashes do not match provenance.")


def _write_coil_cache_manifest(
    path: Path,
    *,
    contract: Mapping[str, Any],
    wcc_path: Path,
    csm_full_path: Path,
) -> None:
    """Write a hash-bound coil-calibration cache manifest.

    Args:
        path: Destination JSON path.
        contract: Immutable source, geometry, and algorithm contract.
        wcc_path: Written coil-compression matrix path.
        csm_full_path: Written full-resolution sensitivity-map path.

    Returns:
        None.
    """

    payload = {
        "contract": dict(contract),
        "artifacts": {
            "coil_compression_matrix": {
                "path": wcc_path.name,
                "sha256": _file_sha256(wcc_path),
            },
            "full_resolution_csm": {
                "path": csm_full_path.name,
                "sha256": _file_sha256(csm_full_path),
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    espirit_crop: float,
    espirit_calib_mode: str,
    espirit_cpu_workers: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    paths = _coil_cache_paths(out_folder, file_tag, ncc, espirit_calib_mode)
    required_cache = (paths["wcc"], paths["csm_full"], paths["manifest"])
    if reuse_coil_calib and all(path.is_file() for path in required_cache):
        print("Loading cached coil-compression matrix and sensitivity maps...")
        logical_acs = _load_logical_integrated_acs(twix_file, cfg)
        ncoil = int(logical_acs.shape[-1])
        expected_contract = _coil_calibration_contract(
            twix_file=twix_file,
            cfg=cfg,
            logical_acs=logical_acs,
            ncc=ncc,
            espirit_calib_mode=espirit_calib_mode,
            espirit_crop=espirit_crop,
        )
        _validate_coil_cache_manifest(
            paths["manifest"],
            expected_contract=expected_contract,
            wcc_path=paths["wcc"],
            csm_full_path=paths["csm_full"],
        )
        wcc = np.load(paths["wcc"], allow_pickle=False)
        csm_full = np.load(paths["csm_full"], allow_pickle=False)
        _check_cc_and_csm(wcc, csm_full, ncoil=ncoil, ncc=ncc, cfg=cfg)
        if not np.isfinite(wcc).all() or not np.isfinite(csm_full).all():
            raise ValueError("Cached coil calibration contains non-finite values.")
        print(
            "Reused hash-bound alias-free coil calibration with centered "
            "image-domain readout cropping."
        )
        return wcc, csm_full, ncoil

    if reuse_coil_calib:
        present = [path for path in required_cache if path.exists()]
        if present:
            raise ValueError(
                "Alias-free coil-calibration cache is incomplete; use a new output "
                "folder or file tag rather than mixing cache generations."
            )
        print(
            "No compatible alias-free coil-calibration cache exists; recomputing "
            "from integrated ACS. Historical stride-derived cache names are ignored."
        )
    return generate_coil_sens(
        twix_file=twix_file,
        cfg=cfg,
        out_folder=out_folder,
        file_tag=file_tag,
        ncc=ncc,
        espirit_device=espirit_device,
        espirit_gpu_index=espirit_gpu_index,
        espirit_crop=espirit_crop,
        espirit_calib_mode=espirit_calib_mode,
        espirit_cpu_workers=espirit_cpu_workers,
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
    espirit_crop: float,
    espirit_calib_mode: str,
    espirit_cpu_workers: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    espirit_calib_mode = _normalize_espirit_calib_mode(espirit_calib_mode)
    print(f"ESPIRiT calibration mode: {espirit_calib_mode}")
    print(f"ESPIRiT crop threshold: {espirit_crop:g}")
    if espirit_calib_mode == "slice2d":
        if espirit_device == "gpu":
            raise ValueError(
                "slice2d ESPIRiT is CPU-only. Use --espirit-device cpu or auto, "
                "or select the 3d calibration mode for GPU execution."
            )
        device, using_gpu = sp.Device(-1), False
        print("ESPIRiT device: CPU (required by slice2d)")
    else:
        device, using_gpu = _select_espirit_device(espirit_device, espirit_gpu_index)
    kspace_acs = _load_logical_integrated_acs(twix_file, cfg)
    nacs = int(cfg["Nacs"])
    nx, ny_acs, nz_acs, ncoil = map(int, kspace_acs.shape)
    if ncc > ncoil:
        raise ValueError(f"Requested ncc={ncc}, but the TWIX refscan has only {ncoil} coils.")
    if nx != int(cfg["Nx"]):
        raise ValueError(
            f"Logical ACS readout {nx} does not match sequence Nx={cfg['Nx']}."
        )
    print(f"Alias-free logical integrated ACS shape: {tuple(kspace_acs.shape)}")

    wcc, _, cc_energy = estimate_cc_matrix_coillast(
        kspace_acs,
        ncc=ncc,
        acs=min(ny_acs, nz_acs),
    )
    print(f"Coil-compression matrix: {wcc.shape}")
    print(f"Energy retained by {ncc} coils: {float(cc_energy[ncc - 1]):.6f}")

    kspace_np = (
        kspace_acs.permute(3, 0, 1, 2)
        .contiguous()
        .numpy()
        .astype(np.complex64, copy=False)
    )
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

    calib_width = min(24, low_y, low_z)
    csm_low_cc_np, espirit_info = estimate_espirit_maps(
        kspace_low_cc_np,
        mode=espirit_calib_mode,
        device=device,
        crop=espirit_crop,
        calib_width=calib_width,
        thresh=0.02,
        kernel_width=6,
        max_iter=100,
        cpu_workers=espirit_cpu_workers,
    )
    if espirit_info.mode == "slice2d":
        print(
            "Completed slice2d ESPIRiT with "
            f"{espirit_info.cpu_workers} CPU worker(s)."
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
    csm_full = np.divide(csm_full, rss, out=np.zeros_like(csm_full), where=rss > 1e-6)

    paths = _coil_cache_paths(out_folder, file_tag, ncc, espirit_calib_mode)
    np.save(paths["wcc"], np.asarray(wcc))
    np.save(paths["csm_low"], csm_low_cc_np)
    np.save(paths["csm_full"], csm_full)
    contract = _coil_calibration_contract(
        twix_file=twix_file,
        cfg=cfg,
        logical_acs=kspace_acs,
        ncc=ncc,
        espirit_calib_mode=espirit_calib_mode,
        espirit_crop=espirit_crop,
    )
    _write_coil_cache_manifest(
        paths["manifest"],
        contract=contract,
        wcc_path=paths["wcc"],
        csm_full_path=paths["csm_full"],
    )

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


def _build_bart_calibration_kspace(
    *,
    twix_file: Path,
    cfg: Mapping[str, Any],
    wcc: np.ndarray,
) -> np.ndarray:
    """Return compressed, alias-free ACS on BART's full image grid.

    Args:
        twix_file: Integrated Wave-GRE TWIX source.
        cfg: Validated GRE sequence configuration.
        wcc: Physical-to-virtual coil-compression matrix.

    Returns:
        Logical-readout, coil-last calibration k-space embedded on the native
        BART LIN/PAR grid.
    """

    nacs = int(cfg["Nacs"])
    kspace_acs = _load_logical_integrated_acs(twix_file, cfg)
    kspace_acs_cc = apply_cc_coillast_torch(kspace_acs, wcc, x_chunk=8)

    sx = int(cfg["Nx"])
    sy = int(cfg["Ny"])
    sz = int(cfg["Nz"])
    nc = int(wcc.shape[1])
    if tuple(kspace_acs_cc.shape) != (sx, nacs, nacs, nc):
        raise ValueError(
            "Unexpected compressed BART calibration shape: "
            f"received {tuple(kspace_acs_cc.shape)}, expected {(sx, nacs, nacs, nc)}."
        )
    if nacs > sy or nacs > sz:
        raise ValueError(
            f"ACS size {nacs} does not fit the BART image grid {(sx, sy, sz)}."
        )

    full = torch.zeros((sx, sy, sz, nc), dtype=torch.complex64)
    y0 = (sy - nacs) // 2
    z0 = (sz - nacs) // 2
    full[:, y0 : y0 + nacs, z0 : z0 + nacs, :] = kspace_acs_cc
    return full.numpy()


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


def _projection_fit_quality_summary(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Summarize wrapped-plane fit quality on the readout grid.

    Args:
        result: Mapping returned by ``fit_wrapped_phase_planes`` with quality
            maps enabled.

    Returns:
        Per-readout vectors describing support, residuals, skipped fits, and
        median coherence over the final fit mask.
    """

    mask = torch.as_tensor(result["mask"], dtype=torch.bool).detach().cpu()

    def masked_median(values: Any) -> np.ndarray:
        """Reduce a quality map over accepted pixels for each readout sample.

        Args:
            values: Readout-by-projection quality map.

        Returns:
            Per-readout median values over the final wrapped-plane fit mask.
        """

        array = torch.as_tensor(values).detach().cpu()
        medians = np.full(mask.shape[0], np.nan, dtype=np.float64)
        for index in range(mask.shape[0]):
            selected = array[index][mask[index]]
            selected = selected[torch.isfinite(selected)]
            if selected.numel():
                medians[index] = float(torch.median(selected).item())
        return medians

    summary = {
        name: torch.as_tensor(result[name]).detach().cpu().numpy()
        for name in ("wrapped_rms", "valid_pixels", "masked_ratio", "skipped")
    }
    summary["median_phase_coherence"] = masked_median(result["phase_coherence"])
    summary["median_residual_coherence"] = masked_median(
        result["residual_coherence"]
    )
    return summary


def fit_wave_psf_deviation_from_projection(
    *,
    twix_file: Path,
    calib_lines: np.ndarray,
    cfg: Mapping[str, Any],
    out_folder: Path,
    file_tag: str,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, ...]:
    """Fit one shared raw coefficient solution from integrated calibration.

    Args:
        twix_file: Integrated Siemens TWIX file whose refscan is calibrated.
        calib_lines: Calibration ADC trajectory with shape
            ``(3, calibration_lines, Nx_os)``.
        cfg: Validated GRE sequence and acquisition configuration.
        out_folder: Destination for raw coefficient arrays.
        file_tag: Optional filename suffix token.
        return_diagnostics: Append projection-quality evidence for automatic
            coefficient processing.

    Returns:
        Raw ``a``, ``b``, and ``c`` vectors. If ``return_diagnostics`` is true,
        a fourth evidence mapping is appended. This fit is independent of GRE
        echo number and is shared by all echo-specific PSFs.
    """
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
    projection_quality: dict[str, dict[str, np.ndarray]] = {}

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
        a_result = torch.as_tensor(result["a_fit_all"]).detach().cpu()
        b_result = torch.as_tensor(result["b_fit_all"]).detach().cpu()
        c_result = torch.as_tensor(result["c_fit_all"]).detach().cpu()

        a_fit_all.append(a_result)
        b_fit_all.append(b_result)
        c_fit_all.append(c_result)
        projection_quality[wave_mode] = _projection_fit_quality_summary(result)

        suffix = _cache_suffix(file_tag)
        np.save(out_folder / f"a_fit_all_{tag}{suffix}.npy", a_result.numpy())
        np.save(out_folder / f"b_fit_all_{tag}{suffix}.npy", b_result.numpy())
        np.save(out_folder / f"c_fit_all_{tag}{suffix}.npy", c_result.numpy())

    a_fit = a_fit_all[0]
    b_fit = b_fit_all[1]
    c_fit = c_fit_all[0] + c_fit_all[1]
    if return_diagnostics:
        evidence = {
            "readout_index": np.arange(nx_os, dtype=np.int64),
            "projection_quality": projection_quality,
            "component_projection_sources": {
                "a": ["sin"],
                "b": ["cos"],
                "c": ["sin", "cos"],
            },
            "shared_across_echoes": True,
            "source": "integrated_refscan_projection_calibration",
        }
        return a_fit, b_fit, c_fit, evidence
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
        coeff = torch.stack(
            (a_fit[kx_idx], b_fit[kx_idx], c_fit[kx_idx])
        ).to(dtype=design.dtype)
        correction[kx_idx] = (design @ coeff).view(ny, nz)
    return torch.nan_to_num(correction, nan=0.0).to(torch.float32)


def _sine_line_model(t, A, w, phi, C1, C2):
    """Evaluate A*sin(w*t + phi) + C1*t + C2.

    Args:
        t: Readout sample coordinates.
        A: Sine amplitude.
        w: Angular frequency in radians per sample.
        phi: Sine phase at readout index zero.
        C1: Linear slope.
        C2: Linear intercept.

    Returns:
        Model values at ``t``.
    """

    return sine_line_model(t, A, w, phi, C1, C2)


def _fit_sine_plus_line(t, values):
    """Fit one coefficient with the shared sine-plus-line implementation.

    Args:
        t: Readout sample coordinates.
        values: Raw coefficient samples.

    Returns:
        Fitted parameters and numerical diagnostics.
    """

    return fit_sine_plus_line(t, values)


def _process_psf_coefficients(
    a_raw,
    b_raw,
    c_raw,
    *,
    nx_os,
    processing="smooth",
    fit_kx_min=None,
    fit_kx_max=None,
    fit_quality=None,
    out_folder=None,
    file_tag="",
    return_diagnostics=False,
):
    """Apply mutually exclusive smoothing or sine-line coefficient processing.

    Args:
        a_raw: Raw LIN phase coefficient vector.
        b_raw: Raw PAR phase coefficient vector.
        c_raw: Raw constant phase coefficient vector.
        nx_os: Oversampled readout length.
        processing: ``smooth`` or ``sine-line``.
        fit_kx_min: Optional inclusive manual range start.
        fit_kx_max: Optional exclusive manual range stop.
        fit_quality: Sin/cos projection evidence used for automatic selection.
        out_folder: Optional diagnostics destination.
        file_tag: Diagnostics filename tag.
        return_diagnostics: Append processing diagnostics to the output tuple.

    Returns:
        Processed ``a``, ``b``, and ``c`` tensors, optionally followed by a
        JSON-compatible diagnostics mapping.
    """

    mode = str(processing).strip().lower()
    if mode == "smooth":
        outputs = (
            smooth_1d_nan(a_raw, window=9),
            smooth_1d_nan(b_raw, window=9),
            smooth_1d_nan(c_raw, window=9),
        )
        diagnostics = {
            "coefficient_processing": "smooth",
            "fit_range_selection": None,
            "kx_range": None,
            "kx_range_convention": "half-open [min, max)",
        }
        return (*outputs, diagnostics) if return_diagnostics else outputs
    if mode != "sine-line":
        raise ValueError("processing must be 'smooth' or 'sine-line'.")
    if (fit_kx_min is None) != (fit_kx_max is None):
        raise ValueError("sine-line PSF processing requires both manual fit bounds or neither.")
    if fit_kx_min is None:
        if fit_quality is None:
            raise ValueError(
                "automatic sine-line PSF processing requires projection fit quality evidence."
            )
        selected, selection_diagnostics = select_automatic_kx_range(
            (a_raw, b_raw, c_raw), fit_quality
        )
        fit_kx_min, fit_kx_max = selected
        fit_sample_mask = np.asarray(
            selection_diagnostics.pop("_fit_sample_mask"), dtype=bool
        )
        fit_range_selection = "automatic"
    else:
        fit_kx_min = int(fit_kx_min)
        fit_kx_max = int(fit_kx_max)
        fit_range_selection = "manual"
        selection_diagnostics = {
            "name": "manual-half-open-range",
            "version": 1,
            "selected_interval": [fit_kx_min, fit_kx_max],
        }
        fit_sample_mask = np.ones(int(nx_os), dtype=bool)
    if not (0 <= fit_kx_min < fit_kx_max <= int(nx_os)):
        raise ValueError(
            f"PSF fit range must satisfy 0 <= min < max <= Nx_os; got "
            f"[{fit_kx_min}, {fit_kx_max}) with nx_os={nx_os}."
        )

    interval_width = fit_kx_max - fit_kx_min
    kx_all = np.arange(int(nx_os), dtype=float)
    fit_indices = np.flatnonzero(fit_sample_mask)
    fit_indices = fit_indices[
        (fit_indices >= fit_kx_min) & (fit_indices < fit_kx_max)
    ]
    if fit_range_selection == "automatic":
        fit_input_processing = {
            "name": "quality-masked-nan-aware-moving-average",
            "window_samples": AUTO_FIT_PREFILTER_WINDOW,
        }
    else:
        fit_input_processing = {"name": "raw", "window_samples": None}
    outputs = []
    diagnostics = {}
    validation_passed = True
    for name, raw in (("a", a_raw), ("b", b_raw), ("c", c_raw)):
        # Keep this branch tensor-native so dtype/device information remains
        # available when the full fitted curve is converted back to PyTorch.
        raw_1d = torch.as_tensor(raw).detach().squeeze()
        if raw_1d.ndim != 1:
            raise ValueError(
                f"{name}_raw should reduce to a 1D vector after squeeze; "
                f"got shape {tuple(raw_1d.shape)}"
            )
        if raw_1d.numel() != int(nx_os):
            raise ValueError(
                f"{name}_raw has {raw_1d.numel()} samples; expected nx_os={nx_os}."
            )
        fit_indices_tensor = torch.as_tensor(
            fit_indices, dtype=torch.long, device=raw_1d.device
        )
        if fit_range_selection == "automatic":
            fit_mask_tensor = torch.as_tensor(
                fit_sample_mask, dtype=torch.bool, device=raw_1d.device
            )
            masked_raw = raw_1d.clone()
            masked_raw[~fit_mask_tensor] = torch.nan
            fit_input_1d = smooth_1d_nan(
                masked_raw,
                window=AUTO_FIT_PREFILTER_WINDOW,
            )
        else:
            fit_input_1d = raw_1d

        params = _fit_sine_plus_line(
            fit_indices.astype(float),
            fit_input_1d[fit_indices_tensor].cpu().numpy(),
        )
        fitted = _sine_line_model(
            kx_all,
            params["A"],
            params["w"],
            params["phi"],
            params["C1"],
            params["C2"],
        )
        raw_fit_values = raw_1d[fit_indices_tensor].cpu().numpy()
        raw_fit_prediction = _sine_line_model(
            fit_indices.astype(float),
            params["A"],
            params["w"],
            params["phi"],
            params["C1"],
            params["C2"],
        )
        raw_residual = raw_fit_prediction - raw_fit_values
        params["raw_observation_residual_rmse"] = float(
            np.sqrt(np.mean(raw_residual**2))
        )
        params["raw_observation_residual_rmse_relative_to_range"] = float(
            np.sqrt(np.mean(raw_residual**2))
            / max(float(np.ptp(raw_fit_values)), np.finfo(float).eps)
        )
        params["fit_input_processing"] = fit_input_processing
        outputs.append(torch.as_tensor(fitted, dtype=raw_1d.dtype, device=raw_1d.device))
        trim = max(2, int(np.ceil(0.05 * interval_width)))
        stability = {
            "trim_samples_per_side": trim,
            "refit_success": False,
            "refit_standardized_jacobian_condition_number": None,
            "full_readout_relative_l2_difference": None,
            "full_readout_max_difference_relative_to_fit_range": None,
        }
        trimmed_indices = fit_indices[
            (fit_indices >= fit_kx_min + trim)
            & (fit_indices < fit_kx_max - trim)
        ]
        if trimmed_indices.size >= 6:
            try:
                trimmed_indices_tensor = torch.as_tensor(
                    trimmed_indices, dtype=torch.long, device=raw_1d.device
                )
                trimmed_params = _fit_sine_plus_line(
                    trimmed_indices.astype(float),
                    fit_input_1d[trimmed_indices_tensor].cpu().numpy(),
                )
                trimmed_fitted = _sine_line_model(
                    kx_all,
                    trimmed_params["A"],
                    trimmed_params["w"],
                    trimmed_params["phi"],
                    trimmed_params["C1"],
                    trimmed_params["C2"],
                )
                difference = fitted - trimmed_fitted
                fit_values = fit_input_1d[fit_indices_tensor].cpu().numpy()
                stability.update(
                    {
                        "refit_success": bool(
                            trimmed_params["success"]
                            and np.isfinite(
                                trimmed_params["standardized_jacobian_condition_number"]
                            )
                            and trimmed_params["standardized_jacobian_condition_number"]
                            <= 1.0e12
                        ),
                        "refit_standardized_jacobian_condition_number": trimmed_params[
                            "standardized_jacobian_condition_number"
                        ],
                        "full_readout_relative_l2_difference": float(
                            np.linalg.norm(difference)
                            / max(np.linalg.norm(fitted), np.finfo(float).eps)
                        ),
                        "full_readout_max_difference_relative_to_fit_range": float(
                            np.max(np.abs(difference))
                            / max(float(np.ptp(fit_values)), np.finfo(float).eps)
                        ),
                    }
                )
            except ValueError as exc:
                stability["error"] = str(exc)
        gates = {
            "optimizer_converged": bool(params["success"]),
            "condition_number_at_most_1e12": bool(
                params["standardized_jacobian_condition_number"] <= 1.0e12
            ),
            "residual_rmse_relative_to_range_at_most_0p5": bool(
                params["residual_rmse_relative_to_range"] <= 0.5
            ),
            "period_at_least_4_samples": bool(params["period_samples"] >= 4.0),
            "frequency_not_on_search_boundary": bool(
                params["frequency_boundary_fraction"] >= 0.005
            ),
            "endpoint_trim_refit_succeeded": bool(stability["refit_success"]),
            "endpoint_trim_full_readout_relative_l2_at_most_1": bool(
                stability["full_readout_relative_l2_difference"] is not None
                and stability["full_readout_relative_l2_difference"] <= 1.0
            ),
        }
        params["endpoint_trim_stability"] = stability
        params["validation_gates"] = gates
        params["validation_passed"] = all(gates.values())
        validation_passed &= params["validation_passed"]
        diagnostics[name] = params

    if out_folder is not None:
        diag_path = Path(out_folder) / (
            f"psf_sine_line_fit{_cache_suffix(file_tag)}.json"
        )
        with diag_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "model": "A*sin(w*kx+phi)+C1*kx+C2",
                    "kx_range": [fit_kx_min, fit_kx_max],
                    "kx_range_convention": "half-open [min, max)",
                    "fit_range_selection": fit_range_selection,
                    "fit_input_processing": fit_input_processing,
                    "range_selection_diagnostics": selection_diagnostics,
                    "coefficients": diagnostics,
                    "validation_passed": validation_passed,
                },
                stream,
                indent=2,
            )
        print(f"Saved sine-line PSF fit diagnostics: {diag_path}")
    processing_diagnostics = {
        "coefficient_processing": "sine-line",
        "model": "A*sin(w*kx+phi)+C1*kx+C2",
        "fit_range_selection": fit_range_selection,
        "fit_input_processing": fit_input_processing,
        "kx_range": [fit_kx_min, fit_kx_max],
        "kx_range_convention": "half-open [min, max)",
        "range_selection_diagnostics": selection_diagnostics,
        "coefficients": diagnostics,
        "validation_passed": validation_passed,
    }
    if fit_range_selection == "automatic" and not validation_passed:
        raise ValueError(
            "Automatic sine-line PSF fitting failed one or more numerical or "
            "extrapolation-stability gates; inspect the saved fit diagnostics."
        )
    return (*outputs, processing_diagnostics) if return_diagnostics else tuple(outputs)


def generate_calibrated_psfs(
    *,
    twix_file: Path,
    image_lines: np.ndarray,
    calib_lines: np.ndarray,
    cfg: Mapping[str, Any],
    out_folder: Path,
    file_tag: str,
    psf_plot: bool = True,
    coefficient_processing: str = "smooth",
    fit_kx_min: int | None = None,
    fit_kx_max: int | None = None,
    return_diagnostics: bool = False,
) -> tuple[Any, ...]:
    """Generate per-echo PSFs from one shared coefficient calibration.

    Args:
        twix_file: Integrated Siemens TWIX file.
        image_lines: Image ADC trajectories interleaved by echo.
        calib_lines: Integrated projection-calibration ADC trajectories.
        cfg: Validated GRE sequence and acquisition configuration.
        out_folder: Destination for coefficient and plot diagnostics.
        file_tag: Optional filename suffix token.
        psf_plot: Write the processed coefficient assessment PNG.
        coefficient_processing: ``smooth`` or ``sine-line``.
        fit_kx_min: Optional inclusive manual sine-line bound.
        fit_kx_max: Optional exclusive manual sine-line bound.
        return_diagnostics: Append processing and shared-calibration provenance.

    Returns:
        Calibrated and theoretical PSF tensors with echo as the first axis.
        If ``return_diagnostics`` is true, a third diagnostics mapping is
        appended. The coefficient fit is performed exactly once and shared;
        only the sequence-derived theoretical trajectory varies by echo.
    """

    a_raw, b_raw, c_raw, calibration_evidence = fit_wave_psf_deviation_from_projection(
        twix_file=twix_file,
        calib_lines=calib_lines,
        cfg=cfg,
        out_folder=out_folder,
        file_tag=file_tag,
        return_diagnostics=True,
    )
    a_fit, b_fit, c_fit, processing_diagnostics = _process_psf_coefficients(
        a_raw,
        b_raw,
        c_raw,
        nx_os=int(cfg["Nx_os"]),
        processing=coefficient_processing,
        fit_kx_min=fit_kx_min,
        fit_kx_max=fit_kx_max,
        fit_quality=calibration_evidence["projection_quality"],
        out_folder=out_folder,
        file_tag=file_tag,
        return_diagnostics=True,
    )
    processing_diagnostics["calibration_scope"] = {
        "coefficient_fit_count": 1,
        "shared_across_echoes": True,
        "echo_count": int(cfg["Necho"]),
        "source": "integrated_refscan_projection_calibration",
        "echo_specific_component": "sequence_theoretical_trajectory",
    }
    processing_diagnostics["requested_fit_kx_range"] = (
        None
        if fit_kx_min is None
        else [int(fit_kx_min), int(fit_kx_max)]
    )
    if psf_plot:
        plt.figure(figsize=(7, 4))
        plt.plot(a_fit, label="a(t)")
        plt.plot(b_fit, label="b(t)")
        plt.plot(c_fit, label="c(t)")
        selected_range = processing_diagnostics.get("kx_range")
        if selected_range is not None:
            plt.axvspan(
                selected_range[0],
                selected_range[1],
                alpha=0.12,
                label=f"{processing_diagnostics['fit_range_selection']} fit region",
            )
        plt.axvline(len(a_fit) // 2, linestyle="--", color="k")
        plt.axhline(0, linestyle="--", color="k")
        plt.title(
            "Integrated PSF coefficient processing: "
            f"{coefficient_processing}"
        )
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
    outputs = (
        torch.stack(psf_calib_echoes, dim=0),
        torch.stack(psf_theory_echoes, dim=0),
    )
    if return_diagnostics:
        return (*outputs, processing_diagnostics)
    return outputs

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


# GRE_RECON_UPDATE_2026_07_30
def _first_finite_definition(defs: Mapping[str, Any], *keys: str) -> float | None:
    """Return the first finite scalar definition without adding metadata defaults."""
    for key in keys:
        if key not in defs:
            continue
        try:
            value = float(defs[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _derive_nifti_voxel_size_mm(cfg: Mapping[str, Any]) -> tuple[float, float, float]:
    """Convert the TRA sequence spacing from metres to millimetres exactly once."""
    spacing = np.asarray(cfg["res_xyz_m"], dtype=float) * 1e3
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid NIfTI voxel size derived from .seq: {spacing.tolist()} mm")
    if np.any(spacing < 0.05) or np.any(spacing > 20.0):
        raise ValueError(
            "Implausible NIfTI voxel size in millimetres: "
            f"{spacing.tolist()}. This commonly indicates an m/mm/um conversion error."
        )
    return tuple(float(v) for v in spacing)


def _coerce_twix_fov_mm(raw_value, expected_mm):
    """Choose the TWIX FOV interpretation that best matches the sequence FOV."""
    if raw_value is None:
        return None, "missing"
    raw = float(raw_value)
    candidates = ((raw, "raw-as-mm"), (raw * 1e3, "raw-as-m-converted-to-mm"))
    value, interpretation = min(
        candidates,
        key=lambda item: abs(item[0] - expected_mm) / max(abs(expected_mm), 1e-12),
    )
    return float(value), interpretation


def _direction_patient_string(vector_ras):
    """Describe the positive direction of a RAS vector as an anatomical arrow."""
    if vector_ras is None:
        return "unknown"
    vector = np.asarray(vector_ras, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
        return "unknown"
    axis = int(np.argmax(np.abs(vector)))
    positive = vector[axis] >= 0
    if axis == 0:
        return "L->R" if positive else "R->L"
    if axis == 1:
        return "P->A" if positive else "A->P"
    return "I->S" if positive else "S->I"


def _report_seq_twix_geometry(
    *,
    twix_file: Path,
    cfg: Mapping[str, Any],
    received_image_shape: Sequence[int],
    voxel_size_mm: Sequence[float],
    twix_array_axis_roles: Sequence[str],
    twix_array_axis_flips: Sequence[bool],
    twix_coord_system: str,
    twix_inplane_rot_sign: float,
) -> dict[str, Any]:
    """Print warning-only TRA sequence/TWIX geometry and PE diagnostics."""
    from utils.nifti_export_twix import make_nifti_affine_from_twix

    expected_shape = (int(cfg["Nx"]), int(cfg["Ny"]), int(cfg["Nz"]))
    try:
        _, _, twix_info = make_nifti_affine_from_twix(
            twix_file=twix_file,
            npy_shape=expected_shape,
            twix_array_axis_roles=twix_array_axis_roles,
            twix_array_axis_flips=(False, False, False),
            twix_coord_system=twix_coord_system,
            twix_inplane_rot_sign=twix_inplane_rot_sign,
            twix_use_fov_for_voxel_size=False,
            voxel_size_mm=voxel_size_mm,
        )
    except Exception as exc:
        message = f"Unable to read TWIX geometry ({type(exc).__name__}: {exc})"
        print("Sequence/TWIX geometry diagnostics (warning-only)")
        print(f"  WARNING: {message}")
        print("  Reconstruction will continue.")
        return {
            "Status": "warning",
            "Passed": False,
            "SequenceOrientation": "TRA",
            "Error": message,
            "Directions": {
                "ReadoutPhysicalAxis": "x",
                "LINPhaseEncodingPhysicalAxis": "y",
                "PARPhaseEncodingPhysicalAxis": "z",
            },
        }

    expected_fov_mm = {
        "readout": float(cfg["FOVxyz_m"][0]) * 1e3,
        "phase": float(cfg["FOVxyz_m"][1]) * 1e3,
        "slice": float(cfg["FOVxyz_m"][2]) * 1e3,
    }
    raw_fov = twix_info.get("FOV", {})
    fov_checks = {}
    passed = True
    for role in ("readout", "phase", "slice"):
        observed_mm, interpretation = _coerce_twix_fov_mm(
            raw_fov.get(role), expected_fov_mm[role]
        )
        match = observed_mm is not None and np.isclose(
            observed_mm, expected_fov_mm[role], rtol=0.01, atol=0.5
        )
        passed = passed and bool(match)
        fov_checks[role] = {
            "SequenceMm": expected_fov_mm[role],
            "TwixRaw": None if raw_fov.get(role) is None else float(raw_fov[role]),
            "TwixInterpretedMm": observed_mm,
            "TwixUnitInterpretation": interpretation,
            "Match": bool(match),
        }

    received = tuple(int(v) for v in received_image_shape)
    matrix_checks = {
        "ExpectedReadoutOversampled": int(cfg["Nx_os"]),
        "ExpectedLINMeasured": int(cfg["Ny_meas"]),
        "ExpectedPARMeasured": int(cfg["Nz_meas"]),
        "ExpectedLINLabelExtent": int(cfg["Ny_label_extent"]),
        "ExpectedPARLabelExtent": int(cfg["Nz_label_extent"]),
        "ExpectedEchoCount": int(cfg["Necho"]),
        "ReceivedReadoutSamples": received[0],
        "ReceivedLINExtent": received[1],
        "ReceivedPARExtent": received[2],
        "ReceivedEchoCount": received[3],
        "ReadoutSamplesMatch": received[0] == int(cfg["Nx_os"]),
        "LINExtentMatch": received[1] == int(cfg["Ny_label_extent"]),
        "PARExtentMatch": received[2] == int(cfg["Nz_label_extent"]),
        "EchoCountMatch": received[3] == int(cfg["Necho"]),
    }
    passed = passed and all(
        matrix_checks[key]
        for key in (
            "ReadoutSamplesMatch",
            "LINExtentMatch",
            "PARExtentMatch",
            "EchoCountMatch",
        )
    )

    normal_ras = np.asarray(twix_info.get("NormalRAS", [np.nan, np.nan, np.nan]), dtype=float)
    tra_normal_match = (
        normal_ras.shape == (3,)
        and np.all(np.isfinite(normal_ras))
        and int(np.argmax(np.abs(normal_ras))) == 2
    )
    passed = passed and bool(tra_normal_match)

    direction_by_role = {
        "readout": twix_info.get("ReadoutDirectionRAS"),
        "phase": twix_info.get("PhaseDirectionRAS"),
        "slice": twix_info.get("SliceDirectionRAS"),
    }
    stored_by_role = {
        role: None if vector is None else np.asarray(vector, dtype=float)
        for role, vector in direction_by_role.items()
    }
    for axis, role in enumerate(twix_array_axis_roles):
        if bool(twix_array_axis_flips[axis]) and stored_by_role[role] is not None:
            stored_by_role[role] = -stored_by_role[role]
    axis_for_role = {role: axis for axis, role in enumerate(twix_array_axis_roles)}
    directions = {
        "ReadoutPhysicalAxis": "x",
        "LINPhaseEncodingPhysicalAxis": "y",
        "PARPhaseEncodingPhysicalAxis": "z",
        "ReadoutNIfTIAxisIndex": int(axis_for_role["readout"]),
        "LINPhaseEncodingNIfTIAxisIndex": int(axis_for_role["phase"]),
        "PARPhaseEncodingNIfTIAxisIndex": int(axis_for_role["slice"]),
        "ReadoutDirectionPatient": _direction_patient_string(direction_by_role["readout"]),
        "LINPhaseEncodingDirectionPatient": _direction_patient_string(direction_by_role["phase"]),
        "PARPhaseEncodingDirectionPatient": _direction_patient_string(direction_by_role["slice"]),
        "ReadoutStoredPositiveDirectionPatient": _direction_patient_string(stored_by_role["readout"]),
        "LINStoredPositiveDirectionPatient": _direction_patient_string(stored_by_role["phase"]),
        "PARStoredPositiveDirectionPatient": _direction_patient_string(stored_by_role["slice"]),
        "ReadoutDirectionRAS": direction_by_role["readout"],
        "LINPhaseEncodingDirectionRAS": direction_by_role["phase"],
        "PARPhaseEncodingDirectionRAS": direction_by_role["slice"],
    }

    print("Sequence/TWIX geometry diagnostics (warning-only)")
    print(f"  Orientation: .seq=TRA, TWIX transverse-normal match={tra_normal_match}")
    for role, check in fov_checks.items():
        status = "MATCH" if check["Match"] else "WARNING"
        print(
            f"  FOV {role:7s}: seq={check['SequenceMm']:g} mm, "
            f"twix={check['TwixInterpretedMm']} mm "
            f"({check['TwixUnitInterpretation']}) [{status}]"
        )
    print(
        "  Matrix: "
        f"received RO_os/LIN/PAR/Echo={received[:4]}, "
        f"expected extents=({cfg['Nx_os']}, {cfg['Ny_label_extent']}, "
        f"{cfg['Nz_label_extent']}, {cfg['Necho']}); measured LIN/PAR counts="
        f"({cfg['Ny_meas']}, {cfg['Nz_meas']})"
    )
    print(f"  Readout direction: {directions['ReadoutDirectionPatient']}")
    print(f"  LIN phase-encoding direction: {directions['LINPhaseEncodingDirectionPatient']}")
    print(f"  PAR phase-encoding direction: {directions['PARPhaseEncodingDirectionPatient']}")
    if not passed:
        print("  WARNING: one or more .seq/TWIX geometry checks did not match.")
        print("  Reconstruction will continue.")
    else:
        print("  Overall geometry status: MATCH")

    return {
        "Status": "match" if passed else "warning",
        "Passed": bool(passed),
        "SequenceOrientation": "TRA",
        "TwixTransverseNormalMatch": bool(tra_normal_match),
        "SequenceFOVMmXYZ": [float(v) * 1e3 for v in cfg["FOVxyz_m"]],
        "SequenceVoxelSizeMmXYZ": [float(v) for v in voxel_size_mm],
        "FOVChecks": fov_checks,
        "MatrixChecks": matrix_checks,
        "Directions": directions,
    }

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
    voxel_size_mm: Sequence[float] | None = None,
    geometry_diagnostics: Mapping[str, Any] | None = None,
    psf_coefficient_processing: str | None = None,
    psf_fit_kx_range: tuple[int | None, int | None] | None = None,
    psf_processing_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata without feeding sidecar values back into reconstruction.

    Args:
        cfg: Validated GRE sequence configuration.
        mode: Resolved reconstruction mode.
        twix_file: Source Siemens TWIX path.
        seq_file: Matching Pulseq sequence path.
        echo_idx: Optional zero-based echo index.
        voxel_size_mm: Optional output voxel spacing override.
        geometry_diagnostics: Optional sequence/TWIX geometry comparison.
        psf_coefficient_processing: Backward-compatible processing mode.
        psf_fit_kx_range: Backward-compatible manual fit interval.
        psf_processing_diagnostics: Full shared PSF fit provenance.

    Returns:
        JSON-compatible GRE reconstruction metadata.
    """
    defs = cfg["defs"]
    voxel_size_mm = (
        tuple(float(v) for v in voxel_size_mm)
        if voxel_size_mm is not None
        else tuple(float(v) * 1e3 for v in cfg["res_xyz_m"])
    )
    metadata: dict[str, Any] = {
        "Modality": "MR",
        "MRAcquisitionType": "3D",
        "SequenceType": "3D multi-echo Wave-GRE with integrated FLASH calibration",
        "SequenceName": cfg["sequence_name"],
        "SourceTwix": twix_file.name,
        "SourcePulseq": seq_file.name,
        "ReconstructionMode": mode,
        "OrientationMapping": cfg["orientation"],
        "ReadoutPhysicalAxis": "x",
        "LINPhaseEncodingPhysicalAxis": "y",
        "PARPhaseEncodingPhysicalAxis": "z",
        "MatrixSizeXYZ": [cfg["Nx"], cfg["Ny"], cfg["Nz"]],
        "MeasuredMatrixLINPAR": [cfg["Ny_meas"], cfg["Nz_meas"]],
        "ReadoutOversampledSize": cfg["Nx_os"],
        "ReadoutOversamplingFactor": cfg["os_factor"],
        "FOVMetersXYZ": cfg["FOVxyz_m"],
        "FOVMillimetersXYZ": [float(v) * 1e3 for v in cfg["FOVxyz_m"]],
        "VoxelSizeMillimetersXYZ": list(voxel_size_mm),
        "NIfTIVoxelSizeSource": "Pulseq FOV/matrix converted from m to mm",
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

    # Metadata-only sequence lookups. No values here alter reconstruction.
    tr_s = _first_finite_definition(defs, "TR")
    if tr_s is not None:
        metadata["RepetitionTime"] = tr_s
        metadata["RepetitionTimeUnits"] = "s"
    flip_angle_deg = _first_finite_definition(defs, "FlipAngle")
    if flip_angle_deg is not None:
        metadata["FlipAngle"] = flip_angle_deg
        metadata["FlipAngleUnits"] = "degree"
    readout_duration_s = _first_finite_definition(defs, "ReadoutDuration")
    if readout_duration_s is not None:
        metadata["ReadoutDuration"] = readout_duration_s
        metadata["ReadoutDurationUnits"] = "s"
    calibration_te_s = _first_finite_definition(defs, "CalibrationTE")
    if calibration_te_s is not None:
        metadata["CalibrationEchoTime"] = calibration_te_s
        metadata["CalibrationEchoTimeUnits"] = "s"
    calibration_tr_s = _first_finite_definition(defs, "CalibrationTR")
    if calibration_tr_s is not None:
        metadata["CalibrationRepetitionTime"] = calibration_tr_s
        metadata["CalibrationRepetitionTimeUnits"] = "s"

    optional_scalar_keys = {
        "ReadoutPolarity": "ReadoutPolarity",
        "WaveSinChannel": "WaveSinChannel",
        "WaveCosChannel": "WaveCosChannel",
        "WaveAmplitude_mTm": "WaveAmplitudeMilliteslaPerMeter",
        "WaveSlew_Tms": "WaveSlewTeslaPerMeterPerSecond",
        "WaveCycles": "WaveCycles",
        "SliceOversampling": "SliceOversampling",
        "PhaseResolution": "PhaseResolution",
        "PartitionResolution": "PartitionResolution",
        "UseFullInitialFC": "UseFullInitialFlowCompensation",
        "UseFullInterEchoFC": "UseFullInterEchoFlowCompensation",
        "CalibrationNSets": "CalibrationSetCount",
        "CalibrationACSSetID": "CalibrationACSSetID",
    }
    for seq_key, meta_key in optional_scalar_keys.items():
        if seq_key in defs:
            metadata[meta_key] = _json_safe(defs[seq_key])

    target_fov = defs.get("TargetFOV")
    if target_fov is not None:
        target = np.asarray(target_fov, dtype=float).reshape(-1)
        if target.size >= 3 and np.all(np.isfinite(target[:3])):
            metadata["TargetFOVMetersXYZ"] = target[:3].tolist()
            metadata["TargetFOVMillimetersXYZ"] = (target[:3] * 1e3).tolist()

    if geometry_diagnostics is not None:
        metadata["GeometryDiagnostics"] = geometry_diagnostics
        metadata["PhaseEncodingDirections"] = geometry_diagnostics.get("Directions", {})

    if mode == "wave" and psf_processing_diagnostics is not None:
        metadata["PSFCoefficientProcessing"] = str(
            psf_processing_diagnostics.get("coefficient_processing", "unknown")
        )
        metadata["PSFCoefficientProcessingDiagnostics"] = _json_safe(
            psf_processing_diagnostics
        )
    elif mode == "wave" and psf_coefficient_processing is not None:
        metadata["PSFCoefficientProcessing"] = str(psf_coefficient_processing)
        if psf_coefficient_processing == "sine-line" and psf_fit_kx_range is not None:
            metadata["PSFFitKxRange"] = [
                int(psf_fit_kx_range[0]),
                int(psf_fit_kx_range[1]),
            ]
            metadata["PSFFitKxRangeConvention"] = "half-open [min, max)"
            metadata["PSFFitModel"] = "A*sin(w*kx+phi)+C1*kx+C2"

    if echo_idx is not None:
        metadata["EchoNumber"] = int(echo_idx + 1)
        metadata["EchoTime"] = float(cfg["TE_s"][echo_idx])
        metadata["EchoTimeUnits"] = "s"
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
    voxel_size_mm: Sequence[float],
    magnitude_normalization_scale: float,
    crop_readout_os: int | None = None,
) -> list[tuple[Path, Path]]:
    from utils.nifti_export_twix import (
        apply_array_axis_flips,
        crop_readout_oversampling,
        make_nifti_affine_from_twix,
        normalize_magnitude,
        prepare_image_array,
        save_nifti_with_json,
    )

    img_np = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    if img_np.ndim != 3:
        raise ValueError(f"Expected a 3D echo image for NIfTI export, got {img_np.shape}.")
    crop_factor = (
        int(cfg["os_factor"])
        if crop_readout_os is None
        else int(crop_readout_os)
    )
    img_crop = crop_readout_oversampling(
        img_np,
        crop_readout_os=crop_factor,
    )
    readout_processing = (
        "after readout-oversampling crop"
        if crop_factor > 1
        else "without an additional readout crop"
    )

    magnitude = prepare_image_array(img_crop, part="mag")
    magnitude, magnitude_normalization = normalize_magnitude(
        magnitude,
        percentile=99.0,
        scale=magnitude_normalization_scale,
    )

    print(
        f"NIfTI echo {echo_idx + 1} magnitude normalization: "
        f"input p99={magnitude_normalization['InputPercentileValue']:.6g}, "
        f"shared scale={magnitude_normalization['NormalizationScale']:.6g}, "
        f"output p99={magnitude_normalization['OutputPercentileValue']:.6g} "
        "(no clipping)"
    )

    outputs: list[tuple[str, np.ndarray]] = [
        ("mag", magnitude)
    ]
    if save_phase:
        outputs.append(
            ("phase", prepare_image_array(img_crop, part="phase"))
        )
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
        voxel_size_mm=tuple(float(v) for v in voxel_size_mm),
    )
    out_folder.mkdir(parents=True, exist_ok=True)
    base = f"sub-{nifti_sub}_echo-{echo_idx + 1:02d}_acq-{mode}"
    saved: list[tuple[Path, Path]] = []

    for part, arr in outputs:
        nii_path = out_folder / f"{base}_part-{part}_{suffix}.nii.gz"
        json_path = out_folder / f"{base}_part-{part}_{suffix}.json"

        # Create the per-file sidecar before adding part-specific metadata.
        sidecar = dict(metadata)
        sidecar.update(
            {
                "ImagePart": part,
                "SavedVoxelSizeMillimeters": list(voxel_size_affine),
                "TwixGeometry": twix_info,
                "TwixArrayAxisRoles": list(twix_array_axis_roles),
                "AppliedArrayAxisFlips": [
                    bool(v) for v in twix_array_axis_flips
                ],
                "NIfTIVoxelSizeSource": (
                    "TWIX FOV divided by saved image matrix"
                    if twix_use_fov_for_voxel_size
                    else "Pulseq FOV/matrix converted from m to mm"
                ),
            }
        )

        if part == "phase":
            sidecar["Units"] = "rad"
            sidecar["ImageProcessing"] = (
                f"angle(complex_image), {readout_processing}"
            )
        else:
            sidecar["Units"] = "relative"
            sidecar["MagnitudeNormalization"] = {
                **magnitude_normalization,
                "SharedAcrossEchoes": True,
                "ReferenceEchoNumber": 1,
            }
            sidecar["ImageProcessing"] = (
                f"abs(complex_image), {readout_processing}; "
                "divided by the positive-finite 99th-percentile magnitude "
                "of echo 1; the same scale is applied to every echo; "
                "values are not clipped"
            )

        expected_saved_spacing = (
            tuple(float(v) for v in voxel_size_affine)
            if twix_use_fov_for_voxel_size
            else tuple(float(v) for v in voxel_size_mm)
        )

        saved.append(
            save_nifti_with_json(
                arr,
                affine,
                nii_path,
                json_path,
                sidecar,
                expected_voxel_size_mm=expected_saved_spacing,
            )
        )

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
    if runtime["save_bart_inputs"] and mode != "wave":
        raise ValueError("--save-bart-inputs requires a wave acquisition.")
    _print_sequence_summary(cfg, detected_mode=mode)
    nifti_voxel_size_mm = _derive_nifti_voxel_size_mm(cfg)
    print(
        "  NIfTI voxel size from .seq: "
        f"{nifti_voxel_size_mm[0]:g} x {nifti_voxel_size_mm[1]:g} x "
        f"{nifti_voxel_size_mm[2]:g} mm"
    )
    if runtime["validate_only"]:
        print("Sequence validation completed successfully.")
        return 0

    print("Importing GRE image data from integrated TWIX file...")
    img = _normalize_gre_image_data(load_img(str(runtime["twix_file"])), cfg)
    ncoil = int(img.shape[-1])
    print(f"Normalized image shape: {tuple(img.shape)}")
    geometry_diagnostics = _report_seq_twix_geometry(
        twix_file=runtime["twix_file"],
        cfg=cfg,
        received_image_shape=tuple(int(v) for v in img.shape),
        voxel_size_mm=nifti_voxel_size_mm,
        twix_array_axis_roles=runtime["nifti_axis_roles"],
        twix_array_axis_flips=runtime["nifti_axis_flips"],
        twix_coord_system=runtime["twix_coord_system"],
        twix_inplane_rot_sign=runtime["twix_inplane_rot_sign"],
    )

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
        espirit_crop=runtime["espirit_crop"],
        espirit_calib_mode=runtime["espirit_calib_mode"],
        espirit_cpu_workers=runtime["espirit_cpu_workers"],
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
    psf_processing_diagnostics: dict[str, Any] | None = None
    if mode == "wave":
        print("Generating echo-specific calibrated PSFs from integrated calibration...")
        (
            psf_calib_echoes,
            psf_theory_echoes,
            psf_processing_diagnostics,
        ) = generate_calibrated_psfs(
            twix_file=runtime["twix_file"],
            image_lines=image_lines,
            calib_lines=calib_lines,
            cfg=cfg,
            out_folder=runtime["out_folder"],
            file_tag=runtime["file_tag"],
            coefficient_processing=runtime["psf_coefficient_processing"],
            fit_kx_min=runtime["psf_fit_kx_min"],
            fit_kx_max=runtime["psf_fit_kx_max"],
            return_diagnostics=True,
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
        if runtime["save_bart_inputs"]:
            bart_folder = runtime["out_folder"] / (
                "bart_inputs" + _cache_suffix(runtime["file_tag"])
            )
            kspace_calib = _build_bart_calibration_kspace(
                twix_file=runtime["twix_file"],
                cfg=cfg,
                wcc=wcc,
            )
            manifest_path = export_wave_inputs(
                bart_folder,
                wave_kspace=kspace_cc.numpy(),
                calibrated_psf=psf_calib_echoes.numpy(),
                coil_sens=csm_full,
                kspace_calib=kspace_calib,
                psf_calibration=psf_processing_diagnostics,
                coil_calibration={
                    "source": "integrated refscan set 4",
                    "readout_oversampling_removal": {
                        **COIL_CALIBRATION_READOUT_OVERSAMPLING_REMOVAL,
                        "input_readout": int(cfg["Nx_os"]),
                        "output_readout": int(cfg["Nx"]),
                        "oversampling_factor": int(cfg["os_factor"]),
                    },
                    "kspace_calib_shape": list(kspace_calib.shape),
                    "finite": bool(np.isfinite(kspace_calib).all()),
                },
            )
            print(f"Saved BART Wave-CAIPI inputs: {manifest_path}")
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
        voxel_size_mm=nifti_voxel_size_mm,
        geometry_diagnostics=geometry_diagnostics,
        psf_coefficient_processing=(
            runtime["psf_coefficient_processing"] if mode == "wave" else None
        ),
        psf_fit_kx_range=(runtime["psf_fit_kx_min"], runtime["psf_fit_kx_max"]),
        psf_processing_diagnostics=psf_processing_diagnostics,
    )
    metadata_path = image_path.with_suffix(".json")
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved reconstruction metadata: {metadata_path}")

    # Calculate one shared magnitude-normalization scale after reconstruction.
    # Echo 1 defines the scale, and the same scale is used for every echo so
    # relative inter-echo signal differences are preserved.
    shared_nifti_magnitude_scale: float | None = None

    if runtime["save_nifti"]:
        from utils.nifti_export_twix import (
            crop_readout_oversampling,
            normalize_magnitude,
            prepare_image_array,
        )

        first_echo_np = images[:, :, :, 0].detach().cpu().numpy()
        first_echo_crop = crop_readout_oversampling(
            first_echo_np,
            crop_readout_os=int(cfg["os_factor"]),
        )
        first_echo_magnitude = prepare_image_array(
            first_echo_crop,
            part="mag",
        )

        _, reference_normalization = normalize_magnitude(
            first_echo_magnitude,
            percentile=99.0,
        )
        shared_nifti_magnitude_scale = float(
            reference_normalization["NormalizationScale"]
        )

        print(
            "Shared GRE NIfTI magnitude normalization: "
            f"echo 1 positive-voxel p99="
            f"{shared_nifti_magnitude_scale:.6g} -> 1.0; "
            "the same scale will be applied to all echoes."
        )

    for echo_idx in range(int(cfg["Necho"])):
        echo_image = images[:, :, :, echo_idx]
        if runtime["save_echo_npy"]:
            _save_complex_npy(
                runtime["out_folder"]
                / f"image_cg_integrated_calib_{stem}_echo{echo_idx + 1:02d}.npy",
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
                voxel_size_mm=nifti_voxel_size_mm,
                geometry_diagnostics=geometry_diagnostics,
                psf_coefficient_processing=(
                    runtime["psf_coefficient_processing"] if mode == "wave" else None
                ),
                psf_fit_kx_range=(runtime["psf_fit_kx_min"], runtime["psf_fit_kx_max"]),
                psf_processing_diagnostics=psf_processing_diagnostics,
            )
            if shared_nifti_magnitude_scale is None:
                raise RuntimeError(
                    "Shared NIfTI magnitude normalization scale was not initialized."
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
                voxel_size_mm=nifti_voxel_size_mm,
                magnitude_normalization_scale=shared_nifti_magnitude_scale,
            )
    print("Reconstruction completed successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
