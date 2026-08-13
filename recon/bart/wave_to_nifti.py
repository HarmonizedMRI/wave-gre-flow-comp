#!/usr/bin/env python3
"""Convert per-echo BART Wave-CAIPI CFL images to geometry-correct NIfTI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

RECON_ROOT = Path(__file__).resolve().parents[1]
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from bart.bart_utils.bart_io import read_cfl


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def discover_bart_echoes(input_dir: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    """Resolve matched manifest, k-space, and reconstructed image files."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    manifest_path = input_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"BART input manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("echoes")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"BART manifest contains no echo entries: {manifest_path}")

    resolved: list[dict[str, Any]] = []
    for expected_echo, entry in enumerate(entries, start=1):
        echo = int(entry.get("echo", -1))
        if echo != expected_echo:
            raise ValueError(
                f"BART manifest echoes must be consecutive from 1; "
                f"expected {expected_echo}, found {echo}."
            )
        kspace_name = str(entry.get("wave_kspace", ""))
        expected_name = (
            "wave_kspace"
            if len(entries) == 1
            else f"wave_kspace_echo-{echo:02d}"
        )
        if kspace_name != expected_name:
            raise ValueError(
                f"Invalid wave_kspace basename for echo {echo}: expected "
                f"{expected_name!r}, found {kspace_name!r}."
            )
        suffix = kspace_name[len("wave_kspace") :]
        image_name = f"image_wave{suffix}"
        kspace_base = input_path / kspace_name
        image_base = output_path / image_name
        for base in (kspace_base, image_base):
            if not base.with_suffix(".hdr").is_file() or not base.with_suffix(".cfl").is_file():
                raise FileNotFoundError(f"Missing BART CFL pair: {base}.{{hdr,cfl}}")
        resolved.append(
            {
                "echo": echo,
                "wave_kspace": kspace_base,
                "image": image_base,
            }
        )
    return resolved


def restore_bart_intensity(image: np.ndarray, wave_kspace: np.ndarray) -> tuple[np.ndarray, float]:
    """Undo the per-input k-space normalization performed by ``bart wave``."""

    scale = float(np.linalg.norm(np.asarray(wave_kspace, dtype=np.complex64)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"BART wave k-space norm must be positive and finite; got {scale}.")
    return np.asarray(image, dtype=np.complex64) * scale, scale


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert BART Wave-GRE images to NIfTI using matching TWIX/Pulseq geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bart-input-dir", required=True)
    parser.add_argument("--bart-output-dir", required=True)
    parser.add_argument("--twix", required=True, help="Matching Siemens TWIX .dat file.")
    parser.add_argument("--seq", required=True, help="Matching Pulseq .seq file.")
    parser.add_argument("--out", required=True, help="NIfTI output directory.")
    parser.add_argument("--save-phase", action="store_true")
    parser.add_argument("--nifti-sub", default=None)
    parser.add_argument("--nifti-suffix", default="GRE")
    parser.add_argument(
        "--nifti-axis-roles",
        nargs=3,
        default=("readout", "phase", "slice"),
        metavar=("AXIS0", "AXIS1", "AXIS2"),
    )
    parser.add_argument(
        "--nifti-axis-flips",
        nargs=3,
        type=_parse_bool,
        default=(False, True, False),
        metavar=("FLIP0", "FLIP1", "FLIP2"),
    )
    parser.add_argument("--twix-coord-system", choices=("LPS", "RAS"), default="LPS")
    parser.add_argument("--twix-inplane-rot-sign", type=float, default=-1.0)
    parser.add_argument("--twix-use-fov-for-voxel-size", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_dir = Path(args.bart_input_dir).expanduser().resolve()
    output_dir = Path(args.bart_output_dir).expanduser().resolve()
    twix_file = Path(args.twix).expanduser().resolve()
    seq_file = Path(args.seq).expanduser().resolve()
    nifti_out = Path(args.out).expanduser().resolve()
    for path, label in ((twix_file, "TWIX"), (seq_file, "Pulseq")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    import recon_wave_gre_from_twix_integrated_nifti as native
    from utils.nifti_export_twix import normalize_magnitude, prepare_image_array

    seq = native._load_sequence(seq_file)
    cfg = native._derive_gre_config(seq, yflip_override=None, zflip_override=None)
    echo_files = discover_bart_echoes(input_dir, output_dir)
    if len(echo_files) != int(cfg["Necho"]):
        raise ValueError(
            f"BART manifest contains {len(echo_files)} echo(es), but the Pulseq "
            f"sequence defines {cfg['Necho']}."
        )

    expected_shape = (int(cfg["Nx"]), int(cfg["Ny"]), int(cfg["Nz"]))
    restored_images: list[np.ndarray] = []
    kspace_norms: list[float] = []
    for entry in echo_files:
        image = read_cfl(entry["image"])
        if image.ndim != 3:
            raise ValueError(
                f"BART image for echo {entry['echo']} must reduce to one 3D map; "
                f"got {image.shape}. Multiple ESPIRiT map sets are not supported."
            )
        if image.shape != expected_shape:
            raise ValueError(
                f"BART image for echo {entry['echo']} has shape {image.shape}; "
                f"expected logical image shape {expected_shape}."
            )
        restored, norm = restore_bart_intensity(image, read_cfl(entry["wave_kspace"]))
        restored_images.append(restored)
        kspace_norms.append(norm)

    voxel_size_mm = native._derive_nifti_voxel_size_mm(cfg)
    diagnostic_cfg = dict(cfg)
    diagnostic_cfg["Nx_os"] = int(cfg["Nx"])
    diagnostic_cfg["os_factor"] = 1
    geometry_diagnostics = native._report_seq_twix_geometry(
        twix_file=twix_file,
        cfg=diagnostic_cfg,
        received_image_shape=(*expected_shape, len(restored_images)),
        voxel_size_mm=voxel_size_mm,
        twix_array_axis_roles=tuple(args.nifti_axis_roles),
        twix_array_axis_flips=tuple(args.nifti_axis_flips),
        twix_coord_system=args.twix_coord_system,
        twix_inplane_rot_sign=float(args.twix_inplane_rot_sign),
    )
    geometry_diagnostics["BARTOutputLogicalShape"] = list(expected_shape)
    geometry_diagnostics["BARTReadoutAlreadyDeoversampled"] = True

    _, normalization = normalize_magnitude(
        prepare_image_array(restored_images[0], part="mag"),
        percentile=99.0,
    )
    shared_scale = float(normalization["NormalizationScale"])
    nifti_sub = native._sanitize_token(args.nifti_sub or twix_file.stem)
    nifti_suffix = native._sanitize_token(args.nifti_suffix)
    for echo_idx, (entry, image, kspace_norm) in enumerate(
        zip(echo_files, restored_images, kspace_norms)
    ):
        metadata = native._build_gre_metadata(
            cfg=cfg,
            mode="wave",
            twix_file=twix_file,
            seq_file=seq_file,
            echo_idx=echo_idx,
            voxel_size_mm=voxel_size_mm,
            geometry_diagnostics=geometry_diagnostics,
        )
        metadata.update(
            {
                "ReconstructionSoftware": "BART wave",
                "BARTImageInput": entry["image"].name,
                "BARTWaveKspaceInput": entry["wave_kspace"].name,
                "BARTWaveKspaceNormRestored": kspace_norm,
                "BARTInternalNormalizationRestored": True,
                "BARTOutputAlreadyReadoutDeoversampled": True,
            }
        )
        native.save_gre_echo_to_nifti(
            image=image,
            twix_file=twix_file,
            out_folder=nifti_out,
            nifti_sub=nifti_sub,
            suffix=nifti_suffix,
            mode="wave",
            echo_idx=echo_idx,
            cfg=cfg,
            save_phase=bool(args.save_phase),
            twix_array_axis_roles=tuple(args.nifti_axis_roles),
            twix_array_axis_flips=tuple(args.nifti_axis_flips),
            twix_coord_system=args.twix_coord_system,
            twix_inplane_rot_sign=float(args.twix_inplane_rot_sign),
            twix_use_fov_for_voxel_size=bool(args.twix_use_fov_for_voxel_size),
            metadata=metadata,
            voxel_size_mm=voxel_size_mm,
            magnitude_normalization_scale=shared_scale,
            crop_readout_os=1,
        )
    print(f"Converted {len(restored_images)} BART echo image(s) to {nifti_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
