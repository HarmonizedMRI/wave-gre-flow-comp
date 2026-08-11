"""BART CFL export helpers for Wave-CAIPI reconstruction inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_cfl(path: str | Path, array: np.ndarray) -> Path:
    """Write a complex array as a BART ``.hdr``/``.cfl`` pair.

    BART stores arrays in column-major order. ``path`` is the shared basename;
    a supplied ``.hdr`` or ``.cfl`` suffix is removed before writing.
    """

    base = Path(path)
    if base.suffix in {".hdr", ".cfl"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(array, dtype=np.complex64)
    if data.ndim < 1:
        raise ValueError("BART CFL output must have at least one dimension.")
    if any(int(size) < 1 for size in data.shape):
        raise ValueError(f"BART CFL output contains an empty dimension: {data.shape}.")

    header_path = base.with_suffix(".hdr")
    data_path = base.with_suffix(".cfl")
    header_path.write_text(
        "# Dimensions\n" + " ".join(str(int(size)) for size in data.shape) + "\n",
        encoding="utf-8",
    )
    with data_path.open("wb") as stream:
        np.ravel(data, order="F").tofile(stream)
    return base


def _complex64(name: str, array: Any, ndim: int) -> np.ndarray:
    result = np.asarray(array, dtype=np.complex64)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D; received shape {result.shape}.")
    if any(int(size) < 1 for size in result.shape):
        raise ValueError(f"{name} contains an empty dimension: {result.shape}.")
    return result


def export_wave_inputs(
    out_folder: str | Path,
    *,
    wave_kspace: np.ndarray,
    calibrated_psf: np.ndarray,
    coil_sens: np.ndarray,
    kspace_calib: np.ndarray,
) -> Path:
    """Export all inputs expected by BART's ``wave`` and ``ecalib`` tools.

    Input conventions before conversion are the reconstruction's native ones:

    * ``wave_kspace``: ``(wx, sy, sz, echo, coil)``
    * ``calibrated_psf``: ``(echo, wx, sy, sz)``
    * ``coil_sens``: ``(coil, sx, sy, sz)``
    * ``kspace_calib``: ``(sx, sy, sz, coil)``

    Single-echo acquisitions use the canonical ``wave_kspace`` and ``psf``
    basenames. Multi-echo acquisitions receive matching ``_echo-NN`` suffixes.
    """

    destination = Path(out_folder)
    destination.mkdir(parents=True, exist_ok=True)

    kspace = _complex64("wave_kspace", wave_kspace, 5)
    psf = _complex64("calibrated_psf", calibrated_psf, 4)
    maps = _complex64("coil_sens", coil_sens, 4)
    calib = _complex64("kspace_calib", kspace_calib, 4)

    wx, sy, sz, necho, nc = map(int, kspace.shape)
    if psf.shape != (necho, wx, sy, sz):
        raise ValueError(
            "calibrated_psf shape must be (echo, wx, sy, sz); "
            f"expected {(necho, wx, sy, sz)}, received {psf.shape}."
        )
    if maps.shape[0] != nc:
        raise ValueError(
            f"coil_sens has {maps.shape[0]} coils, but wave_kspace has {nc}."
        )
    sx = int(maps.shape[1])
    if maps.shape[2:] != (sy, sz):
        raise ValueError(
            "coil_sens spatial phase dimensions must match wave_kspace; "
            f"received {maps.shape[2:]} and {(sy, sz)}."
        )
    if calib.shape != (sx, sy, sz, nc):
        raise ValueError(
            "kspace_calib shape must be (sx, sy, sz, coil); "
            f"expected {(sx, sy, sz, nc)}, received {calib.shape}."
        )

    exported_maps = np.moveaxis(maps, 0, 3)[..., None]
    write_cfl(destination / "coil_sens", exported_maps)
    write_cfl(destination / "kspace_calib", calib)

    files: list[dict[str, Any]] = []
    for echo_index in range(necho):
        suffix = "" if necho == 1 else f"_echo-{echo_index + 1:02d}"
        kspace_name = f"wave_kspace{suffix}"
        psf_name = f"psf{suffix}"
        exported_kspace = kspace[:, :, :, echo_index, :, None]
        exported_psf = psf[echo_index, :, :, :, None, None]
        write_cfl(destination / kspace_name, exported_kspace)
        write_cfl(destination / psf_name, exported_psf)
        files.append(
            {
                "echo": echo_index + 1,
                "wave_kspace": kspace_name,
                "wave_kspace_shape": list(exported_kspace.shape),
                "psf": psf_name,
                "psf_shape": list(exported_psf.shape),
            }
        )

    manifest = {
        "format": "BART CFL",
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "coil_sens": "coil_sens",
        "coil_sens_shape": list(exported_maps.shape),
        "kspace_calib": "kspace_calib",
        "kspace_calib_shape": list(calib.shape),
        "echoes": files,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
