"""BART CFL I/O and Wave-CAIPI input export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _cfl_base(path: str | Path) -> Path:
    base = Path(path)
    return base.with_suffix("") if base.suffix in {".hdr", ".cfl"} else base


def write_cfl(path: str | Path, array: np.ndarray) -> Path:
    """Write a complex array as a column-major BART ``.hdr``/``.cfl`` pair."""

    base = _cfl_base(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype=np.complex64)
    if data.ndim < 1 or any(int(size) < 1 for size in data.shape):
        raise ValueError(f"BART CFL output must have non-empty dimensions: {data.shape}.")
    base.with_suffix(".hdr").write_text(
        "# Dimensions\n" + " ".join(str(int(size)) for size in data.shape) + "\n",
        encoding="utf-8",
    )
    with base.with_suffix(".cfl").open("wb") as stream:
        np.ravel(data, order="F").tofile(stream)
    return base


def read_cfl(path: str | Path, *, trim_trailing_singletons: bool = True) -> np.ndarray:
    """Read a BART CFL pair, preserving axis order and validating byte count."""

    base = _cfl_base(path)
    header_path = base.with_suffix(".hdr")
    data_path = base.with_suffix(".cfl")
    if not header_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"Missing BART CFL pair: {base}.{{hdr,cfl}}")

    dimension_line = next(
        (
            line.strip()
            for line in header_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        None,
    )
    if dimension_line is None:
        raise ValueError(f"BART header contains no dimensions: {header_path}")
    try:
        shape = tuple(int(value) for value in dimension_line.split())
    except ValueError as exc:
        raise ValueError(f"Invalid BART dimensions in {header_path}: {dimension_line!r}") from exc
    if not shape or any(size < 1 for size in shape):
        raise ValueError(f"BART dimensions must be positive: {shape}")

    expected_elements = int(np.prod(shape, dtype=np.int64))
    data = np.fromfile(data_path, dtype=np.complex64)
    if data.size != expected_elements:
        raise ValueError(
            f"BART CFL size mismatch for {data_path}: header expects "
            f"{expected_elements} complex64 values, file contains {data.size}."
        )
    array = data.reshape(shape, order="F")
    if trim_trailing_singletons:
        trimmed_shape = list(array.shape)
        while len(trimmed_shape) > 1 and trimmed_shape[-1] == 1:
            trimmed_shape.pop()
        array = array.reshape(tuple(trimmed_shape), order="F")
    return array


def _cfl_pair_times(path: str | Path) -> tuple[int, int] | None:
    """Return earliest/latest pair mtimes after validating dimensions and size."""

    base = _cfl_base(path)
    header = base.with_suffix(".hdr")
    data = base.with_suffix(".cfl")
    if not header.is_file() or not data.is_file():
        return None
    dimension_line = next(
        (
            line.strip()
            for line in header.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        None,
    )
    if dimension_line is None:
        return None
    try:
        shape = tuple(int(value) for value in dimension_line.split())
    except ValueError:
        return None
    if not shape or any(size < 1 for size in shape):
        return None
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.complex64).itemsize
    if data.stat().st_size != expected_bytes:
        return None
    mtimes = (header.stat().st_mtime_ns, data.stat().st_mtime_ns)
    return min(mtimes), max(mtimes)


def bart_reconstruction_is_current(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    source_twix: str | Path,
    source_seq: str | Path,
) -> bool:
    """Return true when BART maps and every manifest echo are complete/current."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    twix_path = Path(source_twix).resolve()
    seq_path = Path(source_seq).resolve()
    try:
        manifest = json.loads(
            (input_path / "manifest.json").read_text(encoding="utf-8")
        )
        provenance = manifest.get("coil_calibration", {})
        if provenance.get("source_twix") != str(twix_path):
            return False
        if provenance.get("source_seq") != str(seq_path):
            return False
        source_latest = max(twix_path.stat().st_mtime_ns, seq_path.stat().st_mtime_ns)
        calib_times = _cfl_pair_times(input_path / "kspace_calib")
        maps_times = _cfl_pair_times(output_path / "coil_sens_bart")
        if calib_times is None or maps_times is None:
            return False
        if maps_times[0] < max(source_latest, calib_times[1]):
            return False
        echoes = manifest.get("echoes")
        if not isinstance(echoes, list) or not echoes:
            return False
        for entry in echoes:
            kspace_name = entry.get("wave_kspace")
            psf_name = entry.get("psf")
            if not isinstance(kspace_name, str) or not isinstance(psf_name, str):
                return False
            if not kspace_name.startswith("wave_kspace"):
                return False
            suffix = kspace_name[len("wave_kspace") :]
            kspace_times = _cfl_pair_times(input_path / kspace_name)
            psf_times = _cfl_pair_times(input_path / psf_name)
            image_times = _cfl_pair_times(output_path / f"image_wave{suffix}")
            if kspace_times is None or psf_times is None or image_times is None:
                return False
            newest_input = max(
                source_latest,
                maps_times[1],
                kspace_times[1],
                psf_times[1],
            )
            if image_times[0] < newest_input:
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _complex64(name: str, array: Any, ndim: int) -> np.ndarray:
    result = np.asarray(array, dtype=np.complex64)
    if result.ndim != ndim or any(int(size) < 1 for size in result.shape):
        raise ValueError(f"{name} must be a non-empty {ndim}D array; got {result.shape}.")
    return result


def export_wave_inputs(
    out_folder: str | Path,
    *,
    wave_kspace: np.ndarray,
    calibrated_psf: np.ndarray,
    coil_sens: np.ndarray | None,
    kspace_calib: np.ndarray,
    psf_calibration: dict[str, Any] | None = None,
    coil_calibration: dict[str, Any] | None = None,
) -> Path:
    """Export reconstruction-native arrays for BART ``ecalib`` and ``wave``.

    Args:
        out_folder: Destination directory for CFL pairs and their manifest.
        wave_kspace: Multi-echo Wave k-space in ``(RO, LIN, PAR, echo, coil)``.
        calibrated_psf: Per-echo calibrated PSFs in
            ``(echo, RO, LIN, PAR)``.
        coil_sens: Optional coil-first sensitivity maps. The default BART
            ecalib path omits these maps.
        kspace_calib: Coil-last ACS calibration k-space.
        psf_calibration: Optional JSON-compatible shared coefficient-fit
            provenance.
        coil_calibration: Optional JSON-compatible ACS and sensitivity-map
            calibration provenance.

    Returns:
        Path to the written JSON manifest.
    """

    destination = Path(out_folder)
    destination.mkdir(parents=True, exist_ok=True)
    kspace = _complex64("wave_kspace", wave_kspace, 5)
    psf = _complex64("calibrated_psf", calibrated_psf, 4)
    calib = _complex64("kspace_calib", kspace_calib, 4)

    wx, sy, sz, necho, nc = map(int, kspace.shape)
    if psf.shape != (necho, wx, sy, sz):
        raise ValueError(
            "calibrated_psf shape must be (echo, wx, sy, sz); "
            f"expected {(necho, wx, sy, sz)}, received {psf.shape}."
        )
    sx = int(calib.shape[0])
    if calib.shape != (sx, sy, sz, nc):
        raise ValueError(f"kspace_calib must have shape {(sx, sy, sz, nc)}; got {calib.shape}.")

    exported_maps = None
    if coil_sens is not None:
        maps = _complex64("coil_sens", coil_sens, 4)
        if maps.shape != (nc, sx, sy, sz):
            raise ValueError(
                f"coil_sens must have shape {(nc, sx, sy, sz)}; got {maps.shape}."
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
                "wave_kspace_norm": float(np.linalg.norm(exported_kspace)),
                "psf": psf_name,
                "psf_shape": list(exported_psf.shape),
            }
        )

    manifest = {
        "format": "BART CFL",
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "kspace_calib": "kspace_calib",
        "kspace_calib_shape": list(calib.shape),
        "echoes": files,
    }
    if exported_maps is not None:
        manifest["coil_sens"] = "coil_sens"
        manifest["coil_sens_shape"] = list(exported_maps.shape)
    if psf_calibration is not None:
        manifest["psf_calibration"] = psf_calibration
    if coil_calibration is not None:
        manifest["coil_calibration"] = coil_calibration
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
