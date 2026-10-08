#!/usr/bin/env python3
"""Convert BART sensitivity maps to a lightweight logical-axis RSS NIfTI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np


def _cfl_base(path: str | Path) -> Path:
    """Return a BART basename after removing an optional CFL/HDR suffix."""
    value = Path(path).expanduser().resolve()
    return value.with_suffix("") if value.suffix in {".cfl", ".hdr"} else value


def _read_bart_dimensions(header: Path) -> tuple[int, ...]:
    """Read BART dimensions and remove only trailing singleton axes."""
    dimension_line = next(
        (
            line.strip()
            for line in header.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        None,
    )
    if dimension_line is None:
        raise ValueError(f"BART header contains no dimensions: {header}")
    try:
        dimensions = [int(value) for value in dimension_line.split()]
    except ValueError as exc:
        raise ValueError(
            f"Invalid BART dimensions in {header}: {dimension_line!r}"
        ) from exc
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError(f"BART dimensions must be positive: {dimensions}")
    while len(dimensions) > 1 and dimensions[-1] == 1:
        dimensions.pop()
    return tuple(dimensions)


def compute_csm_rss(csm: str | Path, *, partition_chunk: int = 4) -> np.ndarray:
    """Compute coil RSS from a BART CSM pair with bounded working memory.

    Args:
        csm: BART sensitivity-map basename or its ``.hdr``/``.cfl`` path.
        partition_chunk: Number of partition planes processed per chunk.

    Returns:
        Float32 RSS volume in logical ``(RO, LIN, PAR)`` order.

    Raises:
        ValueError: If dimensions, byte count, chunk size, or values are invalid.
        FileNotFoundError: If either member of the BART pair is missing.
    """
    if partition_chunk < 1:
        raise ValueError("partition_chunk must be positive.")
    base = _cfl_base(csm)
    header = base.with_suffix(".hdr")
    data_file = base.with_suffix(".cfl")
    if not header.is_file() or not data_file.is_file():
        raise FileNotFoundError(f"Missing BART CFL pair: {base}.{{hdr,cfl}}")

    dimensions = _read_bart_dimensions(header)
    if len(dimensions) != 4:
        raise ValueError(
            "Expected BART sensitivity maps in (RO, LIN, PAR, coil) order; "
            f"received dimensions {dimensions}."
        )
    expected_bytes = (
        int(np.prod(dimensions, dtype=np.int64)) * np.dtype(np.complex64).itemsize
    )
    actual_bytes = data_file.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"CFL size mismatch for {data_file}: expected {expected_bytes} bytes, "
            f"found {actual_bytes}."
        )

    maps = np.memmap(
        data_file,
        dtype=np.complex64,
        mode="r",
        shape=dimensions,
        order="F",
    )
    rss = np.empty(dimensions[:3], dtype=np.float32)
    for start in range(0, dimensions[2], partition_chunk):
        stop = min(start + partition_chunk, dimensions[2])
        power = np.zeros(
            (dimensions[0], dimensions[1], stop - start), dtype=np.float32
        )
        for coil in range(dimensions[3]):
            block = np.asarray(maps[:, :, start:stop, coil])
            power += block.real * block.real + block.imag * block.imag
        np.sqrt(power, out=rss[:, :, start:stop])
    if not np.all(np.isfinite(rss)):
        raise ValueError("CSM RSS contains non-finite values.")
    return rss


def save_logical_csm_rss(
    *,
    csm: str | Path,
    output: str | Path,
    voxel_size_mm: Sequence[float] = (1.0, 1.0, 1.0),
    partition_chunk: int = 4,
) -> Path:
    """Save BART coil-map RSS without requiring acquisition geometry.

    Args:
        csm: BART sensitivity-map basename or CFL/HDR path.
        output: Exact output ``.nii`` or ``.nii.gz`` path.
        voxel_size_mm: Synthetic spacing for logical RO, LIN, and PAR axes.
        partition_chunk: Number of partition planes processed per CSM chunk.

    Returns:
        Resolved path of the newly written NIfTI.

    Raises:
        FileExistsError: If the output already exists.
        ValueError: If the output suffix or synthetic voxel sizes are invalid.

    Side Effects:
        Writes one float32 NIfTI. Its diagonal affine is synthetic and does not
        encode scanner or anatomical orientation.
    """
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    if not output_path.name.endswith(".nii.gz") and output_path.suffix != ".nii":
        raise ValueError("output must end in .nii or .nii.gz")

    spacing = np.asarray(tuple(voxel_size_mm), dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("voxel_size_mm must contain three positive finite values.")

    rss = compute_csm_rss(csm, partition_chunk=partition_chunk)
    affine = np.diag((*spacing.tolist(), 1.0))
    image = nib.Nifti1Image(rss, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_xyzt_units("mm")
    image.header["descrip"] = (
        b"BART CSM RSS; synthetic logical RO/LIN/PAR affine"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, output_path)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the lightweight conversion command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert BART coil sensitivities to an RSS NIfTI in unmodified "
            "logical (RO, LIN, PAR) axis order. The affine is synthetic and "
            "must not be interpreted as anatomical orientation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csm", required=True, help="BART CSM basename or CFL/HDR path.")
    parser.add_argument("--out", required=True, help="Exact .nii or .nii.gz output path.")
    parser.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("RO_MM", "LIN_MM", "PAR_MM"),
        help="Synthetic logical-axis voxel sizes in millimeters.",
    )
    parser.add_argument(
        "--partition-chunk",
        type=int,
        default=4,
        help="Partition planes processed at once while calculating RSS.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run lightweight CSM conversion and report the synthetic geometry."""
    args = _build_parser().parse_args(argv)
    output = save_logical_csm_rss(
        csm=args.csm,
        output=args.out,
        voxel_size_mm=args.voxel_size,
        partition_chunk=args.partition_chunk,
    )
    saved = nib.load(output)
    print(f"Saved: {output}")
    print(f"Shape in logical (RO, LIN, PAR) order: {saved.shape}")
    print(f"Synthetic voxel size [mm]: {saved.header.get_zooms()[:3]}")
    print("Warning: the affine is synthetic and is not anatomy-orientation accurate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
