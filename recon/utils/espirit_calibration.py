"""Encapsulated ESPIRiT calibration backends for Wave-GRE reconstruction.

The public :func:`estimate_espirit_maps` entry point supports:

``3d``
    Native SigPy 3D ESPIRiT calibration. This remains the reference/default
    method and can run on either CPU or GPU.
``slice2d``
    CPU-only hybrid-space calibration. The logical readout axis is transformed
    to image space, independent 2D ESPIRiT calibrations are run across the two
    phase-encoding dimensions, and the maps are stacked along logical readout.

The slice-wise implementation lives outside the main reconstruction script so
worker scheduling, axis handling, and map assembly remain implementation
details rather than being mixed into the GRE reconstruction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import sigpy as sp
import sigpy.mri as mr
from joblib import Parallel, cpu_count, delayed, parallel_config
from scipy.ndimage import binary_closing, binary_dilation, label


@dataclass(frozen=True)
class EspiritCalibrationInfo:
    """Execution details returned with an ESPIRiT map estimate."""

    mode: str
    cpu_workers: Optional[int]
    logical_ro_slices: int
    zero_input_slices: tuple[int, ...]


def estimate_espirit_maps(
    kspace: np.ndarray,
    *,
    mode: str = "3d",
    device: Optional[sp.Device] = None,
    crop: float = 0.8,
    calib_width: int = 24,
    thresh: float = 0.02,
    kernel_width: int = 6,
    max_iter: int = 100,
    cpu_workers: Optional[int] = None,
) -> tuple[np.ndarray, EspiritCalibrationInfo]:
    """Estimate coil-sensitivity maps using a selected ESPIRiT backend.

    Parameters
    ----------
    kspace
        Coil-first logical k-space with shape ``(coil, RO, LIN, PAR)``.
        Readout oversampling must already have been removed.
    mode
        ``"3d"`` for native 3D SigPy ESPIRiT or ``"slice2d"`` for parallel
        hybrid-space 2D ESPIRiT along logical RO.
    device
        SigPy device used by the 3D backend. The slice2d backend is CPU-only.
    crop, calib_width, thresh, kernel_width, max_iter
        Parameters passed directly to ``sigpy.mri.app.EspiritCalib``.
    cpu_workers
        Number of process workers for slice2d. ``None`` selects the available
        physical-core count, limited by the number of logical RO slices.
    """

    kspace = np.asarray(kspace, dtype=np.complex64)
    if kspace.ndim != 4:
        raise ValueError(
            "ESPIRiT input must have shape (coil, RO, LIN, PAR); "
            f"received {kspace.shape}."
        )
    if any(int(size) < 1 for size in kspace.shape):
        raise ValueError(f"ESPIRiT input contains an empty dimension: {kspace.shape}.")

    mode = str(mode).strip().lower()
    if mode not in ("3d", "slice2d"):
        raise ValueError("ESPIRiT calibration mode must be '3d' or 'slice2d'.")

    crop = float(crop)
    if not np.isfinite(crop) or not 0.0 <= crop <= 1.0:
        raise ValueError("ESPIRiT crop must be a finite value between 0 and 1.")
    calib_width = _positive_int("calib_width", calib_width)
    kernel_width = _positive_int("kernel_width", kernel_width)
    max_iter = _positive_int("max_iter", max_iter)
    thresh = float(thresh)
    if not np.isfinite(thresh) or thresh < 0.0:
        raise ValueError("ESPIRiT thresh must be a finite non-negative value.")

    if mode == "3d":
        maps = _estimate_3d(
            kspace,
            device=sp.Device(-1) if device is None else device,
            crop=crop,
            calib_width=calib_width,
            thresh=thresh,
            kernel_width=kernel_width,
            max_iter=max_iter,
        )
        return maps, EspiritCalibrationInfo(
            mode="3d",
            cpu_workers=None,
            logical_ro_slices=int(kspace.shape[1]),
            zero_input_slices=(),
        )

    if device is not None and int(device.id) != -1:
        raise ValueError(
            "slice2d ESPIRiT is CPU-only. Use a CPU SigPy device or select "
            "the 3d calibration mode for GPU execution."
        )

    return _estimate_slice2d(
        kspace,
        crop=crop,
        calib_width=calib_width,
        thresh=thresh,
        kernel_width=kernel_width,
        max_iter=max_iter,
        cpu_workers=cpu_workers,
    )


def _estimate_3d(
    kspace: np.ndarray,
    *,
    device: sp.Device,
    crop: float,
    calib_width: int,
    thresh: float,
    kernel_width: int,
    max_iter: int,
) -> np.ndarray:
    """Run native 3D SigPy ESPIRiT calibration."""

    kspace_device = sp.to_device(kspace, device)
    maps_device = mr.app.EspiritCalib(
        kspace_device,
        calib_width=calib_width,
        thresh=thresh,
        kernel_width=kernel_width,
        crop=crop,
        max_iter=max_iter,
        device=device,
        show_pbar=True,
    ).run()
    maps = np.asarray(sp.to_device(maps_device, sp.Device(-1)), dtype=np.complex64)
    _validate_output(maps, expected_shape=kspace.shape, label="3D ESPIRiT")
    return maps


def _estimate_slice2d(
    kspace: np.ndarray,
    *,
    crop: float,
    calib_width: int,
    thresh: float,
    kernel_width: int,
    max_iter: int,
    cpu_workers: Optional[int],
) -> tuple[np.ndarray, EspiritCalibrationInfo]:
    """Run parallel 2D ESPIRiT over logical-readout hybrid-space slices."""

    # Input ordering is (coil, RO, LIN, PAR). GRE readout oversampling has
    # already been removed by the caller before this transform.
    hybrid = sp.ifft(kspace, axes=(1,))
    hybrid = np.ascontiguousarray(hybrid, dtype=np.complex64)
    nro = int(hybrid.shape[1])
    active_ro, ro_slice_rms, ro_threshold = _detect_active_ro_slices(
        hybrid,
        noise_fraction=0.15,
        noise_multiplier=5.0,
        relative_floor=1e-3,
        padding_slices=2,
    )

    inactive_indices = np.flatnonzero(~active_ro)

    print(
        "slice2d RO support detection: "
        f"{int(np.count_nonzero(active_ro))}/{nro} active slices, "
        f"threshold={ro_threshold:.6g}, "
        f"peak RMS={float(np.max(ro_slice_rms)):.6g}."
    )

    if inactive_indices.size:
        print(
            "Skipping low-signal logical-RO slices: "
            + ", ".join(str(int(index)) for index in inactive_indices)
        )
    
    workers = _resolve_worker_count(cpu_workers, nro)

    print(
        "ESPIRiT slice2d backend: "
        f"{nro} logical-RO slices, {workers} CPU process worker(s)."
    )
    print(
        "Each worker calibrates one (coil, LIN, PAR) hybrid-space plane; "
        "native BLAS threads are limited to one per worker."
    )

    with parallel_config(
        backend="loky",
        n_jobs=workers,
        inner_max_num_threads=1,
    ):
        results = Parallel(batch_size=1, verbose=10)(
            delayed(_calibrate_single_ro_slice)(
                ro_index,
                hybrid[:, ro_index, :, :],
                active=bool(active_ro[ro_index]),
                crop=crop,
                calib_width=calib_width,
                thresh=thresh,
                kernel_width=kernel_width,
                max_iter=max_iter,
            )
            for ro_index in range(nro)
        )

    # Joblib preserves input order. Stacking on axis 1 restores
    # (coil, RO, LIN, PAR) without manual indexed assignment.
    maps = np.stack([result[1] for result in results], axis=1)
    maps = np.asarray(maps, dtype=np.complex64)

    # Enforce exact zeros outside the detected RO support.
    maps[:, ~active_ro, :, :] = 0
    _validate_output(maps, expected_shape=kspace.shape, label="slice2d ESPIRiT")

    zero_slices = tuple(result[0] for result in results if result[2])
    if zero_slices:
        print(
            "slice2d ESPIRiT skipped exactly-zero logical-RO planes: "
            + ", ".join(str(index) for index in zero_slices)
        )

    return maps, EspiritCalibrationInfo(
        mode="slice2d",
        cpu_workers=workers,
        logical_ro_slices=nro,
        zero_input_slices=zero_slices,
    )


def _calibrate_single_ro_slice(
    ro_index: int,
    kspace_slice: np.ndarray,
    *,
    active: bool,
    crop: float,
    calib_width: int,
    thresh: float,
    kernel_width: int,
    max_iter: int,
) -> tuple[int, np.ndarray, bool]:
    """Calibrate one logical-RO hybrid-space plane in a worker process."""

    kspace_slice = np.ascontiguousarray(
        kspace_slice,
        dtype=np.complex64,
    )

    if not active or not np.any(kspace_slice):
        return ro_index, np.zeros_like(kspace_slice), True

    maps = mr.app.EspiritCalib(
        kspace_slice,
        calib_width=calib_width,
        thresh=thresh,
        kernel_width=kernel_width,
        crop=crop,
        max_iter=max_iter,
        device=sp.Device(-1),
        show_pbar=False,
    ).run()
    maps = np.asarray(maps, dtype=np.complex64)
    _validate_output(
        maps,
        expected_shape=kspace_slice.shape,
        label=f"slice2d ESPIRiT RO slice {ro_index}",
    )
    return ro_index, maps, False


def _resolve_worker_count(requested: Optional[int], task_count: int) -> int:
    """Resolve automatic or requested workers without a hard-coded count."""

    task_count = _positive_int("task_count", task_count)
    available = max(1, int(cpu_count(only_physical_cores=True)))

    if requested is None:
        return min(available, task_count)

    requested = _positive_int("cpu_workers", requested)
    if requested > available:
        print(
            f"Requested {requested} ESPIRiT CPU workers, but joblib reports "
            f"{available} available physical core(s); using {available}."
        )
    return min(requested, available, task_count)


def _validate_output(
    maps: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    label: str,
) -> None:
    """Validate map shape and numerical finiteness."""

    if tuple(maps.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"{label} returned shape {maps.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(maps)):
        raise FloatingPointError(f"{label} produced non-finite sensitivity-map values.")


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _detect_active_ro_slices(
    hybrid: np.ndarray,
    *,
    noise_fraction: float = 0.15,
    noise_multiplier: float = 5.0,
    relative_floor: float = 1e-3,
    padding_slices: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Detect logical-RO planes containing calibration signal.

    Parameters
    ----------
    hybrid
        Hybrid-space calibration data with shape (coil, RO, LIN, PAR).
    noise_fraction
        Fraction of lowest-energy RO planes used to estimate the noise floor.
    noise_multiplier
        Required slice RMS relative to the estimated noise floor.
    relative_floor
        Additional threshold relative to the maximum slice RMS.
    padding_slices
        Number of RO slices added to both sides of the detected support.
    """

    if hybrid.ndim != 4:
        raise ValueError(
            "Expected hybrid calibration shape (coil, RO, LIN, PAR), "
            f"got {hybrid.shape}."
        )

    # By Parseval's theorem, summing energy over kLIN/kPAR is equivalent
    # to summing over the corresponding image-space plane.
    slice_rms = np.sqrt(
        np.mean(
            np.abs(hybrid) ** 2,
            axis=(0, 2, 3),
            dtype=np.float64,
        )
    )

    nro = int(slice_rms.size)
    if nro == 0:
        raise ValueError("Hybrid calibration contains no RO slices.")

    noise_count = max(4, int(np.ceil(noise_fraction * nro)))
    noise_count = min(noise_count, nro)

    lowest = np.partition(slice_rms, noise_count - 1)[:noise_count]
    noise_floor = float(np.median(lowest))
    peak = float(np.max(slice_rms))

    threshold = max(
        noise_multiplier * noise_floor,
        relative_floor * peak,
    )

    active = slice_rms > threshold

    if not np.any(active):
        print(
            "WARNING: automatic RO support detection found no active slices; "
            "disabling the support guard for this calibration."
        )
        return np.ones(nro, dtype=bool), slice_rms, threshold

    # Fill one-slice holes inside the object support.
    active = binary_closing(
        active,
        structure=np.ones(3, dtype=bool),
    )

    # Retain the strongest contiguous RO component. For a head acquisition,
    # the object should form one contiguous R-to-L support interval.
    labels, component_count = label(active)

    if component_count > 1:
        component_scores = [
            float(np.sum(slice_rms[labels == component]))
            for component in range(1, component_count + 1)
        ]
        strongest_component = int(np.argmax(component_scores)) + 1
        active = labels == strongest_component

    if padding_slices > 0:
        active = binary_dilation(
            active,
            structure=np.ones(3, dtype=bool),
            iterations=padding_slices,
        )

    return np.asarray(active, dtype=bool), slice_rms, threshold