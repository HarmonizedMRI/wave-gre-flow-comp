"""Generic Twix-based NIfTI export utilities.

These helpers are intentionally sequence-agnostic. Sequence-specific naming,
metadata, and reconstruction-array conventions should live in the calling
reconstruction script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _normalize(v: Sequence[float], name: str = "vector") -> np.ndarray:
    """Return a unit vector and raise if the vector has zero norm."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError(f"{name} has zero norm.")
    return v / n


def _twix_get(yaps: Any, key: tuple[str, ...], default: Any = None) -> Any:
    """Robust MeasYaps getter for mapVBVD-style nested tuple keys."""
    try:
        val = yaps.get(key, default)
    except Exception:
        try:
            val = yaps[key]
        except Exception:
            val = default

    if val is None:
        return default
    return val


def _get_twix_scan_and_hdr(twix_file: str | Path | None = None, twix_obj: Any = None, scan_index: int = -1):
    """Return scan, hdr, and MeasYaps from a Twix filename or mapVBVD object."""
    if twix_obj is None:
        if twix_file is None:
            raise ValueError("Either twix_file or twix_obj must be provided.")
        try:
            import mapvbvd
        except ImportError as exc:
            raise ImportError(
                "Saving NIfTI from Twix orientation requires the Python package 'mapvbvd'."
            ) from exc
        twix_obj = mapvbvd.mapVBVD(str(twix_file))

    if isinstance(twix_obj, (list, tuple)):
        scan = twix_obj[scan_index]
    else:
        scan = twix_obj

    try:
        hdr = scan["hdr"]
    except Exception:
        hdr = scan.hdr

    yaps = hdr["MeasYaps"]
    return scan, hdr, yaps


def _sct_to_ras(v_sct: Sequence[float], twix_coord_system: str = "LPS") -> np.ndarray:
    """Convert Siemens Sag/Cor/Tra vector to NIfTI RAS coordinates."""
    v_sct = np.asarray(v_sct, dtype=float)

    if twix_coord_system.upper() == "LPS":
        # +Sag = Left, +Cor = Posterior, +Tra = Superior.
        return np.array([-v_sct[0], -v_sct[1], v_sct[2]], dtype=float)
    if twix_coord_system.upper() == "RAS":
        return np.array([v_sct[0], v_sct[1], v_sct[2]], dtype=float)
    raise ValueError("twix_coord_system must be 'LPS' or 'RAS'.")


def _make_inplane_basis_from_normal_ras(
    normal_ras: Sequence[float],
    inplane_rot: float = 0.0,
    inplane_rot_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build readout, phase, and slice directions from a slice normal.

    The output is right-handed: readout x phase = slice. This practical
    Siemens-style construction uses R as the default readout reference for
    non-sagittal planes and A as the default readout reference for sagittal-like
    planes. The in-plane rotation sign is exposed because Siemens/Twix
    conventions can differ between pipelines.
    """
    n = _normalize(normal_ras, "slice normal RAS")

    R = np.array([1.0, 0.0, 0.0])
    A = np.array([0.0, 1.0, 0.0])

    dominant_axis = int(np.argmax(np.abs(n)))
    ref = A if dominant_axis == 0 else R

    read0 = ref - n * np.dot(ref, n)
    if np.linalg.norm(read0) < 1e-8:
        ref = A if np.allclose(ref, R) else R
        read0 = ref - n * np.dot(ref, n)

    read0 = _normalize(read0, "unrotated readout direction")
    phase0 = _normalize(np.cross(n, read0), "unrotated phase direction")

    phi = float(inplane_rot_sign) * float(inplane_rot)
    read = np.cos(phi) * read0 + np.sin(phi) * phase0
    phase = -np.sin(phi) * read0 + np.cos(phi) * phase0

    read = _normalize(read, "readout direction")
    phase = _normalize(phase, "phase direction")

    return read, phase, n


def make_nifti_affine_from_twix(
    twix_file: str | Path | None = None,
    twix_obj: Any = None,
    scan_index: int = -1,
    slice_index: int = 0,
    npy_shape: Sequence[int] | None = None,
    twix_array_axis_roles: Sequence[str] = ("readout", "phase", "slice"),
    twix_array_axis_flips: Sequence[bool] = (False, False, False),
    twix_coord_system: str = "LPS",
    twix_inplane_rot_sign: float = 1.0,
    twix_use_fov_for_voxel_size: bool = True,
    voxel_size_mm: Sequence[float] | None = None,
    twix_fov_override: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    """Build a NIfTI RAS affine from Siemens Twix MeasYaps geometry.

    Parameters
    ----------
    npy_shape
        Shape of the 3D image array being saved.
    twix_array_axis_roles
        Role of each image array axis; must contain one each of
        "readout", "phase", and "slice".
    twix_array_axis_flips
        Whether each image array axis is reversed relative to the Twix physical
        direction. If an axis is flipped, the affine origin is shifted so voxel
        physical locations remain correct.
    twix_use_fov_for_voxel_size
        If True, voxel sizes are inferred from Twix FOV / npy_shape. If False,
        voxel_size_mm is used directly.
    """
    if npy_shape is None:
        raise ValueError("npy_shape must be provided.")
    npy_shape = tuple(int(v) for v in npy_shape)
    if len(npy_shape) != 3:
        raise ValueError(f"npy_shape must have length 3, got {npy_shape}.")

    if len(twix_array_axis_roles) != 3:
        raise ValueError("twix_array_axis_roles must have length 3.")
    if len(twix_array_axis_flips) != 3:
        raise ValueError("twix_array_axis_flips must have length 3.")

    valid_roles = {"readout", "phase", "slice"}
    for role in twix_array_axis_roles:
        if role not in valid_roles:
            raise ValueError(f"Invalid Twix axis role '{role}'. Use {sorted(valid_roles)}.")
    if len(set(twix_array_axis_roles)) != 3:
        raise ValueError(
            "twix_array_axis_roles must contain each role once. "
            f"Got {tuple(twix_array_axis_roles)}"
        )

    _, hdr, yaps = _get_twix_scan_and_hdr(
        twix_file=twix_file,
        twix_obj=twix_obj,
        scan_index=scan_index,
    )

    Ns = str(slice_index)
    fov = {
        "readout": _twix_get(yaps, ("sSliceArray", "asSlice", Ns, "dReadoutFOV"), None),
        "phase": _twix_get(yaps, ("sSliceArray", "asSlice", Ns, "dPhaseFOV"), None),
        "slice": _twix_get(yaps, ("sSliceArray", "asSlice", Ns, "dThickness"), None),
    }
    if twix_fov_override is not None:
        fov.update(dict(twix_fov_override))

    normal_sct = np.array([
        _twix_get(yaps, ("sSliceArray", "asSlice", Ns, "sNormal", "dSag"), 0.0),
        _twix_get(yaps, ("sSliceArray", "asSlice", Ns, "sNormal", "dCor"), 0.0),
        _twix_get(yaps, ("sSliceArray", "asSlice", Ns, "sNormal", "dTra"), 0.0),
    ], dtype=float)

    position_sct = np.array([
        _twix_get(yaps, ("sSliceArray", "asSlice", Ns, "sPosition", "dSag"), 0.0),
        _twix_get(yaps, ("sSliceArray", "asSlice", Ns, "sPosition", "dCor"), 0.0),
        _twix_get(yaps, ("sSliceArray", "asSlice", Ns, "sPosition", "dTra"), 0.0),
    ], dtype=float)

    inplane_rot = (
        hdr.get("Meas", {}).get("dInPlaneRot")
        or hdr.get("Protocol", {}).get("dInPlaneRot")
        or _twix_get(yaps, ("sSliceArray", "asSlice", Ns, "dInPlaneRot"), 0.0)
        or 0.0
    )

    normal_ras = _normalize(_sct_to_ras(normal_sct, twix_coord_system), "Twix normal RAS")
    center_ras = _sct_to_ras(position_sct, twix_coord_system)

    read_dir, phase_dir, slice_dir = _make_inplane_basis_from_normal_ras(
        normal_ras,
        inplane_rot=float(inplane_rot),
        inplane_rot_sign=twix_inplane_rot_sign,
    )

    direction_by_role = {
        "readout": read_dir,
        "phase": phase_dir,
        "slice": slice_dir,
    }
    axis_for_role = {role: ax for ax, role in enumerate(twix_array_axis_roles)}

    if twix_use_fov_for_voxel_size:
        spacing_by_role = {}
        for role in ("readout", "phase", "slice"):
            if fov[role] is None:
                raise ValueError(
                    f"Twix FOV for role '{role}' is missing. Provide twix_fov_override, "
                    "or set twix_use_fov_for_voxel_size=False and provide voxel_size_mm."
                )
            ax = axis_for_role[role]
            spacing_by_role[role] = float(fov[role]) / float(npy_shape[ax])
    else:
        if voxel_size_mm is None:
            raise ValueError("voxel_size_mm must be provided when twix_use_fov_for_voxel_size=False.")
        voxel_size_mm = tuple(float(x) for x in voxel_size_mm)
        if len(voxel_size_mm) != 3:
            raise ValueError("voxel_size_mm must have length 3.")
        spacing_by_role = {}
        for ax, role in enumerate(twix_array_axis_roles):
            spacing_by_role[role] = voxel_size_mm[ax]

    role_vectors = {
        role: direction_by_role[role] * spacing_by_role[role]
        for role in ("readout", "phase", "slice")
    }

    affine = np.eye(4, dtype=float)
    origin_ras = center_ras.copy()

    for ax, role in enumerate(twix_array_axis_roles):
        v = role_vectors[role]
        origin_ras -= v * ((npy_shape[ax] - 1) / 2.0)

    affine[:3, 3] = origin_ras

    for ax, role in enumerate(twix_array_axis_roles):
        v = role_vectors[role].copy()
        if bool(twix_array_axis_flips[ax]):
            affine[:3, 3] += v * (npy_shape[ax] - 1)
            v = -v
        affine[:3, ax] = v

    try:
        import nibabel as nib
        voxel_size_mm_out = tuple(float(x) for x in nib.affines.voxel_sizes(affine))
    except ImportError:
        voxel_size_mm_out = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))

    twix_info = {
        "FOV": {k: None if v is None else float(v) for k, v in fov.items()},
        "NormalSagCorTra": normal_sct.tolist(),
        "PositionSagCorTra": position_sct.tolist(),
        "NormalRAS": normal_ras.tolist(),
        "CenterRAS": center_ras.tolist(),
        "InPlaneRotationRad": float(inplane_rot),
        "InPlaneRotationSignUsed": float(twix_inplane_rot_sign),
        "TwixCoordinateSystemAssumption": twix_coord_system,
        "TwixArrayAxisRoles": list(twix_array_axis_roles),
        "TwixArrayAxisFlips": [bool(x) for x in twix_array_axis_flips],
        "ReadoutDirectionRAS": read_dir.tolist(),
        "PhaseDirectionRAS": phase_dir.tolist(),
        "SliceDirectionRAS": slice_dir.tolist(),
    }

    return affine, voxel_size_mm_out, twix_info


def apply_array_axis_flips(images: Iterable[np.ndarray], axis_flips: Sequence[bool]) -> list[np.ndarray]:
    """Physically reverse selected stored array axes for all images."""
    if len(axis_flips) != 3:
        raise ValueError("axis_flips must contain exactly three booleans.")
    if not all(isinstance(x, (bool, np.bool_)) for x in axis_flips):
        raise TypeError("axis_flips must contain booleans, e.g. (False, True, False).")

    output = []
    for image in images:
        corrected = image
        for axis, should_flip in enumerate(axis_flips):
            if should_flip:
                corrected = np.flip(corrected, axis=axis)
        output.append(np.ascontiguousarray(corrected))
    return output


def crop_readout_oversampling(arr: np.ndarray, crop_readout_os: int | None = 1) -> np.ndarray:
    """Center-crop readout oversampling along axis 0."""
    arr = np.asarray(arr)
    crop_readout_os = 1 if crop_readout_os is None else int(crop_readout_os)
    if crop_readout_os <= 1:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D image before readout crop, got shape {arr.shape}.")
    nx_os = arr.shape[0]
    if nx_os % crop_readout_os != 0:
        raise ValueError(
            f"Readout dimension {nx_os} is not divisible by crop_readout_os={crop_readout_os}."
        )
    nx = nx_os // crop_readout_os
    start = nx_os // 2 - nx // 2
    stop = start + nx
    return np.ascontiguousarray(arr[start:stop, :, :])


def prepare_image_array(arr: np.ndarray, part: str = "mag") -> np.ndarray:
    """Convert a 3D complex/real image to magnitude or phase float32 data."""
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array with shape (Nx, Ny, Nz), got {arr.shape}")

    if part == "mag":
        out = np.abs(arr) if np.iscomplexobj(arr) else arr.astype(np.float32)
    elif part == "phase":
        out = np.angle(arr) if np.iscomplexobj(arr) else arr.astype(np.float32)
    else:
        raise ValueError("part must be either 'mag' or 'phase'.")

    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def clean_magnitude(
    arr: np.ndarray,
    percentile: float = 99.0,
    outlier_factor: float = 20.0,
    remove_mode: str = "zero",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert complex image to magnitude and remove extreme outlier pixels."""
    mag = np.abs(arr).astype(np.float32)
    mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

    finite_positive = mag[np.isfinite(mag) & (mag > 0)]
    if finite_positive.size == 0:
        raise ValueError("Magnitude image has no positive finite pixels.")

    p_bound = np.percentile(finite_positive, percentile)
    threshold = outlier_factor * p_bound
    outlier_mask = mag > threshold
    n_outliers = int(np.sum(outlier_mask))

    if remove_mode == "zero":
        mag[outlier_mask] = 0.0
    elif remove_mode == "clip":
        mag[outlier_mask] = threshold
    else:
        raise ValueError("remove_mode must be 'zero' or 'clip'.")

    info = {
        "percentile": float(percentile),
        "percentile_bound": float(p_bound),
        "outlier_factor": float(outlier_factor),
        "outlier_threshold": float(threshold),
        "n_outliers_removed": n_outliers,
        "remove_mode": remove_mode,
    }
    return mag, info


def prepare_phase(arr: np.ndarray, phase_mask: np.ndarray | None = None) -> np.ndarray:
    """Convert complex/real image to phase in radians."""
    phase = np.angle(arr).astype(np.float32) if np.iscomplexobj(arr) else np.asarray(arr, dtype=np.float32)
    phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)
    if phase_mask is not None:
        phase = phase.copy()
        phase[~phase_mask] = 0.0
    return phase


def save_nifti_with_json(
    image: np.ndarray,
    affine: np.ndarray,
    nii_path: str | Path,
    json_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save one 3D float image to NIfTI plus a JSON sidecar."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Saving NIfTI files requires the Python package 'nibabel'.") from exc

    nii_path = Path(nii_path)
    json_path = Path(json_path)
    nii_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    img = nib.Nifti1Image(np.asarray(image, dtype=np.float32), np.asarray(affine, dtype=float))
    img.header.set_xyzt_units(xyz="mm", t="sec")
    voxel_size_mm = tuple(float(x) for x in nib.affines.voxel_sizes(img.affine))
    img.header.set_zooms(voxel_size_mm)
    img.set_qform(img.affine, code=1)
    img.set_sform(img.affine, code=1)
    nib.save(img, str(nii_path))

    sidecar = dict(metadata or {})
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Saved NIfTI: {nii_path}")
    print(f"Saved JSON:  {json_path}")
    print(f"Saved shape: {img.shape}")
    print(f"Saved voxel size: {voxel_size_mm}")
    print(f"Saved orientation: {nib.aff2axcodes(img.affine)}")
    return nii_path, json_path


def print_twix_orientation_summary(affine: np.ndarray, twix_info: Mapping[str, Any]) -> None:
    """Print a compact Twix-derived affine/orientation summary."""
    try:
        import nibabel as nib
        print("Twix-derived NIfTI orientation:", nib.aff2axcodes(affine))
        print("Twix-derived voxel sizes:", nib.affines.voxel_sizes(affine))
    except ImportError:
        print("Twix-derived voxel sizes:", [np.linalg.norm(affine[:3, i]) for i in range(3)])
    print("Twix FOV:", twix_info.get("FOV"))
    print("Twix normal Sag/Cor/Tra:", twix_info.get("NormalSagCorTra"))
    print("Twix normal RAS:", twix_info.get("NormalRAS"))
    print("Twix center RAS:", twix_info.get("CenterRAS"))
    print("Twix in-plane rotation rad:", twix_info.get("InPlaneRotationRad"))
    print("Readout dir RAS:", twix_info.get("ReadoutDirectionRAS"))
    print("Phase dir RAS:", twix_info.get("PhaseDirectionRAS"))
    print("Slice dir RAS:", twix_info.get("SliceDirectionRAS"))
    print("Affine:")
    print(affine)
