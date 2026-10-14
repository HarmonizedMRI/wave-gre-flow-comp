"""
MRI k-space import helpers.

Assumed data conventions
------------------------
1. Raw k-space tensors are assumed to use coil-last layout:

       kspace.shape == (Nx, Ny, Nz, Ncoil)

   where:
       Nx    : readout dimension, possibly oversampled
       Ny    : first phase-encode dimension
       Nz    : second phase-encode / partition dimension
       Ncoil : physical receive channels, e.g. 32

Functions
---------
load_merged_img_ref(filename, Nx_os, Ny, Nz, ncoil, acs)
    Load Siemens Twix image and refscan data using mapVBVD, insert image data
    into a full k-space tensor, then overwrite the central ACS ky-kz block
    with REF scan data. Also returns the resulting 2D ky-kz sampling mask.

Typical workflow
----------------
1. Load and merge image + ACS/reference k-space:

       kspace_echo, mask_2d = load_merged_img_ref(...)

2. Estimate coil compression from the merged full k-space:
3. Build a low-resolution coil-first k-space for ESPIRiT:
4. Run ESPIRiT on the compressed low-resolution k-space:
5. Compress full-resolution k-space, if needed:

"""

import numpy as np
import mapvbvd
import torch


def _read_selected_dimensions(stream, selections, retained_dims):
    """Read selected mapVBVD dimensions without materializing unused axes.

    Args:
        stream: mapVBVD image or refscan stream.
        selections: Mapping from Siemens dimension names to indices or slices.
        retained_dims: Dimension names to preserve, in the requested order.

    Returns:
        NumPy array containing only the retained dimensions.

    Raises:
        ValueError: If a required dimension is absent.
        RuntimeError: If mapVBVD returns an unexpected array layout.
    """

    data_dims = list(stream.dataDims)
    missing = sorted(set(retained_dims).difference(data_dims))
    if missing:
        raise ValueError(f"TWIX stream is missing dimensions: {missing}")
    stream.squeeze = False
    selection = [selections.get(name, 0) for name in data_dims]
    values = np.asarray(stream[tuple(selection)])
    if values.ndim != len(data_dims):
        raise RuntimeError(
            "mapVBVD returned an unexpected rank for explicit selection: "
            f"shape={values.shape}, dimensions={data_dims}."
        )
    retained_axes = [data_dims.index(name) for name in retained_dims]
    trailing_axes = [axis for axis in range(values.ndim) if axis not in retained_axes]
    reordered = np.transpose(values, retained_axes + trailing_axes)
    retained_shape = tuple(int(values.shape[axis]) for axis in retained_axes)
    if int(np.prod(reordered.shape[len(retained_axes) :], dtype=np.int64)) != 1:
        raise RuntimeError(
            "Unselected TWIX dimensions were not singleton after selection: "
            f"shape={values.shape}, dimensions={data_dims}."
        )
    return reordered.reshape(retained_shape)


def load_ref_set(filename, set_index, *, lin_count=None, par_count=None):
    """Load one REFSCAN SET and optional leading LIN/PAR region.

    Args:
        filename: Siemens TWIX input path.
        set_index: Zero-based REFSCAN SET index.
        lin_count: Optional number of leading LIN samples to read.
        par_count: Optional number of leading PAR samples to read.

    Returns:
        Complex64 tensor in ``(Col, Lin, Par, coil)`` order.

    Raises:
        ValueError: If the REFSCAN stream or required dimensions are absent.
        IndexError: If the requested SET or region is outside the stream.
    """

    twix_obj = mapvbvd.mapVBVD(filename)
    tw = twix_obj[-1] if isinstance(twix_obj, list) else twix_obj
    ref_obj = tw.refscan
    if ref_obj is None:
        raise ValueError(f"Twix file has no REFSCAN stream: {filename}")
    ref_obj.flagRemoveOS = False
    data_dims = list(ref_obj.dataDims)
    required = {"Col", "Cha", "Lin", "Par", "Set"}
    missing = sorted(required.difference(data_dims))
    if missing:
        raise ValueError(f"REFSCAN stream is missing dimensions: {missing}")

    sizes = {name: int(ref_obj.fullSize[data_dims.index(name)]) for name in required}
    resolved_set = int(set_index)
    if resolved_set < 0:
        resolved_set += sizes["Set"]
    if not 0 <= resolved_set < sizes["Set"]:
        raise IndexError(
            f"set_index={set_index} is outside REFSCAN SET range "
            f"0...{sizes['Set'] - 1}"
        )
    lin_stop = sizes["Lin"] if lin_count is None else int(lin_count)
    par_stop = sizes["Par"] if par_count is None else int(par_count)
    if not 1 <= lin_stop <= sizes["Lin"] or not 1 <= par_stop <= sizes["Par"]:
        raise IndexError(
            "Requested REFSCAN LIN/PAR region exceeds stored dimensions: "
            f"requested={(lin_stop, par_stop)}, stored="
            f"{(sizes['Lin'], sizes['Par'])}."
        )
    ref = _read_selected_dimensions(
        ref_obj,
        {
            "Col": slice(None),
            "Cha": slice(None),
            "Lin": slice(0, lin_stop),
            "Par": slice(0, par_stop),
            "Set": resolved_set,
        },
        ("Col", "Cha", "Lin", "Par"),
    )
    coil_last = np.ascontiguousarray(np.transpose(ref, (0, 2, 3, 1)))
    return torch.from_numpy(coil_last).to(torch.complex64)


def load_img_coil_compressed(
    filename,
    compression_matrix,
    *,
    output_shape,
    par_chunk=1,
):
    """Read IMAGE data in PAR chunks and compress physical coils immediately.

    Args:
        filename: Siemens TWIX input path.
        compression_matrix: Physical-to-virtual coil matrix shaped
            ``(physical_coils, virtual_coils)``.
        output_shape: Target ``(Col, Lin, Par, Echo)`` shape before the virtual
            coil dimension. The target may pad stored LIN/PAR extents.
        par_chunk: Number of stored PAR indices to materialize per read.

    Returns:
        Tuple containing compressed complex64 data, the unpadded physical-coil
        image shape, and the physical coil count.

    Raises:
        ValueError: If dimensions, target shape, or compression basis mismatch.
    """

    target = tuple(int(value) for value in output_shape)
    if len(target) != 4 or any(value < 1 for value in target):
        raise ValueError(f"output_shape must contain four positive values: {target}")
    chunk = int(par_chunk)
    if chunk < 1:
        raise ValueError("par_chunk must be positive.")
    basis = np.asarray(compression_matrix, dtype=np.complex64)
    if basis.ndim != 2:
        raise ValueError("compression_matrix must be two-dimensional.")

    twix_obj = mapvbvd.mapVBVD(filename)
    tw = twix_obj[-1] if isinstance(twix_obj, list) else twix_obj
    image_obj = tw.image
    if image_obj is None:
        raise ValueError(f"Twix file has no IMAGE stream: {filename}")
    image_obj.flagRemoveOS = False
    data_dims = list(image_obj.dataDims)
    required = {"Col", "Cha", "Lin", "Par", "Eco"}
    missing = sorted(required.difference(data_dims))
    if missing:
        raise ValueError(f"IMAGE stream is missing dimensions: {missing}")
    sizes = {name: int(image_obj.fullSize[data_dims.index(name)]) for name in required}
    acquired_shape = (
        sizes["Col"],
        sizes["Lin"],
        sizes["Par"],
        sizes["Eco"],
        sizes["Cha"],
    )
    if basis.shape[0] != sizes["Cha"]:
        raise ValueError(
            "Compression basis physical-coil count does not match IMAGE data: "
            f"basis={basis.shape}, image_coils={sizes['Cha']}."
        )
    if (
        target[0] != sizes["Col"]
        or target[3] != sizes["Eco"]
        or any(target[index] < acquired_shape[index] for index in (1, 2))
    ):
        raise ValueError(
            f"Target image shape {target} cannot contain acquired shape "
            f"{acquired_shape[:-1]}."
        )

    output = torch.zeros((*target, int(basis.shape[1])), dtype=torch.complex64)
    if chunk == 1 and hasattr(image_obj, "Par"):
        skip_par = int(getattr(image_obj, "skipPar", 0))
        stored_par = np.unique(np.asarray(image_obj.Par, dtype=np.int64) - skip_par)
        par_ranges = [
            (int(index), int(index) + 1)
            for index in stored_par
            if 0 <= int(index) < sizes["Par"]
        ]
    else:
        par_ranges = [
            (start, min(start + chunk, sizes["Par"]))
            for start in range(0, sizes["Par"], chunk)
        ]
    if not par_ranges:
        raise ValueError("IMAGE stream contains no acquired PAR indices.")
    for echo_index in range(sizes["Eco"]):
        for par_start, par_stop in par_ranges:
            physical = _read_selected_dimensions(
                image_obj,
                {
                    "Col": slice(None),
                    "Cha": slice(None),
                    "Lin": slice(None),
                    "Par": slice(par_start, par_stop),
                    "Eco": echo_index,
                },
                ("Col", "Cha", "Lin", "Par"),
            )
            coil_last = np.ascontiguousarray(
                np.transpose(physical, (0, 2, 3, 1)), dtype=np.complex64
            )
            compressed = coil_last.reshape(-1, sizes["Cha"]) @ basis
            compressed = compressed.reshape(
                sizes["Col"], sizes["Lin"], par_stop - par_start, basis.shape[1]
            )
            output[
                :, : sizes["Lin"], par_start:par_stop, echo_index, :
            ] = torch.from_numpy(compressed)
    return output, acquired_shape, sizes["Cha"]

def load_img(filename):
    twixObj = mapvbvd.mapVBVD(filename)
    tw = twixObj[1] if isinstance(twixObj, list) else twixObj

    img_obj = tw.image
    img_obj.flagRemoveOS = False

    # mapVBVD order should be: Nx, coil, ky, kz
    img = np.asarray(img_obj['']).squeeze()
    if len(img.shape) == 4:
        # (Nx, Ny, Nz, Ncoil)
        kspace = torch.from_numpy(np.ascontiguousarray(img)).permute(0, 2, 3, 1).to(torch.cfloat)
    elif len(img.shape) == 5:
        # (Nx, Ny, Nz, Navg, Ncoil)
        kspace = torch.from_numpy(np.ascontiguousarray(img)).permute(0, 2, 3, 4, 1).to(torch.cfloat)
    else:
        raise RuntimeError('Please try manual import!')

    return kspace


def load_ref(filename):
    twixObj = mapvbvd.mapVBVD(filename)
    tw = twixObj[1] if isinstance(twixObj, list) else twixObj

    ref_obj = tw.refscan
    ref_obj.flagRemoveOS = False

    # mapVBVD order should be: Nx, coil, ky, kz
    ref = np.asarray(ref_obj['']).squeeze()
    if len(ref.shape) == 4:
        # (Nx, Ny, Nz, Ncoil)
        kspace = torch.from_numpy(np.ascontiguousarray(ref)).permute(0, 2, 3, 1).to(torch.cfloat)
    elif len(ref.shape) == 5:
        # (Nx, Ny, Nz, Navg, Ncoil)
        kspace = torch.from_numpy(np.ascontiguousarray(ref)).permute(0, 2, 3, 4, 1).to(torch.cfloat)
    else:
        raise RuntimeError('Please try manual import!')

    return kspace

# To-Do: Fix the mask
def load_merged_img_ref(filename, Nx_os, Ny, Nz, ncoil, acs):
    twixObj = mapvbvd.mapVBVD(filename)
    tw = twixObj[1] if isinstance(twixObj, list) else twixObj

    img_obj = tw.image
    ref_obj = tw.refscan

    img_obj.flagRemoveOS = False
    ref_obj.flagRemoveOS = False

    # mapVBVD sorted order usually: Nx, coil, ky, kz
    img = np.asarray(img_obj['']).squeeze()
    ref = np.asarray(ref_obj['']).squeeze()

    img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(0, 2, 3, 1).to(torch.cfloat)
    ref_t = torch.from_numpy(np.ascontiguousarray(ref)).permute(0, 2, 3, 1).to(torch.cfloat)

    kspace = torch.zeros((Nx_os, Ny, Nz, ncoil), dtype=torch.cfloat)

    # Fill image data
    kspace[:, :img_t.shape[1], :img_t.shape[2], :] = img_t

    # ACS center indices
    cy, cz = Ny // 2, Nz // 2
    y0, y1 = cy - acs // 2, cy + acs // 2
    z0, z1 = cz - acs // 2, cz + acs // 2

    # Overwrite center ACS with REF
    if ref_t.shape[1] == acs and ref_t.shape[2] == acs:
        # REF is compact ACS: Nx x acs x acs x coil
        kspace[:, y0:y1, z0:z1, :] = ref_t
    else:
        # REF is full-size / zero-filled
        kspace[:, y0:y1, z0:z1, :] = ref_t[:, y0:y1, z0:z1, :]

    # Actual sampling mask from merged k-space
    # collapse readout and coil dimensions: Nx, Ny, Nz, coil -> Ny, Nz
    mask_2d = torch.sum(torch.abs(kspace) ** 2, dim=(0, 3)) > 0
    mask_2d = mask_2d.cpu().numpy().astype(np.float32)

    return kspace, mask_2d
