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