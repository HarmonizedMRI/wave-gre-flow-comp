#!/usr/bin/env python3
"""
MRI k-space coil-compression helpers.

This module provides utilities for preparing Siemens Twix/mapVBVD k-space data
for low-resolution ESPIRiT coil sensitivity calibration and reconstruction.

Assumed data conventions
------------------------
1. Raw k-space tensors are assumed to use coil-last layout:

       kspace.shape == (Nx, Ny, Nz, Ncoil)

   where:
       Nx    : readout dimension, possibly oversampled
       Ny    : first phase-encode dimension
       Nz    : second phase-encode / partition dimension
       Ncoil : physical receive channels, e.g. 32

2. SigPy MRI routines generally expect coil-first layout:

       kspace_sigpy.shape == (Ncoil, Nx, Ny, Nz)

3. Complex data should be stored as complex64 whenever possible to reduce
   memory use.

4. For accelerated acquisitions with separate ACS/reference scans, the image
   data may contain only the undersampled Ry x Rz grid, while the fully sampled
   ACS block may live in the Twix REF/REFSCAN stream. The ACS region should be
   merged into the image k-space before ESPIRiT calibration.

Functions
---------

estimate_cc_matrix_coillast(kspace, ncc=12, acs=24, x_step=4, eps=1e-8)
    Estimate a coil-compression matrix W from a small central ACS calibration
    block using the coil covariance matrix. The input is coil-last k-space
    shaped (Nx, Ny, Nz, Ncoil). The returned W has shape (Ncoil, Ncc).

apply_cc_coillast_torch(kspace, W, x_chunk=8)
    Apply a previously estimated coil-compression matrix to a full coil-last
    PyTorch tensor in readout chunks to avoid excessive memory use. Returns
    compressed k-space shaped (Nx, Ny, Nz, Ncc).

apply_cc_coilfirst_np(kspace_cf, W)
    Apply the same compression matrix to a coil-first NumPy array shaped
    (Ncoil, Nx, Ny, Nz), returning a compressed array shaped
    (Ncc, Nx, Ny, Nz). This is useful before passing low-resolution k-space
    to SigPy ESPIRiT.

Typical workflow
----------------
1. Load and merge image + ACS/reference k-space:

       kspace_echo, mask_2d = load_merged_img_ref(...)

2. Estimate coil compression from the merged full k-space:

       Wcc, svals, energy = estimate_cc_matrix_coillast(
           kspace_echo,
           ncc=12,
           acs=24,
           x_step=4,
       )

3. Build a low-resolution coil-first k-space for ESPIRiT:

       kspace_np = (
           kspace_echo
           .permute(3, 0, 1, 2)
           .contiguous()
           .numpy()
           .astype(np.complex64, copy=False)
       )

       kspace_low_np = sp.resize(kspace_np, (Ncoil, 64, 64, 48))
       kspace_low_cc_np = apply_cc_coilfirst_np(kspace_low_np, Wcc)

4. Run ESPIRiT on the compressed low-resolution k-space:

       kspace_low_cc_sp = sp.to_device(kspace_low_cc_np, sp.Device(0))

       csm_low_cc = mr.app.EspiritCalib(
           kspace_low_cc_sp,
           calib_width=24,
           device=sp.Device(0),
           crop=0,
           show_pbar=True,
       ).run()

5. Compress full-resolution k-space, if needed:

       kspace_cc = apply_cc_coillast_torch(kspace_echo, Wcc, x_chunk=8)

Notes
-----
- Coil compression is estimated from the central ACS region but can be applied
  to the full k-space.
- Use the same compression matrix W for wave and no-wave datasets if they were
  acquired with the same coil configuration and ordering.
- ESPIRiT memory scales roughly as:

       Nx * Ny * Nz * Ncoil^2

  so reducing 32 coils to 12 coils greatly reduces GPU memory usage.
- For a 32-channel complex64 dataset shaped (32, 1024, 256, 192), the full
  coil-first array is approximately 12.9 GB. Avoid unnecessary full copies.
"""

import numpy as np
import torch
import scipy.linalg as la

def estimate_cc_matrix_coillast(kspace, ncc=12, acs=24, x_step=4, eps=1e-8):
    """
    Estimate SVD coil compression matrix from central ACS.

    kspace: torch or numpy, shape (Nx, Ny, Nz, ncoil)
    returns W: numpy complex64, shape (ncoil, ncc)
    """
    Nx, Ny, Nz, ncoil = kspace.shape

    cy, cz = Ny // 2, Nz // 2
    y0, y1 = cy - acs // 2, cy + acs // 2
    z0, z1 = cz - acs // 2, cz + acs // 2

    calib = kspace[::x_step, y0:y1, z0:z1, :]

    if isinstance(calib, torch.Tensor):
        calib = calib.detach().cpu().numpy()

    calib = np.asarray(calib, dtype=np.complex64)
    X = calib.reshape(-1, ncoil)

    # Remove zero-filled samples
    power = np.sum(np.abs(X) ** 2, axis=1)
    keep = power > eps * power.max()
    X = X[keep]

    # 32 x 32 covariance, much faster than full SVD
    C = X.conj().T @ X
    C = 0.5 * (C + C.conj().T)

    evals, evecs = la.eigh(C)
    idx = np.argsort(evals)[::-1]

    evals = evals[idx]
    evecs = evecs[:, idx]

    W = evecs[:, :ncc].astype(np.complex64, copy=False)
    svals = np.sqrt(np.maximum(evals, 0))

    energy = np.cumsum(evals) / np.sum(evals)

    return W, svals, energy


def apply_cc_coillast_torch(kspace, W, x_chunk=8):
    """
    Apply coil compression to full k-space in chunks.

    kspace: torch tensor, shape (Nx, Ny, Nz, ncoil)
    W: numpy array, shape (ncoil, ncc)
    returns: torch tensor, shape (Nx, Ny, Nz, ncc)
    """
    assert isinstance(kspace, torch.Tensor)

    Nx, Ny, Nz, ncoil = kspace.shape
    ncc = W.shape[1]

    W_t = torch.as_tensor(W, dtype=kspace.dtype, device=kspace.device)
    out = torch.empty((Nx, Ny, Nz, ncc), dtype=kspace.dtype, device=kspace.device)

    with torch.no_grad():
        for x0 in range(0, Nx, x_chunk):
            x1 = min(x0 + x_chunk, Nx)

            block = kspace[x0:x1]  # (chunk, Ny, Nz, ncoil)
            X = block.reshape(-1, ncoil)
            Xcc = X @ W_t

            out[x0:x1] = Xcc.reshape(x1 - x0, Ny, Nz, ncc)

    return out


def apply_cc_coilfirst_np(kspace_cf, W):
    """
    kspace_cf: numpy, shape (ncoil, x, y, z)
    W: shape (ncoil, ncc)
    returns: shape (ncc, x, y, z)
    """
    return np.einsum(
        "cxyz,cn->nxyz",
        kspace_cf,
        W,
        optimize=True,
    ).astype(np.complex64, copy=False)