"""Coil-sensitivity plotting helpers.

This module provides lightweight Matplotlib utilities for visualizing
coil sensitivity maps (CSMs) estimated during MRI reconstruction. The
functions assume coil-first CSM arrays with shape ``(ncoils, nx, ny, nz)``
and plot either magnitude or phase for a selected axial slice.

Typical usage
-------------
    from utils.plot_coil_sens import plot_csm_magnitude_grid, plot_csm_phase_grid

    plot_csm_magnitude_grid(csm_full_cc_np, z=csm_full_cc_np.shape[-1] // 2)
    plot_csm_phase_grid(csm_full_cc_np, z=csm_full_cc_np.shape[-1] // 2)

Notes
-----
- Magnitude plots use a percentile-based display maximum for robust scaling.
- Phase plots mask low-RSS background before displaying phase in ``[-pi, pi]``.
- These helpers are intended for quick quality-control figures, not for
  quantitative analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_csm_magnitude_grid(csm, z=None, max_coils=32, vmax_pct=99):
    """
    Plot coil sensitivity magnitudes for one axial slice.

    csm shape: (ncoils, nx, ny, nz)
    """
    ncoils, nx, ny, nz = csm.shape

    if z is None:
        z = nz // 2

    nplot = min(ncoils, max_coils)
    ncols = int(np.ceil(np.sqrt(nplot)))
    nrows = int(np.ceil(nplot / ncols))

    mag = np.abs(csm[:nplot, :, :, z])
    vmax = np.percentile(mag, vmax_pct)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))
    axes = np.asarray(axes).ravel()

    for i in range(nplot):
        axes[i].imshow(np.rot90(mag[i]), cmap="gray", vmin=0, vmax=vmax)
        axes[i].set_title(f"Coil {i}")
        axes[i].axis("off")

    for i in range(nplot, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"CSM magnitude, axial slice z={z}", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_csm_phase_grid(csm, z=None, max_coils=32, mag_thresh=0.05):
    """
    Plot coil sensitivity phases for one axial slice,
    masking background where RSS magnitude is small.
    """
    ncoils, nx, ny, nz = csm.shape

    if z is None:
        z = nz // 2

    nplot = min(ncoils, max_coils)
    ncols = int(np.ceil(np.sqrt(nplot)))
    nrows = int(np.ceil(nplot / ncols))

    rss = np.sqrt(np.sum(np.abs(csm) ** 2, axis=0))
    mask = rss[:, :, z] > mag_thresh * rss[:, :, z].max()

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))
    axes = np.asarray(axes).ravel()

    for i in range(nplot):
        phase = np.angle(csm[i, :, :, z])
        phase = np.where(mask, phase, np.nan)

        im = axes[i].imshow(
            np.rot90(phase),
            cmap="gray",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[i].set_title(f"Coil {i}")
        axes[i].axis("off")

    for i in range(nplot, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"CSM phase, axial slice z={z}", fontsize=14)
    plt.tight_layout()
    plt.show()