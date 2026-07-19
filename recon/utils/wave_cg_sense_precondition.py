"""
wave_cg_sense_precondition.py

Helper utilities for 3D FFTs and Wave-CG-SENSE reconstruction.

This version keeps the original Wave-CG-SENSE operator order unchanged:

    E = M * F_y * F_z * PSF[y,z] * F_x * S

and adds optional acceleration features:

    1. Combined centered FFT over y/z.
    2. Cached conjugates, unsqueezed PSF, and coil sensitivity power.
    3. Optional direct full-sampling reconstruction shortcut.
    4. Optional diagonal preconditioned CG.
    5. Optional initialization choice: zero, adjoint, or adjoint_precond.
    6. Separate image-magnitude normalization helper for display/comparison.

Defaults are conservative:

    - init="zero"
    - use_preconditioner=False
    - use_direct_if_full=False

So the default CG behavior remains close to the original solver.

Author: Yiyun Dong
Affiliation: Athinoula A. Martinos Center for Biomedical Imaging
License: MIT License
"""

import torch


def _dim_tuple(dim):
    """Return dim as a tuple while accepting either int or tuple/list."""
    if isinstance(dim, int):
        return (dim,)
    return tuple(dim)


def ifft3call(x, dim=(0, 1, 2)):
    """
    Perform 3D IFFT with proper shifting and normalization.

    Args:
        x: Input tensor of shape (Nx, Ny, Nz, ...)

    Returns:
        Result of 3D IFFT with sqrt(Nz*Nx*Ny) normalization
    """
    fctr = x.shape[0] * x.shape[1] * x.shape[2]

    # Combined 3D IFFT with shifting along all spatial dimensions
    res = torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(x, dim=dim),
            dim=dim
        ),
        dim=dim
    ) * (fctr ** 0.5)

    return res


def fft3call(x, dim=(0, 1, 2)):
    """
    Perform 3D FFT with proper shifting and normalization.

    Args:
        x: Input tensor of shape (Nx, Ny, Nz, ...)

    Returns:
        Result of 3D FFT with sqrt(Nz*Nx*Ny) normalization
    """
    fctr = x.shape[0] * x.shape[1] * x.shape[2]

    # Combined 3D FFT with shifting along all spatial dimensions
    res = torch.fft.ifftshift(
        torch.fft.fftn(
            torch.fft.fftshift(x, dim=dim),
            dim=dim
        ),
        dim=dim
    ) * (fctr ** 0.5)

    return res


def fftc_dim(x, dim):
    """Centered orthonormal 1D FFT along one dimension."""
    return torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(x, dim=(dim,)), dim=dim, norm='ortho'),
        dim=(dim,))


def ifftc_dim(x, dim):
    """Centered orthonormal 1D IFFT along one dimension."""
    return torch.fft.fftshift(
        torch.fft.ifft(torch.fft.ifftshift(x, dim=(dim,)), dim=dim, norm='ortho'),
        dim=(dim,))


def fftc_nd(x, dim):
    """Centered orthonormal ND FFT over one or more dimensions."""
    dim = _dim_tuple(dim)
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=dim), dim=dim, norm='ortho'),
        dim=dim)


def ifftc_nd(x, dim):
    """Centered orthonormal ND IFFT over one or more dimensions."""
    dim = _dim_tuple(dim)
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=dim), dim=dim, norm='ortho'),
        dim=dim)


class WaveCGSenseOperator:
    """
    Encapsulated Wave-CG-SENSE forward, adjoint, and CG solver.

    Expected shapes:
        sens:        (nc, Nx_os, Ny, Nz)
        psf_to_use:  (Nx_os, Ny, Nz)
        mask_t:      broadcastable to (nc, Nx_os, Ny, Nz)
        y:           (nc, Nx_os, Ny, Nz)
        x:           (Nx_os, Ny, Nz)

    Args:
        sens: Coil sensitivity maps.
        psf_to_use: Wave PSF term in hybrid space.
        mask_t: Sampling mask, broadcastable to measured k-space shape.
        combine_yz_fft: If True, use one centered 2D FFT over y/z instead of
            two sequential 1D FFTs. This is mathematically equivalent up to
            small floating-point differences.
    """

    def __init__(self, sens, psf_to_use, mask_t, combine_yz_fft=True):
        self.sens = sens
        self.psf_to_use = psf_to_use
        self.mask_t = mask_t
        self.combine_yz_fft = combine_yz_fft

        # Cached quantities to avoid repeated conj/unsqueeze/sum operations.
        self.sens_conj = torch.conj(sens)
        self.sens_power = torch.sum(torch.abs(sens) ** 2, dim=0).real
        self.psf_to_use_unsq = psf_to_use.unsqueeze(0)
        self.psf_to_use_conj_unsq = torch.conj(psf_to_use).unsqueeze(0)

        self.Nx_os = sens.shape[1]
        self.Ny = sens.shape[2]
        self.Nz = sens.shape[3]

    def _coil_lambda(self, coil_lambda=None):
        """
        Return a small denominator stabilizer for coil-power division.

        If coil_lambda is None, use 1e-8 * max(sum_c |S_c|^2).
        """
        if coil_lambda is not None:
            return coil_lambda
        return 1e-8 * self.sens_power.max()

    def apply_coil_preconditioner(self, x, coil_lambda=None):
        """
        Apply diagonal coil-power preconditioner:

            M^{-1} x = x / (sum_c |S_c|^2 + lambda)
        """
        lam = self._coil_lambda(coil_lambda)
        return x / (self.sens_power + lam + 1e-24)

    def is_fully_sampled(self, atol=0.0):
        """
        Check whether mask_t is effectively all ones/True.

        Args:
            atol: Absolute tolerance for floating-point masks.

        Returns:
            bool
        """
        mask = self.mask_t
        if mask.dtype == torch.bool:
            return bool(torch.all(mask).item())

        mask_abs = torch.abs(mask) if torch.is_complex(mask) else mask
        ones = torch.ones_like(mask_abs)
        return bool(torch.all(torch.isclose(mask_abs, ones, rtol=0.0, atol=atol)).item())

    def E_wave(self, x):
        """Forward operator: x -> k-space coils (nc, Nx_os, Ny, Nz)."""
        img_coils = self.sens * x.unsqueeze(0)                      # S
        hybrid = fftc_dim(img_coils, dim=1)                         # F_x
        hybrid = hybrid * self.psf_to_use_unsq                      # PSF[y,z]

        if self.combine_yz_fft:
            kspace = fftc_nd(hybrid, dim=(2, 3))                    # F_y, F_z
        else:
            kspace = fftc_dim(hybrid, dim=2)                        # F_y
            kspace = fftc_dim(kspace, dim=3)                        # F_z

        return kspace * self.mask_t                                 # M

    def EH_wave(self, k):
        """Adjoint operator: k-space coils -> image."""
        if self.combine_yz_fft:
            hybrid = ifftc_nd(k * self.mask_t, dim=(2, 3))          # F_y^H, F_z^H, M
        else:
            hybrid = ifftc_dim(k * self.mask_t, dim=2)              # F_y^H * M
            hybrid = ifftc_dim(hybrid, dim=3)                       # F_z^H * M

        hybrid = hybrid * self.psf_to_use_conj_unsq                 # PSF[y,z]^H
        img_coils = ifftc_dim(hybrid, dim=1)                        # F_x^H
        return (self.sens_conj * img_coils).sum(dim=0)              # S^H

    def normal_operator(self, x):
        """Apply A x = E^H E x."""
        return self.EH_wave(self.E_wave(x))

    def direct_recon_if_full(self, y, coil_lambda=None, mask_full_sampled_atol=0.0):
        """
        Direct reconstruction for fully sampled data, if mask_t is all ones.

        For full sampling and unitary FFT/PSF terms:

            E^H E = S^H S

        so the solution is approximately:

            x = E^H y / (sum_c |S_c|^2 + lambda)

        Returns:
            x if fully sampled, otherwise None.
        """
        if not self.is_fully_sampled(atol=mask_full_sampled_atol):
            return None

        b = self.EH_wave(y)
        return self.apply_coil_preconditioner(b, coil_lambda=coil_lambda)

    def _initial_guess(self, b, init, coil_lambda=None):
        """
        Build the CG initial guess and residual.

        Choices:
            init="zero":            x0 = 0, r0 = b
            init="adjoint":         x0 = E^H y, r0 = b - A x0
            init="adjoint_precond": x0 = M^{-1} E^H y, r0 = b - A x0
        """
        if init == "zero":
            x = torch.zeros((self.Nx_os, self.Ny, self.Nz), dtype=torch.complex64, device=b.device)
            r = b.clone()
            return x, r

        if init == "adjoint":
            x = b.clone()
            r = b - self.normal_operator(x)
            return x, r

        if init == "adjoint_precond":
            x = self.apply_coil_preconditioner(b, coil_lambda=coil_lambda)
            r = b - self.normal_operator(x)
            return x, r

        raise ValueError('init must be one of: "zero", "adjoint", "adjoint_precond"')

    def cg_sense_wave(
        self,
        y,
        n_iter=50,
        tol=1e-6,
        init="zero",
        use_preconditioner=False,
        use_direct_if_full=False,
        coil_lambda=None,
        mask_full_sampled_atol=0.0,
    ):
        """
        Solve (E^H E)x = E^H y with conjugate gradient.

        Args:
            y: Measured k-space data, shape (nc, Nx_os, Ny, Nz).
            n_iter: Number of CG iterations.
            tol: Relative residual tolerance. Default unchanged from original.
            init: "zero", "adjoint", or "adjoint_precond". Default is "zero".
            use_preconditioner: If True, use diagonal coil-power preconditioned CG.
            use_direct_if_full: If True and mask_t is fully sampled, bypass CG and
                return E^H y / (sum_c |S_c|^2 + lambda).
            coil_lambda: Optional denominator stabilizer for coil-power division.
                If None, uses 1e-8 * max(sum_c |S_c|^2).
            mask_full_sampled_atol: Tolerance used when checking whether mask_t is all ones.

        Returns:
            Reconstructed image, shape (Nx_os, Ny, Nz).
        """
        if use_direct_if_full:
            x_direct = self.direct_recon_if_full(
                y,
                coil_lambda=coil_lambda,
                mask_full_sampled_atol=mask_full_sampled_atol,
            )
            if x_direct is not None:
                print("Using direct full-sampling reconstruction shortcut.")
                return x_direct

        b = self.EH_wave(y)
        x, r = self._initial_guess(b, init=init, coil_lambda=coil_lambda)

        bb = torch.vdot(b.reshape(-1), b.reshape(-1)).real + 1e-24

        if use_preconditioner:
            z = self.apply_coil_preconditioner(r, coil_lambda=coil_lambda)
            p = z.clone()
            rz = torch.vdot(r.reshape(-1), z.reshape(-1)).real

            for i in range(n_iter):
                print(f'{i}/{n_iter}')
                Ap = self.normal_operator(p)
                pAp = torch.vdot(p.reshape(-1), Ap.reshape(-1)).real + 1e-24
                alpha = rz / pAp
                x = x + alpha * p
                r = r - alpha * Ap

                rr_new = torch.vdot(r.reshape(-1), r.reshape(-1)).real
                rel = torch.sqrt(rr_new / bb)
                if rel < tol:
                    print(f"CG converged at iter {i + 1}, rel-res={rel.item():.2e}")
                    return x

                z = self.apply_coil_preconditioner(r, coil_lambda=coil_lambda)
                rz_new = torch.vdot(r.reshape(-1), z.reshape(-1)).real
                beta = rz_new / (rz + 1e-24)
                p = z + beta * p
                rz = rz_new

            rr = torch.vdot(r.reshape(-1), r.reshape(-1)).real
            print(f"CG reached max_iter={n_iter}, final rel-res={torch.sqrt(rr / bb).item():.2e}")
            return x

        # Original vanilla CG path.
        p = r.clone()
        rr = torch.vdot(r.reshape(-1), r.reshape(-1)).real

        for i in range(n_iter):
            print(f'{i}/{n_iter}')
            Ap = self.normal_operator(p)
            pAp = torch.vdot(p.reshape(-1), Ap.reshape(-1)).real + 1e-24
            alpha = rr / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = torch.vdot(r.reshape(-1), r.reshape(-1)).real

            rel = torch.sqrt(rr_new / bb)
            if rel < tol:
                print(f"CG converged at iter {i + 1}, rel-res={rel.item():.2e}")
                return x

            beta = rr_new / (rr + 1e-24)
            p = r + beta * p
            rr = rr_new

        print(f"CG reached max_iter={n_iter}, final rel-res={torch.sqrt(rr / bb).item():.2e}")
        return x


def cg_sense_wave(
    y,
    sens,
    psf_to_use,
    mask_t,
    n_iter=50,
    tol=1e-6,
    init="zero",
    use_preconditioner=False,
    use_direct_if_full=False,
    coil_lambda=None,
    mask_full_sampled_atol=0.0,
    combine_yz_fft=True,
):
    """
    Convenience wrapper for one-shot reconstruction.

    Example:
        from wave_cg_sense_accel import cg_sense_wave

        x = cg_sense_wave(
            y=y_meas,
            sens=sens,
            psf_to_use=psf_to_use,
            mask_t=mask_t,
            n_iter=50,
            tol=1e-6,
            init="zero",
            use_preconditioner=False,
            use_direct_if_full=False,
        )

    Optional acceleration examples:
        # Direct shortcut if mask_t is fully sampled:
        x = cg_sense_wave(y, sens, psf_to_use, mask_t, use_direct_if_full=True)

        # Preconditioned CG:
        x = cg_sense_wave(y, sens, psf_to_use, mask_t, use_preconditioner=True)

        # Adjoint initialization:
        x = cg_sense_wave(y, sens, psf_to_use, mask_t, init="adjoint")
    """
    op = WaveCGSenseOperator(
        sens=sens,
        psf_to_use=psf_to_use,
        mask_t=mask_t,
        combine_yz_fft=combine_yz_fft,
    )
    return op.cg_sense_wave(
        y=y,
        n_iter=n_iter,
        tol=tol,
        init=init,
        use_preconditioner=use_preconditioner,
        use_direct_if_full=use_direct_if_full,
        coil_lambda=coil_lambda,
        mask_full_sampled_atol=mask_full_sampled_atol,
    )


def normalize_image_magnitude(
    x,
    method="p99",
    mask=None,
    percentile=None,
    eps=1e-12,
    return_scale=False,
):
    """
    Normalize image magnitude for display/comparison without changing recon physics.

    Args:
        x: Complex or real image tensor.
        method: One of "p99", "p95", "percentile", "max", "mean", "median".
        mask: Optional boolean mask selecting voxels used to estimate scale.
        percentile: Quantile in [0, 1] when method="percentile". If method is
            "p99" or "p95", this argument is ignored.
        eps: Small denominator stabilizer.
        return_scale: If True, return (x_normalized, scale).

    Returns:
        x_normalized, or (x_normalized, scale) if return_scale=True.
    """
    mag = torch.abs(x)

    if mask is not None:
        vals = mag[mask]
    else:
        vals = mag.reshape(-1)

    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        raise ValueError("No finite voxels available for normalization scale estimation.")

    if method == "p99":
        scale = torch.quantile(vals, 0.99)
    elif method == "p95":
        scale = torch.quantile(vals, 0.95)
    elif method == "percentile":
        if percentile is None:
            raise ValueError('percentile must be provided when method="percentile"')
        scale = torch.quantile(vals, percentile)
    elif method == "max":
        scale = torch.max(vals)
    elif method == "mean":
        scale = torch.mean(vals)
    elif method == "median":
        scale = torch.median(vals)
    else:
        raise ValueError('method must be one of: "p99", "p95", "percentile", "max", "mean", "median"')

    x_norm = x / (scale + eps)
    if return_scale:
        return x_norm, scale
    return x_norm
