# Reconstruction

## Entry point

```text
recon/recon_wave_gre_from_twix_integrated_nifti.py
```

Run it from the repository root so the relative `recon/utils/` imports resolve consistently:

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py --help
```

## Required inputs

A normal reconstruction requires:

- one integrated Siemens TWIX `.dat` file;
- the exact matching integrated Pulseq `.seq` file;
- an output folder.

The command-line paths are direct paths; there is no separate shared data-folder option:

```text
--twix PATH
--seq PATH
--out PATH
```

The TWIX containers are expected to be:

```text
image    -> GRE image k-space
refscan  -> FLASH projection calibration and ACS
```

## Supported acquisition assumptions

The current verified reconstruction expects:

- transverse geometry;
- readout on x, LIN/sine on y, and PAR/cosine on z;
- exactly one average;
- one or more GRE echoes;
- two-axis wave imaging or fully no-wave imaging;
- integrated calibration SETs 0–4;
- calibration projection width compatible with the sequence definitions;
- ACS in SET 4 by default;
- matching image and calibration k-space ordering.

Sine-only and cosine-only wave image acquisitions are rejected.

At the beginning of a normal reconstruction, the updated code prints warning-only `.seq`/TWIX geometry diagnostics. These compare FOV and received dimensions, retain the verified transverse assertion, and report readout, LIN phase-encoding, and PAR phase-encoding directions. For accelerated image data, the diagnostic reports the measured LIN/PAR counts separately from the mapVBVD array extents implied by the maximum global LIN/PAR labels. A mismatch is reported for investigation but does not stop reconstruction.

## Basic commands

### Automatic wave/no-wave detection

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py \
    --twix /path/to/scan.dat \
    --seq /path/to/scan.seq \
    --out /path/to/recon \
    --wave-mode auto
```

`--mode` remains an accepted alias for `--wave-mode`.

### Force no-wave consistency checking

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py \
    --twix /path/to/scan.dat \
    --seq /path/to/scan.seq \
    --out /path/to/recon \
    --wave-mode nowave
```

### Validate sequence-derived configuration

The existing parser still requires `--twix`, `--seq`, and `--out` even in validation-only mode. The TWIX file is not read when `--validate-only` is active, so a placeholder path is acceptable:

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py \
    --twix unused.dat \
    --seq /path/to/scan.seq \
    --out /tmp/wave-gre-validation \
    --wave-mode auto \
    --validate-only
```

This checks the sequence trajectory and definitions without loading imaging data.

## Pipeline

1. Read the Pulseq sequence and its definitions.
2. Validate matrix dimensions, oversampling, FOV, echoes, averages, acceleration, calibration layout, orientation, wave channels, and k-space ordering.
3. Split the sequence ADC trajectory into image and appended calibration lines.
4. Inspect the image trajectory and resolve wave/no-wave mode.
5. Load the integrated ACS from TWIX `refscan` SET 4 and remove readout
   oversampling with a centered readout IFFT, central nominal-FOV image crop,
   and centered FFT.
6. Estimate the requested coil-compression matrix from that alias-free logical
   ACS on CPU.
7. Generate low-resolution ESPIRiT maps using native 3D calibration or CPU-parallel logical-RO slice2d calibration.
8. Interpolate and normalize the sensitivity maps.
9. Load the multi-echo GRE image k-space and apply coil compression on CPU.
10. For wave data, fit FLASH projection phase deviations, process the fitted PSF coefficients using the selected method, and construct calibrated echo-specific PSFs.
11. Run wave or no-wave CG-SENSE on CPU for each echo.
12. Save NumPy arrays, diagnostic plots, geometry metadata, and optional NIfTI outputs.

## CPU and GPU behavior

The full reconstruction is not moved to GPU.

| Operation | Current device |
|---|---|
| Coil-compression estimation | CPU / NumPy and SciPy |
| Coil-compression application | CPU / PyTorch tensor |
| ESPIRiT calibration | native `3d`: SigPy CPU/GPU; `slice2d`: CPU processes |
| Wave and no-wave CG-SENSE | CPU / PyTorch tensor |

### ESPIRiT selection

```text
--espirit-device auto   use a visible compatible GPU, otherwise CPU
--espirit-device cpu    always use SigPy CPU
--espirit-device gpu    require a usable CuPy/CUDA device
--espirit-gpu-index N   select GPU index; default 0
```

In `auto` mode, missing CuPy, CUDA initialization failures, or the absence of a CUDA GPU cause a reported CPU fallback. Explicit `gpu` mode raises an error when GPU execution is unavailable.

CuPy is therefore optional for CPU reconstruction. Install the `gpu` dependency group only on a CUDA 12 system where GPU-assisted ESPIRiT is desired.

For data acquired with more than 32 physical receiver channels, consider explicitly using `--espirit-device cpu`, depending on available CPU RAM, CPU performance, and GPU memory. GPU ESPIRiT can still be appropriate on a sufficiently large GPU, but `cpu` avoids GPU-memory pressure and CUDA-specific failures.

<!-- ESPIRIT-SLICE2D-RECONSTRUCTION -->
### ESPIRiT calibration backend

```text
--espirit-calib-mode 3d       native SigPy 3D calibration; default/reference
--espirit-calib-mode slice2d  CPU-parallel 2D calibration over logical-RO hybrid slices
--espirit-crop VALUE          eigenvalue support crop; default 0.8
--espirit-cpu-workers N       optional slice2d process limit; default automatic
```

The GRE ACS is converted to logical k-space with centered image-domain cropping
before coil compression or either ESPIRiT backend. Direct readout k-space
striding is not equivalent: it aliases signal outside the nominal readout FOV
into the calibration image and is therefore forbidden. In `slice2d` mode, only
after this correction is logical RO transformed to image space. Each worker
receives one `(coil, LIN, PAR)` plane, so calibration remains joint across the
two phase-encoding dimensions.

`slice2d` is CPU-only. `--espirit-device auto` and `cpu` are accepted; explicit `gpu` is rejected. Native `3d` remains the method to use for GPU calibration and the reference for method comparisons.

`--espirit-crop` is passed directly to SigPy in both modes. Testing in the associated Wave-MPRAGE workflow found **0.8–0.9** to be a reasonable practical range, and the same range is recommended as the initial GRE range:

- `0.8`: broader sensitivity support, including more low-SNR anatomy;
- `0.9`: stricter support with more background suppression.

Changing crop requires recalculating maps. Do not use `--reuse-coil-calib` when evaluating a new crop value.

When `--espirit-cpu-workers` is omitted, Joblib selects the available physical-core count and limits it by the number of logical-RO slices. An explicit value is useful on a shared node or when memory bandwidth limits scaling. Start conservatively, then benchmark; using every logical CPU is not necessarily fastest.

## Main reconstruction options

| Option | Default | Purpose |
|---|---:|---|
| `--wave-mode {auto,wave,nowave}` | `auto` | Select or detect the image reconstruction model |
| `--file-tag TEXT` | empty | Append a sanitized tag to cache and result filenames |
| `--ncc N` | `12` | Number of virtual coils retained |
| `--reuse-coil-calib` | off | Reuse compatible cached coil compression and CSM files |
| `--espirit-device {auto,cpu,gpu}` | `auto` | Select ESPIRiT execution device |
| `--espirit-gpu-index N` | `0` | Select CUDA GPU index |
| `--espirit-calib-mode {3d,slice2d}` | `3d` | Select native 3D or CPU-parallel slice2d ESPIRiT |
| `--espirit-crop VALUE` | `0.8` | Set ESPIRiT eigenvalue support crop; practical initial range 0.8–0.9 |
| `--espirit-cpu-workers N` | automatic | Limit slice2d process workers |
| `--cg-iters N` | `50` | Maximum CG iterations |
| `--cg-tol VALUE` | `1e-6` | Relative CG stopping tolerance |
| `--yflip {-1,1}` | sequence-derived | Override LIN PSF sign |
| `--zflip {-1,1}` | sequence-derived | Override PAR PSF sign |
| `--psf-coefficient-processing {smooth,sine-line}` | `smooth` | Select PSF coefficient post-processing for wave data |
| `--psf-fit-kx-min N` | none | Inclusive manual `sine-line` bound; omit both bounds for automatic selection |
| `--psf-fit-kx-max N` | none | Exclusive manual `sine-line` bound; omit both bounds for automatic selection |
| `--save-echo-npy` | off | Save one complex NumPy file per echo |
| `--save-bart-inputs` | off | Export calibrated Wave-CAIPI inputs as BART CFL pairs |
| `--validate-only` | off | Validate sequence-derived configuration without reading TWIX |

Use `--help` as the authoritative complete argument reference for the checked-out code.

## BART Wave-CAIPI input export

Add `--save-bart-inputs` to a wave reconstruction to write BART-compatible
`.hdr`/`.cfl` pairs under `<out>/bart_inputs`. When `--file-tag` is set, the
folder becomes `bart_inputs_<tag>`.

The export uses BART's `READ`, `PHS1`, `PHS2`, `COIL`, and `MAPS` dimension
order:

| Basename | Shape | Contents |
|---|---|---|
| `wave_kspace` | `(Nx_os, Ny, Nz, Ncc, 1)` | Coil-compressed acquired k-space |
| `psf` | `(Nx_os, Ny, Nz, 1, 1)` | Calibrated wave PSF |
| `coil_sens` | `(Nx, Ny, Nz, Ncc, 1)` | Sensitivity maps from this reconstruction |
| `kspace_calib` | `(Nx, Ny, Nz, Ncc)` | Coil-compressed, centered integrated ACS for BART `ecalib` |

Multi-echo acquisitions write matching `_echo-01`, `_echo-02`, and so on
suffixes for `wave_kspace` and `psf`. Common sensitivity and calibration data
are written once. `manifest.json` records every basename and shape plus the
shared PSF coefficient-processing request, selected range, algorithm/version,
fit diagnostics, validation result, and multi-echo calibration scope.

Run BART ESPIRiT calibration, reconstruct every exported echo, and convert the
results to NIfTI with:

```bash
recon/bart/run_wave_recon.sh \
    --bart-input /path/to/reconstruction/bart_inputs \
    --bart-output /path/to/reconstruction/bart_output \
    --maps-source bart \
    --twix /path/to/meas_wave_gre.dat \
    --seq /path/to/matching_wave_gre.seq \
    --save-phase \
    --ecalib-options -c 0.8 --end-ecalib-options \
    --wave-options -w -r 0.001 -f -i 100 -t 1e-6 --end-wave-options
```

The flags between `--ecalib-options` and `--end-ecalib-options` are passed
unchanged to `bart ecalib`. Likewise, everything in the `--wave-options`
section is passed unchanged to `bart wave`; the example requests wavelet
regularization and FISTA. The helper prints each complete command before
running it. If `--nifti-output` is omitted, converted files are written to
`BART_OUTPUT/nifti`; pass the option only to override that location. Conversion
uses `python` from the active Conda environment or virtual environment. Set
`PYTHON_BIN` to select a different interpreter.

To skip `ecalib` and use the exported Python sensitivity maps, select existing
maps explicitly:

```bash
recon/bart/run_wave_recon.sh \
    --bart-input /path/to/reconstruction/bart_inputs \
    --bart-output /path/to/reconstruction/bart_output \
    --maps-source existing \
    --existing-maps /path/to/reconstruction/bart_inputs/coil_sens \
    --twix /path/to/meas_wave_gre.dat \
    --seq /path/to/matching_wave_gre.seq \
    --nifti-output /path/to/custom/nifti \
    --wave-options -l -r 0.002 -b 8 -f -i 100 --end-wave-options
```

The final conversion uses TWIX geometry and Pulseq metadata. It restores the
k-space norm removed internally by `bart wave`, applies one echo-1 magnitude
scale to every echo, and does not crop BART's already de-oversampled readout.
Use `--help` for every supported BART wave flag and the optional direct NIfTI
option section. Set `BART_BIN` when the executable is not named `bart`.

## PSF coefficient processing

Wave calibration first estimates the readout-dependent phase-plane coefficients `a(kx)`, `b(kx)`, and `c(kx)`. The reconstruction provides two mutually exclusive post-processing methods.

### `smooth` — default

```text
--psf-coefficient-processing smooth
```

This preserves the normal reconstruction path: each raw coefficient curve is processed with the existing NaN-aware moving-average smoothing. No kx bounds are required.

### `sine-line` — optional automatic fit

If the default smooth-coefficient diagnostic looks unreliable, first omit
both bounds to request automatic selection:

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py \
    --twix /path/to/scan.dat \
    --seq /path/to/scan.seq \
    --out /path/to/recon \
    --wave-mode auto \
    --psf-coefficient-processing sine-line
```

Automatic selection uses the common finite `a/b/c` support and sin/cos
projection quality, excludes unreliable edges and transient/high-variance
coefficient regions, requires a reliable central core, and selects one shared
contiguous half-open interval. Reliable samples inside that interval are
smoothed with the same nine-sample NaN-aware window before fitting. The fit
must pass convergence, residual, conditioning, frequency, and endpoint-trim
extrapolation-stability gates. Failure stops processing; it never silently
falls back to `smooth`.

### Manual `sine-line` override

If the automatic result is also unsatisfactory, provide a reviewed interval:

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py \
    --twix /path/to/scan.dat \
    --seq /path/to/scan.seq \
    --out /path/to/recon \
    --wave-mode auto \
    --psf-coefficient-processing sine-line \
    --psf-fit-kx-min 200 \
    --psf-fit-kx-max 512
```

The interval follows the half-open convention:

```text
[kx_min, kx_max)
```

The manual sine-plus-line model is fitted independently to raw `a(kx)`,
`b(kx)`, and `c(kx)` inside that interval, then evaluated over the full
oversampled readout. It replaces smoothing and is not smoothed again.

Before using `sine-line`, first rule out a mismatched `.seq` file, incorrect
PSF signs, incomplete FOV coverage, and neck/shoulder signal contamination.
Inspect `psf_integrated_calib_fit*.png`; its shaded interval and legend identify
the automatic or manual fit selection. Automatic diagnostics are also written
to `psf_sine_line_fit*.json`.

Supplying only one bound is an error. Passing bounds while using `smooth` is
also rejected so options are not silently ignored.

The integrated refscan produces one `a/b/c` coefficient fit, shared across all
echoes. It is combined separately with each echo's sequence-derived
theoretical trajectory, which may differ because of flow compensation. The
code never refits `a/b/c` from later echo images, so reconstruction PSF
calibration does not absorb inter-echo object phase evolution used for
delta-B0 analysis.

## Coil-calibration cache

The output folder can contain files such as:

```text
coil_compression_matrix_ncc<N>_roimgcrop-v1<tag>.npy
csm_acs_ncc<N>_roimgcrop-v1<tag>.npy
csm_full_ncc<N>_roimgcrop-v1<tag>.npy
csm_full_mag_ncc<N>_roimgcrop-v1<tag>.png
csm_full_phase_ncc<N>_roimgcrop-v1<tag>.png
coil_calibration_ncc<N>_roimgcrop-v1<tag>.json
```

`--reuse-coil-calib` requires the matrix, full-resolution CSM, and versioned
manifest. Reuse recomputes the logical ACS identity and validates the source,
geometry, centered-crop algorithm, ACS hash, ESPIRiT settings, dimensions,
finite values, and artifact hashes. Historical stride-derived cache filenames
are ignored and cannot be reused as corrected calibration.

Reuse cached files only when the following are unchanged:

- receiver-coil configuration;
- integrated ACS data;
- acquisition matrix and FOV;
- number of compressed coils;
- sequence geometry;
- relevant preprocessing assumptions.

Use a distinct `--file-tag` for different scans or configurations sharing an output location.


### Mode-specific CSM caches

Native 3D uses:

```text
csm_acs_ncc<N>_roimgcrop-v1<tag>.npy
csm_full_ncc<N>_roimgcrop-v1<tag>.npy
```

Slice2d uses:

```text
csm_acs_ncc<N>_slice2d_roimgcrop-v1<tag>.npy
csm_full_ncc<N>_slice2d_roimgcrop-v1<tag>.npy
```

The coil-compression matrix remains shared because the ESPIRiT backend does
not change coil compression. The mode-specific manifest binds parameters that
are not encoded in the filename, including the ESPIRiT crop threshold.

## NIfTI export

Enable one magnitude NIfTI plus JSON sidecar per echo with:

```text
--save-nifti
```

Also export phase in radians with:

```text
--save-nifti-phase
```

`--save-nifti-phase` implies magnitude NIfTI export.

Important defaults:

```text
output directory:       <out>/nifti
subject token:          TWIX filename stem
suffix:                 GRE
axis roles:             readout phase slice
axis flips:             false false false
Twix coordinate system: LPS
in-plane rotation sign: -1.0
```

Relevant options:

```text
--nifti-out
--nifti-sub
--nifti-suffix
--nifti-axis-roles AXIS0 AXIS1 AXIS2
--nifti-axis-flips BOOL0 BOOL1 BOOL2
--twix-coord-system {LPS,RAS}
--twix-inplane-rot-sign VALUE
--twix-use-fov-for-voxel-size
```

The NIfTI helper uses Siemens TWIX geometry to build the affine and center-crops readout oversampling for NIfTI output. Spatial units are written in millimetres and temporal units in seconds. Each saved NIfTI is reopened to validate header spacing, affine spacing, qform/sform codes, authoritative sform agreement, and qform orientation/voxel size. Expected quaternion round-off is tolerated.

The JSON sidecar records sequence-derived metadata, warning-only `.seq`/TWIX geometry diagnostics, and readout/LIN/PAR direction information. Verify orientation in an independent viewer before quantitative use.

## Outputs

The output folder may contain:

- coil-compression matrix;
- low- and full-resolution coil-sensitivity maps;
- CSM magnitude and phase plots;
- coil-compressed multi-echo k-space;
- fitted PSF phase-deviation arrays and diagnostics for wave data;
- reconstructed complex multi-echo image arrays;
- optional per-echo complex NumPy files;
- optional magnitude and phase NIfTI files with JSON sidecars.

The primary array shapes documented by the script are:

```text
compressed k-space: Nx_os × Ny × Nz × Necho × Ncc
reconstructed image: Nx_os × Ny × Nz × Necho
```
