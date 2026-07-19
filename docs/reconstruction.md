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
5. Load the integrated ACS from TWIX `refscan` SET 4.
6. Estimate the requested coil-compression matrix on CPU.
7. Generate low-resolution ESPIRiT maps on the selected CPU or GPU device.
8. Interpolate and normalize the sensitivity maps.
9. Load the multi-echo GRE image k-space and apply coil compression on CPU.
10. For wave data, fit FLASH projection phase deviations and construct calibrated echo-specific PSFs.
11. Run wave or no-wave CG-SENSE on CPU for each echo.
12. Save NumPy arrays, diagnostic plots, and optional NIfTI outputs.

## CPU and GPU behavior

The full reconstruction is not moved to GPU.

| Operation | Current device |
|---|---|
| Coil-compression estimation | CPU / NumPy and SciPy |
| Coil-compression application | CPU / PyTorch tensor |
| ESPIRiT calibration | selectable SigPy CPU or GPU |
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

## Main reconstruction options

| Option | Default | Purpose |
|---|---:|---|
| `--wave-mode {auto,wave,nowave}` | `auto` | Select or detect the image reconstruction model |
| `--file-tag TEXT` | empty | Append a sanitized tag to cache and result filenames |
| `--ncc N` | `12` | Number of virtual coils retained |
| `--reuse-coil-calib` | off | Reuse compatible cached coil compression and CSM files |
| `--espirit-device {auto,cpu,gpu}` | `auto` | Select ESPIRiT execution device |
| `--espirit-gpu-index N` | `0` | Select CUDA GPU index |
| `--cg-iters N` | `50` | Maximum CG iterations |
| `--cg-tol VALUE` | `1e-6` | Relative CG stopping tolerance |
| `--yflip {-1,1}` | sequence-derived | Override LIN PSF sign |
| `--zflip {-1,1}` | sequence-derived | Override PAR PSF sign |
| `--save-echo-npy` | off | Save one complex NumPy file per echo |
| `--validate-only` | off | Validate sequence-derived configuration without reading TWIX |

Use `--help` as the authoritative complete argument reference for the checked-out code.

## Coil-calibration cache

The output folder can contain files such as:

```text
coil_compression_matrix_ncc<N><tag>.npy
csm_acs_ncc<N><tag>.npy
csm_full_ncc<N><tag>.npy
csm_full_mag_ncc<N><tag>.png
csm_full_phase_ncc<N><tag>.png
```

`--reuse-coil-calib` reuses the coil-compression matrix and full-resolution CSM only when both required cache files are present. The script validates their dimensions before use.

Reuse cached files only when the following are unchanged:

- receiver-coil configuration;
- integrated ACS data;
- acquisition matrix and FOV;
- number of compressed coils;
- sequence geometry;
- relevant preprocessing assumptions.

Use a distinct `--file-tag` for different scans or configurations sharing an output location.

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

The NIfTI helper uses Siemens TWIX geometry to build the affine and center-crops readout oversampling for NIfTI output. Verify orientation in an independent viewer before quantitative use.

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
