# Troubleshooting

## Installation

### `uv sync --locked` reports that the lockfile is stale

Confirm that `pyproject.toml` and `uv.lock` came from the same update bundle and were copied to the repository root. Do not edit `uv.lock` manually.

To intentionally refresh dependency resolution after changing `pyproject.toml`:

```bash
uv lock
uv sync
```

Commit both files together.

### `pip install --group recon` is not recognized

Dependency-group installation requires pip 25.1 or newer:

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group recon
```

### Python version mismatch

The environment is defined for Python 3.11 through 3.13, with `.python-version` selecting Python 3.11 by default for `uv` and compatible version managers.

Check:

```bash
python --version
uv run python --version
```

### Import errors when running from another folder

Run the reconstruction from the repository root:

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py --help
```

The script imports utilities from `recon/utils/` using its current verified layout.

## CUDA and ESPIRiT

### CuPy is not installed

This is expected in the default CPU-capable environment. Use:

```text
--espirit-device auto
```

or:

```text
--espirit-device cpu
```

The reconstruction remains usable without CuPy; ESPIRiT runs on CPU.

### GPU requested but unavailable

Explicit GPU mode is intentionally strict:

```text
--espirit-device gpu
```

It fails when CuPy cannot import, CUDA cannot initialize, no GPU is visible, or the selected index is invalid.

Check:

```bash
uv run python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

Also verify the NVIDIA driver outside Python.

### Wrong CuPy CUDA family

The optional `gpu` group installs `cupy-cuda12x`. It is intended for a CUDA 12-compatible host. Do not install multiple CuPy variants in the same environment.

For CPU-only systems, use the default `recon` group and omit `gpu`.

### CPU ESPIRiT is slow

CPU fallback prioritizes portability rather than speed. ESPIRiT operates on a reduced, coil-compressed calibration volume, but it can still take noticeably longer than GPU execution.

After generating compatible maps once, `--reuse-coil-calib` can avoid repeating calibration.

### More than 32 receiver channels

With more than 32 physical receiver channels, ESPIRiT may place substantial pressure on GPU memory even though the calibration volume is reduced and coil-compressed. Depending on the available hardware, consider:

```text
--espirit-device cpu
```

CPU execution may be slower, but it avoids GPU-memory limitations and CUDA-specific failures. A large-memory GPU can still be preferable; select the device based on available CPU RAM, CPU performance, GPU memory, and expected runtime.

## Input and sequence validation

### `--validate-only` still asks for `--twix` and `--out`

This reflects the current verified parser. Supply a placeholder TWIX path and a temporary output folder:

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py \
    --twix unused.dat \
    --seq /path/to/scan.seq \
    --out /tmp/wave-gre-validation \
    --validate-only
```

The placeholder TWIX file is not read in validation-only mode.

### Pulseq sequence file not found

Use the exact generated `.seq` file that was executed on the scanner. Relative paths are resolved from the shell's current working directory.

### Required sequence definition is missing

The reconstruction depends on metadata embedded in the matching `.seq` file. A sequence from an older or unrelated workflow may not contain all required definitions.

Do not substitute a similarly named sequence file from another scan.

### Unsupported orientation

The reconstruction currently supports transverse geometry only. It expects:

```text
readout = x
LIN/sine = y
PAR/cosine = z
```

A sagittal or coronal sequence will be rejected.

### More than one average

The current reconstruction supports exactly one average. A sequence reporting `Averages > 1` is rejected rather than combining averages implicitly.

### Sine-only or cosine-only wave acquisition

The supported modes are:

- two-axis wave: sine and cosine enabled;
- no-wave: sine and cosine disabled.

One-axis wave acquisitions are intentionally rejected.

### Explicit wave mode conflicts with trajectory

`--wave-mode wave` and `--wave-mode nowave` are consistency checks. Use the mode that matches the sequence trajectory, or use the recommended default:

```text
--wave-mode auto
```

### Image ADC line count mismatch

This usually means the `.seq` file does not match the TWIX measurement or the acquisition definitions are inconsistent. Verify:

- matrix size;
- acceleration factors;
- measured LIN/PAR counts;
- echo count;
- average count;
- appended calibration line count.

### Integrated refscan shape mismatch

The expected refscan is five-dimensional after loading and contains at least five SETs. Verify that the measurement includes the appended FLASH calibration and that TWIX routing placed it in `refscan`.

Default SET meanings are:

```text
0 no-wave LIN projection
1 sine-wave LIN projection
2 no-wave PAR projection
3 cosine-wave PAR projection
4 no-wave ACS
```

### K-space ordering or PSF sign looks wrong

The reconstruction derives `yflip` and `zflip` from `KspaceOrdering`. Manual overrides should be used only when independently justified:

```text
--yflip -1|1
--zflip -1|1
```

A mismatch often indicates that the wrong `.seq` file was paired with the TWIX data.

### PSF coefficient fit becomes unstable or blows up

The default processing is:

```text
--psf-coefficient-processing smooth
```

First verify the matching `.seq` file, sequence-derived PSF signs, calibration SET layout, and raw PSF diagnostic plots. Also check acquisition setup: neck or shoulder signal contamination and anatomy outside the prescribed FOV can corrupt part of the projection calibration.

When `a(kx)`, `b(kx)`, or `c(kx)` is reliable only over a known high-fidelity readout interval, use the optional sine-plus-line substitution:

```bash
--psf-coefficient-processing sine-line \
--psf-fit-kx-min 200 \
--psf-fit-kx-max 512
```

The range is half-open, `[kx_min, kx_max)`, and refers to oversampled-readout indices. Choose the bounds from the stable part of the raw coefficient curves. The model is fitted inside that interval and extrapolated across the complete readout. It replaces smoothing and is not followed by another smoothing step. Both bounds are required.

Do not use `sine-line` merely to hide a mismatched sequence, incorrect ordering, or poor acquisition coverage.

## Acquisition setup

### Neck or shoulder signal contaminates wave calibration

When neck-coil elements are not required for the target anatomy, disable them in the scanner protocol UI. Strong neck or shoulder signal can enter the FLASH projection calibration and contaminate the fitted PSF coefficients. Follow local coil-selection and safety procedures.

### Anatomy extends outside the FOV box

Prescribe the FOV box to cover the complete signal-producing volume. Signal outside the FOV can wrap into the image, and the corresponding wave-induced aliasing may not be fully resolvable by the reconstruction. Increasing only the reconstruction matrix does not recover anatomy that was acquired outside the prescribed FOV.

### Phase-encoding direction is not R-to-L

The verified transverse Wave-GRE protocol uses right-to-left phase encoding (`R -> L`). Confirm this setting on the scanner UI before acquisition. If a different direction was used, review the sequence/TWIX direction diagnostics and do not assume the verified orientation and aliasing behavior applies unchanged.

## Coil-calibration cache

### Reconstruction failed after ESPIRiT maps were generated

When coil compression and ESPIRiT completed successfully but a later PSF, CG-SENSE, or output step failed, rerun with:

```text
--reuse-coil-calib
```

The correct option name is `--reuse-coil-calib`—not `--reuse-exist-calib`. Use the same output folder, `--file-tag`, `--ncc`, receiver-coil configuration, integrated ACS, matrix, FOV, and sequence geometry. The script checks cache dimensions, but the filename cannot encode every acquisition property.

### Cached calibration shape error

Delete or stop reusing the cache when acquisition geometry or coil configuration changes. The cache filename includes `ncc` and the optional file tag, but it cannot encode every acquisition property.

Use a new tag:

```text
--file-tag scan02
```

or rerun without:

```text
--reuse-coil-calib
```

### Requested `ncc` exceeds physical coil count

Lower `--ncc` to a value no greater than the number of coils in the integrated refscan. The default is 12.

## Memory and runtime

### Out-of-memory error

The full CG-SENSE reconstruction runs on CPU, and multi-echo arrays can be large. Possible mitigations include:

- close other memory-intensive applications;
- reduce the number of compressed coils with care;
- reconstruct on a machine with more RAM;
- avoid unnecessary duplicate output arrays;
- confirm the matrix and oversampling definitions are correct.

Changing `--cg-iters` affects runtime but usually does not materially reduce the peak size of the principal arrays.

### CG reconstruction does not converge as expected

Confirm the `.seq` file, wave mode, coil maps, calibration cache, PSF signs, and acquisition ordering before changing numerical parameters. Then review:

```text
--cg-iters
--cg-tol
```

Do not reuse sensitivity maps from a different coil setup or geometry.

## NIfTI output

### NiBabel import error

Install or resynchronize the reconstruction environment:

```bash
uv sync --locked
```

or:

```bash
python -m pip install --group recon
```

### NIfTI spacing appears to be in micrometres

The updated export path writes spatial units as millimetres and reopens each file to verify header and affine spacing. Review the printed NIfTI validation report and the JSON sidecar. For a nominal 1 mm acquisition, header and affine voxel sizes should be close to `1.0`, not `0.001`.

Use `--twix-use-fov-for-voxel-size` only as a deliberate diagnostic override. The normal spacing source is the sequence geometry after comparison with TWIX.

### NIfTI appears flipped or rotated

First verify that the correct TWIX and `.seq` files were used. Then inspect:

```text
--nifti-axis-roles
--nifti-axis-flips
--twix-coord-system
--twix-inplane-rot-sign
--twix-use-fov-for-voxel-size
```

The GRE defaults apply no additional array flips because the verified image LIN/PAR ordering is negative-to-positive. Validate the final orientation in an independent NIfTI viewer using known anatomical landmarks.

### Phase NIfTI was not written

Use:

```text
--save-nifti-phase
```

This also enables magnitude NIfTI output.
