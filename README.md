# Wave-encoded multi-echo GRE with flow compensation

Pulseq-based 3D single- or multi-echo Wave-GRE sequence generation with integrated FLASH wave calibration and Python reconstruction from Siemens TWIX data.

The repository has two intentionally separate parts:

- MATLAB generates the integrated GRE + FLASH calibration sequence.
- Python reconstructs wave or no-wave GRE data from the resulting Siemens TWIX measurement.

`pyproject.toml` defines only the Python reconstruction environment. This repository is not configured as an installable Python package and does not need to be published to PyPI.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── docs/
│   ├── sequence.md
│   ├── reconstruction.md
│   └── troubleshooting.md
├── seq/
│   ├── gre_3d_wave_with_flash_calibration.m
│   └── utils/
├── recon/
│   ├── recon_wave_gre_from_twix_integrated_nifti.py
│   └── utils/
└── external/
```

Recommended entry points:

```text
seq/gre_3d_wave_with_flash_calibration.m
recon/recon_wave_gre_from_twix_integrated_nifti.py
```

## Requirements

### Sequence generation

- MATLAB
- Pulseq MATLAB toolbox

Optional scanner-safety checks can use Safe PNS Prediction, a scanner `.asc` file, and an existing `forbiddenFreqCheck.m` helper.

### Reconstruction

- Python 3.11
- CPU reconstruction dependencies defined in `pyproject.toml`
- Optional NVIDIA GPU and CUDA 12-compatible CuPy for faster ESPIRiT calibration

Current device behavior:

| Step | Device |
|---|---|
| Coil-compression estimation and application | CPU |
| ESPIRiT sensitivity-map calibration | GPU when available, otherwise CPU |
| Wave/no-wave CG-SENSE | CPU |

A GPU is optional. With `--espirit-device auto`, the reconstruction uses a compatible visible GPU when available and otherwise falls back to CPU.

## Clone

```bash
git clone --recurse-submodules https://github.com/HarmonizedMRI/wave-gre-flow-comp.git
cd wave-gre-flow-comp
```

## Install the reconstruction environment

### Recommended: uv

Install `uv`, then create the locked CPU-capable environment:

```bash
uv sync --locked
```

Run commands inside the environment with `uv run`:

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py --help
```

To include CUDA 12 CuPy for GPU-assisted ESPIRiT:

```bash
uv sync --locked --group gpu
```

The committed `uv.lock` records the exact resolved Python dependencies. The host NVIDIA driver and GPU remain system requirements and are not included in the lockfile.

### Alternative: pip

`pip` 25.1 or newer can install the standardized dependency groups from `pyproject.toml`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group recon
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

For CUDA 12 CuPy:

```bash
python -m pip install --group gpu
```

The `pip` route resolves compatible versions at installation time. Use `uv sync --locked` when exact lockfile reproduction is important.

## Generate the integrated sequence

Open MATLAB and run:

```matlab
cd seq
gre_3d_wave_with_flash_calibration
```

On the first run, enter the requested paths. Machine-specific settings are saved beside the script in:

```text
seq/gre_flash_path_settings.json
```

Leaving the output path blank uses MATLAB's current folder. Generated sequence files are written under:

```text
generated_seq_v141/
generated_seq_v151/
```

See [Sequence generation](docs/sequence.md) for acquisition order, calibration SET layout, geometry, flow compensation, path handling, and output behavior.

## Reconstruct an integrated acquisition

The measurement is expected to contain:

```text
image    -> single- or multi-echo GRE k-space
refscan  -> four FLASH projection-calibration sets plus one ACS set
```

Review all command-line arguments:

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py --help
```

Typical reconstruction with automatic wave/no-wave detection, automatic ESPIRiT device selection, and NIfTI export:

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py \
    --twix /path/to/meas_wave_gre.dat \
    --seq /path/to/matching_wave_gre.seq \
    --out /path/to/reconstruction \
    --wave-mode auto \
    --espirit-device auto \
    --save-nifti \
    --save-nifti-phase
```

Force a fully CPU-capable run:

```bash
uv run python recon/recon_wave_gre_from_twix_integrated_nifti.py \
    --twix /path/to/meas_wave_gre.dat \
    --seq /path/to/matching_wave_gre.seq \
    --out /path/to/reconstruction \
    --wave-mode auto \
    --espirit-device cpu
```

Use `--espirit-device gpu --espirit-gpu-index 0` to require a specific GPU. Explicit GPU mode raises an error instead of silently falling back when the requested GPU is unavailable.

See [Reconstruction](docs/reconstruction.md) for supported acquisition assumptions, the pipeline, complete argument guidance, cache reuse, outputs, and NIfTI conventions.

## Documentation

- [Sequence generation](docs/sequence.md)
- [Reconstruction](docs/reconstruction.md)
- [Troubleshooting](docs/troubleshooting.md)

## Attribution

Author: Yiyun Dong, Athinoula A. Martinos Center for Biomedical Imaging.

The GRE implementation is based on Berkin's GRE code in the HarmonizedMRI `megre_label` repository. The integrated calibration organization and path-setting workflow follow the Wave-MPRAGE project.

## License

MIT License. See `LICENSE`.
