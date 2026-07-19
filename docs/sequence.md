# Sequence generation

## Entry point

```text
seq/gre_3d_wave_with_flash_calibration.m
```

Run from MATLAB:

```matlab
cd seq
gre_3d_wave_with_flash_calibration
```

The sequence code is intentionally kept separate from the Python reconstruction environment. `uv` and `pip` manage only reconstruction dependencies; they do not install MATLAB or Pulseq.

## Path configuration

The script discovers its own folder, adds `seq/utils/`, and asks for machine-specific paths on first use. The resulting settings are stored locally in:

```text
seq/gre_flash_path_settings.json
```

The configuration contains paths for:

- Pulseq
- optional Safe PNS Prediction code
- output root
- optional scanner `.asc` file

The JSON file is ignored by Git because these paths are specific to each workstation. Later runs can reuse or update the saved settings. Leaving the output path blank uses MATLAB's current folder.

## Output formats

The script can write:

```text
generated_seq_v141/   legacy Pulseq v1.4.1 files
generated_seq_v151/   current Pulseq v1.5.x files
```

The current script enables legacy v1.4.1 output as well as the current format. Confirm the format supported by the scanner interpreter before use.

## Integrated acquisition order

One `.seq` file contains two consecutive acquisitions.

### 1. Multi-echo GRE image acquisition

```text
REF = false
IMA = false
SET = 0
ECO = 0 ... Nechoes-1
AVG = 0 ... naverage-1
```

The image data are expected in the Siemens TWIX `image` container.

### 2. FLASH wave-calibration acquisition

```text
REF = true
IMA = false
ECO = 0
AVG = 0
SET = 0 ... 4
```

The calibration data are expected in the TWIX `refscan` container.

## Calibration SET layout

The default integrated calibration convention is:

| SET | Acquisition | Logical local size |
|---:|---|---:|
| 0 | no-wave, LIN-wide / PAR-narrow projection | 72 × 1 |
| 1 | sine-wave, LIN-wide / PAR-narrow projection | 72 × 1 |
| 2 | no-wave, PAR-wide / LIN-narrow projection | 1 × 72 |
| 3 | cosine-wave, PAR-wide / LIN-narrow projection | 1 × 72 |
| 4 | no-wave ACS | 32 × 32 |

The logical calibration extent is therefore:

```text
LIN × PAR × SET = 72 × 72 × 5
```

Depending on loader axis ordering, the raw refscan commonly resembles:

```text
Nx_os × Ncoil × 72 × 72 × 5
```

The reconstruction validates this integrated layout against the definitions stored in the matching `.seq` file.

## Geometry

The verified implementation uses transverse mapping:

```text
readout       -> x
LIN / sine    -> y
PAR / cosine  -> z
slab select   -> z
```

The GRE reconstruction currently supports this transverse orientation only. Use the exact `.seq` file that was executed for the measurement.

The calibration uses the same FOV and slab-selective excitation convention as the GRE acquisition. Its slab rephaser is placed in a standalone block rather than overlapping the following phase-encoding, readout, or wave gradients.

## Wave and no-wave acquisitions

The sequence can generate a two-axis wave acquisition or a fully no-wave acquisition. The reconstruction supports:

- both sine and cosine wave gradients enabled;
- both wave gradients disabled.

One-axis wave acquisitions—sine only or cosine only—are rejected by the current reconstruction.

The reconstruction's default `--wave-mode auto` inspects the image trajectory and selects wave or no-wave processing. Explicit `wave` or `nowave` mode also acts as a consistency check.

## K-space ordering

The integrated implementation uses matching GRE and calibration ordering. The reconstruction reads `KspaceOrdering` from the `.seq` definitions and derives the default PSF signs:

```text
negative_to_positive -> yflip = +1, zflip = +1
positive_to_negative -> yflip = -1, zflip = -1
```

Manual `--yflip` and `--zflip` overrides exist for controlled debugging, but the sequence-derived values should normally be used.

## Flow compensation

The GRE implementation includes flow-compensation controls for the initial and inter-echo readout/wave/phase-encoding modules. The current default sequence configuration enables the verified flow-compensation path, including readout, sine, cosine, LIN, PAR, slab-rephaser, and inter-echo corrections where applicable.

Inter-echo modules are instantiated when more than one echo is requested. The low-PNS system definition remains part of the flow-compensated waveform design.

The reconstruction reads and reports the `UseFlowComp` sequence definition but does not redesign or alter the acquisition waveforms.

## Sequence definitions used by reconstruction

The matching `.seq` file supplies acquisition metadata such as:

- `Name`
- `Nx`, `Ny`, `Nz`, and optionally `Nx_os`
- `FOV`
- `ReadoutOversamplingFactor`
- `Nechoes` and `TE`
- `Averages`
- `Ry`, `Rz`, `Ny_meas`, and `Nz_meas`
- calibration dimensions and ACS SET ID
- `WaveSinChannel` and `WaveCosChannel`
- `KspaceOrdering`
- `UseFlowComp`
- orientation mapping

Do not reconstruct using a `.seq` file from a different scan or a differently configured sequence generation.

## Before scanning

Review the generated sequence using the validation tools appropriate for the scanner and local safety workflow. In particular, confirm:

- sequence timing passes;
- gradient and slew limits pass;
- expected echo times and TR are achieved;
- PNS/CNS checks pass when enabled;
- forbidden-frequency checks pass when available;
- the output Pulseq format is supported by the scanner;
- the filename can be interpreted by the scanner environment;
- the prescribed orientation is transverse;
- the `.seq` file is retained with the acquired TWIX data.
