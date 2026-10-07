# Sequence generation

## Entry points

```text
seq/gre_3d_wave_with_flash_calibration_tra.m
seq/gre_3d_wave_with_flash_calibration_sag.m
```

Run from MATLAB:

```matlab
cd seq
gre_3d_wave_with_flash_calibration_tra  % transverse
gre_3d_wave_with_flash_calibration_sag  % sagittal
```

The sequence code is intentionally kept separate from the Python reconstruction environment. `uv` and `pip` manage only reconstruction dependencies; they do not install MATLAB or Pulseq.

## Path configuration

Each entry discovers its own folder and adds `seq/utils/`. Its initial path
block contains the machine-specific locations for:

- Pulseq
- optional Safe PNS Prediction code
- the independent evaluation output root
- optional scanner `.asc` file

Review that block before running on another workstation. Generated files are
not written beside MATLAB source files.

## Output formats

The parity-aware arbitrary-gradient boundary samples require Pulseq v1.5.x.
The entries write v1.5.1 files below:

```text
evaluation/output/v1.5.1/high_slew_wave_gre_tra/
evaluation/output/v1.5.1/high_slew_wave_gre_sag/
```

Legacy v1.4.1 output is disabled because those boundary values do not survive
a write/read round trip. Confirm v1.5.x support in the scanner interpreter.

## High-slew parity-aware full-FC wave cases

Both entry points support the coupled C10/A12.732, C20/A6.3662, and
C25/A5.093 mT/m cases. Set `Ncycles` to 10, 20, or 25 and
set `centerWaveAroundNowave` to `false` (`sinzero`) or `true` (`sinctr`)
before running either source; the default is `sinzero`. Both sources deliberately require every initial,
inter-echo, slab, readout, LIN, PAR, sine, and cosine flow-compensation
component to remain enabled; partial-FC combinations are rejected.

The TRA entry treats TE1=10 ms, echo spacing=10 ms, and TR=30 ms as
minimum timing targets. It applies the smallest raster-exact increase needed
by the selected full-FC case and either centering state; currently C10 uses
TE=[10, 20.14] ms while
C20/C25 retain TE=[10, 20] ms. The emitted values are stored in the sequence
definitions and checked after reload.

Only active sine/cosine samples use the 180 T/m/s physical envelope. Wave
ramps, PE, spoilers, rephasers, and prescribed-M0/M1 FC lobes use the
63 T/m/s low-PNS envelope. Generated v1.5.1 files are written below
`evaluation/output/v1.5.1/high_slew_wave_gre_tra/` or
`evaluation/output/v1.5.1/high_slew_wave_gre_sag/`. Run the matching
`evaluation/validate_wave_gre_tra.m` or `evaluation/validate_wave_gre_sag.m`
after generating all six case/state combinations; each validator reloads the
files and checks every echo's center-line sine M0/M1 and the integrated
calibration contract. The SAG validator additionally matches the appended
calibration tail against the accepted standalone calibration files.

PNS and forbidden-frequency checks are deliberately outside that validator.
Passing its timing, trajectory, labels, and hardware-envelope checks does not
make a generated file scanner-safe.

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

The transverse entry uses:

```text
readout       -> x
LIN / sine    -> y
PAR / cosine  -> z
slab select   -> z
```

The sagittal entry uses RO=z, LIN=y/sine, PAR=x/cosine, and slab select=x.
The GRE reconstruction currently supports the transverse orientation only.
Use the exact `.seq` file that was executed for the measurement.

The calibration uses the same FOV and slab-selective excitation convention as the GRE acquisition. Its slab rephaser is placed in a standalone block rather than overlapping the following phase-encoding, readout, or wave gradients.

## Scanner protocol UI recommendations

For the verified transverse acquisition:

- set the in-plane phase-encoding direction to right-to-left (`R -> L`);
- disable neck-coil elements when they are not required for the target anatomy, because strong neck or shoulder signal can contaminate the projection calibration and wave reconstruction;
- ensure that the prescribed FOV box covers the complete signal-producing anatomy. Signal outside the FOV can wrap into the acquisition, and the resulting wave-induced aliasing may not be fully resolvable.

Confirm the displayed phase-encoding direction on the scanner UI and follow local coil-selection and safety procedures. Keep the exact generated `.seq` file with the acquired TWIX data.

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
- the scanner phase-encoding direction is `R -> L`;
- unnecessary neck-coil elements are disabled;
- the FOV box covers the complete signal-producing volume;
- the `.seq` file is retained with the acquired TWIX data.
