# Wave-encoded multi-echo GRE with integrated FLASH calibration

This bundle reorganizes the supplied multi-echo wave-GRE sequence into an open-source layout and appends a slab-selective FLASH wave-calibration acquisition to the same Pulseq sequence.

## Main sequence

Run:

```matlab
cd seq
gre_3d_wave_with_flash_calibration
```

The first run prompts for the Pulseq path, optional Safe PNS path, output path, and optional scanner `.asc` file. Settings are saved locally in `seq/gre_flash_path_settings.json`. Pressing Enter for the output path uses MATLAB's current folder. Later runs can reuse all saved settings or update selected entries.

Generated sequence files are written to `generated_seq_v141/` and `generated_seq_v151/` under the selected output root.

## Acquisition order and TWIX routing

1. Multi-echo GRE image acquisition
   - `REF=false`, `IMA=false`, `SET=0`
   - global `LIN`/`PAR`, `ECO=0:(Nechoes-1)`, `AVG=0:(naverage-1)`
   - stored in TWIX `image`
2. FLASH calibration acquisition
   - `REF=true`, `IMA=false`, `ECO=0`, `AVG=0`
   - compact local `LIN`/`PAR` within `SET=0:4`
   - stored in TWIX `refscan`

Default calibration sets:

| SET | Acquisition | Local size |
|---:|---|---:|
| 0 | no-wave, LIN-wide / PAR-narrow | 72 × 1 |
| 1 | sine-wave, LIN-wide / PAR-narrow | 72 × 1 |
| 2 | no-wave, PAR-wide / LIN-narrow | 1 × 72 |
| 3 | cosine-wave, PAR-wide / LIN-narrow | 1 × 72 |
| 4 | no-wave ACS | 32 × 32 |

The logical calibration extent is `LIN × PAR × SET = 72 × 72 × 5`. Depending on loader axis order, a typical raw layout is approximately `Nx_os × Ncoil × 72 × 72 × 5`.

## Geometry

Both GRE and calibration use transverse mapping:

- readout: `x`
- LIN / sine wave: `y`
- PAR / cosine wave / slab selection: `z`

The calibration uses the same slab-selective sinc excitation parameters and FOV as GRE. Its slab rephaser is placed in a standalone block and is not overlapped with the following PE/readout/wave gradients.

## Flow compensation

All GRE flow-compensation flags default to `true`, including initial readout, sine, cosine, PE-y, PAR-z, slab-rephaser, inter-echo, and cosine endpoint M1 corrections. `sys_lowPNS2` is preserved for the FC waveform design. Inter-echo modules are instantiated when the selected `TE` array contains more than one echo.

## Helper organization

- GRE wave and flow-compensation helpers were extracted from the supplied GRE script into individual files without changing their numerical bodies, except that the ADC block helper now accepts an explicit `SET` label.
- Calibration wave helpers are based on the Wave-MPRAGE utility implementations and are renamed with `4Calib` to avoid collision with the GRE helpers:
  - `defineCosineWaveGradient4Calib.m`
  - `defineSineWaveGradient4Calib.m`
  - `makeFixedDurationPreRamp4Calib.m`
  - `makeExtendedTrapezoidAndWaveform4Calib.m`

## Attribution

Author: Yiyun Dong, Athinoula A. Martinos Center for Biomedical Imaging.

The GRE implementation is built based on Berkin's GRE code:

`https://github.com/HarmonizedMRI/megre_label/blob/main/script_writeGradientEcho3D_label_spoil_github_v0.m`

The integration pattern, path-setting behavior, and calibration utility organization follow:

`https://github.com/HarmonizedMRI/wave-mprage`

