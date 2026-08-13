#!/usr/bin/env bash
set -euo pipefail

# File/workflow arguments belong to this wrapper. Options inside the three
# delimited sections are passed unchanged to BART or wave_to_nifti.py.

usage() {
    cat <<'EOF'
Usage:
  run_wave_recon.sh \
    --bart-input PATH \
    --bart-output PATH \
    --maps-source bart|existing \
    --twix FILE.dat \
    --seq FILE.seq \
    [--nifti-output PATH] \
    [--existing-maps CFL_BASENAME] \
    [--save-phase] \
    [--ecalib-options BART_OPTIONS... --end-ecalib-options] \
    [--wave-options BART_OPTIONS... --end-wave-options] \
    [--nifti-options OPTIONS... --end-nifti-options]

Required wrapper arguments:
  --bart-input PATH       Directory containing manifest.json and exported CFLs.
  --bart-output PATH      Directory for BART maps and reconstructed images.
  --maps-source SOURCE    'bart' runs ecalib; 'existing' skips ecalib.
  --twix FILE.dat         Matching Siemens TWIX file for NIfTI geometry.
  --seq FILE.seq          Matching Pulseq file for NIfTI metadata.

Optional wrapper arguments:
  --nifti-output PATH     Directory for converted NIfTI files. Defaults to
                          BART_OUTPUT/nifti.
  --existing-maps BASE    Existing ESPIRiT CFL basename. Defaults to
                          BART_INPUT/coil_sens with --maps-source existing.
  --save-phase            Also write phase NIfTI files.
  -h, --help              Show this help.

Direct option sections:
  --ecalib-options ... --end-ecalib-options
      Passed unchanged to `bart ecalib`. The command visibly supplies `-m 1`
      because the NIfTI converter accepts one ESPIRiT map set.

  --wave-options ... --end-wave-options
      Passed unchanged to `bart wave`. Common BART wave options are:
        -w          wavelet regularization
        -l          locally low-rank (LLR) regularization
        -r VALUE    regularization strength
        -b VALUE    LLR block size
        -f          FISTA instead of IST
        -H          Hogwild IST/FISTA
        -s VALUE    step size
        -i VALUE    maximum iterations
        -t VALUE    convergence tolerance
        -c VALUE    continuation value
        -e VALUE    known maximum eigenvalue
        -g          GPU
        -v          split real and imaginary components

  --nifti-options ... --end-nifti-options
      Passed unchanged to wave_to_nifti.py.

Example using BART ecalib and wavelet/FISTA reconstruction:
  run_wave_recon.sh \
    --bart-input ./bart_inputs \
    --bart-output ./bart_output \
    --maps-source bart \
    --twix ./meas.dat \
    --seq ./sequence.seq \
    --ecalib-options -c 0.8 --end-ecalib-options \
    --wave-options -w -r 0.001 -f -i 100 -t 1e-6 --end-wave-options

Example using existing maps and LLR/FISTA reconstruction:
  run_wave_recon.sh \
    --bart-input ./bart_inputs \
    --bart-output ./bart_output \
    --maps-source existing \
    --existing-maps ./bart_inputs/coil_sens \
    --twix ./meas.dat \
    --seq ./sequence.seq \
    --nifti-output ./custom-nifti \
    --wave-options -l -r 0.002 -b 8 -f -i 100 --end-wave-options

Environment overrides:
  BART_BIN  BART executable (default: bart)
  PYTHON_BIN  Python interpreter used for conversion (default: python from
              the active Conda environment or virtual environment)
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 2
}

require_value() {
    [[ -n "${2:-}" ]] || fail "$1 requires a value."
}

require_cfl_pair() {
    [[ -f "$1.hdr" && -f "$1.cfl" ]] ||
        fail "Missing BART CFL pair: $1.{hdr,cfl}"
}

strip_cfl_extension() {
    local path="$1"
    path="${path%.hdr}"
    path="${path%.cfl}"
    printf '%s\n' "$path"
}

array_contains() {
    local wanted="$1"
    shift
    local item
    for item in "$@"; do
        [[ "$item" == "$wanted" ]] && return 0
    done
    return 1
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

BART_INPUT=""
BART_OUTPUT=""
MAPS_SOURCE=""
EXISTING_MAPS=""
TWIX_FILE=""
SEQUENCE_FILE=""
NIFTI_OUTPUT=""
SAVE_PHASE=0
ECALIB_OPTIONS=()
WAVE_OPTIONS=()
NIFTI_OPTIONS=()

while (($#)); do
    case "$1" in
        --bart-input)
            require_value "$1" "${2:-}"; BART_INPUT="${2%/}"; shift 2 ;;
        --bart-output)
            require_value "$1" "${2:-}"; BART_OUTPUT="${2%/}"; shift 2 ;;
        --maps-source)
            require_value "$1" "${2:-}"; MAPS_SOURCE="$2"; shift 2 ;;
        --existing-maps)
            require_value "$1" "${2:-}"; EXISTING_MAPS="$2"; shift 2 ;;
        --twix)
            require_value "$1" "${2:-}"; TWIX_FILE="$2"; shift 2 ;;
        --seq)
            require_value "$1" "${2:-}"; SEQUENCE_FILE="$2"; shift 2 ;;
        --nifti-output)
            require_value "$1" "${2:-}"; NIFTI_OUTPUT="${2%/}"; shift 2 ;;
        --save-phase)
            SAVE_PHASE=1; shift ;;
        --ecalib-options)
            shift
            while (($#)) && [[ "$1" != "--end-ecalib-options" ]]; do
                ECALIB_OPTIONS+=("$1"); shift
            done
            (($#)) || fail "--ecalib-options requires --end-ecalib-options."
            shift
            ;;
        --wave-options)
            shift
            while (($#)) && [[ "$1" != "--end-wave-options" ]]; do
                WAVE_OPTIONS+=("$1"); shift
            done
            (($#)) || fail "--wave-options requires --end-wave-options."
            shift
            ;;
        --nifti-options)
            shift
            while (($#)) && [[ "$1" != "--end-nifti-options" ]]; do
                NIFTI_OPTIONS+=("$1"); shift
            done
            (($#)) || fail "--nifti-options requires --end-nifti-options."
            shift
            ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            fail "Unknown wrapper argument: $1. Put BART flags inside their option section."
            ;;
    esac
done

[[ -n "$BART_INPUT" ]] || fail "--bart-input is required."
[[ -n "$BART_OUTPUT" ]] || fail "--bart-output is required."
[[ -n "$MAPS_SOURCE" ]] || fail "--maps-source is required."
[[ -n "$TWIX_FILE" ]] || fail "--twix is required."
[[ -n "$SEQUENCE_FILE" ]] || fail "--seq is required."
[[ "$MAPS_SOURCE" == "bart" || "$MAPS_SOURCE" == "existing" ]] ||
    fail "--maps-source must be 'bart' or 'existing'."
[[ -f "$BART_INPUT/manifest.json" ]] ||
    fail "BART input manifest not found: $BART_INPUT/manifest.json"
[[ -f "$TWIX_FILE" ]] || fail "TWIX file not found: $TWIX_FILE"
[[ -f "$SEQUENCE_FILE" ]] || fail "Pulseq file not found: $SEQUENCE_FILE"

BART_EXECUTABLE="${BART_BIN:-bart}"
PYTHON_EXECUTABLE="${PYTHON_BIN:-python}"
[[ -n "$NIFTI_OUTPUT" ]] || NIFTI_OUTPUT="$BART_OUTPUT/nifti"
command -v "$BART_EXECUTABLE" >/dev/null 2>&1 ||
    fail "BART executable not found: $BART_EXECUTABLE"
command -v "$PYTHON_EXECUTABLE" >/dev/null 2>&1 ||
    fail "Python interpreter not found: $PYTHON_EXECUTABLE. Activate the intended Conda environment or virtual environment first."

# A second -m could conflict with the visible, required one-map setting below.
array_contains -m "${ECALIB_OPTIONS[@]}" &&
    fail "Do not pass -m in --ecalib-options; this workflow explicitly uses -m 1."

USES_WAVELET=0
USES_LLR=0
array_contains -w "${WAVE_OPTIONS[@]}" && USES_WAVELET=1
array_contains -l "${WAVE_OPTIONS[@]}" && USES_LLR=1
((USES_WAVELET && USES_LLR)) &&
    fail "BART wave options -w and -l are mutually exclusive."
if ((!USES_WAVELET && !USES_LLR)); then
    for regularized_option in -r -b -f -H -s -c; do
        array_contains "$regularized_option" "${WAVE_OPTIONS[@]}" &&
            fail "$regularized_option requires -w or -l in --wave-options."
    done
fi
array_contains -b "${WAVE_OPTIONS[@]}" && ((!USES_LLR)) &&
    fail "BART wave option -b is only valid with LLR regularization (-l)."

mkdir -p "$BART_OUTPUT" "$NIFTI_OUTPUT"

if [[ "$MAPS_SOURCE" == "bart" ]]; then
    require_cfl_pair "$BART_INPUT/kspace_calib"
    ESPIRIT_MAPS="$BART_OUTPUT/coil_sens_bart"

    echo "Running BART ESPIRiT calibration:"
    print_command \
        "$BART_EXECUTABLE" ecalib -m 1 "${ECALIB_OPTIONS[@]}" \
        "$BART_INPUT/kspace_calib" "$ESPIRIT_MAPS"

    "$BART_EXECUTABLE" ecalib -m 1 "${ECALIB_OPTIONS[@]}" \
        "$BART_INPUT/kspace_calib" \
        "$ESPIRIT_MAPS"
else
    [[ -n "$EXISTING_MAPS" ]] || EXISTING_MAPS="$BART_INPUT/coil_sens"
    ESPIRIT_MAPS="$(strip_cfl_extension "$EXISTING_MAPS")"
    require_cfl_pair "$ESPIRIT_MAPS"
    echo "Skipping BART ecalib; using existing ESPIRiT maps: $ESPIRIT_MAPS"
fi

# manifest.json is authoritative for echo order and input basenames.
MANIFEST_KSPACE_OUTPUT="$(python3 - "$BART_INPUT/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    echoes = json.load(stream).get("echoes")
if not isinstance(echoes, list) or not echoes:
    raise SystemExit("manifest.json contains no echo entries")
for expected_echo, entry in enumerate(echoes, start=1):
    if entry.get("echo") != expected_echo:
        raise SystemExit("manifest echoes must be consecutive and start at 1")
    basename = entry.get("wave_kspace")
    if not isinstance(basename, str) or not basename:
        raise SystemExit(f"manifest echo {expected_echo} has no wave_kspace basename")
    print(basename)
PY
)" || fail "Unable to read echo inputs from manifest.json."
mapfile -t WAVE_KSPACE_BASENAMES <<<"$MANIFEST_KSPACE_OUTPUT"

for WAVE_KSPACE_NAME in "${WAVE_KSPACE_BASENAMES[@]}"; do
    [[ "$WAVE_KSPACE_NAME" == wave_kspace* ]] ||
        fail "Invalid wave k-space basename in manifest: $WAVE_KSPACE_NAME"

    ECHO_SUFFIX="${WAVE_KSPACE_NAME#wave_kspace}"
    WAVE_KSPACE="$BART_INPUT/$WAVE_KSPACE_NAME"
    WAVE_PSF="$BART_INPUT/psf$ECHO_SUFFIX"
    WAVE_IMAGE="$BART_OUTPUT/image_wave$ECHO_SUFFIX"
    require_cfl_pair "$WAVE_KSPACE"
    require_cfl_pair "$WAVE_PSF"

    echo "Running BART Wave-CAIPI reconstruction (${ECHO_SUFFIX:-single echo}):"
    print_command \
        "$BART_EXECUTABLE" wave "${WAVE_OPTIONS[@]}" \
        "$ESPIRIT_MAPS" "$WAVE_PSF" "$WAVE_KSPACE" "$WAVE_IMAGE"

    "$BART_EXECUTABLE" wave "${WAVE_OPTIONS[@]}" \
        "$ESPIRIT_MAPS" \
        "$WAVE_PSF" \
        "$WAVE_KSPACE" \
        "$WAVE_IMAGE"
done

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NIFTI_CONVERTER="$SCRIPT_DIRECTORY/wave_to_nifti.py"
[[ -f "$NIFTI_CONVERTER" ]] || fail "NIfTI converter not found: $NIFTI_CONVERTER"

NIFTI_COMMAND=(
    "$PYTHON_EXECUTABLE" "$NIFTI_CONVERTER"
    --bart-input-dir "$BART_INPUT"
    --bart-output-dir "$BART_OUTPUT"
    --twix "$TWIX_FILE"
    --seq "$SEQUENCE_FILE"
    --out "$NIFTI_OUTPUT"
)
((SAVE_PHASE)) && NIFTI_COMMAND+=(--save-phase)
NIFTI_COMMAND+=("${NIFTI_OPTIONS[@]}")

echo "Converting BART output to NIfTI:"
print_command "${NIFTI_COMMAND[@]}"
"${NIFTI_COMMAND[@]}"

echo "Completed BART reconstruction and NIfTI conversion."
