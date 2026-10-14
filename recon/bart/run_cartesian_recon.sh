#!/usr/bin/env bash
set -euo pipefail

# Run conventional Cartesian BART reconstruction for no-wave GRE inputs.

usage() {
    cat <<'EOF'
Usage:
  run_cartesian_recon.sh \
    --bart-input PATH --bart-output PATH --maps-source bart|existing \
    --twix FILE.dat --seq FILE.seq \
    [--nifti-output PATH] [--existing-maps CFL_BASENAME] \
    [--save-phase] [--skip-nifti] [--resume] \
    [--ecalib-options BART_OPTIONS... --end-ecalib-options] \
    [--pics-options BART_OPTIONS... --end-pics-options] \
    [--nifti-options OPTIONS... --end-nifti-options]

When --pics-options is omitted, the default is:
  -l1 -r 1e-5 --fista -i 300 -g -S

This uses 3D wavelet regularization with FISTA on the GPU. The -S option
restores BART's internal data scaling in the reconstructed image.
EOF
}

fail() { echo "Error: $*" >&2; exit 2; }
require_value() { [[ -n "${2:-}" ]] || fail "$1 requires a value."; }
require_cfl_pair() {
    cfl_pair_is_complete "$1" ||
        fail "Missing, malformed, or truncated BART CFL pair: $1.{hdr,cfl}"
}

cfl_pair_is_complete() {
    local base="$1"
    python3 - "$base" <<'PY'
import math
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
header = base.with_suffix(".hdr")
data = base.with_suffix(".cfl")
try:
    line = next(
        value.strip()
        for value in header.read_text(encoding="utf-8").splitlines()
        if value.strip() and not value.lstrip().startswith("#")
    )
    dimensions = [int(value) for value in line.split()]
    valid = bool(dimensions) and all(value > 0 for value in dimensions)
    valid = valid and data.stat().st_size == math.prod(dimensions) * 8
except (FileNotFoundError, OSError, StopIteration, ValueError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

cfl_pair_is_current() {
    local output="$1"
    shift
    cfl_pair_is_complete "$output" || return 1
    local input
    for input in "$@"; do
        cfl_pair_is_complete "$input" || return 1
        [[ ! "$input.hdr" -nt "$output.hdr" ]] || return 1
        [[ ! "$input.hdr" -nt "$output.cfl" ]] || return 1
        [[ ! "$input.cfl" -nt "$output.hdr" ]] || return 1
        [[ ! "$input.cfl" -nt "$output.cfl" ]] || return 1
    done
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
SKIP_NIFTI=0
RESUME=0
ECALIB_OPTIONS=()
PICS_OPTIONS=()
NIFTI_OPTIONS=()

while (($#)); do
    case "$1" in
        --bart-input) require_value "$1" "${2:-}"; BART_INPUT="${2%/}"; shift 2 ;;
        --bart-output) require_value "$1" "${2:-}"; BART_OUTPUT="${2%/}"; shift 2 ;;
        --maps-source) require_value "$1" "${2:-}"; MAPS_SOURCE="$2"; shift 2 ;;
        --existing-maps) require_value "$1" "${2:-}"; EXISTING_MAPS="$2"; shift 2 ;;
        --twix) require_value "$1" "${2:-}"; TWIX_FILE="$2"; shift 2 ;;
        --seq) require_value "$1" "${2:-}"; SEQUENCE_FILE="$2"; shift 2 ;;
        --nifti-output) require_value "$1" "${2:-}"; NIFTI_OUTPUT="${2%/}"; shift 2 ;;
        --save-phase) SAVE_PHASE=1; shift ;;
        --skip-nifti) SKIP_NIFTI=1; shift ;;
        --resume) RESUME=1; shift ;;
        --ecalib-options)
            shift
            while (($#)) && [[ "$1" != "--end-ecalib-options" ]]; do
                ECALIB_OPTIONS+=("$1"); shift
            done
            (($#)) || fail "--ecalib-options requires --end-ecalib-options."
            shift
            ;;
        --pics-options)
            shift
            while (($#)) && [[ "$1" != "--end-pics-options" ]]; do
                PICS_OPTIONS+=("$1"); shift
            done
            (($#)) || fail "--pics-options requires --end-pics-options."
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
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown wrapper argument: $1." ;;
    esac
done

[[ -n "$BART_INPUT" ]] || fail "--bart-input is required."
[[ -n "$BART_OUTPUT" ]] || fail "--bart-output is required."
[[ -n "$MAPS_SOURCE" ]] || fail "--maps-source is required."
[[ -n "$TWIX_FILE" ]] || fail "--twix is required."
[[ -n "$SEQUENCE_FILE" ]] || fail "--seq is required."
[[ "$MAPS_SOURCE" == "bart" || "$MAPS_SOURCE" == "existing" ]] ||
    fail "--maps-source must be 'bart' or 'existing'."
[[ -f "$BART_INPUT/manifest.json" ]] || fail "Missing manifest: $BART_INPUT/manifest.json"
[[ -f "$TWIX_FILE" ]] || fail "TWIX file not found: $TWIX_FILE"
[[ -f "$SEQUENCE_FILE" ]] || fail "Pulseq file not found: $SEQUENCE_FILE"

BART_EXECUTABLE="${BART_BIN:-bart}"
PYTHON_EXECUTABLE="${PYTHON_BIN:-python}"
[[ -n "$NIFTI_OUTPUT" ]] || NIFTI_OUTPUT="$BART_OUTPUT/nifti"
command -v "$BART_EXECUTABLE" >/dev/null 2>&1 || fail "BART executable not found."
if ((!SKIP_NIFTI)); then
    command -v "$PYTHON_EXECUTABLE" >/dev/null 2>&1 || fail "Python interpreter not found."
fi
array_contains -m "${ECALIB_OPTIONS[@]}" &&
    fail "Do not pass -m in --ecalib-options; this workflow uses -m 1."

if ((${#PICS_OPTIONS[@]} == 0)); then
    PICS_OPTIONS=(-l1 -r 1e-5 --fista -i 300 -g -S)
    echo "No --pics-options supplied; using BART GPU defaults: ${PICS_OPTIONS[*]}"
fi

mkdir -p "$BART_OUTPUT"
if ((!SKIP_NIFTI)); then
    mkdir -p "$NIFTI_OUTPUT"
fi

if [[ "$MAPS_SOURCE" == "bart" ]]; then
    require_cfl_pair "$BART_INPUT/kspace_calib"
    ESPIRIT_MAPS="$BART_OUTPUT/coil_sens_bart"
    if ((RESUME)) && cfl_pair_is_current "$ESPIRIT_MAPS" "$BART_INPUT/kspace_calib"; then
        echo "Resume: reusing current BART ESPIRiT maps: $ESPIRIT_MAPS"
    else
        echo "Running BART ESPIRiT calibration:"
        print_command "$BART_EXECUTABLE" ecalib -m 1 "${ECALIB_OPTIONS[@]}" \
            "$BART_INPUT/kspace_calib" "$ESPIRIT_MAPS"
        "$BART_EXECUTABLE" ecalib -m 1 "${ECALIB_OPTIONS[@]}" \
            "$BART_INPUT/kspace_calib" "$ESPIRIT_MAPS"
    fi
else
    [[ -n "$EXISTING_MAPS" ]] || EXISTING_MAPS="$BART_INPUT/coil_sens"
    ESPIRIT_MAPS="$(strip_cfl_extension "$EXISTING_MAPS")"
    require_cfl_pair "$ESPIRIT_MAPS"
fi

MANIFEST_KSPACE_OUTPUT="$(python3 - "$BART_INPUT/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
reconstruction = manifest.get("reconstruction", {})
if reconstruction.get("mode") != "cartesian" or reconstruction.get("tool") != "pics":
    raise SystemExit("manifest is not a Cartesian BART pics reconstruction")
if reconstruction.get("readout_oversampling_removed") is not True:
    raise SystemExit("manifest does not confirm readout oversampling removal")
echoes = manifest.get("echoes")
if not isinstance(echoes, list) or not echoes:
    raise SystemExit("manifest contains no echo entries")
for expected_echo, entry in enumerate(echoes, start=1):
    if entry.get("echo") != expected_echo:
        raise SystemExit("manifest echoes must be consecutive and start at 1")
    basename = entry.get("cartesian_kspace")
    if not isinstance(basename, str) or not basename.startswith("cartesian_kspace"):
        raise SystemExit("manifest has an invalid Cartesian k-space basename")
    print(basename)
PY
)" || fail "Unable to read Cartesian inputs from manifest.json."

while IFS= read -r KSPACE_NAME; do
    [[ -n "$KSPACE_NAME" ]] || continue
    SUFFIX="${KSPACE_NAME#cartesian_kspace}"
    KSPACE="$BART_INPUT/$KSPACE_NAME"
    IMAGE="$BART_OUTPUT/image_nowave$SUFFIX"
    require_cfl_pair "$KSPACE"
    if ((RESUME)) && cfl_pair_is_current "$IMAGE" "$ESPIRIT_MAPS" "$KSPACE"; then
        echo "Resume: reusing current BART Cartesian image (${SUFFIX:-single echo}): $IMAGE"
    else
        echo "Running BART Cartesian PICS reconstruction (${SUFFIX:-single echo}):"
        print_command "$BART_EXECUTABLE" pics "${PICS_OPTIONS[@]}" \
            "$KSPACE" "$ESPIRIT_MAPS" "$IMAGE"
        "$BART_EXECUTABLE" pics "${PICS_OPTIONS[@]}" \
            "$KSPACE" "$ESPIRIT_MAPS" "$IMAGE"
    fi
done <<<"$MANIFEST_KSPACE_OUTPUT"

if ((SKIP_NIFTI)); then
    echo "Completed BART Cartesian reconstruction; NIfTI conversion was skipped."
    exit 0
fi

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NIFTI_COMMAND=(
    "$PYTHON_EXECUTABLE" "$SCRIPT_DIRECTORY/wave_to_nifti.py"
    --bart-input-dir "$BART_INPUT"
    --bart-output-dir "$BART_OUTPUT"
    --twix "$TWIX_FILE"
    --seq "$SEQUENCE_FILE"
    --out "$NIFTI_OUTPUT"
)
((SAVE_PHASE)) && NIFTI_COMMAND+=(--save-phase)
NIFTI_COMMAND+=("${NIFTI_OPTIONS[@]}")
echo "Converting BART Cartesian output to NIfTI:"
print_command "${NIFTI_COMMAND[@]}"
"${NIFTI_COMMAND[@]}"
echo "Completed BART Cartesian reconstruction and NIfTI conversion."
