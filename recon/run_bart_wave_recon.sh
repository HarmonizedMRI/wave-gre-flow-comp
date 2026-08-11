#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_bart_wave_recon.sh [options] BART_INPUT_DIR OUTPUT_DIR

Run BART ESPIRiT calibration and Wave-CAIPI reconstruction from files exported
by recon_wave_gre_from_twix_integrated_nifti.py --save-bart-inputs.

Options:
  --maps-source bart|exported  Use BART ecalib maps (default) or exported maps.
  --ecalib-crop VALUE          ESPIRiT eigenvalue crop (default: 0.8).
  --ecalib-maps N              Number of ESPIRiT map sets (default: 1).
  --wave-iters N               BART wave maximum iterations (default: 50).
  --wave-tol VALUE             BART wave convergence tolerance (default: 1e-6).
  --gpu                        Pass -g to BART ecalib and wave.
  -h, --help                   Show this help.

Set BART_BIN to override the BART executable (default: bart).
EOF
}

maps_source="bart"
ecalib_crop="0.8"
ecalib_maps="1"
wave_iters="50"
wave_tol="1e-6"
use_gpu=0

while (($#)); do
    case "$1" in
        --maps-source)
            maps_source="${2:?missing value for --maps-source}"
            shift 2
            ;;
        --ecalib-crop)
            ecalib_crop="${2:?missing value for --ecalib-crop}"
            shift 2
            ;;
        --ecalib-maps)
            ecalib_maps="${2:?missing value for --ecalib-maps}"
            shift 2
            ;;
        --wave-iters)
            wave_iters="${2:?missing value for --wave-iters}"
            shift 2
            ;;
        --wave-tol)
            wave_tol="${2:?missing value for --wave-tol}"
            shift 2
            ;;
        --gpu)
            use_gpu=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

if (($# != 2)); then
    usage >&2
    exit 2
fi
if [[ "$maps_source" != "bart" && "$maps_source" != "exported" ]]; then
    echo "--maps-source must be 'bart' or 'exported'." >&2
    exit 2
fi

input_dir="${1%/}"
output_dir="${2%/}"
bart_bin="${BART_BIN:-bart}"

if ! command -v "$bart_bin" >/dev/null 2>&1; then
    echo "BART executable not found: $bart_bin" >&2
    exit 127
fi
for base in kspace_calib coil_sens; do
    if [[ ! -f "$input_dir/$base.hdr" || ! -f "$input_dir/$base.cfl" ]]; then
        echo "Missing BART input pair: $input_dir/$base.{hdr,cfl}" >&2
        exit 1
    fi
done

mkdir -p "$output_dir"
gpu_args=()
if ((use_gpu)); then
    gpu_args=(-g)
fi

echo "Running BART ESPIRiT calibration..."
"$bart_bin" ecalib "${gpu_args[@]}" -m "$ecalib_maps" -c "$ecalib_crop" \
    "$input_dir/kspace_calib" "$output_dir/coil_sens_bart"

if [[ "$maps_source" == "bart" ]]; then
    maps="$output_dir/coil_sens_bart"
else
    maps="$input_dir/coil_sens"
fi

shopt -s nullglob
kspace_headers=("$input_dir"/wave_kspace*.hdr)
if ((${#kspace_headers[@]} == 0)); then
    echo "No wave_kspace*.hdr inputs found in $input_dir." >&2
    exit 1
fi

for header in "${kspace_headers[@]}"; do
    kspace="${header%.hdr}"
    stem="${kspace##*/}"
    suffix="${stem#wave_kspace}"
    psf="$input_dir/psf$suffix"
    if [[ ! -f "$psf.hdr" || ! -f "$psf.cfl" ]]; then
        echo "Missing PSF pair for $stem: $psf.{hdr,cfl}" >&2
        exit 1
    fi
    echo "Running BART wave reconstruction for ${suffix:-single echo}..."
    "$bart_bin" wave "${gpu_args[@]}" -i "$wave_iters" -t "$wave_tol" \
        "$maps" "$psf" "$kspace" "$output_dir/image_wave$suffix"
done

echo "BART outputs written to $output_dir"
