#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  MODEL_GRID_GLOB="/path/to/grid/*.photoz" \
  OBS_PHOT="/path/to/obs_phot.dat" \
  TEMPLATE_PROB="/path/to/template_prob.dat" \
  OUTPUT_REDSHIFTS="/path/to/output_redshifts.out" \
  OUTPUT_PDFS="/path/to/output_pdfs.out" \
  ./run_jax_photoz.sh [--gpu N] [--batch-obj N] [--skip-verify]

Optional env vars:
  CPP_PDFS="/path/to/cpp_pdfs.out"     # If set and --skip-verify is not used, runs verify mode

Notes:
  - This wrapper runs fit mode first.
  - Verification runs only when CPP_PDFS is set.
EOF
}

GPU_DEVICE="${GPU_DEVICE:-}"
BATCH_OBJ="${BATCH_OBJ:-10000}"
SKIP_VERIFY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            if [[ $# -lt 2 ]]; then
                echo "Error: --gpu requires a device index" >&2
                exit 1
            fi
            GPU_DEVICE="$2"
            shift 2
            ;;
        --batch-obj)
            if [[ $# -lt 2 ]]; then
                echo "Error: --batch-obj requires a value" >&2
                exit 1
            fi
            BATCH_OBJ="$2"
            shift 2
            ;;
        --skip-verify)
            SKIP_VERIFY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

for required_var in MODEL_GRID_GLOB OBS_PHOT TEMPLATE_PROB OUTPUT_REDSHIFTS OUTPUT_PDFS; do
    if [[ -z "${!required_var:-}" ]]; then
        echo "Error: environment variable ${required_var} is required." >&2
        usage >&2
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_with_optional_gpu() {
    if [[ -n "${GPU_DEVICE}" ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" "$@"
    else
        "$@"
    fi
}

echo "Running fit mode..."
run_with_optional_gpu "${PYTHON_BIN}" "${SCRIPT_DIR}/photoz_jax.py" fit \
    "${MODEL_GRID_GLOB}" \
    "${OBS_PHOT}" \
    "${TEMPLATE_PROB}" \
    "${OUTPUT_REDSHIFTS}" \
    --pdfs "${OUTPUT_PDFS}" \
    --batch-obj "${BATCH_OBJ}"

if [[ "${SKIP_VERIFY}" -eq 1 ]]; then
    echo "Skipping verify step (--skip-verify)."
    exit 0
fi

if [[ -z "${CPP_PDFS:-}" ]]; then
    echo "CPP_PDFS not set; skipping verify mode."
    exit 0
fi

echo "Running verify mode..."
run_with_optional_gpu "${PYTHON_BIN}" "${SCRIPT_DIR}/photoz_jax.py" verify \
    "${MODEL_GRID_GLOB}" \
    "${OBS_PHOT}" \
    "${TEMPLATE_PROB}" \
    --cpp-pdfs "${CPP_PDFS}" \
    --batch-obj "${BATCH_OBJ}"

echo "Done."
