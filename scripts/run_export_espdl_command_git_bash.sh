#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/Scripts/python.exe}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/artifacts/exports}"
CALIBRATION_DIR="${CALIBRATION_DIR:-$ROOT_DIR/data/calibration}"
IMAGE_SIZE="${IMAGE_SIZE:-96}"
NUM_CLASSES="${NUM_CLASSES:-3}"
ESP_QUANT="${ESP_QUANT:-int8}"

if [[ -z "${ESPDL_COMMAND:-}" ]]; then
  cat <<'EOF'
ESPDL_COMMAND is not set.

Example:
export ESPDL_COMMAND='python /abs/path/to/converter.py --input "{input_onnx}" --input-data "{input_onnx_data}" --output "{output_espdl}" --calib "{calibration_dir}" --quant {quantization}'
EOF
  exit 1
fi

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(mobilenet_v3_small_reduced mobilenet_v3_small mobilenet_v2_w0_40 mobilenet_v2_w0_28)
fi

"$PYTHON_BIN" "$ROOT_DIR/src/main.py" \
  --models "${MODELS[@]}" \
  --num-classes "$NUM_CLASSES" \
  --image-size "$IMAGE_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --export-formats onnx espdl \
  --espdl-backend command \
  --calibration-dir "$CALIBRATION_DIR" \
  --espdl-quantization "$ESP_QUANT" \
  --espdl-command "$ESPDL_COMMAND"
