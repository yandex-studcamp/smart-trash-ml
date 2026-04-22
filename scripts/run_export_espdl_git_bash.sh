#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/Scripts/python.exe}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/artifacts/exports}"
CALIBRATION_DIR="${CALIBRATION_DIR:-$ROOT_DIR/data/calibration}"
IMAGE_SIZE="${IMAGE_SIZE:-96}"
NUM_CLASSES="${NUM_CLASSES:-3}"
ESP_TARGET="${ESP_TARGET:-c}"
ESP_QUANT="${ESP_QUANT:-int8}"
ESP_DEVICE="${ESP_DEVICE:-cuda}"
ESP_CALIB_STEPS="${ESP_CALIB_STEPS:-32}"
ESP_BATCH_SIZE="${ESP_BATCH_SIZE:-1}"

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
  --espdl-backend esp-ppq \
  --calibration-dir "$CALIBRATION_DIR" \
  --espdl-target "$ESP_TARGET" \
  --espdl-quantization "$ESP_QUANT" \
  --espdl-device "$ESP_DEVICE" \
  --espdl-calib-steps "$ESP_CALIB_STEPS" \
  --espdl-batch-size "$ESP_BATCH_SIZE" \
  --espdl-export-test-values
