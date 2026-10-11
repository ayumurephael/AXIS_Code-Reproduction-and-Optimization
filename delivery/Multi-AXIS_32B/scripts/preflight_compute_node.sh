#!/usr/bin/env bash
set -euo pipefail

: "${MULTI_AXIS_DELIVERY_ROOT:?set MULTI_AXIS_DELIVERY_ROOT on compute storage}"
SOURCE_ROOT="$MULTI_AXIS_DELIVERY_ROOT/source/checkout"
export PYTHONPATH="$SOURCE_ROOT"
python -m tools.multi_axis.hardware_preflight \
  --model-dir "$MULTI_AXIS_DELIVERY_ROOT/models/Qwen3-VL-32B-Instruct-0cfaf48183f5" \
  --work-dir "${MULTI_AXIS_WORK_DIR:-$MULTI_AXIS_DELIVERY_ROOT}"
