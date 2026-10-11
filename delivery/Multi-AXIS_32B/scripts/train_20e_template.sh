#!/usr/bin/env bash
set -euo pipefail

: "${MULTI_AXIS_DELIVERY_ROOT:?set MULTI_AXIS_DELIVERY_ROOT on compute storage}"
: "${MULTI_AXIS_RUN_ROOT:?set an empty, dedicated MULTI_AXIS_RUN_ROOT}"
: "${MULTI_AXIS_TORCHRUN_PREFIX:?set after hardware preflight, e.g. torchrun --nnodes=1 --nproc_per_node=8}"

SOURCE_ROOT="$MULTI_AXIS_DELIVERY_ROOT/source/checkout"
CONFIG="$SOURCE_ROOT/experiments/multi_axis/formal_qwen3_vl_32b_20epochs_hardware_template.json"
export PYTHONPATH="$SOURCE_ROOT"
export MULTI_AXIS_MODEL_PATH="$MULTI_AXIS_DELIVERY_ROOT/models/Qwen3-VL-32B-Instruct-0cfaf48183f5"
export MULTI_AXIS_TIMES_FONT_DIR="$MULTI_AXIS_DELIVERY_ROOT/assets/fonts"

test ! -e "$MULTI_AXIS_RUN_ROOT/TRAINING_COMPLETE"
test -f "$MULTI_AXIS_MODEL_PATH/SNAPSHOT_COMPLETE"
test -f "$MULTI_AXIS_DELIVERY_ROOT/images/image_manifest.jsonl"

# This formal first run deliberately has no --resume flag.
eval "$MULTI_AXIS_TORCHRUN_PREFIX" "$SOURCE_ROOT/tools/multi_axis/train.py" \
  --config "$CONFIG" \
  --data-root "$MULTI_AXIS_DELIVERY_ROOT/datasets/extracted" \
  --manifest-dir "$MULTI_AXIS_DELIVERY_ROOT/manifests" \
  --image-root "$MULTI_AXIS_DELIVERY_ROOT/images" \
  --timercd-checkpoint "$MULTI_AXIS_DELIVERY_ROOT/assets/timercd/pretrain_checkpoint_best_multi.pth" \
  --question-semantic-cache-dir "$MULTI_AXIS_RUN_ROOT/question_semantics_32b" \
  --output-dir "$MULTI_AXIS_RUN_ROOT/training"
