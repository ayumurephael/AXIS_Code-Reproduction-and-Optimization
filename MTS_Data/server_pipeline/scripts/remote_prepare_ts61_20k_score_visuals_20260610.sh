#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
ROOT="${ROOT:-$WORKROOT/multiaxis_runtime_train}"
TRAIN_ROOT="${TRAIN_ROOT:-$WORKROOT/data/ts61_train_20k_20260610a}"
OUT_ROOT="${OUT_ROOT:-$WORKROOT/data/ts61_train_20k_visuals_20260610a}"
CFG="${CFG:-$ROOT/configs/torch_fixed60_qa600_timercd.json}"
CKPT="${CKPT:-$WORKROOT/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt}"
PY="${PY:-python}"
LOGFILE="${LOGFILE:-$WORKROOT/logs/prepare_ts61_20k_score_visuals_20260610.log}"

mkdir -p "$OUT_ROOT" "$(dirname "$LOGFILE")"

if [ ! -f "$TRAIN_ROOT/questions_20000.jsonl" ]; then
  echo "missing merged training data: $TRAIN_ROOT/questions_20000.jsonl" >&2
  exit 1
fi
if [ ! -f "$CFG" ]; then
  echo "missing axis config: $CFG" >&2
  exit 1
fi
if [ ! -f "$CKPT" ]; then
  echo "missing encoder checkpoint: $CKPT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

"$PY" -B "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" \
  --data "$TRAIN_ROOT/questions_20000.jsonl" \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --output-root "$OUT_ROOT" \
  --modes score \
  > "$LOGFILE" 2>&1
