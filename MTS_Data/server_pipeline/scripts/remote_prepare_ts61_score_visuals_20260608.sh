#!/usr/bin/env bash
set -euo pipefail

ROOT="/WORK/fit/zhangchen/axis_project/multiaxis_runtime_train"
OUT_ROOT="/WORK/fit/zhangchen/axis_project/data/ts61_train_visuals_20260608a"
PY="${PY:-python}"

mkdir -p "$OUT_ROOT"

export CUDA_VISIBLE_DEVICES=0

"$PY" -B "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" \
  --data /WORK/fit/zhangchen/axis_project/data/ts61_train/questions_2000.jsonl \
  --config "$ROOT/configs/torch_fixed60_qa600_timercd.json" \
  --checkpoint /WORK/fit/zhangchen/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt \
  --output-root "$OUT_ROOT" \
  --modes score \
  > /WORK/fit/zhangchen/axis_project/logs/prepare_ts61_score_visuals_20260608.log 2>&1
