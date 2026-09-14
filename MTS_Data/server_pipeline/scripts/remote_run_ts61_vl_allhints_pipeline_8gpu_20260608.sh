#!/usr/bin/env bash
set -euo pipefail

ROOT="/WORK/fit/zhangchen/axis_project/multiaxis_runtime_train"
WORKROOT="/WORK/fit/zhangchen/axis_project"
LOGROOT="$WORKROOT/logs"
RUNROOT="$WORKROOT/runs/ts61_vl_allhints_nexttoken_train_20260608a"

mkdir -p "$LOGROOT" "$RUNROOT/checkpoints" "$RUNROOT/reports" "$RUNROOT/logs"

source /etc/profile >/dev/null 2>&1 || true
if command -v module >/dev/null 2>&1; then
  module load soft/anaconda3/config || true
fi
if ! command -v python >/dev/null 2>&1; then
  source /apps/soft/anaconda3/bin/activate || true
fi

PY_BIN="$(command -v python || true)"
if [ -z "$PY_BIN" ]; then
  echo "python not found in remote_run_ts61_vl_allhints_pipeline_8gpu_20260608.sh" >&2
  exit 1
fi
export PY="$PY_BIN"

for required in \
  "$ROOT/scripts/remote_prepare_ts61_score_visuals_20260608.sh" \
  "$ROOT/scripts/remote_start_ts61_vl_allhints_nexttoken_smoke_8gpu_20260608.sh" \
  "$ROOT/scripts/remote_start_ts61_vl_allhints_nexttoken_full_8gpu_20260608.sh" \
  "$ROOT/configs/local_qwen25_vl7b_ts61_vl_allhints_ct12_scale010_8gpu.json" \
  "$WORKROOT/data/ts61_train/questions_2000.jsonl" \
  "$WORKROOT/data/ts61_train/teacher_gpt55.jsonl" \
  "$WORKROOT/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1.pt" \
  "$WORKROOT/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt"; do
  if [ ! -e "$required" ]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

date > "$LOGROOT/pipeline_started_8gpu_$(date +%Y%m%d_%H%M%S).txt"

bash "$ROOT/scripts/remote_prepare_ts61_score_visuals_20260608.sh"
date > "$LOGROOT/pipeline_prepare_done.txt"

bash "$ROOT/scripts/remote_start_ts61_vl_allhints_nexttoken_smoke_8gpu_20260608.sh"
date > "$LOGROOT/pipeline_smoke_done.txt"

bash "$ROOT/scripts/remote_start_ts61_vl_allhints_nexttoken_full_8gpu_20260608.sh"
date > "$LOGROOT/pipeline_full_done.txt"
