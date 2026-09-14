#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
ROOT="${ROOT:-$WORKROOT/multiaxis_runtime_train}"
TRAIN_ROOT="${TRAIN_ROOT:-$WORKROOT/data/ts61_train_20k_20260610a}"
OUT_ROOT="${OUT_ROOT:-$WORKROOT/data/ts61_train_20k_visuals_20260618_highlighta}"
PY="${PY:-python}"
LOGFILE="${LOGFILE:-$WORKROOT/logs/prepare_ts61_20k_highlight_visuals_20260618.log}"
CONDA_SH="${CONDA_SH:-/home/fit/zhangchen/WORK/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-Allava}"

if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
  PY=python
fi

mkdir -p "$OUT_ROOT" "$(dirname "$LOGFILE")"

if [ ! -f "$TRAIN_ROOT/questions_20000.jsonl" ]; then
  echo "missing merged training data: $TRAIN_ROOT/questions_20000.jsonl" >&2
  exit 1
fi

"$PY" -B "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" \
  --data "$TRAIN_ROOT/questions_20000.jsonl" \
  --output-root "$OUT_ROOT" \
  --modes raw \
  > "$LOGFILE" 2>&1
