#!/usr/bin/env bash
set -euo pipefail

AXIS_ROOT="${AXIS_ROOT:-/WORK/fit/zhangchen/axis_project/multiaxis_runtime_train}"
DATA_ROOT="${DATA_ROOT:-/WORK/fit/zhangchen/axis_project/data/swat_axis_160}"
OUT_ROOT="${OUT_ROOT:-/WORK/fit/zhangchen/axis_project/runs/chat_baselines_swat_160_20260610}"
CHATTS_CKPT_DIR="${CHATTS_CKPT_DIR:-/WORK/fit/zhangchen/axis_project/checkpoints/chatts_legacy14b}"
CHATTIME_MODEL_PATH="${CHATTIME_MODEL_PATH:-/WORK/fit/zhangchen/axis_project/checkpoints/chattime_1_7b_chat}"
HF_HOME="${HF_HOME:-/WORK/fit/zhangchen/axis_project/models/hf}"
PY_BIN="${PY_BIN:-python3}"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PY_BIN=python
  else
    echo "No python interpreter found via PY_BIN/python3/python" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_ROOT" "$HF_HOME"
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PYTHONPATH="$AXIS_ROOT/baseline/ChatTS/src:$AXIS_ROOT/baseline/ChatTS:$AXIS_ROOT:${PYTHONPATH:-}"

if [ ! -f "$CHATTS_CKPT_DIR/config.json" ]; then
  echo "ChatTS checkpoint not found at $CHATTS_CKPT_DIR" >&2
  exit 1
fi

if [ ! -f "$AXIS_ROOT/baseline/ChatTS/src/models/Baselines/chatts_mvaxis_runner.py" ]; then
  echo "ChatTS baseline code not found under $AXIS_ROOT/baseline/ChatTS" >&2
  exit 1
fi

if [ ! -f "$CHATTIME_MODEL_PATH/config.json" ]; then
  echo "ChatTime model not found at $CHATTIME_MODEL_PATH" >&2
  exit 1
fi

run_split() {
  local split="$1"
  local q_dir="$2"
  local q_json="$DATA_ROOT/$q_dir/questions_160.jsonl"
  local teacher_json="$DATA_ROOT/$q_dir/teacher_gpt55.jsonl"
  local split_out="$OUT_ROOT/$split"
  local chatts_out="$split_out/chatts"
  local chattime_out="$split_out/chattime"

  mkdir -p "$split_out"

  echo "==== [$split] ChatTS ===="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PY_BIN" -B "$AXIS_ROOT/baseline/ChatTS/src/models/Baselines/chatts_mvaxis_runner.py" \
    --input_jsonl "$q_json" \
    --output_dir "$chatts_out" \
    --ckpt_path "$CHATTS_CKPT_DIR" \
    --batch_size 1 \
    --device cuda:0 \
    --run_name "chatts_swat_${split}"

  echo "==== [$split] ChatTime ===="
  rm -rf "$AXIS_ROOT/experiments/logs/ChatTime"
  mkdir -p "$AXIS_ROOT/experiments/logs/ChatTime"
  (
    cd "$AXIS_ROOT"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    MVAXIS_INPUT_JSONL="$q_json" \
    MVAXIS_SAMPLE_LIMIT=160 \
    CHATTIME_MODEL_PATH="$CHATTIME_MODEL_PATH" \
    "$PY_BIN" -B "$AXIS_ROOT/baseline/ChatTS/src/models/Baselines/chattime_test.py"
  )

  local latest_chattime
  latest_chattime="$(find "$AXIS_ROOT/experiments/logs/ChatTime" -maxdepth 1 -type d -name 'results_*' | sort | tail -n 1)"
  if [ -z "$latest_chattime" ]; then
    echo "Failed to locate ChatTime result directory for $split" >&2
    exit 1
  fi
  rm -rf "$chattime_out"
  mkdir -p "$chattime_out"
  cp -a "$latest_chattime/." "$chattime_out/"

  echo "==== [$split] Teacher-aligned eval ===="
  "$PY_BIN" -B "$AXIS_ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
    --run-root "$split_out" \
    --questions "$q_json" \
    --teacher "$teacher_json" \
    --runs chatts chattime

  "$PY_BIN" -B "$AXIS_ROOT/scripts/mvaxis_compute_metrics_with_open_decision.py" \
    --root "$split_out" \
    --questions "$q_json" \
    --teacher "$teacher_json" \
    --groups chatts chattime

  "$PY_BIN" -B "$AXIS_ROOT/scripts/mvaxis_build_answer_review_html.py" \
    --root "$split_out" \
    --groups chatts chattime \
    --questions "$q_json" \
    --teacher "$teacher_json"

  "$PY_BIN" -B "$AXIS_ROOT/scripts/mvaxis_build_eval_summary_html.py" \
    --summary-json "$split_out/teacher_aligned_summary.json" \
    --title "SWAT_160_${split}_ChatTS_ChatTime_20260610" \
    --output-html "$split_out/one_page_report.html"
}

run_split regular question_swat_axis_regular_160_0605b
run_split hard question_swat_axis_hard_160_0605a

echo "All done. Reports under $OUT_ROOT"
