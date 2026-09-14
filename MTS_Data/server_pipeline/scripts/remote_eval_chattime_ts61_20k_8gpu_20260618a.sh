#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
AXIS_ROOT="${AXIS_ROOT:-$WORKROOT/multiaxis_runtime_train}"
DATA_ROOT="${DATA_ROOT:-$WORKROOT/data/ts61_20k_partial_teacher_eval_20260617a}"
OUT_ROOT="${OUT_ROOT:-$WORKROOT/runs/chattime_ts61_20k_8gpu_parallel_20260618a}"
CHATTIME_MODEL_PATH="${CHATTIME_MODEL_PATH:-$WORKROOT/checkpoints/chattime_1_7b_chat}"
HF_HOME="${HF_HOME:-$WORKROOT/models/hf}"
CONDA_ROOT="${CONDA_ROOT:-/home/fit/zhangchen/WORK/miniconda3}"
CONDA_SH="${CONDA_SH:-/home/fit/zhangchen/WORK/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-Allava}"
ALLAVA_PY="${ALLAVA_PY:-$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin/python}"
PY_BIN="${PY_BIN:-$ALLAVA_PY}"
RUN_TAG="${RUN_TAG:-chattime_ts61_20k_8gpu_20260618a}"

if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  if command -v python >/dev/null 2>&1; then
    PY_BIN=python
  fi
fi

mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs" "$OUT_ROOT/full_20k/chattime" "$HF_HOME"
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PYTHONPATH="$AXIS_ROOT/baseline/ChatTS/src:$AXIS_ROOT/baseline/ChatTS:$AXIS_ROOT:${PYTHONPATH:-}"

FULL_Q="$DATA_ROOT/questions_20000.jsonl"
TEACHER_JSON="$DATA_ROOT/teacher_available.jsonl"
SCRIPT_PATH="$AXIS_ROOT/baseline/ChatTS/src/models/Baselines/chattime_test.py"
RUN_ROOT="$OUT_ROOT/full_20k"
MERGED_DIR="$RUN_ROOT/chattime"

if [ ! -f "$FULL_Q" ]; then
  echo "missing full questions: $FULL_Q" >&2
  exit 1
fi
if [ ! -f "$TEACHER_JSON" ]; then
  echo "missing teacher jsonl: $TEACHER_JSON" >&2
  exit 1
fi
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "missing chattime_test.py: $SCRIPT_PATH" >&2
  exit 1
fi
if [ ! -f "$CHATTIME_MODEL_PATH/config.json" ]; then
  echo "missing ChatTime config: $CHATTIME_MODEL_PATH/config.json" >&2
  exit 1
fi
if [ ! -f "$CHATTIME_MODEL_PATH/model-00001-of-00003.safetensors" ] || \
   [ ! -f "$CHATTIME_MODEL_PATH/model-00002-of-00003.safetensors" ] || \
   [ ! -f "$CHATTIME_MODEL_PATH/model-00003-of-00003.safetensors" ]; then
  echo "missing one or more ChatTime shards under $CHATTIME_MODEL_PATH" >&2
  exit 1
fi

GPU_IDS_RAW="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd',' -)"
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_RAW"
GPU_COUNT="${#GPU_IDS[@]}"
if [ "$GPU_COUNT" -lt 1 ]; then
  echo "no visible GPUs" >&2
  exit 1
fi

TOTAL_LINES="$("$PY_BIN" - <<'PY'
from pathlib import Path
path = Path("/WORK/fit/zhangchen/axis_project/data/ts61_20k_partial_teacher_eval_20260617a/questions_20000.jsonl")
count = 0
with path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            count += 1
print(count)
PY
)"

echo "[chattime-20k] visible GPUs: $GPU_IDS_RAW"
echo "[chattime-20k] total questions: $TOTAL_LINES"

for gpu in "${GPU_IDS[@]}"; do
  rm -rf "$RUN_ROOT/shards/shard_$gpu"
done
mkdir -p "$RUN_ROOT/shards"

PIDS=()
for idx in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$idx]}"
  start=$(( TOTAL_LINES * idx / GPU_COUNT ))
  end=$(( TOTAL_LINES * (idx + 1) / GPU_COUNT ))
  limit=$(( end - start ))
  shard_dir="$RUN_ROOT/shards/shard_$gpu"
  shard_log="$OUT_ROOT/logs/shard_$gpu.log"
  mkdir -p "$shard_dir"
  echo "[chattime-20k] shard=$gpu start=$start limit=$limit"
  (
    cd "$AXIS_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" \
    MVAXIS_INPUT_JSONL="$FULL_Q" \
    MVAXIS_SAMPLE_OFFSET="$start" \
    MVAXIS_SAMPLE_LIMIT="$limit" \
    MVAXIS_GLOBAL_INDEX_OFFSET="$start" \
    CHATTIME_MODEL_PATH="$CHATTIME_MODEL_PATH" \
    CHATTIME_RUN_TAG="${RUN_TAG}_shard_${gpu}" \
    CHATTIME_RESULTS_DIR="$shard_dir" \
    CHATTIME_LOG_FILE="$shard_log" \
    "$PY_BIN" -B "$SCRIPT_PATH"
  ) &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done
if [ "$FAIL" -ne 0 ]; then
  echo "[chattime-20k] one or more shards failed" >&2
  exit 1
fi

MERGED_JSONL="$MERGED_DIR/qwen_raw_answers.jsonl"
>"$MERGED_JSONL"
for gpu in "${GPU_IDS[@]}"; do
  shard_json="$RUN_ROOT/shards/shard_$gpu/qwen_raw_answers.jsonl"
  if [ ! -f "$shard_json" ]; then
    echo "missing shard output: $shard_json" >&2
    exit 1
  fi
  cat "$shard_json" >> "$MERGED_JSONL"
done

"$PY_BIN" - <<'PY'
import json
from pathlib import Path
path = Path("/WORK/fit/zhangchen/axis_project/runs/chattime_ts61_20k_8gpu_parallel_20260618a/full_20k/chattime/qwen_raw_answers.jsonl")
rows = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
rows.sort(key=lambda row: int(row["index"]))
with path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(len(rows))
PY

"$PY_BIN" -B "$AXIS_ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
  --run-root "$RUN_ROOT" \
  --questions "$FULL_Q" \
  --teacher "$TEACHER_JSON" \
  --runs chattime

"$PY_BIN" -B "$AXIS_ROOT/scripts/mvaxis_compute_metrics_with_open_decision.py" \
  --root "$RUN_ROOT" \
  --questions "$FULL_Q" \
  --teacher "$TEACHER_JSON" \
  --groups chattime

"$PY_BIN" -B "$AXIS_ROOT/scripts/mvaxis_build_answer_review_html.py" \
  --root "$RUN_ROOT" \
  --groups chattime \
  --questions "$FULL_Q" \
  --teacher "$TEACHER_JSON"

"$PY_BIN" -B "$AXIS_ROOT/scripts/mvaxis_build_eval_summary_html.py" \
  --summary-json "$RUN_ROOT/teacher_aligned_summary.json" \
  --title "TS61_20K_ChatTime_20260618A" \
  --output-html "$RUN_ROOT/one_page_report.html"

echo "[chattime-20k] done: $RUN_ROOT"
