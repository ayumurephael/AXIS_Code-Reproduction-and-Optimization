#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="question4000_ts61_eval_20260609a"
BASE="/dev/shm/${RUN_NAME}"
RUNTIME="$BASE/runtime/multiaxis"
INPUTS="$BASE/inputs"
OUTROOT="$BASE/student_runs"
CFGROOT="$BASE/configs"
PIPELINE_LOG="$BASE/pipeline.log"

PY="/root/shared-nvme/axis_project/envs/axis/bin/python"
if [ ! -x "$PY" ]; then
  PY="/root/hfenv/bin/python"
fi

BASE_LLM_CFG="$RUNTIME/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1.json"
AXIS_CFG="$RUNTIME/configs/torch_fixed60_qa600_timercd.json"
BRIDGE_ENCODER="/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt"

mkdir -p "$OUTROOT" "$CFGROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "[question4000] start $(date --iso-8601=seconds)"
echo "[question4000] runtime=$RUNTIME"
echo "[question4000] inputs=$INPUTS"
echo "[question4000] outroot=$OUTROOT"
echo "[question4000] python=$PY"

if [ ! -f "$BASE_LLM_CFG" ]; then
  echo "[question4000] missing base llm config: $BASE_LLM_CFG" >&2
  exit 1
fi
if [ ! -f "$AXIS_CFG" ]; then
  echo "[question4000] missing axis config: $AXIS_CFG" >&2
  exit 1
fi
if [ ! -f "$BRIDGE_ENCODER" ]; then
  echo "[question4000] missing encoder checkpoint: $BRIDGE_ENCODER" >&2
  exit 1
fi

"$PY" - <<'PYCODE'
import json
from pathlib import Path

base = Path("/dev/shm/question4000_ts61_eval_20260609a/runtime/multiaxis/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1.json")
dst = Path("/dev/shm/question4000_ts61_eval_20260609a/configs/all_hints_raw_image_1gpu.json")
cfg = json.loads(base.read_text(encoding="utf-8"))
cfg["device"] = "cuda:0"
cfg["device_map"] = "auto"
cfg["max_memory"] = {"0": "23500MiB", "cpu": "100GiB"}
cfg["min_pixels"] = 401408
cfg["max_pixels"] = 1204224
cfg["max_tokens"] = 1024
axis = dict(cfg.get("axis_hints") or {})
axis["enabled"] = True
axis["injection_scale"] = 0.1
axis["global_injection_scale"] = 0.1
axis["channel_injection_scale"] = 0.1
axis["max_global_tokens"] = 8
axis["max_channel_tokens"] = 16
axis["max_local_tokens"] = 16
cfg["axis_hints"] = axis
dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[question4000] wrote {dst}")
PYCODE

run_dataset() {
  local gpu="$1"
  local dataset="$2"
  local q="$INPUTS/${dataset}.questions_4000_student.jsonl"
  local t="$INPUTS/${dataset}.teacher_gpt55.jsonl"
  local raw_manifest="$INPUTS/${dataset}.raw_manifest.json"
  local run_dir="$OUTROOT/$dataset/all_hints"
  local run_log="$OUTROOT/$dataset/all_hints.log"
  local eval_log="$OUTROOT/$dataset/eval_all_hints.log"

  mkdir -p "$run_dir"
  echo "[question4000] launch dataset=$dataset gpu=$gpu at $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -B "$RUNTIME/scripts/mvaxis_run_raw_qwen_answers.py" \
    --data "$q" \
    --output-dir "$run_dir" \
    --llm-config "$CFGROOT/all_hints_raw_image_1gpu.json" \
    --config "$AXIS_CFG" \
    --interval-proposer-checkpoint "$BRIDGE_ENCODER" \
    --use-axis-embedding-hints \
    --include-raw-image-note \
    --raw-image-manifest "$raw_manifest" \
    --max-window-rows 24 \
    --digits 3 \
    --progress-percent-step 5 \
    > "$run_log" 2>&1

  "$PY" "$RUNTIME/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
    --run-root "$OUTROOT/$dataset" \
    --questions "$q" \
    --teacher "$t" \
    --runs all_hints \
    > "$eval_log" 2>&1

  echo "[question4000] finished dataset=$dataset gpu=$gpu at $(date --iso-8601=seconds)"
}

DATASETS=(
  question_4000_ts61_0000_1000_0609a
  question_4000_ts61_1000_2000_0609a
  question_4000_ts61_2000_3000_0609a
  question_4000_ts61_3000_4000_0609a
  question_4000_ts61_4000_5000_0609a
)

FREE_GPUS=(0 1 2 3)
declare -A PID_TO_GPU=()
declare -A PID_TO_DATASET=()
next_idx=0

while [ "$next_idx" -lt "${#DATASETS[@]}" ] || [ "${#PID_TO_GPU[@]}" -gt 0 ]; do
  while [ "$next_idx" -lt "${#DATASETS[@]}" ] && [ "${#FREE_GPUS[@]}" -gt 0 ]; do
    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    dataset="${DATASETS[$next_idx]}"
    run_dataset "$gpu" "$dataset" &
    pid=$!
    PID_TO_GPU[$pid]="$gpu"
    PID_TO_DATASET[$pid]="$dataset"
    echo "[question4000] pid=$pid dataset=$dataset gpu=$gpu"
    next_idx=$((next_idx + 1))
  done

  sleep 20
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        echo "[question4000] pid=$pid dataset=${PID_TO_DATASET[$pid]} completed"
      else
        echo "[question4000] pid=$pid dataset=${PID_TO_DATASET[$pid]} failed" >&2
        exit 1
      fi
      FREE_GPUS+=("${PID_TO_GPU[$pid]}")
      unset 'PID_TO_GPU[$pid]'
      unset 'PID_TO_DATASET[$pid]'
    fi
  done
done

echo "[question4000] all datasets finished $(date --iso-8601=seconds)"
