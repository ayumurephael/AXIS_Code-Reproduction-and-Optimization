#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/shared-nvme/axis_project/repos/multiaxis"
PY="/root/shared-nvme/axis_project/envs/axis/bin/python"

PROMPT_CAP="${PROMPT_CAP:-2048}"
RUN_ROOT="${RUN_ROOT:-/tmp/question1000_promptcap_smoke_20260603a/run_cap_${PROMPT_CAP}}"
DATA_PATH="${DATA_PATH:-/tmp/question1000_promptcap_smoke_20260603a/questions_top20_longest_prompt.jsonl}"
TEACHER_PATH="${TEACHER_PATH:-/tmp/question1000_promptcap_smoke_20260603a/teacher_top20_longest_prompt.jsonl}"
SCRIPT_PATH="${SCRIPT_PATH:-/tmp/mvaxis_train_nexttoken_hints.py}"

mkdir -p "$RUN_ROOT"
LLM_CONFIG="$RUN_ROOT/llm_config_shard.json"

export OMP_NUM_THREADS=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29631}"
export WORLD_SIZE=2
export PYTHONPATH="/tmp:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PY" - <<'PY' "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1.json" "$LLM_CONFIG" "$RUN_ROOT/offload"
import json, pathlib, sys
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
offload = pathlib.Path(sys.argv[3])
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["device"] = "cuda:0"
cfg["device_map"] = "auto"
cfg["max_memory"] = {"0": "22000MiB", "1": "22000MiB", "cpu": "100GiB"}
cfg["offload_folder"] = str(offload)
cfg["low_cpu_mem_usage"] = True
offload.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(dst)
PY

COMMON_ARGS=(
  "$SCRIPT_PATH"
  --data "$DATA_PATH"
  --teacher "$TEACHER_PATH"
  --teacher-overrides /tmp/nonexistent_teacher_overrides.jsonl
  --config "$ROOT/configs/torch_fixed60_qa600_timercd.json"
  --llm-config "$LLM_CONFIG"
  --interval-proposer-checkpoint /root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt
  --init-bridge-checkpoint /root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1.pt
  --interval-source target
  --include-anomaly-score-text
  --no-anomaly-score-image-note
  --preserve-device-map-in-distributed
  --limit 20
  --epochs 1
  --learning-rate 5e-5
  --weight-decay 0.0
  --gradient-accumulation-steps 1
  --max-grad-norm 1.0
  --answer-loss-weight 0.2
  --window-size 32
  --stride 8
  --max-window-rows 48
  --max-prompt-tokens "$PROMPT_CAP"
  --max-target-tokens 192
  --output-checkpoint /tmp/nonexistent_smoke_checkpoint.pt
  --output-encoder-checkpoint /tmp/nonexistent_smoke_encoder.pt
  --report "$RUN_ROOT/report.json"
  --run-note "2026-06-03 smoke prompt cap $PROMPT_CAP; question_1000_0602 longest prompts; VL Qwen2.5; score-text + global/channel hints; no image; 2 workers x 2 GPU model sharding; max_window_rows=48 max_target_tokens=192"
  --progress-percent-step 25
)

echo "[launcher] starting prompt-cap smoke at $(date -Is) cap=$PROMPT_CAP" | tee "$RUN_ROOT/launch.log"

(
  export CUDA_VISIBLE_DEVICES=0,1
  export RANK=0
  export LOCAL_RANK=0
  "$PY" -u "${COMMON_ARGS[@]}" > "$RUN_ROOT/worker0.log" 2>&1
) &
PID0=$!

(
  export CUDA_VISIBLE_DEVICES=2,3
  export RANK=1
  export LOCAL_RANK=0
  "$PY" -u "${COMMON_ARGS[@]}" > "$RUN_ROOT/worker1.log" 2>&1
) &
PID1=$!

echo "[launcher] worker0_pid=$PID0 worker1_pid=$PID1" | tee -a "$RUN_ROOT/launch.log"

wait $PID0
wait $PID1

echo "[launcher] completed at $(date -Is)" | tee -a "$RUN_ROOT/launch.log"
