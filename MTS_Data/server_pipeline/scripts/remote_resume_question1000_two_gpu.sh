#!/usr/bin/env bash
set -euo pipefail

# Resume the question_1000_0602 denoised 9-group run by:
# 1) waiting for the currently running group to finish,
# 2) stopping the old single-GPU parent runner before it starts the remaining groups,
# 3) launching the last two groups on two GPUs in parallel,
# 4) rerunning the teacher-aligned evaluation across all nine groups.

ROOT="/root/shared-nvme/axis_project/repos/multiaxis"
PY="/root/shared-nvme/axis_project/envs/axis/bin/python"

RUN_ROOT="/tmp/denoised_hard1000_9group_20260602a"
DATA="/tmp/qhard1000_data_20260602a/questions_1000_student.jsonl"
TEACHER="/tmp/qhard1000_data_20260602a/teacher_gpt55.jsonl"
AXIS_CFG="/root/shared-nvme/axis_project/repos/multiaxis/configs/torch_fixed60_qa600_timercd.json"
INTERVAL_CKPT="/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt"
SCORE_MANIFEST="/tmp/qhard1000_visuals_20260602a/score_images/manifest.json"

LLM_AXIS_ON="$RUN_ROOT/_llm_configs/llm_axis_denoised.json"
LLM_AXIS_OFF="$RUN_ROOT/_llm_configs/llm_no_axis.json"

CURRENT_GROUP_REPORT="$RUN_ROOT/score_text_image_no_global_hints/qwen_raw_report.json"
GROUP8_DIR="$RUN_ROOT/score_text_image_no_channel_hints"
GROUP9_DIR="$RUN_ROOT/score_text_nohints"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[two-gpu-resume] waiting for current group to finish: $CURRENT_GROUP_REPORT"
while [[ ! -f "$CURRENT_GROUP_REPORT" ]]; do
  sleep 15
done

echo "[two-gpu-resume] current group finished; stopping old single-GPU parent if still alive"
pkill -f "mvaxis_run_denoised_student_answer_eval.py --data $DATA" || true
pkill -f "$GROUP8_DIR" || true
pkill -f "$GROUP9_DIR" || true
sleep 5

for d in "$GROUP8_DIR" "$GROUP9_DIR"; do
  if [[ -d "$d" && ! -f "$d/qwen_raw_report.json" ]]; then
    mv "$d" "${d}.stale_${STAMP}"
  fi
done

mkdir -p "$GROUP8_DIR" "$GROUP9_DIR"

echo "[two-gpu-resume] launching remaining groups in parallel on GPU0 and GPU1"
nohup env CUDA_VISIBLE_DEVICES=0 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
  --data "$DATA" \
  --llm-config "$LLM_AXIS_ON" \
  --output-dir "$GROUP8_DIR" \
  --limit 1000 \
  --max-tokens 1024 \
  --prompt-examples 4 \
  --raw-examples 8 \
  --max-window-rows 0 \
  --digits 3 \
  --progress-percent-step 5 \
  --use-axis-embedding-hints \
  --config "$AXIS_CFG" \
  --interval-proposer-checkpoint "$INTERVAL_CKPT" \
  --hide-channel-hints \
  --include-anomaly-score-text \
  --include-anomaly-score-image-note \
  --score-image-manifest "$SCORE_MANIFEST" \
  > "$RUN_ROOT/score_text_image_no_channel_hints.gpu0.log" 2>&1 &
PID0=$!

nohup env CUDA_VISIBLE_DEVICES=1 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
  --data "$DATA" \
  --llm-config "$LLM_AXIS_OFF" \
  --output-dir "$GROUP9_DIR" \
  --limit 1000 \
  --max-tokens 1024 \
  --prompt-examples 4 \
  --raw-examples 8 \
  --max-window-rows 0 \
  --digits 3 \
  --progress-percent-step 5 \
  --config "$AXIS_CFG" \
  --interval-proposer-checkpoint "$INTERVAL_CKPT" \
  --include-anomaly-score-text \
  --score-image-manifest "$SCORE_MANIFEST" \
  > "$RUN_ROOT/score_text_nohints.gpu1.log" 2>&1 &
PID1=$!

echo "[two-gpu-resume] gpu0 pid=$PID0 gpu1 pid=$PID1"
wait "$PID0"
wait "$PID1"

echo "[two-gpu-resume] remaining groups finished; recomputing teacher-aligned eval"
"$PY" -B "$ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
  --run-root "$RUN_ROOT" \
  --questions "$DATA" \
  --teacher "$TEACHER" \
  --runs \
    all_hints \
    no_global_hints \
    no_channel_hints \
    score_image_note_all_hints \
    score_text_all_hints \
    raw_image \
    score_text_image_no_global_hints \
    score_text_image_no_channel_hints \
    score_text_nohints \
  | tee "$RUN_ROOT/teacher_aligned_eval_rerun.log"

echo "[two-gpu-resume] done"
