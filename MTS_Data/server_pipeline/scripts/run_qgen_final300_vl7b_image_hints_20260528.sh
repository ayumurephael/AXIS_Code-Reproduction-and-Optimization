#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/axis_project/repos/multiaxis

PY="/root/shared-nvme/axis_project/envs/axis/bin/python"
DATA="outputs/questiongeneration_final/questions_600.jsonl"
TEACHER="outputs/quetiongeneration_1_teacher_answer/teacher_gpt55.jsonl"
CONFIG="configs/torch_fixed60_qa600_timercd.json"
VL_LLM="configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256.json"
CKPT="outputs/checkpoints/timercd_global_hint_head_truth600_labelglobal_e10_encoder.pt"
SCORE_MANIFEST="outputs/questiongeneration_final/anomaly_score_images_max_legacy_style_v2/manifest.json"
RAW_MANIFEST="outputs/questiongeneration_final/raw_images_legacy_style_v2/manifest.json"
RUN_ROOT="outputs/runs/student_qgen_final300_vl7b_image_hints_20260528_v1"
LIMIT=300

COMMON_ARGS=(
  --data "$DATA"
  --llm-config "$VL_LLM"
  --limit "$LIMIT"
  --max-tokens 1024
  --prompt-examples 3
  --raw-examples 10
  --interval-source target
)
AXIS_ARGS=(
  --config "$CONFIG"
  --interval-proposer-checkpoint "$CKPT"
  --use-axis-embedding-hints
)
SCORE_ARGS=(
  --config "$CONFIG"
  --interval-proposer-checkpoint "$CKPT"
  --anomaly-score-aggregation max
)

mkdir -p "$RUN_ROOT/logs"

run_group() {
  local name="$1"
  shift
  local existing="$RUN_ROOT/$name/qwen_raw_answers.jsonl"
  if [[ -f "$existing" ]]; then
    local lines
    lines=$(wc -l < "$existing" || echo 0)
    if [[ "$lines" -ge "$LIMIT" ]]; then
      echo "===== SKIP $name ($lines/$LIMIT complete) ====="
      return 0
    fi
  fi
  echo "===== RUN $name ====="
  "$PY" -B scripts/mvaxis_run_raw_qwen_answers.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "$RUN_ROOT/$name" \
    "$@" 2>&1 | tee "$RUN_ROOT/logs/${name}.log"
}

run_group "raw_image" \
  --include-raw-image-note \
  --raw-image-manifest "$RAW_MANIFEST"

run_group "score_text_image_no_global_hints" \
  "${AXIS_ARGS[@]}" \
  "${SCORE_ARGS[@]}" \
  --hide-global-hints \
  --include-anomaly-score-text \
  --include-anomaly-score-image-note \
  --score-image-manifest "$SCORE_MANIFEST"

run_group "score_text_image_no_channel_hints" \
  "${AXIS_ARGS[@]}" \
  "${SCORE_ARGS[@]}" \
  --hide-channel-hints \
  --include-anomaly-score-text \
  --include-anomaly-score-image-note \
  --score-image-manifest "$SCORE_MANIFEST"

"$PY" -B scripts/mvaxis_eval_teacher_aligned_student_runs.py \
  --run-root "$RUN_ROOT" \
  --questions "$DATA" \
  --teacher "$TEACHER" \
  --limit "$LIMIT" \
  --runs \
    raw_image \
    score_text_image_no_global_hints \
    score_text_image_no_channel_hints \
  2>&1 | tee "$RUN_ROOT/logs/eval.log"
