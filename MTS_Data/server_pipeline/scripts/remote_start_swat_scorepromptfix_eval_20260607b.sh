set -euo pipefail

ROOT=/tmp/multiaxis_runtime_swat
OUT_BASE=/tmp/swat_scorepromptfix_eval_20260607b
PY=/root/hfenv/bin/python
CFG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
LLM_AXIS="$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_ct16_scale005.json"
CKPT=/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt

run_eval() {
  local name="$1"
  local q="$2"
  local t="$3"
  local raw_manifest="$4"
  local score_root="$5"
  local limit="$6"

  local runroot="$OUT_BASE/$name"
  local score_manifest="$score_root/score_images/manifest.json"
  mkdir -p "$runroot"

  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" \
    --data "$q" \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --output-root "$score_root" \
    --modes score \
    > "$runroot/score_visual_prep.log" 2>&1

  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
    --data "$q" \
    --limit "$limit" \
    --output-dir "$runroot/raw_image" \
    --llm-config "$LLM_AXIS" \
    --include-raw-image-note \
    --raw-image-manifest "$raw_manifest" \
    --max-window-rows 24 \
    --digits 3 \
    > "$runroot/raw_image.log" 2>&1

  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
    --data "$q" \
    --limit "$limit" \
    --output-dir "$runroot/score_text_no_global_hints_ct16_scale005" \
    --llm-config "$LLM_AXIS" \
    --config "$CFG" \
    --interval-proposer-checkpoint "$CKPT" \
    --use-axis-embedding-hints \
    --hide-global-hints \
    --include-anomaly-score-text \
    --include-raw-image-note \
    --raw-image-manifest "$raw_manifest" \
    --max-window-rows 24 \
    --digits 3 \
    > "$runroot/score_text_no_global_hints_ct16_scale005.log" 2>&1

  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
    --data "$q" \
    --limit "$limit" \
    --output-dir "$runroot/score_image_no_global_hints_ct16_scale005" \
    --llm-config "$LLM_AXIS" \
    --config "$CFG" \
    --interval-proposer-checkpoint "$CKPT" \
    --use-axis-embedding-hints \
    --hide-global-hints \
    --include-anomaly-score-image-note \
    --score-image-manifest "$score_manifest" \
    --max-window-rows 24 \
    --digits 3 \
    > "$runroot/score_image_no_global_hints_ct16_scale005.log" 2>&1

  "$PY" "$ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
    --run-root "$runroot" \
    --questions "$q" \
    --teacher "$t" \
    --runs raw_image score_text_no_global_hints_ct16_scale005 score_image_no_global_hints_ct16_scale005 \
    > "$runroot/eval.log" 2>&1

  echo "$name scorepromptfix eval done"
}

run_eval \
  smoke_0606b \
  /tmp/swat_regular_normalclean_smoke_0606b/questions_50_student.jsonl \
  /tmp/swat_smoke_0606b_eval_20260606c/teacher/teacher_gpt55.jsonl \
  /tmp/swat_regular_normalclean_smoke_0606b_visuals_raw/raw_images/manifest.json \
  /tmp/swat_regular_normalclean_smoke_0606b_visuals_score_20260607b \
  50

run_eval \
  regular_160_0605b \
  /tmp/swat_regular_160_0605b/questions_160_student.jsonl \
  /tmp/swat_regular_160_0605b/teacher_gpt55.jsonl \
  /tmp/swat_regular_160_0605b_visuals_raw/raw_images/manifest.json \
  /tmp/swat_regular_160_0605b_visuals_score_20260607b \
  160

run_eval \
  hard_160_0605a \
  /tmp/swat_hard_160_0605a/questions_160_student.jsonl \
  /tmp/swat_hard_160_0605a/teacher_gpt55.jsonl \
  /tmp/swat_hard_160_0605a_visuals_raw/raw_images/manifest.json \
  /tmp/swat_hard_160_0605a_visuals_score_20260607b \
  160
