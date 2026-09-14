set -euo pipefail

ROOT=/tmp/multiaxis_runtime_swat
OUT_BASE=/tmp/swat_160_ct16_scale005_rerun_20260607a
PY=/root/hfenv/bin/python
CFG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
LLM_AXIS="$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_ct16_scale005.json"
CKPT=/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt

for NAME in regular_160_0605b hard_160_0605a; do
  if [ "$NAME" = "regular_160_0605b" ]; then
    Q=/tmp/swat_regular_160_0605b/questions_160_student.jsonl
    T=/tmp/swat_regular_160_0605b/teacher_gpt55.jsonl
    RAW=/tmp/swat_regular_160_0605b_visuals_raw/raw_images/manifest.json
    SCORE_ROOT=/tmp/swat_regular_160_0605b_visuals_score
  else
    Q=/tmp/swat_hard_160_0605a/questions_160_student.jsonl
    T=/tmp/swat_hard_160_0605a/teacher_gpt55.jsonl
    RAW=/tmp/swat_hard_160_0605a_visuals_raw/raw_images/manifest.json
    SCORE_ROOT=/tmp/swat_hard_160_0605a_visuals_score
  fi

  SCORE="$SCORE_ROOT/score_images/manifest.json"
  RUNROOT="$OUT_BASE/$NAME"
  mkdir -p "$RUNROOT"

  if [ ! -f "$SCORE" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" \
      --data "$Q" \
      --config "$CFG" \
      --checkpoint "$CKPT" \
      --output-root "$SCORE_ROOT" \
      --modes score \
      > "$RUNROOT/score_visual_prep.log" 2>&1
  fi

  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
    --data "$Q" \
    --limit 160 \
    --output-dir "$RUNROOT/no_global_hints_ct16_scale005" \
    --llm-config "$LLM_AXIS" \
    --config "$CFG" \
    --interval-proposer-checkpoint "$CKPT" \
    --use-axis-embedding-hints \
    --hide-global-hints \
    --include-raw-image-note \
    --raw-image-manifest "$RAW" \
    --max-window-rows 24 \
    --digits 3 \
    > "$RUNROOT/no_global_hints_ct16_scale005.log" 2>&1

  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
    --data "$Q" \
    --limit 160 \
    --output-dir "$RUNROOT/score_text_no_global_hints_ct16_scale005" \
    --llm-config "$LLM_AXIS" \
    --config "$CFG" \
    --interval-proposer-checkpoint "$CKPT" \
    --use-axis-embedding-hints \
    --hide-global-hints \
    --include-anomaly-score-text \
    --include-raw-image-note \
    --raw-image-manifest "$RAW" \
    --max-window-rows 24 \
    --digits 3 \
    > "$RUNROOT/score_text_no_global_hints_ct16_scale005.log" 2>&1

  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
    --data "$Q" \
    --limit 160 \
    --output-dir "$RUNROOT/score_image_no_global_hints_ct16_scale005" \
    --llm-config "$LLM_AXIS" \
    --config "$CFG" \
    --interval-proposer-checkpoint "$CKPT" \
    --use-axis-embedding-hints \
    --hide-global-hints \
    --include-anomaly-score-image-note \
    --score-image-manifest "$SCORE" \
    --max-window-rows 24 \
    --digits 3 \
    > "$RUNROOT/score_image_no_global_hints_ct16_scale005.log" 2>&1

  "$PY" "$ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
    --run-root "$RUNROOT" \
    --questions "$Q" \
    --teacher "$T" \
    --runs no_global_hints_ct16_scale005 score_text_no_global_hints_ct16_scale005 score_image_no_global_hints_ct16_scale005 \
    > "$RUNROOT/eval.log" 2>&1

  echo "$NAME ct16_scale005 rerun done"
done
