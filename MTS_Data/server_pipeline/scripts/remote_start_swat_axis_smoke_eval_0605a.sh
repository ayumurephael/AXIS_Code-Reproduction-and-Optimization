#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/shared-nvme/axis_project/repos/multiaxis"
PYTHON="/root/shared-nvme/axis_project/envs/axis/bin/python"
CONFIG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
LLM_CONFIG_2GPU="/tmp/local_qwen25_vl7b_cuda_remote_timercd_dproj256_question1000_0603_eval_2gpu.json"
ENCODER_CKPT="/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt"

DATA_ROOT="/tmp/qsmoke_swat_axis_0605a"
VIS_ROOT="/tmp/qsmoke_swat_axis_0605a_visuals_20260605a"
PIPE_ROOT="/tmp/question_swat_axis_smoke_eval_pipeline_20260605a"
OUT_ROOT="/tmp/question_swat_axis_smoke_0605a_newweight_9group_20260605a"
PART_A="${OUT_ROOT}_partA"
PART_B="${OUT_ROOT}_partB"

GROUPS_A=(
  all_hints
  no_global_hints
  no_channel_hints
  score_image_note_all_hints
  score_text_all_hints
)

GROUPS_B=(
  raw_image
  score_text_image_no_global_hints
  score_text_image_no_channel_hints
  score_text_nohints
)

ALL_GROUPS=(
  all_hints
  no_global_hints
  no_channel_hints
  score_image_note_all_hints
  score_text_all_hints
  raw_image
  score_text_image_no_global_hints
  score_text_image_no_channel_hints
  score_text_nohints
)

mkdir -p "$PIPE_ROOT"
exec > >(tee -a "$PIPE_ROOT/pipeline.log") 2>&1

echo "[swat-smoke-eval] start $(date --iso-8601=seconds)"
echo "[swat-smoke-eval] root=$ROOT"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

rm -rf "$VIS_ROOT" "$OUT_ROOT" "$PART_A" "$PART_B"
mkdir -p "$VIS_ROOT" "$OUT_ROOT"

echo "[swat-smoke-eval] prepare visuals"
"$PYTHON" -B scripts/mvaxis_prepare_dataset_visuals.py \
  --data "$DATA_ROOT/questions_50.jsonl" \
  --config "$CONFIG" \
  --checkpoint "$ENCODER_CKPT" \
  --output-root "$VIS_ROOT" \
  --modes raw score \
  --aggregation max

echo "[swat-smoke-eval] launch part A on GPU0,1"
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" -B scripts/mvaxis_run_denoised_student_answer_eval.py \
  --data "$DATA_ROOT/questions_50.jsonl" \
  --teacher "$DATA_ROOT/teacher_gpt55.jsonl" \
  --llm-config "$LLM_CONFIG_2GPU" \
  --config "$CONFIG" \
  --interval-proposer-checkpoint "$ENCODER_CKPT" \
  --output-root "$PART_A" \
  --score-manifest "$VIS_ROOT/score_images/manifest.json" \
  --raw-manifest "$VIS_ROOT/raw_images/manifest.json" \
  --groups "${GROUPS_A[@]}" \
  > "$PIPE_ROOT/partA.log" 2>&1 &
PID_A=$!

echo "[swat-smoke-eval] launch part B on GPU2,3"
CUDA_VISIBLE_DEVICES=2,3 "$PYTHON" -B scripts/mvaxis_run_denoised_student_answer_eval.py \
  --data "$DATA_ROOT/questions_50.jsonl" \
  --teacher "$DATA_ROOT/teacher_gpt55.jsonl" \
  --llm-config "$LLM_CONFIG_2GPU" \
  --config "$CONFIG" \
  --interval-proposer-checkpoint "$ENCODER_CKPT" \
  --output-root "$PART_B" \
  --score-manifest "$VIS_ROOT/score_images/manifest.json" \
  --raw-manifest "$VIS_ROOT/raw_images/manifest.json" \
  --groups "${GROUPS_B[@]}" \
  > "$PIPE_ROOT/partB.log" 2>&1 &
PID_B=$!

echo "[swat-smoke-eval] pids partA=$PID_A partB=$PID_B"
wait "$PID_A"
echo "[swat-smoke-eval] part A finished $(date --iso-8601=seconds)"
wait "$PID_B"
echo "[swat-smoke-eval] part B finished $(date --iso-8601=seconds)"

echo "[swat-smoke-eval] merge outputs"
cp -a "$PART_A/_llm_configs" "$OUT_ROOT/_llm_configs"
for group in "${GROUPS_A[@]}"; do
  cp -a "$PART_A/$group" "$OUT_ROOT/$group"
done
for group in "${GROUPS_B[@]}"; do
  cp -a "$PART_B/$group" "$OUT_ROOT/$group"
done

echo "[swat-smoke-eval] teacher-aligned summary"
"$PYTHON" -B scripts/mvaxis_eval_teacher_aligned_student_runs.py \
  --run-root "$OUT_ROOT" \
  --questions "$DATA_ROOT/questions_50.jsonl" \
  --teacher "$DATA_ROOT/teacher_gpt55.jsonl" \
  --runs "${ALL_GROUPS[@]}"

echo "[swat-smoke-eval] done $(date --iso-8601=seconds)"
