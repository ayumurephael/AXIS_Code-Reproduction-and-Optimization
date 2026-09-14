#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/shared-nvme/axis_project/repos/multiaxis"
PYTHON="/root/shared-nvme/axis_project/envs/axis/bin/python"
CONFIG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
LLM_CONFIG="/tmp/local_qwen25_vl7b_cuda_remote_timercd_dproj256_question1000_0603_eval.json"
ENCODER_CKPT="/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt"
PIPE_ROOT="/tmp/question1000_hard200_smoke_eval_pipeline_20260603a"

mkdir -p "$PIPE_ROOT"

exec > >(tee -a "$PIPE_ROOT/pipeline.log") 2>&1

echo "[pipeline] start $(date --iso-8601=seconds)"
echo "[pipeline] root=$ROOT"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[pipeline] prepare smoke visuals"
"$PYTHON" -B scripts/mvaxis_prepare_dataset_visuals.py \
  --data /tmp/qsmoke_regular_channelname_0603gh/questions_100_student.jsonl \
  --config "$CONFIG" \
  --checkpoint "$ENCODER_CKPT" \
  --output-root /tmp/qsmoke_regular_channelname_0603gh_visuals_20260603a \
  --modes raw score \
  --aggregation max

echo "[pipeline] eval question_1000_0602"
"$PYTHON" -B scripts/mvaxis_run_denoised_student_answer_eval.py \
  --data /tmp/qhard1000_data_20260602a/questions_1000_student.jsonl \
  --teacher /tmp/qhard1000_data_20260602a/teacher_gpt55.jsonl \
  --llm-config "$LLM_CONFIG" \
  --config "$CONFIG" \
  --interval-proposer-checkpoint "$ENCODER_CKPT" \
  --output-root /tmp/question_1000_0602_newweight_9group_20260603a \
  --score-manifest /tmp/qhard1000_visuals_20260602a/score_images/manifest.json \
  --raw-manifest /tmp/qhard1000_visuals_20260602a/raw_images/manifest.json

echo "[pipeline] eval hard200"
"$PYTHON" -B scripts/mvaxis_run_denoised_student_answer_eval.py \
  --data /tmp/qhard200_data_20260603a/questions_200_student.jsonl \
  --teacher /tmp/qhard200_data_20260603a/teacher_gpt55.jsonl \
  --llm-config "$LLM_CONFIG" \
  --config "$CONFIG" \
  --interval-proposer-checkpoint "$ENCODER_CKPT" \
  --output-root /tmp/question_hard_200_0601_newweight_9group_20260603a \
  --score-manifest /tmp/qhard200_score_images_20260601/manifest.json \
  --raw-manifest /tmp/qhard200_raw_images_20260601/manifest.json

echo "[pipeline] eval smoke regular-channelname 0603gh"
"$PYTHON" -B scripts/mvaxis_run_denoised_student_answer_eval.py \
  --data /tmp/qsmoke_regular_channelname_0603gh/questions_100_student.jsonl \
  --teacher /tmp/qsmoke_regular_channelname_0603gh/teacher_gpt55.jsonl \
  --llm-config "$LLM_CONFIG" \
  --config "$CONFIG" \
  --interval-proposer-checkpoint "$ENCODER_CKPT" \
  --output-root /tmp/question_regular_channelname_smoke_0603gh_newweight_9group_20260603a \
  --score-manifest /tmp/qsmoke_regular_channelname_0603gh_visuals_20260603a/score_images/manifest.json \
  --raw-manifest /tmp/qsmoke_regular_channelname_0603gh_visuals_20260603a/raw_images/manifest.json

echo "[pipeline] done $(date --iso-8601=seconds)"
