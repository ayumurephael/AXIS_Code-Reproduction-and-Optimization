#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/shared-nvme/axis_project/repos/multiaxis"
PYTHON="/root/shared-nvme/axis_project/envs/axis/bin/python"
CONFIG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
LLM_CONFIG_2GPU="/tmp/local_qwen25_vl7b_cuda_remote_timercd_dproj256_question1000_0603_eval_2gpu.json"
ENCODER_CKPT="/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt"
PIPE_ROOT="/tmp/eval_parallel_20260604a"

mkdir -p "$PIPE_ROOT"
exec > >(tee -a "$PIPE_ROOT/pipeline.log") 2>&1

echo "[parallel-eval] start $(date --iso-8601=seconds)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

Q1000_GROUPS=(
  no_global_hints
  no_channel_hints
  score_image_note_all_hints
  score_text_all_hints
  raw_image
  score_text_image_no_global_hints
  score_text_image_no_channel_hints
  score_text_nohints
)

echo "[parallel-eval] launch question_1000_0602 on GPU0,1"
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" -B scripts/mvaxis_run_denoised_student_answer_eval.py \
  --data /tmp/qhard1000_data_20260602a/questions_1000_student.jsonl \
  --teacher /tmp/qhard1000_data_20260602a/teacher_gpt55.jsonl \
  --llm-config "$LLM_CONFIG_2GPU" \
  --config "$CONFIG" \
  --interval-proposer-checkpoint "$ENCODER_CKPT" \
  --output-root /tmp/question_1000_0602_newweight_9group_20260603a \
  --score-manifest /tmp/qhard1000_visuals_20260602a/score_images/manifest.json \
  --raw-manifest /tmp/qhard1000_visuals_20260602a/raw_images/manifest.json \
  --groups "${Q1000_GROUPS[@]}" \
  > "$PIPE_ROOT/question1000.log" 2>&1 &
PID_Q1000=$!

echo "[parallel-eval] launch hard200 on GPU2,3"
CUDA_VISIBLE_DEVICES=2,3 "$PYTHON" -B scripts/mvaxis_run_denoised_student_answer_eval.py \
  --data /tmp/qhard200_data_20260603a/questions_200_student.jsonl \
  --teacher /tmp/qhard200_data_20260603a/teacher_gpt55.jsonl \
  --llm-config "$LLM_CONFIG_2GPU" \
  --config "$CONFIG" \
  --interval-proposer-checkpoint "$ENCODER_CKPT" \
  --output-root /tmp/question_hard_200_0601_newweight_9group_20260603a \
  --score-manifest /tmp/qhard200_score_images_20260601/manifest.json \
  --raw-manifest /tmp/qhard200_raw_images_20260601/manifest.json \
  > "$PIPE_ROOT/hard200.log" 2>&1 &
PID_HARD200=$!

echo "[parallel-eval] pids question1000=$PID_Q1000 hard200=$PID_HARD200"
wait "$PID_HARD200"
echo "[parallel-eval] hard200 finished $(date --iso-8601=seconds)"

echo "[parallel-eval] launch smoke on GPU2,3"
CUDA_VISIBLE_DEVICES=2,3 "$PYTHON" -B scripts/mvaxis_run_denoised_student_answer_eval.py \
  --data /tmp/qsmoke_regular_channelname_0603gh/questions_100_student.jsonl \
  --teacher /tmp/qsmoke_regular_channelname_0603gh/teacher_gpt55.jsonl \
  --llm-config "$LLM_CONFIG_2GPU" \
  --config "$CONFIG" \
  --interval-proposer-checkpoint "$ENCODER_CKPT" \
  --output-root /tmp/question_regular_channelname_smoke_0603gh_newweight_9group_20260603a \
  --score-manifest /tmp/qsmoke_regular_channelname_0603gh_visuals_20260603a/score_images/manifest.json \
  --raw-manifest /tmp/qsmoke_regular_channelname_0603gh_visuals_20260603a/raw_images/manifest.json \
  > "$PIPE_ROOT/smoke.log" 2>&1 &
PID_SMOKE=$!

wait "$PID_Q1000"
echo "[parallel-eval] question1000 finished $(date --iso-8601=seconds)"
wait "$PID_SMOKE"
echo "[parallel-eval] smoke finished $(date --iso-8601=seconds)"
echo "[parallel-eval] all done $(date --iso-8601=seconds)"
