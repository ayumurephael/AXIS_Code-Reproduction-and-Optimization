#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/shared-nvme/axis_project/repos/multiaxis"
PY="/root/shared-nvme/axis_project/envs/axis/bin/python"

RUN_ROOT="/tmp/question_1000_0602_nexttoken_train_20260603_scoretext_hints_noimage_tightv3"
mkdir -p "$RUN_ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 "$ROOT/scripts/mvaxis_train_nexttoken_hints.py" \
  --data /tmp/qhard1000_data_20260602a/questions_1000_student.jsonl \
  --teacher /tmp/qhard1000_data_20260602a/teacher_gpt55.jsonl \
  --teacher-overrides /tmp/nonexistent_teacher_overrides.jsonl \
  --config "$ROOT/configs/torch_fixed60_qa600_timercd.json" \
  --llm-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu.json" \
  --interval-proposer-checkpoint /root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt \
  --init-bridge-checkpoint /root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1.pt \
  --interval-source target \
  --include-anomaly-score-text \
  --no-anomaly-score-image-note \
  --limit 1000 \
  --epochs 10 \
  --learning-rate 5e-5 \
  --weight-decay 0.0 \
  --gradient-accumulation-steps 1 \
  --max-grad-norm 1.0 \
  --answer-loss-weight 0.2 \
  --window-size 32 \
  --stride 8 \
  --max-window-rows 24 \
  --max-prompt-tokens 0 \
  --max-target-tokens 96 \
  --output-checkpoint /root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_4gpu_tightv3.pt \
  --output-encoder-checkpoint /root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_4gpu_tightv3_encoder.pt \
  --report "$RUN_ROOT/report.json" \
  --run-note "2026-06-03 question_1000_0602; VL Qwen2.5; score-text + global/channel hints; no image note; no score-image manifest; 4-GPU distributed training; tightv3 with max_window_rows=24 max_target_tokens=96; init from qfinal530_e10_typeheads_v1" \
  --progress-percent-step 10
