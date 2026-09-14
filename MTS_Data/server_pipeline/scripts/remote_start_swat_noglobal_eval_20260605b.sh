#!/usr/bin/env bash
set -euo pipefail
ROOT="/tmp/multiaxis_runtime_swat"
PYTHON="/root/shared-nvme/axis_project/envs/axis/bin/python"
OUT="/tmp/question_swat_axis_smoke_0605a_newweight_noglobal_20260605b"
LOGROOT="/tmp/question_swat_axis_smoke_0605a_newweight_noglobal_20260605b_logs"
mkdir -p "$OUT/no_global_hints" "$LOGROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec > >(tee -a "$LOGROOT/pipeline.log") 2>&1
echo "[swat-noglobal] start $(date --iso-8601=seconds)"
"$PYTHON" -B scripts/mvaxis_run_raw_qwen_answers.py \
  --data /tmp/qsmoke_swat_axis_0605a/questions_50_student.jsonl \
  --llm-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu.json" \
  --config "$ROOT/configs/torch_fixed60_qa600_timercd.json" \
  --interval-proposer-checkpoint /root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt \
  --use-axis-embedding-hints \
  --hide-global-hints \
  --limit 50 \
  --max-window-rows 24 \
  --output-dir "$OUT/no_global_hints" \
  --prompt-examples 4 \
  --raw-examples 6 \
  --progress-percent-step 10
"$PYTHON" -B scripts/mvaxis_eval_teacher_aligned_student_runs.py \
  --run-root "$OUT" \
  --questions /tmp/qsmoke_swat_axis_0605a/questions_50_student.jsonl \
  --teacher /tmp/qsmoke_swat_axis_0605a/teacher_gpt55.jsonl \
  --runs no_global_hints
echo "[swat-noglobal] done $(date --iso-8601=seconds)"
