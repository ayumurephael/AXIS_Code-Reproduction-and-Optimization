#!/usr/bin/env bash
set -euo pipefail
ROOT="/tmp/multiaxis_runtime_swat"
PYTHON="/root/shared-nvme/axis_project/envs/axis/bin/python"
OUT="/tmp/question_swat_axis_smoke_0605a_controls_20260605b"
VIS="/tmp/qsmoke_swat_axis_0605a_visuals_raw_20260605b"
LOGROOT="/tmp/question_swat_axis_smoke_0605a_controls_20260605b_logs"
mkdir -p "$OUT/values_only" "$OUT/raw_image" "$VIS" "$LOGROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec > >(tee -a "$LOGROOT/pipeline.log") 2>&1
echo "[swat-controls] start $(date --iso-8601=seconds)"
"$PYTHON" -B scripts/mvaxis_prepare_dataset_visuals.py \
  --data /tmp/qsmoke_swat_axis_0605a/questions_50_student.jsonl \
  --output-root "$VIS" \
  --modes raw
"$PYTHON" -B scripts/mvaxis_run_raw_qwen_answers.py \
  --data /tmp/qsmoke_swat_axis_0605a/questions_50_student.jsonl \
  --llm-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_noaxis.json" \
  --limit 50 \
  --max-window-rows 24 \
  --output-dir "$OUT/values_only" \
  --prompt-examples 4 \
  --raw-examples 6 \
  --progress-percent-step 10
"$PYTHON" -B scripts/mvaxis_run_raw_qwen_answers.py \
  --data /tmp/qsmoke_swat_axis_0605a/questions_50_student.jsonl \
  --llm-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_noaxis.json" \
  --include-raw-image-note \
  --raw-image-manifest "$VIS/raw_images/manifest.json" \
  --limit 50 \
  --max-window-rows 24 \
  --output-dir "$OUT/raw_image" \
  --prompt-examples 4 \
  --raw-examples 6 \
  --progress-percent-step 10
"$PYTHON" -B scripts/mvaxis_eval_teacher_aligned_student_runs.py \
  --run-root "$OUT" \
  --questions /tmp/qsmoke_swat_axis_0605a/questions_50_student.jsonl \
  --teacher /tmp/qsmoke_swat_axis_0605a/teacher_gpt55.jsonl \
  --runs values_only raw_image
echo "[swat-controls] done $(date --iso-8601=seconds)"
