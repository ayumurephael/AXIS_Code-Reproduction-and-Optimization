set -euo pipefail
ROOT=/tmp/multiaxis_runtime_swat
OUT_BASE=/tmp/swat_160_eval_20260605c
PY=/root/hfenv/bin/python
for NAME in regular_160_0605b hard_160_0605a; do
  if [ "$NAME" = "regular_160_0605b" ]; then
    Q=/tmp/swat_regular_160_0605b/questions_160_student.jsonl
    T=/tmp/swat_regular_160_0605b/teacher_gpt55.jsonl
    RAW=/tmp/swat_regular_160_0605b_visuals_raw/raw_images/manifest.json
  else
    Q=/tmp/swat_hard_160_0605a/questions_160_student.jsonl
    T=/tmp/swat_hard_160_0605a/teacher_gpt55.jsonl
    RAW=/tmp/swat_hard_160_0605a_visuals_raw/raw_images/manifest.json
  fi
  RUNROOT="$OUT_BASE/$NAME"
  mkdir -p "$RUNROOT"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" --data "$Q" --limit 160 --output-dir "$RUNROOT/no_global_hints" --llm-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu.json" --config "$ROOT/configs/torch_fixed60_qa600_timercd.json" --interval-proposer-checkpoint /root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt --use-axis-embedding-hints --hide-global-hints --max-window-rows 24 --digits 3 > "$RUNROOT/no_global_hints.log" 2>&1
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" --data "$Q" --limit 160 --output-dir "$RUNROOT/values_only" --llm-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_noaxis.json" --max-window-rows 24 --digits 3 > "$RUNROOT/values_only.log" 2>&1
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" --data "$Q" --limit 160 --output-dir "$RUNROOT/raw_image" --llm-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_noaxis.json" --raw-image-manifest "$RAW" --include-raw-image-note --max-window-rows 24 --digits 3 > "$RUNROOT/raw_image.log" 2>&1
  "$PY" "$ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" --run-root "$RUNROOT" --questions "$Q" --teacher "$T" --runs no_global_hints values_only raw_image > "$RUNROOT/eval.log" 2>&1
  echo "$NAME done"
done
