set -euo pipefail
ROOT=/tmp/multiaxis_runtime_swat
OUT_BASE=/tmp/swat_160_eval_20260605c
mkdir -p "$OUT_BASE"
python - <<'PY'
import json
from pathlib import Path
pairs = [
    (Path('/tmp/swat_regular_160_0605b/questions_160.jsonl'), Path('/tmp/swat_regular_160_0605b/questions_160_student.jsonl')),
    (Path('/tmp/swat_hard_160_0605a/questions_160.jsonl'), Path('/tmp/swat_hard_160_0605a/questions_160_student.jsonl')),
]
for src, dst in pairs:
    rows=[]
    for line in src.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        if 'evidence_card' not in row or row['evidence_card'] is None:
            row['evidence_card']={}
        rows.append(row)
    dst.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)+'\n', encoding='utf-8')
PY
python "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" --data /tmp/swat_regular_160_0605b/questions_160_student.jsonl --output-root /tmp/swat_regular_160_0605b_visuals_raw --modes raw
python "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" --data /tmp/swat_hard_160_0605a/questions_160_student.jsonl --output-root /tmp/swat_hard_160_0605a_visuals_raw --modes raw
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
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" --input_jsonl "$Q" --output_dir "$RUNROOT/no_global_hints" --client-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu.json" --encoder-checkpoint /root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt --use-axis-embedding-hints --hide-global-hints --max-window-rows 24 --digits 3 > "$RUNROOT/no_global_hints.log" 2>&1
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" --input_jsonl "$Q" --output_dir "$RUNROOT/values_only" --client-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_noaxis.json" --max-window-rows 24 --digits 3 > "$RUNROOT/values_only.log" 2>&1
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" --input_jsonl "$Q" --output_dir "$RUNROOT/raw_image" --client-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_noaxis.json" --raw-image-manifest "$RAW" --include-raw-image-note --max-window-rows 24 --digits 3 > "$RUNROOT/raw_image.log" 2>&1
  python "$ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" --run-root "$RUNROOT" --questions "$Q" --teacher "$T" --runs no_global_hints values_only raw_image > "$RUNROOT/eval.log" 2>&1
  echo "$NAME done"
done
