set -euo pipefail

ROOT=/tmp/multiaxis_runtime_swat
OUT_BASE=/tmp/swat_smoke_0606b_eval_20260606c
RUNROOT="$OUT_BASE/student_runs"
PY=/root/hfenv/bin/python
CFG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
LLM_AXIS="$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu.json"
CKPT=/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt

Q=/tmp/swat_regular_normalclean_smoke_0606b/questions_50_student.jsonl
T="$OUT_BASE/teacher/teacher_gpt55.jsonl"
RAW=/tmp/swat_regular_normalclean_smoke_0606b_visuals_raw/raw_images/manifest.json
SCORE_ROOT=/tmp/swat_regular_normalclean_smoke_0606b_visuals_score
SCORE="$SCORE_ROOT/score_images/manifest.json"

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
  --limit 50 \
  --output-dir "$RUNROOT/score_text_no_global_hints" \
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
  > "$RUNROOT/score_text_no_global_hints.log" 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
  --data "$Q" \
  --limit 50 \
  --output-dir "$RUNROOT/score_image_no_global_hints" \
  --llm-config "$LLM_AXIS" \
  --config "$CFG" \
  --interval-proposer-checkpoint "$CKPT" \
  --use-axis-embedding-hints \
  --hide-global-hints \
  --include-anomaly-score-image-note \
  --score-image-manifest "$SCORE" \
  --max-window-rows 24 \
  --digits 3 \
  > "$RUNROOT/score_image_no_global_hints.log" 2>&1

"$PY" "$ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
  --run-root "$RUNROOT" \
  --questions "$Q" \
  --teacher "$T" \
  --runs no_global_hints values_only raw_image score_text_no_global_hints score_image_no_global_hints \
  > "$RUNROOT/eval.log" 2>&1

"$PY" - <<PYCODE
import json
from pathlib import Path

run_root = Path("$RUNROOT")
lat = {}
for name in [
    "no_global_hints",
    "values_only",
    "raw_image",
    "score_text_no_global_hints",
    "score_image_no_global_hints",
]:
    path = run_root / name / "qwen_raw_answers.jsonl"
    if not path.exists():
        continue
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("latency_seconds") is not None:
            vals.append(float(rec["latency_seconds"]))
    if vals:
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        lat[name] = {
            "count": n,
            "mean_latency_seconds": sum(vals_sorted) / n,
            "median_latency_seconds": vals_sorted[n // 2] if n % 2 == 1 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2,
            "min_latency_seconds": vals_sorted[0],
            "max_latency_seconds": vals_sorted[-1],
        }
(run_root / "latency_summary.json").write_text(json.dumps(lat, ensure_ascii=False, indent=2), encoding="utf-8")
PYCODE
