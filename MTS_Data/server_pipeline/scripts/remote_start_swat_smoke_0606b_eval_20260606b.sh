set -euo pipefail

ROOT=/tmp/multiaxis_runtime_swat
PY=/root/hfenv/bin/python
OUT_BASE=/tmp/swat_smoke_0606b_eval_20260606b
DATA_DIR=/tmp/swat_regular_normalclean_smoke_0606b
Q="$DATA_DIR/questions_50.jsonl"
VIS_ROOT=/tmp/swat_regular_normalclean_smoke_0606b_visuals_raw
RAW_MANIFEST="$VIS_ROOT/raw_images/manifest.json"
TEACHER_DIR="$OUT_BASE/teacher"
TEACHER_JSON="$TEACHER_DIR/teacher_gpt55.jsonl"
RUNROOT="$OUT_BASE/student_runs"

mkdir -p "$TEACHER_DIR" "$RUNROOT" "$VIS_ROOT"

# 1) Generate teacher answers without images to avoid local-only image paths.
"$PY" -B "$ROOT/scripts/mvaxis_generate_teacher_answers.py" \
  --data "$Q" \
  --llm-config "$ROOT/configs/gpt55_question_llm.json" \
  --output "$TEACHER_JSON" \
  --max-window-rows 24 \
  --digits 3 \
  --no-image \
  > "$OUT_BASE/teacher.log" 2>&1

# 2) Build remote raw-image manifest from series payloads in the question JSONL.
"$PY" -B "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" \
  --data "$Q" \
  --output-root "$VIS_ROOT" \
  --modes raw \
  > "$OUT_BASE/raw_visual_prep.log" 2>&1

# 3) Student answer generation: same three-group comparison as prior SWAT smoke.
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
  --data "$Q" \
  --limit 50 \
  --output-dir "$RUNROOT/no_global_hints" \
  --llm-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu.json" \
  --config "$ROOT/configs/torch_fixed60_qa600_timercd.json" \
  --interval-proposer-checkpoint /root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt \
  --use-axis-embedding-hints \
  --hide-global-hints \
  --max-window-rows 24 \
  --digits 3 \
  > "$RUNROOT/no_global_hints.log" 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
  --data "$Q" \
  --limit 50 \
  --output-dir "$RUNROOT/values_only" \
  --llm-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_noaxis.json" \
  --max-window-rows 24 \
  --digits 3 \
  > "$RUNROOT/values_only.log" 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
  --data "$Q" \
  --limit 50 \
  --output-dir "$RUNROOT/raw_image" \
  --llm-config "$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_noaxis.json" \
  --raw-image-manifest "$RAW_MANIFEST" \
  --include-raw-image-note \
  --max-window-rows 24 \
  --digits 3 \
  > "$RUNROOT/raw_image.log" 2>&1

# 4) Teacher-aligned eval.
"$PY" "$ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
  --run-root "$RUNROOT" \
  --questions "$Q" \
  --teacher "$TEACHER_JSON" \
  --runs no_global_hints values_only raw_image \
  > "$RUNROOT/eval.log" 2>&1

# 5) Per-group latency summary.
"$PY" - <<'PY' > "$RUNROOT/latency_summary.json"
import json
import statistics
from pathlib import Path

runroot = Path("/tmp/swat_smoke_0606b_eval_20260606b/student_runs")
summary = {}
for run_name in ("no_global_hints", "values_only", "raw_image"):
    path = runroot / run_name / "qwen_raw_answers.jsonl"
    latencies = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                value = row.get("latency_seconds")
                if isinstance(value, (int, float)):
                    latencies.append(float(value))
    if latencies:
        summary[run_name] = {
            "count": len(latencies),
            "mean_latency_seconds": statistics.fmean(latencies),
            "median_latency_seconds": statistics.median(latencies),
            "min_latency_seconds": min(latencies),
            "max_latency_seconds": max(latencies),
        }
    else:
        summary[run_name] = {"count": 0}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo done
