set -euo pipefail

ROOT=/tmp/multiaxis_runtime_swat
OUT_BASE=/tmp/swat_smoke_0606b_hintformat_20260606f
RUNROOT="$OUT_BASE/student_runs"
PY=/root/hfenv/bin/python
CFG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
LLM_CT8="$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_ct8_scale010.json"
LLM_CT16="$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_ct16_scale005.json"
CKPT=/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt

Q=/tmp/swat_regular_normalclean_smoke_0606b/questions_50_student.jsonl
T=/tmp/swat_smoke_0606b_eval_20260606c/teacher/teacher_gpt55.jsonl
RAW=/tmp/swat_regular_normalclean_smoke_0606b_visuals_raw/raw_images/manifest.json

mkdir -p "$RUNROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
  --data "$Q" \
  --limit 50 \
  --output-dir "$RUNROOT/no_global_hints_ct8_scale010" \
  --llm-config "$LLM_CT8" \
  --config "$CFG" \
  --interval-proposer-checkpoint "$CKPT" \
  --use-axis-embedding-hints \
  --hide-global-hints \
  --include-raw-image-note \
  --raw-image-manifest "$RAW" \
  --max-window-rows 24 \
  --digits 3 \
  > "$RUNROOT/no_global_hints_ct8_scale010.log" 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
  --data "$Q" \
  --limit 50 \
  --output-dir "$RUNROOT/no_global_hints_ct16_scale005" \
  --llm-config "$LLM_CT16" \
  --config "$CFG" \
  --interval-proposer-checkpoint "$CKPT" \
  --use-axis-embedding-hints \
  --hide-global-hints \
  --include-raw-image-note \
  --raw-image-manifest "$RAW" \
  --max-window-rows 24 \
  --digits 3 \
  > "$RUNROOT/no_global_hints_ct16_scale005.log" 2>&1

"$PY" "$ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
  --run-root "$RUNROOT" \
  --questions "$Q" \
  --teacher "$T" \
  --runs no_global_hints_ct8_scale010 no_global_hints_ct16_scale005 \
  > "$RUNROOT/eval.log" 2>&1

"$PY" - <<PYCODE
import json
import re
from pathlib import Path

run_root = Path("$RUNROOT")

def summarize(path: Path):
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    answer_first = 0
    answer_later = 0
    no_answer = 0
    finish_reason_counts = {}
    hit_max_new_tokens = 0
    generated_token_count_values = []
    for rec in records:
        text = str(rec.get("raw_response") or "")
        meta = rec.get("llm_raw_metadata") or {}
        generated = meta.get("generated_token_count")
        if generated is not None:
            generated_token_count_values.append(int(generated))
        reason = str(meta.get("finish_reason") or "missing")
        finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1
        if bool(meta.get("hit_max_new_tokens")):
            hit_max_new_tokens += 1
        if text.startswith("Answer:"):
            answer_first += 1
        elif re.search(r"(?m)^Answer:", text):
            answer_later += 1
        else:
            no_answer += 1
    return {
        "count": len(records),
        "answer_first": answer_first,
        "answer_later": answer_later,
        "no_answer": no_answer,
        "finish_reason_counts": finish_reason_counts,
        "hit_max_new_tokens": hit_max_new_tokens,
        "mean_generated_token_count": (sum(generated_token_count_values) / len(generated_token_count_values)) if generated_token_count_values else None,
        "max_generated_token_count": max(generated_token_count_values) if generated_token_count_values else None,
    }

payload = {}
for name in ["no_global_hints_ct8_scale010", "no_global_hints_ct16_scale005"]:
    payload[name] = summarize(run_root / name / "qwen_raw_answers.jsonl")

(run_root / "format_diagnostics.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PYCODE
