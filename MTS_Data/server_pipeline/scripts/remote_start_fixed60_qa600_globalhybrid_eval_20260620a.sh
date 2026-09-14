#!/usr/bin/env bash
set -euo pipefail

ROOT="/WORK/fit/zhangchen/axis_project/multiaxis_runtime_train"
RUN_ROOT="/WORK/fit/zhangchen/axis_project/runs/fixed60_qa600_globalhybrid_eval_20260620a"
DATA_ROOT="/WORK/fit/zhangchen/axis_project/data/fixed60_qa600_legacy_dag_20260620a"
DATA_JSONL="$DATA_ROOT/train.jsonl"
TEACHER_JSONL="$DATA_ROOT/teacher_from_target.jsonl"
EMPTY_MANIFEST="$DATA_ROOT/empty_manifest.json"
LLM_CONFIG="$ROOT/configs/local_qwen25_vl7b_fixed60_qa600_globalhybrid_ct12_scale010_1gpu_20260620a.json"
AXIS_CONFIG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
INTERVAL_CKPT="/WORK/fit/zhangchen/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt"
CONDA_ROOT="/home/fit/zhangchen/WORK/miniconda3"
CONDA_ENV_NAME="Allava"

mkdir -p "$RUN_ROOT" "$DATA_ROOT"

if [ ! -f "$DATA_JSONL" ]; then
  echo "missing data jsonl: $DATA_JSONL" >&2
  exit 1
fi

if [ ! -f "$INTERVAL_CKPT" ]; then
  echo "missing interval checkpoint: $INTERVAL_CKPT" >&2
  exit 1
fi

if [ ! -f "$TEACHER_JSONL" ]; then
  python3 - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

data_path = Path("/WORK/fit/zhangchen/axis_project/data/fixed60_qa600_legacy_dag_20260620a/train.jsonl")
teacher_path = Path("/WORK/fit/zhangchen/axis_project/data/fixed60_qa600_legacy_dag_20260620a/teacher_from_target.jsonl")
teacher_path.parent.mkdir(parents=True, exist_ok=True)

with data_path.open("r", encoding="utf-8") as src, teacher_path.open("w", encoding="utf-8") as dst:
    for line in src:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        target = row.get("target_output") or {}
        answer = (
            target.get("final_answer")
            or target.get("question_answer")
            or ""
        )
        payload = {
            "sample_id": row.get("sample_id"),
            "teacher_answer_llm": {
                "answer": answer,
            },
            "answer": answer,
        }
        dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
fi

cat >"$EMPTY_MANIFEST" <<'JSON'
{
  "records": []
}
JSON

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -B "$ROOT/scripts/mvaxis_run_denoised_student_answer_eval.py" \
  --data "$DATA_JSONL" \
  --teacher "$TEACHER_JSONL" \
  --llm-config "$LLM_CONFIG" \
  --config "$AXIS_CONFIG" \
  --interval-proposer-checkpoint "$INTERVAL_CKPT" \
  --output-root "$RUN_ROOT" \
  --score-manifest "$EMPTY_MANIFEST" \
  --raw-manifest "$EMPTY_MANIFEST" \
  --groups all_hints \
  --limit 600 \
  --max-tokens 256 \
  --prompt-examples 4 \
  --raw-examples 8 \
  --max-window-rows 0 \
  --digits 3 \
  --progress-percent-step 5 \
  --max-channel-tokens 12 \
  --channel-injection-scale 0.1
