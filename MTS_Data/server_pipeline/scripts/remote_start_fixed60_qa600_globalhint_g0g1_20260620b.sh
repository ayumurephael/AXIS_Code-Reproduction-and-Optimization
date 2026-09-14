#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
ROOT="${ROOT:-$WORKROOT/multiaxis_runtime_train}"
DATA_ROOT="${DATA_ROOT:-$WORKROOT/data/fixed60_qa600_globalhint_train_20260620b}"
VIS_ROOT="${VIS_ROOT:-$WORKROOT/data/fixed60_qa600_globalhint_visuals_20260620b}"
RUN_ROOT="${RUN_ROOT:-$WORKROOT/runs/fixed60_qa600_globalhint_g0g1_20260620b}"

SOURCE_DATA="${SOURCE_DATA:-$WORKROOT/data/fixed60_qa600_legacy_dag_20260620a/train.jsonl}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/torch_fixed60_qa600_timercd.json}"
LLM_CONFIG="${LLM_CONFIG:-$ROOT/configs/local_qwen25_vl7b_fixed60_qa600_globalhint_train_score_ct12_scale010_1gpu_20260620b.json}"
INTERVAL_CKPT="${INTERVAL_CKPT:-$WORKROOT/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt}"
BASE_BRIDGE_CKPT="${BASE_BRIDGE_CKPT:-$WORKROOT/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1.pt}"

BLOCKING_PATTERN="${BLOCKING_PATTERN:-fixed60_qa600_globalhybrid_eval_20260620a}"
POLL_SEC="${POLL_SEC:-300}"
LIMIT="${LIMIT:-600}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

CONDA_SH="${CONDA_SH:-/home/fit/zhangchen/WORK/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-Allava}"
PY="${PY:-python}"

mkdir -p "$DATA_ROOT" "$VIS_ROOT" "$RUN_ROOT/checkpoints" "$RUN_ROOT/reports" "$RUN_ROOT/logs"

if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  PY=python
fi

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

wait_for_previous_eval() {
  while pgrep -af "$BLOCKING_PATTERN" >/dev/null 2>&1; do
    log "waiting for blocker pattern '$BLOCKING_PATTERN' to finish"
    sleep "$POLL_SEC"
  done
}

prepare_training_data() {
  cp -f "$SOURCE_DATA" "$DATA_ROOT/train.jsonl"
  export DATA_ROOT
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
data_path = data_root / "train.jsonl"
teacher_path = data_root / "teacher_from_target.jsonl"

rows = []
with data_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        target = row.get("target_output") or {}
        rows.append(
            {
                "sample_id": row.get("sample_id"),
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "answer": target.get("question_answer") or target.get("final_answer") or target.get("reasoning_summary") or "",
                "final_answer": target.get("final_answer") or target.get("question_answer") or target.get("reasoning_summary") or "",
                "target_output": target,
            }
        )

teacher_path.parent.mkdir(parents=True, exist_ok=True)
with teacher_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print(json.dumps({"teacher_path": str(teacher_path), "num_rows": len(rows)}, ensure_ascii=False))
PY
}

prepare_visuals() {
  if [ -f "$VIS_ROOT/score_images/manifest.json" ]; then
    log "score image manifest already exists at $VIS_ROOT/score_images/manifest.json"
    return
  fi
  log "generating fixed60_qa600 score visuals"
  "$PY" -B "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" \
    --data "$DATA_ROOT/train.jsonl" \
    --config "$CONFIG_PATH" \
    --checkpoint "$INTERVAL_CKPT" \
    --output-root "$VIS_ROOT" \
    --limit "$LIMIT" \
    --aggregation max \
    --modes score \
    > "$RUN_ROOT/logs/visual_prep.log" 2>&1
}

run_g0() {
  log "starting G0 global hint warmup"
  "$PY" -B "$ROOT/scripts/mvaxis_train_globalhint_warmup.py" \
    --data "$DATA_ROOT/train.jsonl" \
    --config "$CONFIG_PATH" \
    --llm-config "$LLM_CONFIG" \
    --interval-proposer-checkpoint "$INTERVAL_CKPT" \
    --limit "$LIMIT" \
    --epochs 1 \
    --learning-rate 1e-4 \
    --weight-decay 0.0 \
    --gradient-accumulation-steps 4 \
    --max-grad-norm 1.0 \
    --global-anom-loss-weight 1.0 \
    --perm-loss-weight 0.05 \
    --pair-loss-weight 0.05 \
    --query-mode hybrid \
    --query-mix-alpha 0.5 \
    --max-question-tokens 128 \
    --output-checkpoint "$RUN_ROOT/checkpoints/global_hint_warmup.pt" \
    --report "$RUN_ROOT/reports/global_hint_warmup_report.json" \
    --run-note "fixed60 qa600 G0 global hint warmup, hybrid query, ct12 scale0.1, base_channel=typeheads_v1" \
    > "$RUN_ROOT/logs/g0_global_hint_warmup.log" 2>&1
}

merge_g0_with_base_bridge() {
  log "merging G0 global weights into base bridge checkpoint"
  export BASE_BRIDGE_CKPT RUN_ROOT
  "$PY" - <<'PY'
import os
from pathlib import Path
import torch

base_path = Path(os.environ["BASE_BRIDGE_CKPT"])
run_root = Path(os.environ["RUN_ROOT"])
g0_path = run_root / "checkpoints" / "global_hint_warmup.pt"
out_path = run_root / "checkpoints" / "bridge_init_g0_merged.pt"

base = torch.load(base_path, map_location="cpu")
g0 = torch.load(g0_path, map_location="cpu")
base["global_hint_head"] = g0.get("global_hint_head")
base["global_anomaly_head"] = g0.get("global_anomaly_head")
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(base, out_path)
print({"merged_checkpoint": str(out_path)})
PY
}

run_g1() {
  log "starting G1 next-token + answer_aux + global_aux training"
  "$PY" -B "$ROOT/scripts/mvaxis_train_nexttoken_hints.py" \
    --data "$DATA_ROOT/train.jsonl" \
    --teacher "$DATA_ROOT/teacher_from_target.jsonl" \
    --teacher-overrides "" \
    --config "$CONFIG_PATH" \
    --llm-config "$LLM_CONFIG" \
    --interval-proposer-checkpoint "$INTERVAL_CKPT" \
    --init-bridge-checkpoint "$RUN_ROOT/checkpoints/bridge_init_g0_merged.pt" \
    --interval-source target \
    --score-image-manifest "$VIS_ROOT/score_images/manifest.json" \
    --image-kind score \
    --limit "$LIMIT" \
    --epochs 1 \
    --learning-rate 5e-5 \
    --weight-decay 0.0 \
    --gradient-accumulation-steps 4 \
    --max-grad-norm 1.0 \
    --answer-loss-weight 0.2 \
    --global-anom-loss-weight 0.2 \
    --perm-loss-weight 0.05 \
    --pair-loss-weight 0.05 \
    --window-size 32 \
    --stride 8 \
    --max-window-rows 32 \
    --max-target-tokens 256 \
    --output-checkpoint "$RUN_ROOT/checkpoints/bridge_g1.pt" \
    --output-encoder-checkpoint "$RUN_ROOT/checkpoints/encoder_g1.pt" \
    --report "$RUN_ROOT/reports/g1_nexttoken_report.json" \
    --run-note "fixed60 qa600 G1 next-token with score-image all-hints, hybrid global query, ct12 scale0.1, base_channel=typeheads_v1" \
    > "$RUN_ROOT/logs/g1_nexttoken.log" 2>&1
}

main() {
  log "launcher started"
  wait_for_previous_eval
  log "blocker cleared; starting pipeline on cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  prepare_training_data
  prepare_visuals
  run_g0
  merge_g0_with_base_bridge
  run_g1
  log "pipeline finished successfully"
}

main "$@"
