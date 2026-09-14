#!/usr/bin/env bash
set -euo pipefail

BASE=/root/shared-nvme/axis_project
cd "$BASE/repos/AXIS_repo/multivariate_axis"

PY=/root/AXIS_repo/AXIS_repo/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$BASE/hf_cache"
export TRANSFORMERS_CACHE="$BASE/hf_cache"
export HF_HUB_CACHE="$BASE/hf_cache/hub"
export XDG_CACHE_HOME="$BASE/cache"
export TMPDIR="$BASE/tmp"

RUN_NAME=${RUN_NAME:-aux_semantic128}
LIMIT=${LIMIT:-128}
EPOCHS=${EPOCHS:-1}
EVAL_LIMIT=${EVAL_LIMIT:-12}
LR=${LR:-6e-5}
AUX_LOSS_WEIGHT=${AUX_LOSS_WEIGHT:-0.2}
AUX_ROOT_WEIGHT=${AUX_ROOT_WEIGHT:-0.4}
AUX_TYPE_WEIGHT=${AUX_TYPE_WEIGHT:-0.4}
AUX_AFFECTED_WEIGHT=${AUX_AFFECTED_WEIGHT:-0.2}
CONFIG=${CONFIG:-configs/torch_semantic_scale.json}
TRAIN_CFG=${TRAIN_CFG:-configs/local_qwen25_7b_cuda_remote_semantic_truth_bridge.json}
RUN_OUTPUT_DIR="outputs/runs/${RUN_NAME}"
mkdir -p "$RUN_OUTPUT_DIR/checkpoints"
EVAL_CFG="$RUN_OUTPUT_DIR/eval_llm_config.json"
CKPT="$RUN_OUTPUT_DIR/checkpoints/qwen25_7b_axis_perceiver_bridge_axis_style_${RUN_NAME}.pt"
TRAIN_REPORT="$RUN_OUTPUT_DIR/train_report.json"
TRUTH_OUTPUT="$RUN_OUTPUT_DIR/truth_limit${EVAL_LIMIT}.jsonl"
TRUTH_REPORT="$RUN_OUTPUT_DIR/truth_limit${EVAL_LIMIT}_report.json"
PROPOSAL_OUTPUT="$RUN_OUTPUT_DIR/proposal_limit${EVAL_LIMIT}.jsonl"
PROPOSAL_REPORT="$RUN_OUTPUT_DIR/proposal_limit${EVAL_LIMIT}_report.json"

echo "== $(date '+%F %T') verify shared workspace =="
pwd
df -h / "$BASE"
$PY -B - <<PY
from pathlib import Path
from src.mvaxis.utils import read_jsonl, load_json
run_cfg = load_json(Path("$CONFIG"))
train_path = Path(run_cfg["data"]["output_dir"]) / "train.jsonl"
print({"config": "$CONFIG", "train_rows": len(read_jsonl(train_path)), "train_path": str(train_path)})
cfg = load_json(Path("$TRAIN_CFG"))
print({"model": cfg["model"], "checkpoint": cfg["axis_hints"].get("checkpoint_path"), "offload": cfg.get("offload_folder")})
PY

echo "== $(date '+%F %T') train AXIS perceiver bridge with semantic aux loss =="
$PY -B scripts/train_axis_embedding_bridge.py \
  --config "$CONFIG" \
  --llm-config "$TRAIN_CFG" \
  --split train \
  --limit "$LIMIT" \
  --epochs "$EPOCHS" \
  --learning-rate "$LR" \
  --gradient-accumulation-steps 4 \
  --interval-source truth \
  --aux-loss-weight "$AUX_LOSS_WEIGHT" \
  --aux-root-weight "$AUX_ROOT_WEIGHT" \
  --aux-type-weight "$AUX_TYPE_WEIGHT" \
  --aux-affected-weight "$AUX_AFFECTED_WEIGHT" \
  --output-checkpoint "$CKPT" \
  --report "$TRAIN_REPORT"

echo "== $(date '+%F %T') create eval config =="
$PY -B - <<PY
import json
from pathlib import Path
src = Path("$TRAIN_CFG")
dst = Path("$EVAL_CFG")
cfg = json.loads(src.read_text())
cfg["axis_hints"]["checkpoint_path"] = "$CKPT"
cfg["axis_hints"]["max_local_tokens"] = 24
cfg["max_tokens"] = 512
dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print({"eval_config": str(dst), "checkpoint": cfg["axis_hints"]["checkpoint_path"]})
PY

echo "== $(date '+%F %T') inference with truth intervals =="
$PY -B scripts/run_llm_on_proposals.py \
  --config "$CONFIG" \
  --llm-config "$EVAL_CFG" \
  --split test \
  --limit "$EVAL_LIMIT" \
  --interval-source truth \
  --window-size 32 \
  --stride 8 \
  --output "$TRUTH_OUTPUT" \
  --report "$TRUTH_REPORT" \
  --raw-log-examples 2 \
  --raw-log-path "$RUN_OUTPUT_DIR/truth_limit${EVAL_LIMIT}_raw_examples.json"

echo "== $(date '+%F %T') inference with anomaly-head proposals =="
$PY -B scripts/run_llm_on_proposals.py \
  --config "$CONFIG" \
  --llm-config "$EVAL_CFG" \
  --split test \
  --limit "$EVAL_LIMIT" \
  --interval-source proposal \
  --window-size 32 \
  --stride 8 \
  --output "$PROPOSAL_OUTPUT" \
  --report "$PROPOSAL_REPORT" \
  --raw-log-examples 2 \
  --raw-log-path "$RUN_OUTPUT_DIR/proposal_limit${EVAL_LIMIT}_raw_examples.json"

echo "== $(date '+%F %T') write run manifest =="
$PY -B - <<PY
import json
from pathlib import Path
run_dir = Path("$RUN_OUTPUT_DIR")
manifest = {
    "run_name": "$RUN_NAME",
    "config": "$CONFIG",
    "train_llm_config": "$TRAIN_CFG",
    "eval_llm_config": "$EVAL_CFG",
    "checkpoint": "$CKPT",
    "train_report": "$TRAIN_REPORT",
    "truth_output": "$TRUTH_OUTPUT",
    "truth_report": "$TRUTH_REPORT",
    "truth_raw_examples": "$RUN_OUTPUT_DIR/truth_limit${EVAL_LIMIT}_raw_examples.json",
    "proposal_output": "$PROPOSAL_OUTPUT",
    "proposal_report": "$PROPOSAL_REPORT",
    "proposal_raw_examples": "$RUN_OUTPUT_DIR/proposal_limit${EVAL_LIMIT}_raw_examples.json",
    "limit": int("$LIMIT"),
    "epochs": int("$EPOCHS"),
    "eval_limit": int("$EVAL_LIMIT"),
    "learning_rate": float("$LR"),
    "aux_loss_weight": float("$AUX_LOSS_WEIGHT"),
}
for key in ["train_report", "truth_report", "proposal_report"]:
    path = Path(manifest[key])
    if path.exists():
        manifest[key + "_metrics"] = json.loads(path.read_text())
(run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(json.dumps({"manifest": str(run_dir / "manifest.json")}, ensure_ascii=False, indent=2))
PY

echo "== $(date '+%F %T') reports =="
cat "$TRAIN_REPORT"
echo
cat "$TRUTH_REPORT"
echo
cat "$PROPOSAL_REPORT"
echo
echo "== $(date '+%F %T') done =="
