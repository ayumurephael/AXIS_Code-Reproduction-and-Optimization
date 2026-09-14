#!/usr/bin/env bash
set -euo pipefail

BASE=${AXIS_PROJECT_BASE:-/root/shared-nvme/axis_project}
REPO="$BASE/repos/AXIS_repo/multivariate_axis"
PY=${AXIS_PYTHON:-/root/AXIS_repo/AXIS_repo/bin/python}

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$BASE/hf_cache"
export TRANSFORMERS_CACHE="$BASE/hf_cache"
export HF_HUB_CACHE="$BASE/hf_cache/hub"
export XDG_CACHE_HOME="$BASE/cache"
export TMPDIR="$BASE/tmp"

echo "== $(date '+%F %T') axis-style smoke: environment =="
$PY -B - <<'PY'
import torch
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()})
PY
df -h "$BASE" || true

echo "== $(date '+%F %T') generate AXIS-style multivariate synthetic data =="
$PY -B scripts/generate_synthetic.py --config configs/torch_smoke.json

echo "== $(date '+%F %T') show one prompt preview =="
$PY -B - <<'PY'
from pathlib import Path
from src.mvaxis.utils import read_jsonl
from src.mvaxis.data_schema import convert_to_model_input
from src.mvaxis.prompts import build_prompt
rows = read_jsonl(Path("data/synthetic_torch_smoke/train.jsonl"))
sample = rows[1]
prompt = build_prompt(
    convert_to_model_input(sample),
    {},
    interval_context={"interval_source": "truth", "hint_delivery": "AXIS embeddings"},
    include_soft_hints=False,
)
print({"sample_id": sample["sample_id"], "has_normal_series": "normal_series" in sample, "windows": len(sample.get("windows", []))})
print(prompt[:2400])
PY

echo "== $(date '+%F %T') train AXIS interval proposer =="
$PY -B scripts/train_axis_interval_proposal.py --config configs/torch_smoke.json

echo "== $(date '+%F %T') train frozen-LLM AXIS perceiver bridge =="
$PY -B scripts/train_axis_embedding_bridge.py \
  --config configs/torch_smoke.json \
  --llm-config configs/local_qwen25_7b_cuda_remote_axis_style_smoke_bridge.json \
  --split train \
  --limit 48 \
  --epochs 1 \
  --learning-rate 8e-5 \
  --gradient-accumulation-steps 4 \
  --interval-source truth \
  --output-checkpoint outputs/checkpoints/qwen25_7b_axis_perceiver_bridge_axis_style_smoke.pt \
  --report outputs/qwen25_7b_axis_style_smoke_bridge_train_report.json

echo "== $(date '+%F %T') inference with oracle truth intervals =="
$PY -B scripts/run_llm_on_proposals.py \
  --config configs/torch_smoke.json \
  --llm-config configs/local_qwen25_7b_cuda_remote_axis_style_smoke_bridge.json \
  --split test \
  --limit 8 \
  --interval-source truth \
  --window-size 32 \
  --stride 8 \
  --output outputs/qwen25_7b_axis_style_truth_limit8.jsonl \
  --report outputs/qwen25_7b_axis_style_truth_limit8_report.json

echo "== $(date '+%F %T') inference with anomaly-head proposals =="
$PY -B scripts/run_llm_on_proposals.py \
  --config configs/torch_smoke.json \
  --llm-config configs/local_qwen25_7b_cuda_remote_axis_style_smoke_bridge.json \
  --split test \
  --limit 8 \
  --interval-source proposal \
  --window-size 32 \
  --stride 8 \
  --output outputs/qwen25_7b_axis_style_proposal_limit8.jsonl \
  --report outputs/qwen25_7b_axis_style_proposal_limit8_report.json

echo "== $(date '+%F %T') reports =="
cat outputs/qwen25_7b_axis_style_smoke_bridge_train_report.json
echo
cat outputs/qwen25_7b_axis_style_truth_limit8_report.json
echo
cat outputs/qwen25_7b_axis_style_proposal_limit8_report.json
echo
echo "== $(date '+%F %T') done =="
