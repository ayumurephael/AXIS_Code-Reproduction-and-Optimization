#!/usr/bin/env bash
set -euo pipefail

AXIS_HOME="${AXIS_HOME:-/root/shared-nvme/axis_project}"
REPO="$AXIS_HOME/repos/multiaxis"
ENV_DIR="$AXIS_HOME/envs/axis"
RUN_DIR="$REPO/outputs/runs/523trail_50"
LOG_DIR="$RUN_DIR/logs"

mkdir -p "$RUN_DIR" "$LOG_DIR"
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME="$AXIS_HOME/hf_cache"
export HF_HUB_CACHE="$AXIS_HOME/hf_cache/hub"
export TRANSFORMERS_CACHE="$AXIS_HOME/hf_cache/transformers"
export TMPDIR="$AXIS_HOME/tmp"

"$ENV_DIR/bin/python" -B scripts/mvaxis_build_multilevel_qa_dataset.py \
  --config configs/torch_legacy_axis50_original.json \
  --source-split test \
  --output-dir outputs/runs/523trail_50/data \
  --output-config outputs/runs/523trail_50/config.json \
  --num-samples 50 \
  --seed 523 \
  > "$LOG_DIR/523trail_50_build_qa.log" 2>&1

"$ENV_DIR/bin/python" -B scripts/mvaxis_run_truth_interval_with_model_hints.py \
  --config outputs/runs/523trail_50/config.json \
  --llm-config configs/523trail_qwen25_7b_axis_embedding.json \
  --split test \
  --limit 50 \
  --interval-proposer-checkpoint external_checkpoints/timercd/best_model/pretrain_checkpoint_best_multi.pth \
  --hint-delivery embedding \
  --output outputs/runs/523trail_50/523trail_50_truth_embedding.jsonl \
  --report outputs/runs/523trail_50/523trail_50_truth_embedding_report.json \
  --raw-log-examples 10 \
  --raw-log-path outputs/runs/523trail_50/523trail_50_raw_examples.json \
  --dump-prompt-examples 3 \
  --prompt-log-path outputs/runs/523trail_50/523trail_50_prompt_examples.json \
  > "$LOG_DIR/523trail_50_llm_truth_embedding.log" 2>&1

"$ENV_DIR/bin/python" -B scripts/export_qa_comparison_txt.py \
  --config outputs/runs/523trail_50/config.json \
  --split test \
  --records outputs/runs/523trail_50/523trail_50_truth_embedding.jsonl \
  --output outputs/runs/523trail_50/523trail_50_case_study_10.txt \
  --limit 10 \
  > "$LOG_DIR/523trail_50_export_case_study.log" 2>&1

cat outputs/runs/523trail_50/523trail_50_truth_embedding_report.json
