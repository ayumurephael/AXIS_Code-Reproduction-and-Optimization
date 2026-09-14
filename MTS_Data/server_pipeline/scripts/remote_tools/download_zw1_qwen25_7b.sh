#!/usr/bin/env bash
set -euo pipefail

AXIS_HOME="${AXIS_HOME:-/root/shared-nvme/axis_project}"
ENV_DIR="$AXIS_HOME/envs/axis"
MODEL_DIR="$AXIS_HOME/models/Qwen2.5-7B-Instruct"
LOG_DIR="$AXIS_HOME/logs"

mkdir -p "$AXIS_HOME"/{models,hf_cache,logs,tmp}

export TMPDIR="$AXIS_HOME/tmp"
export HF_HOME="$AXIS_HOME/hf_cache"
export HF_HUB_CACHE="$AXIS_HOME/hf_cache/hub"
export TRANSFORMERS_CACHE="$AXIS_HOME/hf_cache/transformers"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

"$ENV_DIR/bin/hf" download Qwen/Qwen2.5-7B-Instruct \
  --local-dir "$MODEL_DIR" \
  --max-workers "${HF_MAX_WORKERS:-1}"

"$ENV_DIR/bin/python" - <<'PY'
from pathlib import Path

model_dir = Path("/root/shared-nvme/axis_project/models/Qwen2.5-7B-Instruct")
files = list(model_dir.rglob("*"))
print("model_dir", model_dir)
print("file_count", len([p for p in files if p.is_file()]))
print("has_config", (model_dir / "config.json").exists())
print("has_tokenizer", (model_dir / "tokenizer.json").exists())
print("size_gb", round(sum(p.stat().st_size for p in files if p.is_file()) / 1024**3, 3))
PY
