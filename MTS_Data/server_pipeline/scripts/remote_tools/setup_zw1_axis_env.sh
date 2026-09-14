#!/usr/bin/env bash
set -euo pipefail

AXIS_HOME="${AXIS_HOME:-/root/shared-nvme/axis_project}"
ENV_DIR="$AXIS_HOME/envs/axis"
LOG_DIR="$AXIS_HOME/logs"
TMP_DIR="$AXIS_HOME/tmp"
PIP_CACHE_DIR="$AXIS_HOME/pip_cache"
CONDA_PKGS_DIRS="$AXIS_HOME/conda_pkgs"

mkdir -p "$AXIS_HOME"/{repos,models,hf_cache,logs,tmp,pip_cache,conda_pkgs,envs}

if [ -d /root/hfenv ] && [ ! -e "$AXIS_HOME/envs/hfenv_legacy" ]; then
  mv /root/hfenv "$AXIS_HOME/envs/hfenv_legacy"
fi

if [ ! -d "$ENV_DIR/conda-meta" ]; then
  rm -rf "$ENV_DIR"
  CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" /base/mambaforge/bin/conda create -y \
    -p "$ENV_DIR" \
    -c pytorch \
    -c nvidia \
    python=3.10 \
    pip \
    pytorch \
    pytorch-cuda=12.4
fi

ln -sfn "$ENV_DIR" /root/axis
ln -sfn "$ENV_DIR" /root/hfenv

export TMPDIR="$TMP_DIR"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS"
export HF_HOME="$AXIS_HOME/hf_cache"
export HF_HUB_CACHE="$AXIS_HOME/hf_cache/hub"
export TRANSFORMERS_CACHE="$AXIS_HOME/hf_cache/transformers"

"$ENV_DIR/bin/python" -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120 \
  --retries 10 \
  --upgrade pip setuptools wheel

"$ENV_DIR/bin/python" -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120 \
  --retries 10 \
  accelerate \
  datasets \
  huggingface_hub \
  matplotlib \
  numpy \
  openpyxl \
  pandas \
  protobuf \
  psutil \
  requests \
  safetensors \
  scikit-learn \
  scipy \
  sentencepiece \
  tokenizers \
  tqdm \
  transformers

"$ENV_DIR/bin/python" - <<'PY'
import importlib.util
import torch

print("python_env_ok", flush=True)
print("torch", torch.__version__, flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
print("cuda_device_count", torch.cuda.device_count(), flush=True)
for name in [
    "transformers",
    "accelerate",
    "huggingface_hub",
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
]:
    print(name, bool(importlib.util.find_spec(name)), flush=True)
PY

echo "AXIS_HOME=$AXIS_HOME"
echo "ENV_DIR=$ENV_DIR"
echo "LOG_DIR=$LOG_DIR"
