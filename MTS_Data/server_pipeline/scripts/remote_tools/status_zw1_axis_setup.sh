#!/usr/bin/env bash
set -euo pipefail

AXIS_HOME="${AXIS_HOME:-/root/shared-nvme/axis_project}"
ENV_DIR="$AXIS_HOME/envs/axis"

echo "--- disks ---"
df -hT /root/shared-nvme /base / || true
echo "--- gpu ---"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
echo "--- env ---"
if [ -x "$ENV_DIR/bin/python" ]; then
  "$ENV_DIR/bin/python" - <<'PY'
import importlib.util
print("python ok")
for name in ["torch", "transformers", "accelerate", "huggingface_hub", "numpy", "pandas", "sklearn"]:
    print(name, bool(importlib.util.find_spec(name)))
try:
    import torch
    print("torch_version", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_device_count", torch.cuda.device_count())
except Exception as exc:
    print("torch_error", repr(exc))
PY
else
  echo "missing env: $ENV_DIR"
fi
echo "--- model ---"
du -sh "$AXIS_HOME/models/Qwen2.5-7B-Instruct" 2>/dev/null || true
ls -lah "$AXIS_HOME/models/Qwen2.5-7B-Instruct" 2>/dev/null | sed -n '1,60p' || true
echo "--- logs ---"
ls -lah "$AXIS_HOME/logs" 2>/dev/null | sed -n '1,80p' || true
