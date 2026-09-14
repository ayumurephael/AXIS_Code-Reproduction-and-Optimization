#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
ROOT="${ROOT:-$WORKROOT/multiaxis_runtime_train}"
RUN_ROOT="${RUN_ROOT:-$WORKROOT/runs/ts61_vl_allhints_nexttoken_train_20k_20260610a}"
TRAIN_ROOT="${TRAIN_ROOT:-$WORKROOT/data/ts61_train_20k_20260610a}"
VIS_ROOT="${VIS_ROOT:-$WORKROOT/data/ts61_train_20k_visuals_20260610a}"
CONDA_SH="${CONDA_SH:-/home/fit/zhangchen/WORK/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-Allava}"
PY="${PY:-python}"

if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  PY=python
fi

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/reports" "$RUN_ROOT/logs"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    if [ "${GPU_COUNT:-0}" -gt 0 ]; then
      CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
      export CUDA_VISIBLE_DEVICES
    fi
  fi
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES is empty and GPU count could not be inferred" >&2
  exit 1
fi
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${#VISIBLE_GPU_IDS[@]}"
if [ "$NPROC_PER_NODE" -le 0 ]; then
  echo "no visible gpus parsed from CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi
echo "smoke launch visible_gpus=$CUDA_VISIBLE_DEVICES nproc_per_node=$NPROC_PER_NODE"

"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$ROOT/scripts/mvaxis_train_nexttoken_hints.py" \
  --data "$TRAIN_ROOT/questions_20000.jsonl" \
  --teacher "$TRAIN_ROOT/teacher_gpt55.jsonl" \
  --teacher-overrides "" \
  --config "$ROOT/configs/torch_fixed60_qa600_timercd.json" \
  --llm-config "$ROOT/configs/local_qwen25_vl7b_ts61_vl_allhints_ct12_scale010_8gpu.json" \
  --interval-proposer-checkpoint "$WORKROOT/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt" \
  --init-bridge-checkpoint "$WORKROOT/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1.pt" \
  --interval-source target \
  --score-image-manifest "$VIS_ROOT/score_images/manifest.json" \
  --limit 32 \
  --epochs 1 \
  --learning-rate 5e-5 \
  --gradient-accumulation-steps 4 \
  --answer-loss-weight 0.0 \
  --max-window-rows 32 \
  --preserve-device-map-in-distributed \
  --output-checkpoint "$RUN_ROOT/checkpoints/bridge_smoke.pt" \
  --output-encoder-checkpoint "$RUN_ROOT/checkpoints/encoder_smoke.pt" \
  --report "$RUN_ROOT/reports/train_report_smoke.json" \
  --run-note "ts61 VL score-image all-hints next-token training smoke 20k corpus, dynamic_gpus=${NPROC_PER_NODE}, ct12, scale0.1" \
  > "$RUN_ROOT/logs/smoke_8gpu_20k.log" 2>&1
