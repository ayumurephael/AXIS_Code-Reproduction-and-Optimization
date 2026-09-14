#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
ROOT="${ROOT:-$WORKROOT/multiaxis_runtime_train}"
TRAIN_ROOT="${TRAIN_ROOT:-$WORKROOT/data/fixed60_qa600_globalhint_train_20260620b}"
VIS_ROOT="${VIS_ROOT:-$WORKROOT/data/fixed60_qa600_globalhint_visuals_20260620b}"
MODEL_ROOT="${MODEL_ROOT:-$WORKROOT/runs/fixed60_qa600_globalhint_g0g1_20260620b}"
RUN_ROOT="${RUN_ROOT:-$WORKROOT/runs/fixed60_qa600_globalhint_eval_scoreimage_20260621a}"

DATA_JSONL="${DATA_JSONL:-$TRAIN_ROOT/train.jsonl}"
TEACHER_JSONL="${TEACHER_JSONL:-$TRAIN_ROOT/teacher_from_target.jsonl}"
SCORE_MANIFEST="${SCORE_MANIFEST:-$VIS_ROOT/score_images/manifest.json}"
EMPTY_MANIFEST="${EMPTY_MANIFEST:-$TRAIN_ROOT/empty_manifest.json}"
LLM_CONFIG="${LLM_CONFIG:-$ROOT/configs/local_qwen25_vl7b_fixed60_qa600_globalhint_eval_scoreimage_ct12_scale010_1gpu_20260621a.json}"
AXIS_CONFIG="${AXIS_CONFIG:-$ROOT/configs/torch_fixed60_qa600_timercd.json}"
INTERVAL_CKPT="${INTERVAL_CKPT:-$MODEL_ROOT/checkpoints/encoder_g1.pt}"

CONDA_ROOT="${CONDA_ROOT:-/home/fit/zhangchen/WORK/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-Allava}"
LIMIT="${LIMIT:-600}"

mkdir -p "$RUN_ROOT" "$TRAIN_ROOT"

if [ ! -f "$DATA_JSONL" ]; then
  echo "missing data jsonl: $DATA_JSONL" >&2
  exit 1
fi
if [ ! -f "$TEACHER_JSONL" ]; then
  echo "missing teacher jsonl: $TEACHER_JSONL" >&2
  exit 1
fi
if [ ! -f "$SCORE_MANIFEST" ]; then
  echo "missing score manifest: $SCORE_MANIFEST" >&2
  exit 1
fi
if [ ! -f "$INTERVAL_CKPT" ]; then
  echo "missing interval checkpoint: $INTERVAL_CKPT" >&2
  exit 1
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -B "$ROOT/scripts/mvaxis_run_denoised_student_answer_eval.py" \
  --data "$DATA_JSONL" \
  --teacher "$TEACHER_JSONL" \
  --llm-config "$LLM_CONFIG" \
  --config "$AXIS_CONFIG" \
  --interval-proposer-checkpoint "$INTERVAL_CKPT" \
  --output-root "$RUN_ROOT" \
  --score-manifest "$SCORE_MANIFEST" \
  --raw-manifest "$EMPTY_MANIFEST" \
  --groups score_image_note_all_hints \
  --limit "$LIMIT" \
  --max-tokens 256 \
  --prompt-examples 4 \
  --raw-examples 8 \
  --max-window-rows 0 \
  --digits 3 \
  --progress-percent-step 5 \
  --max-channel-tokens 12 \
  --channel-injection-scale 0.1
