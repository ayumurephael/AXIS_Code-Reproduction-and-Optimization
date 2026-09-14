#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
RUNTIME="${RUNTIME:-$WORKROOT/multiaxis_runtime_train}"
RUN_ROOT="${RUN_ROOT:-$WORKROOT/runs/swat_160_ts61g1_20k4gpu_scoreimage_scale_sweep_20260621c}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-$WORKROOT/runs/swat_160_ts61g1_20k4gpu_scoreimage_eval_20260621a}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$WORKROOT/runs/ts61_vl_globalhints_score_nexttoken_train4gpu_eval4k_20260621a/checkpoints}"
DATA_ROOT="${DATA_ROOT:-$WORKROOT/data/swat_axis_160}"
MODEL_PATH="${MODEL_PATH:-$WORKROOT/models/Qwen2.5-VL-7B-Instruct}"
SLURM_JOB_ID_HINT="${SLURM_JOB_ID_HINT:-355622}"
CONDA_ROOT="${CONDA_ROOT:-/home/fit/zhangchen/WORK/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-Allava}"

DATASET_KEY="${DATASET_KEY:-regular_160_0605b}"
SCALE="${SCALE:-0.2}"
GPU_ID="${GPU_ID:-0}"
LIMIT="${LIMIT:-160}"
MAX_CHANNEL_TOKENS="${MAX_CHANNEL_TOKENS:-12}"
MAX_TOKENS="${MAX_TOKENS:-256}"

case "$DATASET_KEY" in
  regular_160_0605b)
    QUESTION_DIR="$DATA_ROOT/question_swat_axis_regular_160_0605b"
    SCORE_MANIFEST="$BASE_RUN_ROOT/_manifests/question_swat_axis_regular_160_0605b_score_manifest.json"
    ;;
  hard_160_0605a)
    QUESTION_DIR="$DATA_ROOT/question_swat_axis_hard_160_0605a"
    SCORE_MANIFEST="$BASE_RUN_ROOT/_manifests/question_swat_axis_hard_160_0605a_score_manifest.json"
    ;;
  *)
    echo "unsupported DATASET_KEY=$DATASET_KEY" >&2
    exit 1
    ;;
esac

DATA_JSONL="$QUESTION_DIR/questions_160.jsonl"
TEACHER_JSONL="$QUESTION_DIR/teacher_gpt55.jsonl"
AXIS_CONFIG="$RUNTIME/configs/torch_fixed60_qa600_timercd.json"
INTERVAL_CKPT="$CHECKPOINT_ROOT/encoder_g1.pt"
BRIDGE_CKPT="$CHECKPOINT_ROOT/bridge_g1.pt"
RUN_DIR="$RUN_ROOT/$DATASET_KEY/scale${SCALE/./}"
LLM_DIR="$RUN_DIR/_llm_configs"
LLM_CONFIG="$LLM_DIR/local_qwen25_vl7b_ts61g1_swat160_scoreimage_hybrid_ct12_scale${SCALE/./}.json"
EMPTY_MANIFEST="$RUN_DIR/_empty_manifest.json"
LOG_DIR="$RUN_ROOT/_launch_logs"

mkdir -p "$RUN_DIR" "$LLM_DIR" "$LOG_DIR"

for required in "$DATA_JSONL" "$TEACHER_JSONL" "$SCORE_MANIFEST" "$AXIS_CONFIG" "$INTERVAL_CKPT" "$BRIDGE_CKPT"; do
  if [ ! -f "$required" ]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

cat >"$EMPTY_MANIFEST" <<'JSON'
{
  "records": []
}
JSON

cat >"$LLM_CONFIG" <<JSON
{
  "provider": "local_hf_vl",
  "model": "$MODEL_PATH",
  "device": "cuda",
  "device_map": "auto",
  "max_memory": {
    "0": "76000MiB",
    "cpu": "120GiB"
  },
  "offload_folder": "$WORKROOT/tmp/offload/qwen25_vl7b_ts61g1_swat160_scoreimage_scale${SCALE/./}_${DATASET_KEY}",
  "low_cpu_mem_usage": true,
  "torch_dtype": "float16",
  "trust_remote_code": true,
  "temperature": 0.0,
  "do_sample": false,
  "use_cache": true,
  "max_tokens": $MAX_TOKENS,
  "min_pixels": 401408,
  "max_pixels": 802816,
  "axis_hints": {
    "enabled": true,
    "d_proj": 256,
    "num_prototype": 256,
    "num_fixed_tokens": 8,
    "num_global_tokens": 8,
    "max_global_tokens": 8,
    "max_local_tokens": 12,
    "max_channel_tokens": $MAX_CHANNEL_TOKENS,
    "num_heads": 4,
    "injection_scale": $SCALE,
    "global_injection_scale": $SCALE,
    "channel_injection_scale": $SCALE,
    "global_query_mode": "hybrid",
    "global_query_mix_alpha": 0.5,
    "global_sampling_strategy": "uniform",
    "max_question_tokens": 128,
    "checkpoint_path": "$BRIDGE_CKPT"
  }
}
JSON

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$WORKROOT/models/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "[swat160-ts61g1] host=$(hostname) slurm_job=${SLURM_JOB_ID:-$SLURM_JOB_ID_HINT} dataset=$DATASET_KEY scale=$SCALE gpu=$GPU_ID"

python -B "$RUNTIME/scripts/mvaxis_run_denoised_student_answer_eval.py" \
  --data "$DATA_JSONL" \
  --teacher "$TEACHER_JSONL" \
  --llm-config "$LLM_CONFIG" \
  --config "$AXIS_CONFIG" \
  --interval-proposer-checkpoint "$INTERVAL_CKPT" \
  --output-root "$RUN_DIR" \
  --score-manifest "$SCORE_MANIFEST" \
  --raw-manifest "$EMPTY_MANIFEST" \
  --groups score_image_note_all_hints \
  --limit "$LIMIT" \
  --max-tokens "$MAX_TOKENS" \
  --prompt-examples 4 \
  --raw-examples 8 \
  --max-window-rows 0 \
  --digits 3 \
  --progress-percent-step 5 \
  --max-channel-tokens "$MAX_CHANNEL_TOKENS" \
  --channel-injection-scale "$SCALE"

echo "[swat160-ts61g1] done dataset=$DATASET_KEY scale=$SCALE run_dir=$RUN_DIR"
