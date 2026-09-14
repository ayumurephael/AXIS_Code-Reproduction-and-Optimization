#!/usr/bin/env bash
set -euo pipefail

AXIS_HOME="${AXIS_HOME:-/root/shared-nvme/axis_project}"
ROOT="${ROOT:-$AXIS_HOME/repos/multiaxis}"
PYTHON_BIN="${PYTHON_BIN:-$AXIS_HOME/envs/axis/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/axis_project/models/Qwen2.5-VL-7B-Instruct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/root/axis_project/checkpoints/ts61_vl_globalhints_score_nexttoken_train4gpu_eval4k_20260621a}"
DATA_ROOT="${DATA_ROOT:-/root/axis_project/data/swat_axis_160_ts61g1_20260621}"
RUN_ROOT="${RUN_ROOT:-/root/axis_project/runs/swat_160_ts61g1_scoreimage_zw1_20260621}"

DATASET_KEY="${DATASET_KEY:-regular_160_0605b}"
SCALE="${SCALE:-0.2}"
GPU_ID="${GPU_ID:-0}"
LIMIT="${LIMIT:-160}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_WINDOW_ROWS="${MAX_WINDOW_ROWS:-0}"
PROMPT_EXAMPLES="${PROMPT_EXAMPLES:-4}"
RAW_EXAMPLES="${RAW_EXAMPLES:-8}"
PROGRESS_STEP="${PROGRESS_STEP:-5}"
MAX_CHANNEL_TOKENS="${MAX_CHANNEL_TOKENS:-12}"

case "$DATASET_KEY" in
  regular_160_0605b)
    DATA_JSONL="$DATA_ROOT/question_swat_axis_regular_160_0605b/questions_160.jsonl"
    TEACHER_JSONL="$DATA_ROOT/question_swat_axis_regular_160_0605b/teacher_gpt55.jsonl"
    ;;
  hard_160_0605a)
    DATA_JSONL="$DATA_ROOT/question_swat_axis_hard_160_0605a/questions_160.jsonl"
    TEACHER_JSONL="$DATA_ROOT/question_swat_axis_hard_160_0605a/teacher_gpt55.jsonl"
    ;;
  *)
    echo "Unsupported DATASET_KEY=$DATASET_KEY" >&2
    exit 1
    ;;
esac

ENCODER_CKPT="$CHECKPOINT_DIR/encoder_g1.pt"
BRIDGE_CKPT="$CHECKPOINT_DIR/bridge_g1.pt"
AXIS_CONFIG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
RUN_DIR="$RUN_ROOT/${DATASET_KEY}/score_image_note_all_hints_hybrid_ct12_scale${SCALE/./}"
SCORE_DIR="$RUN_DIR/_score_images"
EMPTY_MANIFEST="$RUN_DIR/_empty_manifest.json"
LLM_CONFIG_DIR="$RUN_DIR/_llm_configs"
LLM_CONFIG="$LLM_CONFIG_DIR/local_qwen25_vl7b_ts61g1_scoreimage_hybrid_ct12_scale${SCALE/./}.json"

mkdir -p "$RUN_DIR" "$SCORE_DIR" "$LLM_CONFIG_DIR"

for required in "$DATA_JSONL" "$TEACHER_JSONL" "$ENCODER_CKPT" "$BRIDGE_CKPT" "$AXIS_CONFIG" "$PYTHON_BIN"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
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
  "model": "$MODEL_DIR",
  "device": "cuda",
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
export HF_HOME="${HF_HOME:-/root/axis_project/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/axis_project/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/axis_project/hf_cache/transformers}"

echo "[zw1-swat160] host=$(hostname) dataset=$DATASET_KEY scale=$SCALE gpu=$GPU_ID limit=$LIMIT"
echo "[zw1-swat160] generating score-image manifest into $SCORE_DIR"
"$PYTHON_BIN" -B "$ROOT/scripts/mvaxis_generate_anomaly_score_images.py" \
  --data "$DATA_JSONL" \
  --config "$AXIS_CONFIG" \
  --checkpoint "$ENCODER_CKPT" \
  --output-dir "$SCORE_DIR" \
  --limit "$LIMIT" \
  --aggregation max \
  --mode score

echo "[zw1-swat160] running teacher-aligned eval into $RUN_DIR"
"$PYTHON_BIN" -B "$ROOT/scripts/mvaxis_run_denoised_student_answer_eval.py" \
  --data "$DATA_JSONL" \
  --teacher "$TEACHER_JSONL" \
  --llm-config "$LLM_CONFIG" \
  --config "$AXIS_CONFIG" \
  --interval-proposer-checkpoint "$ENCODER_CKPT" \
  --output-root "$RUN_DIR" \
  --score-manifest "$SCORE_DIR/manifest.json" \
  --raw-manifest "$EMPTY_MANIFEST" \
  --groups score_image_note_all_hints \
  --limit "$LIMIT" \
  --max-tokens "$MAX_TOKENS" \
  --prompt-examples "$PROMPT_EXAMPLES" \
  --raw-examples "$RAW_EXAMPLES" \
  --max-window-rows "$MAX_WINDOW_ROWS" \
  --digits 3 \
  --progress-percent-step "$PROGRESS_STEP" \
  --max-channel-tokens "$MAX_CHANNEL_TOKENS" \
  --channel-injection-scale "$SCALE"

echo "[zw1-swat160] done dataset=$DATASET_KEY scale=$SCALE run_dir=$RUN_DIR"
