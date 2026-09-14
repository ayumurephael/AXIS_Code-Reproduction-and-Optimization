#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
RUNTIME="${RUNTIME:-$WORKROOT/multiaxis_runtime_train}"
LOGROOT="${LOGROOT:-$WORKROOT/logs}"
STATE_ROOT="${STATE_ROOT:-$WORKROOT/state/multi_axis_daemon_20260610a}"
CHATTS_CKPT_DIR="${CHATTS_CKPT_DIR:-$WORKROOT/checkpoints/chatts_legacy14b}"
TRAIN_ROOT="${TRAIN_ROOT:-$WORKROOT/data/ts61_train_20k_20260610a}"
VIS_ROOT="${VIS_ROOT:-$WORKROOT/data/ts61_train_20k_visuals_20260610a}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
ENABLE_TRAIN_SMOKE="${ENABLE_TRAIN_SMOKE:-0}"
HF_HOME="${HF_HOME:-$WORKROOT/models/hf}"

mkdir -p "$LOGROOT" "$STATE_ROOT" "$HF_HOME"

LOGFILE="$LOGROOT/multi_axis_daemon_20260610a.log"
touch "$LOGFILE"
exec >>"$LOGFILE" 2>&1

export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

touch_marker() {
  local name="$1"
  date --iso-8601=seconds > "$STATE_ROOT/$name"
}

log() {
  echo "[daemon] $(date --iso-8601=seconds) $*"
}

fail_stage() {
  local stage="$1"
  touch_marker "${stage}.failed"
  log "stage=$stage failed; marker=$STATE_ROOT/${stage}.failed"
}

stage_done() {
  local stage="$1"
  [ -f "$STATE_ROOT/${stage}.done" ]
}

stage_failed() {
  local stage="$1"
  [ -f "$STATE_ROOT/${stage}.failed" ]
}

chatts_ready() {
  local path expected actual
  local -a required=(
    "config.json:1203"
    "tokenizer.json:11422259"
    "tokenizer_config.json:5330"
    "special_tokens_map.json:354"
    "vocab.json:2776833"
    "merges.txt:1671853"
    "processing_qwen2_ts.py:9083"
    "modeling_qwen2.py:78550"
    "configuration_qwen2.py:18734"
    "generation_config.json:243"
    "added_tokens.json:642"
    "pytorch_model.bin.index.json:48132"
    "pytorch_model-00001-of-00006.bin:4986229446"
    "pytorch_model-00002-of-00006.bin:4954871698"
    "pytorch_model-00003-of-00006.bin:4954871762"
    "pytorch_model-00004-of-00006.bin:4954871762"
    "pytorch_model-00005-of-00006.bin:4954871762"
    "pytorch_model-00006-of-00006.bin:4944481872"
  )
  for item in "${required[@]}"; do
    path="${item%%:*}"
    expected="${item##*:}"
    if [ ! -f "$CHATTS_CKPT_DIR/$path" ]; then
      log "chatts_wait missing=$CHATTS_CKPT_DIR/$path"
      return 1
    fi
    actual="$(stat -c%s "$CHATTS_CKPT_DIR/$path" 2>/dev/null || echo 0)"
    if [ "$actual" != "$expected" ]; then
      log "chatts_wait size_mismatch file=$CHATTS_CKPT_DIR/$path actual=$actual expected=$expected"
      return 1
    fi
  done
  return 0
}

run_stage() {
  local stage="$1"
  shift
  if stage_done "$stage"; then
    return 0
  fi
  if stage_failed "$stage"; then
    log "stage=$stage is marked failed; remove $STATE_ROOT/${stage}.failed to retry"
    return 1
  fi
  log "stage=$stage start"
  if "$@"; then
    touch_marker "${stage}.done"
    log "stage=$stage done"
    return 0
  fi
  fail_stage "$stage"
  return 1
}

run_loop() {
  if ! stage_done "chatts_ready"; then
    if chatts_ready; then
      touch_marker "chatts_ready.done"
      log "chatts checkpoint is complete"
    else
      log "waiting for ChatTS checkpoint upload"
      return 0
    fi
  fi

  run_stage "eval_swat_chatts_chattime" \
    bash "$RUNTIME/scripts/remote_eval_chatts_chattime_swat_160_20260610.sh" || return 0

  run_stage "merge_ts61_train20k" \
    bash "$RUNTIME/scripts/remote_concat_question4000_to_ts61_train20k_20260610.sh" || return 0

  run_stage "prepare_ts61_train20k_visuals" \
    bash "$RUNTIME/scripts/remote_prepare_ts61_20k_score_visuals_20260610.sh" || return 0

  if [ "$ENABLE_TRAIN_SMOKE" = "1" ]; then
    run_stage "train_smoke" \
      bash "$RUNTIME/scripts/remote_start_ts61_vl_allhints_nexttoken_smoke_8gpu_20k_20260610.sh" || return 0
  fi

  run_stage "train_full_20k" \
    bash "$RUNTIME/scripts/remote_start_ts61_vl_allhints_nexttoken_full_8gpu_20k_20260610.sh" || return 0
}

trap 'log "daemon exit on $(hostname)"' EXIT

echo "${SLURM_JOB_ID:-355622}" > "$LOGROOT/current_multi_axis_jobid.txt"
hostname > "$LOGROOT/current_multi_axis_hostname.txt"

log "daemon started host=$(hostname) job_id=${SLURM_JOB_ID:-none} runtime=$RUNTIME"
log "state_root=$STATE_ROOT"

while true; do
  run_loop || true
  if stage_done "train_full_20k"; then
    log "pipeline complete; holding GPUs and waiting for next instruction"
  fi
  sleep "$CHECK_INTERVAL_SECONDS"
done
