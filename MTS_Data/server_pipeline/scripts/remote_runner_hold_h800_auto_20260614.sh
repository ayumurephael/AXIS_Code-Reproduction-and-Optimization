#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
RUNTIME="${RUNTIME:-$WORKROOT/multiaxis_runtime_train}"
LOGROOT="${LOGROOT:-$WORKROOT/logs}"
STATE_ROOT="${STATE_ROOT:-$WORKROOT/state/runner_hold_h800_auto_20260614a}"
HOLD_JOB_ID="${HOLD_JOB_ID:-355622}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
CHATTIME_MODEL_DIR="${CHATTIME_MODEL_DIR:-$WORKROOT/checkpoints/chattime_1_7b_chat}"

mkdir -p "$LOGROOT" "$STATE_ROOT"

log() {
  echo "[runner] $(date --iso-8601=seconds) $*"
}

stage_done() {
  [ -f "$STATE_ROOT/$1.done" ]
}

mark_done() {
  date --iso-8601=seconds > "$STATE_ROOT/$1.done"
}

hold_running() {
  local state
  state="$(sacct -X -j "$HOLD_JOB_ID" --format=State -n 2>/dev/null | head -n 1 | xargs || true)"
  [ "$state" = "RUNNING" ]
}

chatts_ready() {
  [ -f "$WORKROOT/checkpoints/chatts_legacy14b/config.json" ]
}

chattime_ready() {
  local -a required=(
    ".gitattributes:1519"
    "README.md:4928"
    "added_tokens.json:255025"
    "architecture.png:309599"
    "config.json:752"
    "generation_config.json:183"
    "model-00001-of-00003.safetensors:4930735896"
    "model-00002-of-00003.safetensors:4947390888"
    "model-00003-of-00003.safetensors:3762594664"
    "model.safetensors.index.json:23950"
    "special_tokens_map.json:437"
    "tokenizer.model:499723"
    "tokenizer_config.json:1786164"
  )
  local item path expected actual
  for item in "${required[@]}"; do
    path="${item%%:*}"
    expected="${item##*:}"
    if [ ! -f "$CHATTIME_MODEL_DIR/$path" ]; then
      log "chattime_wait missing=$CHATTIME_MODEL_DIR/$path"
      return 1
    fi
    actual="$(stat -c%s "$CHATTIME_MODEL_DIR/$path" 2>/dev/null || echo 0)"
    if [ "$actual" != "$expected" ]; then
      log "chattime_wait size_mismatch file=$CHATTIME_MODEL_DIR/$path actual=$actual expected=$expected"
      return 1
    fi
  done
  return 0
}

run_step() {
  local stage="$1"
  local logfile="$2"
  shift 2

  if stage_done "$stage"; then
    return 0
  fi

  log "stage=$stage start"
  if "$@" >"$logfile" 2>&1; then
    mark_done "$stage"
    log "stage=$stage done log=$logfile"
    return 0
  fi

  log "stage=$stage failed log=$logfile"
  tail -n 40 "$logfile" 2>/dev/null || true
  return 1
}

run_on_hold() {
  local stage="$1"
  local logfile="$2"
  shift 2
  run_step "$stage" "$logfile" \
    srun --jobid="$HOLD_JOB_ID" -N1 -n1 bash -lc "$*"
}

loop_once() {
  if ! hold_running; then
    log "hold job $HOLD_JOB_ID is not RUNNING"
    return 1
  fi

  if ! chatts_ready; then
    log "waiting for ChatTS checkpoint"
    return 0
  fi

  if ! chattime_ready; then
    log "waiting for ChatTime model dir=$CHATTIME_MODEL_DIR"
    return 0
  fi

  run_on_hold \
    eval_swat_chatts_chattime \
    "$LOGROOT/auto_eval_swat_160_20260614b.log" \
    "export CHATTIME_MODEL_PATH='$CHATTIME_MODEL_DIR'; bash '$RUNTIME/scripts/remote_eval_chatts_chattime_swat_160_20260610.sh'" || return 0

  run_on_hold \
    prep_merge_20k \
    "$LOGROOT/auto_concat_20k_20260614b.log" \
    "bash '$RUNTIME/scripts/remote_concat_question4000_to_ts61_train20k_20260610.sh'" || return 0

  run_on_hold \
    prep_visuals_20k \
    "$LOGROOT/auto_prepare_visuals_20k_20260614b.log" \
    "bash '$RUNTIME/scripts/remote_prepare_ts61_20k_score_visuals_20260610.sh'" || return 0

  run_on_hold \
    train_smoke_20k \
    "$LOGROOT/auto_train_smoke_20k_20260614b.log" \
    "bash '$RUNTIME/scripts/remote_start_ts61_vl_allhints_nexttoken_smoke_8gpu_20k_20260610.sh'" || return 0

  run_on_hold \
    train_full_20k \
    "$LOGROOT/auto_train_full_20k_20260614b.log" \
    "bash '$RUNTIME/scripts/remote_start_ts61_vl_allhints_nexttoken_full_8gpu_20k_20260610.sh'" || return 0

  log "pipeline complete"
  return 0
}

trap 'log "runner exit"' EXIT

log "runner start hold_job_id=$HOLD_JOB_ID runtime=$RUNTIME"
log "state_root=$STATE_ROOT"

while true; do
  loop_once || true
  sleep "$CHECK_INTERVAL_SECONDS"
done
