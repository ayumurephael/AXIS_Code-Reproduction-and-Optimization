#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
ROOT="${ROOT:-$WORKROOT/multiaxis_runtime_train}"
HIGHLIGHT_ROOT="${HIGHLIGHT_ROOT:-$WORKROOT/data/ts61_train_20k_visuals_20260618_highlighta}"
STATE_ROOT="${STATE_ROOT:-$WORKROOT/state/noglobalhints_highlight_20260619a}"
LOG_ROOT="${LOG_ROOT:-$WORKROOT/logs}"
PREP_LOG="${PREP_LOG:-$LOG_ROOT/prepare_ts61_20k_highlight_visuals_20260619a.log}"
RUNNER_LOG="${RUNNER_LOG:-$LOG_ROOT/runner_noglobalhints_highlight_20260619a.log}"
SMOKE_SCRIPT="${SMOKE_SCRIPT:-$ROOT/scripts/remote_start_ts61_vl_noglobalhints_highlight_nexttoken_smoke_8gpu_20k_20260618.sh}"
FULL_SCRIPT="${FULL_SCRIPT:-$ROOT/scripts/remote_start_ts61_vl_noglobalhints_highlight_nexttoken_full_8gpu_20k_20260618.sh}"
PREP_SCRIPT="${PREP_SCRIPT:-$ROOT/scripts/remote_prepare_ts61_20k_highlight_visuals_20260618.sh}"

mkdir -p "$STATE_ROOT" "$LOG_ROOT"

timestamp() {
  date '+%F %T'
}

log() {
  echo "$(timestamp) $*" | tee -a "$RUNNER_LOG"
}

MANIFEST="$HIGHLIGHT_ROOT/raw_images/manifest.json"

log "runner start"
log "workroot=$WORKROOT"
log "highlight_root=$HIGHLIGHT_ROOT"

if [ ! -f "$MANIFEST" ]; then
  log "highlight manifest missing; running prepare step"
  OUT_ROOT="$HIGHLIGHT_ROOT" LOGFILE="$PREP_LOG" bash "$PREP_SCRIPT"
  touch "$STATE_ROOT/prep.done"
  log "prepare step complete"
else
  log "highlight manifest already exists; skipping prepare"
fi

if [ ! -f "$MANIFEST" ]; then
  log "prepare failed: manifest still missing at $MANIFEST"
  exit 1
fi

log "starting smoke"
  RUN_ROOT="$WORKROOT/runs/ts61_vl_noglobalhints_highlight_nexttoken_train_20k_20260619a" \
  VIS_ROOT="$HIGHLIGHT_ROOT" \
  bash "$SMOKE_SCRIPT"
touch "$STATE_ROOT/smoke.done"
log "smoke complete"

log "starting full"
  RUN_ROOT="$WORKROOT/runs/ts61_vl_noglobalhints_highlight_nexttoken_train_20k_20260619a" \
  VIS_ROOT="$HIGHLIGHT_ROOT" \
  bash "$FULL_SCRIPT"
touch "$STATE_ROOT/full.done"
log "full complete"
