#!/usr/bin/env bash
set -euo pipefail

BASE=/root/shared-nvme/axis_project
RUN_NAME=${RUN_NAME:-}
if [[ -n "$RUN_NAME" ]]; then
  LOG="$(ls -t "$BASE"/runs/logs/"${RUN_NAME}"_*.log 2>/dev/null | head -1 || true)"
  PID="$BASE/runs/pids/${RUN_NAME}.pid"
else
  LOG="$(ls -t "$BASE"/runs/logs/*.log 2>/dev/null | head -1 || true)"
  PID="$BASE/runs/pids/axis_bridge.pid"
fi
LEGACY_PID="$BASE/runs/aux_semantic.pid"
if [[ -n "$LOG" && -f "$LOG" ]]; then
  echo "log=$LOG"
  tail -n 100 "$LOG"
else
  echo "no aux semantic log yet"
fi
if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "__AXIS_BRIDGE_STATUS__ RUNNING pid=$(cat "$PID")"
elif [[ -f "$LEGACY_PID" ]] && kill -0 "$(cat "$LEGACY_PID")" 2>/dev/null; then
  echo "__AXIS_BRIDGE_STATUS__ RUNNING pid=$(cat "$LEGACY_PID")"
else
  echo "__AXIS_BRIDGE_STATUS__ DONE"
fi
