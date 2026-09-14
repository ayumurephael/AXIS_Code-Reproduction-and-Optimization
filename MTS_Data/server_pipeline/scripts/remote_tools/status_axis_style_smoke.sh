#!/usr/bin/env bash
set -euo pipefail

BASE=${AXIS_PROJECT_BASE:-/root/shared-nvme/axis_project}
LOG="$(ls -t "$BASE"/runs/logs/axis_style_smoke_*.log 2>/dev/null | head -1 || true)"
PID="$BASE/runs/pids/axis_style_smoke.pid"
if [[ -f "$LOG" ]]; then
  tail -n 80 "$LOG"
else
  echo "no axis-style smoke log yet"
fi
if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "__AXIS_STYLE_STATUS__ RUNNING pid=$(cat "$PID")"
else
  echo "__AXIS_STYLE_STATUS__ DONE"
fi
