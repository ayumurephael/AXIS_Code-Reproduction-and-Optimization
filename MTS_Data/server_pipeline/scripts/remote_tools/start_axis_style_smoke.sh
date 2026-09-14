#!/usr/bin/env bash
set -euo pipefail

BASE=${AXIS_PROJECT_BASE:-/root/shared-nvme/axis_project}
SCRIPT_DIR="$BASE/scripts"
LOG_DIR="$BASE/runs/logs"
PID_DIR="$BASE/runs/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

LOG="$LOG_DIR/axis_style_smoke_$(date +%Y%m%d_%H%M%S).log"
PID="$PID_DIR/axis_style_smoke.pid"
rm -f "$PID"
nohup bash "$SCRIPT_DIR/run_axis_style_smoke.sh" > "$LOG" 2>&1 &
echo "$!" > "$PID"
echo "started axis-style smoke pid=$(cat "$PID") log=$LOG"
