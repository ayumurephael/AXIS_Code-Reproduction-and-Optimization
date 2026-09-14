#!/usr/bin/env bash
set -euo pipefail

BASE=/root/shared-nvme/axis_project
SCRIPT_DIR="$BASE/scripts"
LOG_DIR="$BASE/runs/logs"
PID_DIR="$BASE/runs/pids"
RUN_NAME=${RUN_NAME:-axis_bridge}
mkdir -p "$LOG_DIR" "$PID_DIR"
LOG="$LOG_DIR/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
PID="$PID_DIR/${RUN_NAME}.pid"
nohup bash "$SCRIPT_DIR/train_aux_semantic_shared.sh" > "$LOG" 2>&1 &
echo "$!" > "$PID"
echo "started $RUN_NAME pid=$(cat "$PID") log=$LOG"
