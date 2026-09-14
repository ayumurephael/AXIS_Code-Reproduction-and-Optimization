#!/usr/bin/env bash
set -euo pipefail

AXIS_HOME="${AXIS_HOME:-/root/shared-nvme/axis_project}"
REPO="$AXIS_HOME/repos/multiaxis"
RUN_DIR="$REPO/outputs/runs/523trail_50"

echo "--- processes ---"
ps -eo pid,ppid,stat,etime,cmd | grep -E 'run_523trail_50|mvaxis_run_truth|mvaxis_build_multilevel|export_qa|Qwen|python' | grep -v grep || true
echo "--- gpu ---"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo "--- files ---"
find "$RUN_DIR" -maxdepth 3 -type f 2>/dev/null | sort | sed -n '1,120p' || true
echo "--- logs tail ---"
for f in "$RUN_DIR"/logs/*.log; do
  [ -f "$f" ] || continue
  echo "### $f"
  tail -n 80 "$f"
done
echo "--- report ---"
cat "$RUN_DIR/523trail_50_truth_embedding_report.json" 2>/dev/null || true
