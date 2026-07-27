#!/usr/bin/env bash
set -euo pipefail

inference_dir=$1
judge_dir=$2
repo_dir=$3

cd "$repo_dir"
while test ! -s "$inference_dir/predictions.jsonl"; do
    inference_pid=$(cat "$inference_dir/job.pid")
    if ! kill -0 "$inference_pid" 2>/dev/null; then
        printf "inference stopped before predictions.jsonl\n" >&2
        exit 10
    fi
    sleep 30
done

test "$(wc -l < "$inference_dir/predictions.jsonl")" -eq 568
export DEEPSEEK_API_KEY
DEEPSEEK_API_KEY=$(tr -d "\r\n" < "$judge_dir/.deepseek_key")

exec "$HOME/miniconda3/bin/python3.10" \
    -m tools.axis_repro.geval_resilient \
    --predictions "$inference_dir/predictions.jsonl" \
    --output "$judge_dir/scores.jsonl" \
    --pending "$judge_dir/scores.fallback_pending.jsonl" \
    --model deepseek-v4-pro \
    --max-tokens 4096 \
    --primary-workers 12 \
    --fallback-workers 8 \
    --max-retry-rounds 12
