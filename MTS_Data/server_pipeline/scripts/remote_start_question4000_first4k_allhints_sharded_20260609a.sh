#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="question4000_ts61_eval_20260609a"
DATASET="question_4000_ts61_0000_1000_0609a"
BASE="/dev/shm/${RUN_NAME}"
RUNTIME="$BASE/runtime/multiaxis"
INPUTS="$BASE/inputs"
CFGROOT="$BASE/configs"
SHARDROOT="$BASE/shards/${DATASET}"
RUNROOT="$BASE/student_runs/${DATASET}"
PIPELINE_LOG="$BASE/${DATASET}.all_hints_sharded.log"

PY="/root/shared-nvme/axis_project/envs/axis/bin/python"
if [ ! -x "$PY" ]; then
  PY="/root/hfenv/bin/python"
fi

BASE_LLM_CFG="$RUNTIME/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1.json"
AXIS_CFG="$RUNTIME/configs/torch_fixed60_qa600_timercd.json"
BRIDGE_ENCODER="/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt"

mkdir -p "$CFGROOT" "$SHARDROOT" "$RUNROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "[first4k-sharded] start $(date --iso-8601=seconds)"
echo "[first4k-sharded] dataset=$DATASET"
echo "[first4k-sharded] runtime=$RUNTIME"
echo "[first4k-sharded] inputs=$INPUTS"
echo "[first4k-sharded] runroot=$RUNROOT"
echo "[first4k-sharded] python=$PY"

Q_SRC="$INPUTS/${DATASET}.questions_4000_student.jsonl"
T_SRC="$INPUTS/${DATASET}.teacher_gpt55.jsonl"
RAW_MANIFEST_SRC="$INPUTS/${DATASET}.raw_manifest.json"

for path in "$Q_SRC" "$T_SRC" "$RAW_MANIFEST_SRC" "$BASE_LLM_CFG" "$AXIS_CFG" "$BRIDGE_ENCODER"; do
  if [ ! -f "$path" ]; then
    echo "[first4k-sharded] missing required file: $path" >&2
    exit 1
  fi
done

"$PY" - <<'PYCODE'
import json
from pathlib import Path

base = Path("/dev/shm/question4000_ts61_eval_20260609a/runtime/multiaxis/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1.json")
dst = Path("/dev/shm/question4000_ts61_eval_20260609a/configs/all_hints_raw_image_1gpu.json")
cfg = json.loads(base.read_text(encoding="utf-8"))
cfg["device"] = "cuda:0"
cfg["device_map"] = "auto"
cfg["max_memory"] = {"0": "23500MiB", "cpu": "100GiB"}
cfg["min_pixels"] = 401408
cfg["max_pixels"] = 1204224
cfg["max_tokens"] = 1024
axis = dict(cfg.get("axis_hints") or {})
axis["enabled"] = True
axis["injection_scale"] = 0.1
axis["global_injection_scale"] = 0.1
axis["channel_injection_scale"] = 0.1
axis["max_global_tokens"] = 8
axis["max_channel_tokens"] = 16
axis["max_local_tokens"] = 16
cfg["axis_hints"] = axis
dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[first4k-sharded] wrote {dst}")
PYCODE

"$PY" - <<'PYCODE'
import json
from pathlib import Path

base = Path("/dev/shm/question4000_ts61_eval_20260609a")
dataset = "question_4000_ts61_0000_1000_0609a"
q_src = base / "inputs" / f"{dataset}.questions_4000_student.jsonl"
t_src = base / "inputs" / f"{dataset}.teacher_gpt55.jsonl"
m_src = base / "inputs" / f"{dataset}.raw_manifest.json"
shard_root = base / "shards" / dataset
shard_root.mkdir(parents=True, exist_ok=True)

questions = [json.loads(line) for line in q_src.read_text(encoding="utf-8").splitlines() if line.strip()]
teachers = [json.loads(line) for line in t_src.read_text(encoding="utf-8").splitlines() if line.strip()]
manifest = json.loads(m_src.read_text(encoding="utf-8"))
records = manifest.get("records") or []
by_sample = {str(record.get("sample_id")): record for record in records}

sizes = [len(questions) // 4] * 4
for i in range(len(questions) % 4):
    sizes[i] += 1

start = 0
meta = []
for shard_idx, size in enumerate(sizes):
    end = start + size
    q_part = questions[start:end]
    t_part = teachers[start:end]
    samples = {str(row.get("sample_id")) for row in q_part}
    m_part = {
        "dataset": dataset,
        "shard_idx": shard_idx,
        "records": [by_sample[sid] for sid in samples if sid in by_sample],
    }
    q_path = shard_root / f"questions.part{shard_idx}.jsonl"
    t_path = shard_root / f"teacher.part{shard_idx}.jsonl"
    m_path = shard_root / f"raw_manifest.part{shard_idx}.json"
    q_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in q_part), encoding="utf-8")
    t_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in t_part), encoding="utf-8")
    m_path.write_text(json.dumps(m_part, ensure_ascii=False, indent=2), encoding="utf-8")
    meta.append({
        "shard_idx": shard_idx,
        "start": start,
        "end": end,
        "count": len(q_part),
        "question_path": str(q_path),
        "teacher_path": str(t_path),
        "manifest_path": str(m_path),
    })
    start = end

(shard_root / "shards.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[first4k-sharded] wrote shard metadata: {shard_root / 'shards.json'}")
PYCODE

run_shard() {
  local gpu="$1"
  local shard_idx="$2"
  local out_dir="$RUNROOT/all_hints_part${shard_idx}"
  local q="$SHARDROOT/questions.part${shard_idx}.jsonl"
  local raw_manifest="$SHARDROOT/raw_manifest.part${shard_idx}.json"
  mkdir -p "$out_dir"
  echo "[first4k-sharded] launch shard=$shard_idx gpu=$gpu at $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -B "$RUNTIME/scripts/mvaxis_run_raw_qwen_answers.py" \
    --data "$q" \
    --output-dir "$out_dir" \
    --llm-config "$CFGROOT/all_hints_raw_image_1gpu.json" \
    --config "$AXIS_CFG" \
    --interval-proposer-checkpoint "$BRIDGE_ENCODER" \
    --use-axis-embedding-hints \
    --include-raw-image-note \
    --raw-image-manifest "$raw_manifest" \
    --max-window-rows 24 \
    --digits 3 \
    --progress-percent-step 5 \
    > "$RUNROOT/all_hints_part${shard_idx}.log" 2>&1
  echo "[first4k-sharded] finish shard=$shard_idx gpu=$gpu at $(date --iso-8601=seconds)"
}

for shard_idx in 0 1 2 3; do
  run_shard "$shard_idx" "$shard_idx" &
done
wait

"$PY" - <<'PYCODE'
import json
from pathlib import Path

base = Path("/dev/shm/question4000_ts61_eval_20260609a")
dataset = "question_4000_ts61_0000_1000_0609a"
run_root = base / "student_runs" / dataset
shard_meta = json.loads((base / "shards" / dataset / "shards.json").read_text(encoding="utf-8"))
merged_dir = run_root / "all_hints"
merged_dir.mkdir(parents=True, exist_ok=True)
merged_path = merged_dir / "qwen_raw_answers.jsonl"
latencies = []
count = 0
with merged_path.open("w", encoding="utf-8") as fout:
    for item in shard_meta:
        shard_idx = item["shard_idx"]
        index_offset = item["start"]
        shard_path = run_root / f"all_hints_part{shard_idx}" / "qwen_raw_answers.jsonl"
        with shard_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["index"] = int(row.get("index", 0)) + int(index_offset)
                lat = row.get("latency_seconds")
                if lat is not None:
                    try:
                        latencies.append(float(lat))
                    except Exception:
                        pass
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

summary = {
    "dataset": dataset,
    "num_samples": count,
    "latency_seconds_avg": (sum(latencies) / len(latencies)) if latencies else None,
    "latency_seconds_max": max(latencies) if latencies else None,
    "latency_seconds_min": min(latencies) if latencies else None,
    "parts": [f"all_hints_part{item['shard_idx']}" for item in shard_meta],
}
(merged_dir / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[first4k-sharded] merged records into {merged_path}")
PYCODE

"$PY" "$RUNTIME/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
  --run-root "$RUNROOT" \
  --questions "$Q_SRC" \
  --teacher "$T_SRC" \
  --runs all_hints \
  > "$RUNROOT/eval_all_hints.log" 2>&1

echo "[first4k-sharded] all done $(date --iso-8601=seconds)"
