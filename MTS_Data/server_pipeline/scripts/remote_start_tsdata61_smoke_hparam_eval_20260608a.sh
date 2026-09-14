set -euo pipefail

ROOT=/tmp/multiaxis_runtime_swat
OUT_BASE=/tmp/tsdata61_smoke_hparam_eval_20260608a
CFG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
BASE_LLM_CFG="$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu.json"
PY=/root/hfenv/bin/python
PROPOSER_CKPT=/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt

mkdir -p "$OUT_BASE/configs"

write_cfg() {
  local out="$1"
  local ct="$2"
  local scale="$3"
  local axis_enabled="$4"
  "$PY" - <<PYCODE
import json
from pathlib import Path

src = Path("$BASE_LLM_CFG")
dst = Path("$out")
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["min_pixels"] = 401408
cfg["max_pixels"] = 802816
cfg["max_memory"] = {
    "0": "23200MiB",
    "1": "23200MiB",
    "2": "23200MiB",
    "3": "23200MiB",
    "cpu": "100GiB",
}
axis = dict(cfg.get("axis_hints") or {})
axis["enabled"] = bool(int("$axis_enabled"))
axis["max_local_tokens"] = int("$ct")
axis["max_channel_tokens"] = int("$ct")
axis["injection_scale"] = float("$scale")
axis["global_injection_scale"] = float("$scale")
axis["channel_injection_scale"] = float("$scale")
cfg["axis_hints"] = axis
dst.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
PYCODE
}

write_cfg "$OUT_BASE/configs/raw_image_hi_pixels.json" 16 0.1 0

for CT in 8 16; do
  for SCALE in 0 0.05 0.1 0.2 0.5; do
    TAG=$(printf "%03d" "$("$PY" - <<PYCODE
scale = float("$SCALE")
print(int(round(scale * 100)))
PYCODE
)")
    write_cfg "$OUT_BASE/configs/axis_ct${CT}_scale${TAG}.json" "$CT" "$SCALE" 1
  done
done

prepare_visuals() {
  local q="$1"
  local root="$2"
  if [ ! -f "$root/raw_images/manifest.json" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" \
      --data "$q" \
      --config "$CFG" \
      --checkpoint "$PROPOSER_CKPT" \
      --output-root "$root" \
      --modes raw \
      > "$root.prepare_raw.log" 2>&1
  fi
  if [ ! -f "$root/score_images/manifest.json" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_prepare_dataset_visuals.py" \
      --data "$q" \
      --config "$CFG" \
      --checkpoint "$PROPOSER_CKPT" \
      --output-root "$root" \
      --modes score \
      > "$root.prepare_score.log" 2>&1
  fi
}

run_group() {
  local q="$1"
  local out_dir="$2"
  local llm_cfg="$3"
  shift 3
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
    --data "$q" \
    --limit 50 \
    --output-dir "$out_dir" \
    --llm-config "$llm_cfg" \
    --max-window-rows 32 \
    --digits 3 \
    "$@" \
    > "$out_dir.log" 2>&1
}

run_dataset() {
  local name="$1"
  local q="$2"
  local t="$3"
  local visual_root="$4"
  local runroot="$OUT_BASE/$name"
  local raw_manifest="$visual_root/raw_images/manifest.json"
  local score_manifest="$visual_root/score_images/manifest.json"
  mkdir -p "$runroot"

  prepare_visuals "$q" "$visual_root"

  run_group "$q" "$runroot/raw_image" "$OUT_BASE/configs/raw_image_hi_pixels.json" \
    --include-raw-image-note \
    --raw-image-manifest "$raw_manifest"

  for CT in 8 16; do
    for SCALE in 0 0.05 0.1 0.2 0.5; do
      TAG=$(printf "%03d" "$("$PY" - <<PYCODE
scale = float("$SCALE")
print(int(round(scale * 100)))
PYCODE
)")
      CFG_PATH="$OUT_BASE/configs/axis_ct${CT}_scale${TAG}.json"

      run_group "$q" "$runroot/all_hints_ct${CT}_scale${TAG}" "$CFG_PATH" \
        --config "$CFG" \
        --interval-proposer-checkpoint "$PROPOSER_CKPT" \
        --use-axis-embedding-hints \
        --include-raw-image-note \
        --raw-image-manifest "$raw_manifest"

      run_group "$q" "$runroot/score_image_no_global_hints_ct${CT}_scale${TAG}" "$CFG_PATH" \
        --config "$CFG" \
        --interval-proposer-checkpoint "$PROPOSER_CKPT" \
        --use-axis-embedding-hints \
        --hide-global-hints \
        --include-anomaly-score-image-note \
        --score-image-manifest "$score_manifest"
    done
  done

  for SCALE in 0 0.05 0.1 0.2 0.5; do
    TAG=$(printf "%03d" "$("$PY" - <<PYCODE
scale = float("$SCALE")
print(int(round(scale * 100)))
PYCODE
)")
    CFG_PATH="$OUT_BASE/configs/axis_ct16_scale${TAG}.json"

    run_group "$q" "$runroot/score_text_all_hints_ct16_scale${TAG}" "$CFG_PATH" \
      --config "$CFG" \
      --interval-proposer-checkpoint "$PROPOSER_CKPT" \
      --use-axis-embedding-hints \
      --include-anomaly-score-text \
      --include-raw-image-note \
      --raw-image-manifest "$raw_manifest"

    run_group "$q" "$runroot/score_image_all_hints_ct16_scale${TAG}" "$CFG_PATH" \
      --config "$CFG" \
      --interval-proposer-checkpoint "$PROPOSER_CKPT" \
      --use-axis-embedding-hints \
      --include-anomaly-score-image-note \
      --score-image-manifest "$score_manifest"
  done

  "$PY" "$ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
    --run-root "$runroot" \
    --questions "$q" \
    --teacher "$t" \
    --runs \
      raw_image \
      all_hints_ct8_scale000 all_hints_ct8_scale005 all_hints_ct8_scale010 all_hints_ct8_scale020 all_hints_ct8_scale050 \
      all_hints_ct16_scale000 all_hints_ct16_scale005 all_hints_ct16_scale010 all_hints_ct16_scale020 all_hints_ct16_scale050 \
      score_text_all_hints_ct16_scale000 score_text_all_hints_ct16_scale005 score_text_all_hints_ct16_scale010 score_text_all_hints_ct16_scale020 score_text_all_hints_ct16_scale050 \
      score_image_all_hints_ct16_scale000 score_image_all_hints_ct16_scale005 score_image_all_hints_ct16_scale010 score_image_all_hints_ct16_scale020 score_image_all_hints_ct16_scale050 \
      score_image_no_global_hints_ct8_scale000 score_image_no_global_hints_ct8_scale005 score_image_no_global_hints_ct8_scale010 score_image_no_global_hints_ct8_scale020 score_image_no_global_hints_ct8_scale050 \
      score_image_no_global_hints_ct16_scale000 score_image_no_global_hints_ct16_scale005 score_image_no_global_hints_ct16_scale010 score_image_no_global_hints_ct16_scale020 score_image_no_global_hints_ct16_scale050 \
    > "$runroot/eval.log" 2>&1

  echo "$name hyperparameter sweep complete"
}

run_dataset \
  regular_smoke_0606c \
  /tmp/tsdata61_regular_smoke_0606c/questions_50_student.jsonl \
  /tmp/tsdata61_regular_smoke_0606c/teacher_gpt55.jsonl \
  /tmp/tsdata61_regular_smoke_0606c_visuals_20260608a

run_dataset \
  hard_smoke_0606c \
  /tmp/tsdata61_hard_smoke_0606c/questions_50_student.jsonl \
  /tmp/tsdata61_hard_smoke_0606c/teacher_gpt55.jsonl \
  /tmp/tsdata61_hard_smoke_0606c_visuals_20260608a
