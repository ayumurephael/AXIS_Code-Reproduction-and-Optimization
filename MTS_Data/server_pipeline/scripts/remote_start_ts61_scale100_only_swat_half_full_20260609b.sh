set -euo pipefail

ROOT=/tmp/multiaxis_runtime_swat
PY=/root/hfenv/bin/python
CFG="$ROOT/configs/torch_fixed60_qa600_timercd.json"
BASE_LLM_CFG="$ROOT/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu.json"
PROPOSER_CKPT=/root/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_question1000_0603_scoretext_global_channel_noimage_sharded2x2_cap2048_encoder.pt
SWAT_SCALES=(0 0.05 0.1 0.2 0.5 1.0)

TS61_OUT=/tmp/tsdata61_scale100_only_20260609c
SWAT_OUT=/tmp/swat_half_hparam_full_20260609b

mkdir -p "$TS61_OUT/configs" "$SWAT_OUT/configs"

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

make_tag() {
  local scale="$1"
  "$PY" - <<PYCODE
scale = float("$scale")
print(f"{int(round(scale * 100)):03d}")
PYCODE
}

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
  local limit="$4"
  local max_rows="$5"
  shift 5
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -B "$ROOT/scripts/mvaxis_run_raw_qwen_answers.py" \
    --data "$q" \
    --limit "$limit" \
    --output-dir "$out_dir" \
    --llm-config "$llm_cfg" \
    --max-window-rows "$max_rows" \
    --digits 3 \
    "$@" \
    > "$out_dir.log" 2>&1
}

prepare_ts61_cfgs() {
  local tag
  tag="$(make_tag 1.0)"
  write_cfg "$TS61_OUT/configs/axis_ct8_scale${tag}.json" 8 1.0 1
  write_cfg "$TS61_OUT/configs/axis_ct16_scale${tag}.json" 16 1.0 1
}

prepare_swat_cfgs() {
  write_cfg "$SWAT_OUT/configs/raw_image_hi_pixels.json" 16 0.1 0
  for ct in 8 16; do
    for scale in "${SWAT_SCALES[@]}"; do
      tag="$(make_tag "$scale")"
      write_cfg "$SWAT_OUT/configs/axis_ct${ct}_scale${tag}.json" "$ct" "$scale" 1
    done
  done
}

prepare_ts61_cfgs
prepare_swat_cfgs

run_ts61_scale100() {
  local name="$1"
  local q="$2"
  local t="$3"
  local visual_root="$4"
  local runroot="$TS61_OUT/$name"
  local raw_manifest="$visual_root/raw_images/manifest.json"
  local score_manifest="$visual_root/score_images/manifest.json"

  mkdir -p "$runroot"
  prepare_visuals "$q" "$visual_root"

  run_group "$q" "$runroot/all_hints_ct8_scale100" "$TS61_OUT/configs/axis_ct8_scale100.json" 50 32 \
    --config "$CFG" \
    --interval-proposer-checkpoint "$PROPOSER_CKPT" \
    --use-axis-embedding-hints \
    --include-raw-image-note \
    --raw-image-manifest "$raw_manifest"

  run_group "$q" "$runroot/all_hints_ct16_scale100" "$TS61_OUT/configs/axis_ct16_scale100.json" 50 32 \
    --config "$CFG" \
    --interval-proposer-checkpoint "$PROPOSER_CKPT" \
    --use-axis-embedding-hints \
    --include-raw-image-note \
    --raw-image-manifest "$raw_manifest"

  run_group "$q" "$runroot/score_text_all_hints_ct16_scale100" "$TS61_OUT/configs/axis_ct16_scale100.json" 50 32 \
    --config "$CFG" \
    --interval-proposer-checkpoint "$PROPOSER_CKPT" \
    --use-axis-embedding-hints \
    --include-anomaly-score-text \
    --include-raw-image-note \
    --raw-image-manifest "$raw_manifest"

  run_group "$q" "$runroot/score_image_all_hints_ct16_scale100" "$TS61_OUT/configs/axis_ct16_scale100.json" 50 32 \
    --config "$CFG" \
    --interval-proposer-checkpoint "$PROPOSER_CKPT" \
    --use-axis-embedding-hints \
    --include-anomaly-score-image-note \
    --score-image-manifest "$score_manifest"

  run_group "$q" "$runroot/score_image_no_global_hints_ct8_scale100" "$TS61_OUT/configs/axis_ct8_scale100.json" 50 32 \
    --config "$CFG" \
    --interval-proposer-checkpoint "$PROPOSER_CKPT" \
    --use-axis-embedding-hints \
    --hide-global-hints \
    --include-anomaly-score-image-note \
    --score-image-manifest "$score_manifest"

  run_group "$q" "$runroot/score_image_no_global_hints_ct16_scale100" "$TS61_OUT/configs/axis_ct16_scale100.json" 50 32 \
    --config "$CFG" \
    --interval-proposer-checkpoint "$PROPOSER_CKPT" \
    --use-axis-embedding-hints \
    --hide-global-hints \
    --include-anomaly-score-image-note \
    --score-image-manifest "$score_manifest"

  "$PY" "$ROOT/scripts/mvaxis_eval_teacher_aligned_student_runs.py" \
    --run-root "$runroot" \
    --questions "$q" \
    --teacher "$t" \
    --runs \
      all_hints_ct8_scale100 \
      all_hints_ct16_scale100 \
      score_text_all_hints_ct16_scale100 \
      score_image_all_hints_ct16_scale100 \
      score_image_no_global_hints_ct8_scale100 \
      score_image_no_global_hints_ct16_scale100 \
    > "$runroot/eval.log" 2>&1
}

run_swat_full_half() {
  local name="$1"
  local q="$2"
  local t="$3"
  local visual_root="$4"
  local runroot="$SWAT_OUT/$name"
  local raw_manifest="$visual_root/raw_images/manifest.json"
  local score_manifest="$visual_root/score_images/manifest.json"

  mkdir -p "$runroot"
  prepare_visuals "$q" "$visual_root"

  run_group "$q" "$runroot/raw_image" "$SWAT_OUT/configs/raw_image_hi_pixels.json" 80 24 \
    --include-raw-image-note \
    --raw-image-manifest "$raw_manifest"

  for ct in 8 16; do
    for scale in "${SWAT_SCALES[@]}"; do
      tag="$(make_tag "$scale")"
      cfg="$SWAT_OUT/configs/axis_ct${ct}_scale${tag}.json"

      run_group "$q" "$runroot/all_hints_ct${ct}_scale${tag}" "$cfg" 80 24 \
        --config "$CFG" \
        --interval-proposer-checkpoint "$PROPOSER_CKPT" \
        --use-axis-embedding-hints \
        --include-raw-image-note \
        --raw-image-manifest "$raw_manifest"

      run_group "$q" "$runroot/score_image_no_global_hints_ct${ct}_scale${tag}" "$cfg" 80 24 \
        --config "$CFG" \
        --interval-proposer-checkpoint "$PROPOSER_CKPT" \
        --use-axis-embedding-hints \
        --hide-global-hints \
        --include-anomaly-score-image-note \
        --score-image-manifest "$score_manifest"
    done
  done

  for scale in "${SWAT_SCALES[@]}"; do
    tag="$(make_tag "$scale")"
    cfg="$SWAT_OUT/configs/axis_ct16_scale${tag}.json"

    run_group "$q" "$runroot/score_text_all_hints_ct16_scale${tag}" "$cfg" 80 24 \
      --config "$CFG" \
      --interval-proposer-checkpoint "$PROPOSER_CKPT" \
      --use-axis-embedding-hints \
      --include-anomaly-score-text \
      --include-raw-image-note \
      --raw-image-manifest "$raw_manifest"

    run_group "$q" "$runroot/score_image_all_hints_ct16_scale${tag}" "$cfg" 80 24 \
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
      all_hints_ct8_scale000 all_hints_ct8_scale005 all_hints_ct8_scale010 all_hints_ct8_scale020 all_hints_ct8_scale050 all_hints_ct8_scale100 \
      all_hints_ct16_scale000 all_hints_ct16_scale005 all_hints_ct16_scale010 all_hints_ct16_scale020 all_hints_ct16_scale050 all_hints_ct16_scale100 \
      score_text_all_hints_ct16_scale000 score_text_all_hints_ct16_scale005 score_text_all_hints_ct16_scale010 score_text_all_hints_ct16_scale020 score_text_all_hints_ct16_scale050 score_text_all_hints_ct16_scale100 \
      score_image_all_hints_ct16_scale000 score_image_all_hints_ct16_scale005 score_image_all_hints_ct16_scale010 score_image_all_hints_ct16_scale020 score_image_all_hints_ct16_scale050 score_image_all_hints_ct16_scale100 \
      score_image_no_global_hints_ct8_scale000 score_image_no_global_hints_ct8_scale005 score_image_no_global_hints_ct8_scale010 score_image_no_global_hints_ct8_scale020 score_image_no_global_hints_ct8_scale050 score_image_no_global_hints_ct8_scale100 \
      score_image_no_global_hints_ct16_scale000 score_image_no_global_hints_ct16_scale005 score_image_no_global_hints_ct16_scale010 score_image_no_global_hints_ct16_scale020 score_image_no_global_hints_ct16_scale050 score_image_no_global_hints_ct16_scale100 \
    > "$runroot/eval.log" 2>&1
}

run_ts61_scale100 \
  regular_smoke_0606c \
  /tmp/tsdata61_regular_smoke_0606c/questions_50_student.jsonl \
  /tmp/tsdata61_regular_smoke_0606c/teacher_gpt55.jsonl \
  /tmp/tsdata61_regular_smoke_0606c_visuals_20260608a

run_ts61_scale100 \
  hard_smoke_0606c \
  /tmp/tsdata61_hard_smoke_0606c/questions_50_student.jsonl \
  /tmp/tsdata61_hard_smoke_0606c/teacher_gpt55.jsonl \
  /tmp/tsdata61_hard_smoke_0606c_visuals_20260608a

run_swat_full_half \
  regular_160_0605b_half \
  /tmp/swat_regular_160_0605b/questions_160_student.jsonl \
  /tmp/swat_regular_160_0605b/teacher_gpt55.jsonl \
  /tmp/swat_regular_160_0605b_visuals_raw

run_swat_full_half \
  hard_160_0605a_half \
  /tmp/swat_hard_160_0605a/questions_160_student.jsonl \
  /tmp/swat_hard_160_0605a/teacher_gpt55.jsonl \
  /tmp/swat_hard_160_0605a_visuals_raw

echo "ts61 scale100-only + swat half full sweep complete"
