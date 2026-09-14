#!/usr/bin/env bash
set -euo pipefail

ROOT="/WORK/fit/zhangchen/axis_project/multiaxis_runtime_train"
RUN_ROOT="/WORK/fit/zhangchen/axis_project/runs/ts61_vl_allhints_nexttoken_train_20260608a"
PY="${PY:-python}"

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/reports" "$RUN_ROOT/logs"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 "$ROOT/scripts/mvaxis_train_nexttoken_hints.py" \
  --data /WORK/fit/zhangchen/axis_project/data/ts61_train/questions_2000.jsonl \
  --teacher /WORK/fit/zhangchen/axis_project/data/ts61_train/teacher_gpt55.jsonl \
  --teacher-overrides "" \
  --config "$ROOT/configs/torch_fixed60_qa600_timercd.json" \
  --llm-config "$ROOT/configs/local_qwen25_vl7b_ts61_vl_allhints_ct12_scale010_8gpu.json" \
  --interval-proposer-checkpoint /WORK/fit/zhangchen/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1_encoder.pt \
  --init-bridge-checkpoint /WORK/fit/zhangchen/axis_project/checkpoints/qwen25_vl7b_nexttoken_answer_bridge_qfinal530_e10_typeheads_v1.pt \
  --interval-source target \
  --score-image-manifest /WORK/fit/zhangchen/axis_project/data/ts61_train_visuals_20260608a/score_images/manifest.json \
  --limit 32 \
  --epochs 1 \
  --learning-rate 5e-5 \
  --gradient-accumulation-steps 4 \
  --answer-loss-weight 0.0 \
  --max-window-rows 32 \
  --preserve-device-map-in-distributed \
  --output-checkpoint "$RUN_ROOT/checkpoints/bridge_smoke.pt" \
  --output-encoder-checkpoint "$RUN_ROOT/checkpoints/encoder_smoke.pt" \
  --report "$RUN_ROOT/reports/train_report_smoke.json" \
  --run-note "ts61 VL score-image all-hints next-token training smoke, 8gpu, ct12, scale0.1" \
  > "$RUN_ROOT/logs/smoke_8gpu.log" 2>&1
