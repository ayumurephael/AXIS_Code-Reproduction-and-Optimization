# Epoch-17 Multi-AXIS-VL inference-time ablation protocol

This protocol implements Table 3 from `ablation_experiments.md`. Run only one
variant at a time. Multiple approved GPUs may split that variant by complete
`base_sample_id` groups; do not run different variants concurrently.

## 1. Immutable inputs

Use the identities in `BRANCH_PROFILE.md`. The source manifest must have 478
rows with MC/OE/TF counts 176/139/163. All 240 unique time-series images must
already exist and pass the renderer/image hash inventory. Even variants that
remove the image from the VLM input retain this source-image availability
audit, so they cannot silently use a different sample population.

The intervention names are:

```text
observable_only contextual_only wo_visual wo_numeric wo_scale_calibration
wo_step wo_channel wo_joint wo_question_conditioning wo_task_prior
```

The full `multi_axis` result is not rerun. Its user-confirmed Qwen3-30B row and
identity are stored in
`experiments/multi_axis/ablation_epoch17_qwen30b_baseline.json`.

## 2. One-variant inference

Assign all 216 group residues exactly once across the currently approved
80-GB GPUs. Each host launches its own local `torchrun`; `N` is the number of
GPUs used on that host and `A100`/`H100` must match their audited model names.

```bash
torchrun --standalone --nproc_per_node=N tools/multi_axis/infer.py \
  --config /path/to/formal_qwen3_vl_20epochs_8gpu_batch32.json \
  --data-root /path/to/assets \
  --manifest-dir /path/to/manifests \
  --image-root /path/to/rendered-images \
  --timercd-checkpoint /path/to/pretrain_checkpoint_best_multi.pth \
  --hint-checkpoint /path/to/hint_epoch_17.pt \
  --question-semantic-cache-dir /path/to/question-cache \
  --output-dir /path/to/study/VARIANT/workers/HOST \
  --datasets 478new \
  --ablation-variant VARIANT \
  --expected-inference-world-size N \
  --expected-inference-nodes 1 \
  --expected-gpu-substring A100 \
  --inference-shard-count 216 \
  --inference-shard-indices RESIDUES
```

Do not modify the generation section in the formal config. It must resolve to
beam 5, non-sampling, and 1000 maximum new tokens. Keep `model.eval()` and
`torch.inference_mode()`; `infer.py` enforces both.

After every worker completes, merge only mutually disjoint, collectively
complete group shards:

```bash
python tools/multi_axis/merge_sharded_inference_outputs.py \
  --source /path/to/study/VARIANT/workers/HOST_A \
  --source /path/to/study/VARIANT/workers/HOST_B \
  --manifest-dir /path/to/manifests \
  --output-dir /path/to/study/VARIANT/predictions \
  --datasets 478new
```

The merger validates the checkpoint, generation, model, attention backend,
variant specification, inference commit, shard disjointness, full coverage,
and original manifest order.

## 3. Deterministic metrics and Judge inputs

```bash
python tools/multi_axis/label_metrics.py \
  --manifest-dir /path/to/manifests \
  --predictions-dir /path/to/study/VARIANT/predictions \
  --output-dir /path/to/study/VARIANT/label_metrics \
  --datasets 478new --mode VARIANT

python tools/multi_axis/prepare_judge_inputs.py \
  --manifest-dir /path/to/manifests \
  --predictions-dir /path/to/study/VARIANT/predictions \
  --output-dir /path/to/study/VARIANT/judge_inputs \
  --datasets 478new --mode VARIANT
```

## 4. Qwen3-30B G-Eval

Run Judge calls on the approved networked THU_IE host. Inject credentials from
a private mode-0600 environment file; never print, copy into the repository,
or place them on the command line. The profile fixes model ID, temperature 0,
non-thinking mode, top-logprobs 5, and six workers.

First run exactly one probe without writing a partial formal row:

```bash
python tools/multi_axis/geval_runner.py \
  --input /path/to/study/VARIANT/judge_inputs/478new.judge_input.jsonl \
  --rubrics experiments/multi_axis/geval_rubrics.json \
  --profiles experiments/multi_axis/judge_profiles.json \
  --judge qwen3-30b-a3b-instruct-2507 \
  --output /path/to/study/VARIANT/geval/qwen3-30b-a3b-instruct-2507/478new.jsonl \
  --max-missing-score-mass-upper-bound 1e-6 --probe-only
```

Then rerun the same command without `--probe-only`. Resume only into the same
output file. Completion requires 1,095 unique question-dimension scores.

```bash
python tools/multi_axis/aggregate_geval.py \
  --results-root /path/to/study/VARIANT/geval \
  --output-dir /path/to/study/VARIANT/aggregate \
  --judges qwen3-30b-a3b-instruct-2507 --datasets 478new

python tools/multi_axis/audit_evaluation.py \
  --manifest-dir /path/to/manifests \
  --predictions-dir /path/to/study/VARIANT/predictions \
  --judge-input-dir /path/to/study/VARIANT/judge_inputs \
  --geval-results-root /path/to/study/VARIANT/geval \
  --label-metrics-dir /path/to/study/VARIANT/label_metrics \
  --datasets 478new --judges qwen3-30b-a3b-instruct-2507 \
  --expected-checkpoint-epoch 17 --expected-mode VARIANT \
  --output /path/to/study/VARIANT/evaluation_audit.json
```

Do not begin the next variant until the current `evaluation_audit.json` has
`passed=true` and all GPU inference processes for it have exited.

## 5. Final table and study audit

```bash
python tools/multi_axis/aggregate_ablation.py \
  --study-root /path/to/study \
  --baseline experiments/multi_axis/ablation_epoch17_qwen30b_baseline.json \
  --output-dir /path/to/study/final

python tools/multi_axis/audit_ablation_study.py \
  --study-root /path/to/study \
  --manifest /path/to/manifests/eval_478new.jsonl \
  --baseline experiments/multi_axis/ablation_epoch17_qwen30b_baseline.json \
  --expected-inference-commit COMMIT_OF_THIS_BRANCH \
  --output /path/to/study/final/ablation_study_audit.json
```

Only a final audit with `passed=true` may populate the report table.
