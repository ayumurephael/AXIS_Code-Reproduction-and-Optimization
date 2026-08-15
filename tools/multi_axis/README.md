# Multi-AXIS implementation and formal experiment

This directory contains the full multivariate AXIS Phase-II pipeline specified in `多元 AXIS 架构.md`. The implementation keeps TimeRCD and the language model frozen, trains only the Hint Tuner, preserves all valid channel-by-time evidence, and never truncates a prompt.

## Registered formal protocol

- Student LLM: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.
- Also supported by the same model interface: `Qwen/Qwen3.5-9B`.
- Frozen multivariate TimeRCD encoder and anomaly head, loaded strictly from the supplied checkpoint.
- Full Phase D architecture: Step-Local, one Channel-Local overview token per channel, question-conditioned Joint-Local, anomaly evidence, 30 Fixed hints, and 1,024 vocabulary prototypes. The complete Window + Step evidence remains present; Channel Routing is not implemented.
- Numeric windows are approximately reversible: every channel header carries the full-valid-series mean and standard deviation plus the exact normalization epsilon.
- Frozen-LLM question states are detached and disk-cached by model, tokenizer, and exact question text; only the zero-initialized bias-free `A_q` is trainable on the question-semantic path.
- All ordinary cross-attention uses the CUDA FlashAttention backend; CPU exists only as a unit-test fallback.
- Teacher supervision is answer-only NLL over the natural-language `model_answer` field only. The 17 empty `model_answer` rows are filtered; short-label fields are never used as a training fallback.
- Training alignment follows the authoritative 62-shard teacher summary: 67,773 rows are matched one-to-one by normalized question text within each shard, and the remaining 47 are supplied by a SHA-256-audited same-index recovery bundle derived from the raw teacher records.
- Split is by `base_sample_id`, 90/10, seed 42, with zero group overlap.
- Current four-GPU formal experiment: exactly 20 epochs, no early stopping. The preserved five-GPU profile remains exactly 40 epochs. In both cases, select the lowest validation answer-token NLL only after all configured epochs.
- Formal inference covers 478new, SMD, SWaT, LEMMA-RCA, and VTA.
- MC and TF use final-label exact match; OE uses parseability. Results are emitted overall, by question type, and by dataset.
- Five Judges score every test set. GPT-5.4 is the only Judge used for baseline deltas; the other four are used for cross-Judge robustness.

The checked-in formal configuration records the actual world size and effective batch size. Do not change accumulation or world size after launch; a resumed run must use the same configuration.

The formal 4096-dimensional prototype cross-attention uses 16 heads (head_dim=256), the largest head dimension supported by FlashAttention-2. The preserved five-process/40-epoch profile uses micro-batch 1 and six-step accumulation for effective batch size 30. The four-process/20-epoch profile keeps effective batch size 32 and supports the registered benchmark candidates micro-batch/accumulation 1/8, 2/4, and 4/2. Formal launch uses the fastest candidate that fits all four selected GPUs. Model construction audits the resolved LLM backend, every identified LLM attention module, both FlashAttention kernel imports, and all Hint Tuner cross-attention modules; any eager fallback terminates the run.

## Components

- `src/models/MultiAXIS/`: configuration, prompt construction, frozen TimeRCD wrapper, Flash cross-attention, Hint Tuner, and dual-LLM model wrapper.
- `tools/multi_axis/build_training_recovery.py`: reproducibly derives the 47-row, SHA-256-audited recovery bundle from the registered raw teacher records.
- `tools/multi_axis/build_manifests.py`: deterministic pairing, filtering, grouped split, source hashes, random-access indices, and the five evaluation manifests.
- tools/multi_axis/train.py: registered four- or five-process manual data parallelism, bounded batch-aware frozen-encoder cache, exact answer NLL, configured-epoch checkpoints, throughput/memory benchmark mode, FlashAttention runtime audit, and crash resume.
- `tools/multi_axis/infer.py`: configured four- or five-GPU, group-preserving, crash-resumable generation and deterministic merge.
- `tools/multi_axis/label_metrics.py`: exact MC/TF label metrics and OE parse rate.
- `tools/multi_axis/geval_runner.py`: crash-resilient five-Judge scoring with provider-specific probability handling.
- `tools/multi_axis/aggregate_geval.py`: ten metrics for every dataset × Judge plus micro/macro and cross-Judge statistics.
- `tools/multi_axis/audit_experiment.py`: fail-closed protocol, count, checkpoint, prediction, and Judge audit.
- `tools/multi_axis/report_experiment.py`: GPT-5.4 baseline deltas and the final experiment-analysis Markdown.

## Environment

Use Linux, four or five visible NVIDIA H100 GPUs as selected by the formal configuration, BF16, PyTorch 2.5 or newer, Transformers 4.57 or newer, and FlashAttention compatible with the installed Torch/CUDA ABI. On a preconfigured server, preserve its PyTorch build and install the additional packages selectively instead of allowing pip to replace Torch.

```bash
python -m pip install --upgrade "transformers>=4.57,<5" "openai>=1.99" "google-genai>=1.30" "httpx>=0.28" packaging psutil ninja
python -m pip install --no-build-isolation --no-deps "flash-attn==2.7.4.post1"
```

Before any formal work, verify that `torch.cuda.device_count()` equals the configuration's `expected_world_size`, that all visible devices are H100, BF16 is supported, `flash_attn` imports, and PyTorch Flash SDPA dispatch succeeds. Export `PYTHONPATH` to the repository root and put Hugging Face caches on a filesystem with enough space. For an audited offline snapshot, set `MULTI_AXIS_MODEL_PATH` to its local directory; the registered model ID remains unchanged in the configuration and checkpoint metadata.

## Rebuild the audited recovery bundle

This one-time derivation requires the original raw `teacher_gpt55.jsonl` files. The formal asset root only needs the two generated files. The command fails unless the registered counts are exactly 67,820 total, 67,773 direct text matches, 47 same-index recoveries, and 17 empty teacher answers.

```bash
python tools/multi_axis/build_training_recovery.py \
  --source-root /path/to/original-collaborator-assets \
  --output-dir /path/to/formal-assets/derived/training_recovery
```

## Build immutable manifests

```bash
export PYTHONPATH="$PWD"
python tools/multi_axis/build_manifests.py \
  --data-root /path/to/multi-axis-assets \
  --output-dir /path/to/run/manifests \
  --seed 42 \
  --validation-fraction 0.10
```

This must report 62 authoritative shards, 67,773 direct text matches, 47 audited recovery matches, 67,820 aligned rows with 67,820 structured-reference matches, 17 filtered empty answers, 67,803 retained rows, and zero train/validation group overlap. The summary also records the 76,998-row question pool, unused question count, all consumed recovery keys, raw-source provenance hashes, and every materialized input hash. Evaluation counts must be 478, 200, 184, 12, and 200 with the registered per-type counts. The partially covered 478new pool is aligned by normalized question text; the four fully covered real-world sets follow the collaborator scorer's index fallback across bias-neutralized question revisions. Every resulting pair must also match on the structured `windows_0_answer` reference. Input SHA-256 hashes are retained unless `--skip-input-hashes` is explicitly used; the formal run must not use that flag.

## Benchmark and run the registered formal epoch count

Release only the experiment's own verified reservation processes immediately before launch. Never terminate another user's process.

Four-GPU profile (effective batch 32):

```bash
torchrun --standalone --nproc_per_node=4 tools/multi_axis/train.py \
  --config experiments/multi_axis/formal_deepseek_20epochs_4gpu.json \
  --data-root /path/to/multi-axis-assets \
  --manifest-dir /path/to/run/manifests \
  --timercd-checkpoint /path/to/pretrain_checkpoint_best_multi.pth \
  --question-semantic-cache-dir /path/to/run/question_semantics \
  --output-dir /path/to/run/training
```

Preserved five-GPU profile (effective batch 30):

```bash
torchrun --standalone --nproc_per_node=5 tools/multi_axis/train.py \
  --config experiments/multi_axis/formal_deepseek_40epochs.json \
  --data-root /path/to/multi-axis-assets \
  --manifest-dir /path/to/run/manifests \
  --timercd-checkpoint /path/to/pretrain_checkpoint_best_multi.pth \
  --question-semantic-cache-dir /path/to/run/question_semantics \
  --output-dir /path/to/run/training
```

To compare registered four-GPU candidates, add --benchmark-optimizer-steps N and use a distinct output directory for each configuration. Benchmark mode writes benchmark_summary.json and exits without validation or formal checkpoints. To resume a checkpoint produced by this architecture, provide `--resume /path/to/run/training/last_train_state.pt`. To migrate a pre-Channel-Local checkpoint, additionally pass `--migrate-legacy-architecture`; `--new-module-warmup-epochs 1` trains only Channel/A_q modules during the first resumed epoch while preserving the existing optimizer/scheduler/global step by name. A valid completion has TRAINING_COMPLETE, one epochs.jsonl record and one hint checkpoint per configured epoch (20 for the current four-GPU experiment; 40 for the preserved five-GPU profile), and best_checkpoint.json pointing to the minimum validation answer NLL.

## Four- or five-GPU inference

Read the selected checkpoint filename from `best_checkpoint.json`; do not select by test or Judge scores. Inference uses the same registered distributed profile as its training run. The current four-GPU formal run uses:

```bash
torchrun --standalone --nproc_per_node=4 tools/multi_axis/infer.py \
  --config experiments/multi_axis/formal_deepseek_20epochs_4gpu.json \
  --data-root /path/to/multi-axis-assets \
  --manifest-dir /path/to/run/manifests \
  --timercd-checkpoint /path/to/pretrain_checkpoint_best_multi.pth \
  --hint-checkpoint /path/to/run/training/hint_epoch_XX.pt \
  --question-semantic-cache-dir /path/to/run/question_semantics \
  --output-dir /path/to/run/predictions
```

Then compute deterministic task metrics and Judge inputs:

```bash
python tools/multi_axis/label_metrics.py \
  --manifest-dir /path/to/run/manifests \
  --predictions-dir /path/to/run/predictions \
  --output-dir /path/to/run/label_metrics

python tools/multi_axis/prepare_judge_inputs.py \
  --manifest-dir /path/to/run/manifests \
  --predictions-dir /path/to/run/predictions \
  --output-dir /path/to/run/judge_inputs
```

## Five-Judge G-Eval

Run Judge orchestration on the GPU server. Supply endpoints and API keys only through the environment variables named in `experiments/multi_axis/judge_profiles.json`; never place them in a command line, log, result file, or commit. Probe each Judge before the full batch with `--probe-only --limit 1`.

For every Judge and dataset:

```bash
python tools/multi_axis/geval_runner.py \
  --input /path/to/run/judge_inputs/478new.judge_input.jsonl \
  --rubrics experiments/multi_axis/geval_rubrics.json \
  --profiles experiments/multi_axis/judge_profiles.json \
  --judge gpt-5.4 \
  --output /path/to/run/geval/gpt-5.4/478new.jsonl \
  --max-missing-score-mass-upper-bound 1e-6
```

The five registered Judge names are `deepseek-v4-pro`, `gemini-2.5-pro`, `gpt-5.4`, `qwen3.5-397b-a17b`, and `qwen3-30b-a3b-instruct-2507`. The runner resumes from completed `(record_id, dimension)` pairs and fails closed after bounded retries.

DeepSeek uses score-token top-20 log probabilities and falls back to the exact mean of 20 valid samples only when necessary. Both Qwen Judges require bounded top-5 score mass, with an A–E JSON readout if the primary score token is unsuitable. Gemini 2.5 Pro and GPT-5.4 use their integer final scores because usable score-token log probabilities are unavailable.

## Aggregate, audit, and report

```bash
python tools/multi_axis/aggregate_geval.py \
  --results-root /path/to/run/geval \
  --output-dir /path/to/run/aggregate

python tools/multi_axis/audit_experiment.py \
  --config experiments/multi_axis/formal_deepseek_20epochs_4gpu.json \
  --manifest-dir /path/to/run/manifests \
  --run-dir /path/to/run/training \
  --predictions-dir /path/to/run/predictions \
  --judge-input-dir /path/to/run/judge_inputs \
  --geval-results-root /path/to/run/geval \
  --output /path/to/run/audit.json

python tools/multi_axis/report_experiment.py \
  --baseline-tables /path/to/BASELINE_TABLES.md \
  --geval-summary /path/to/run/aggregate/geval_10metrics.json \
  --label-summary /path/to/run/label_metrics/label_metrics_summary.json \
  --run-dir /path/to/run/training \
  --manifest-dir /path/to/run/manifests \
  --inference-dir /path/to/run/predictions \
  --geval-results-root /path/to/run/geval \
  --output-dir /path/to/run/report
```

The final audit must pass before results are reported. The analysis includes all dataset × Judge ten-metric tables, micro/macro summaries, label metrics, full training/intermediate configuration, Judge methods and counts, GPT-5.4 absolute/relative baseline changes, strongest-baseline comparisons, and future-work evidence.

## Tests

```bash
python -m pytest -q tools/axis_repro tools/multi_axis
python -m compileall -q src/models/MultiAXIS tools/multi_axis
```

Formal model training, generation, and Judge scoring are intentionally absent from unit tests and must run only on the authorized GPU server.
