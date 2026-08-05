# Multi-AXIS implementation and formal experiment

This directory contains the full multivariate AXIS Phase-II pipeline specified in `多元 AXIS 架构.md`. The implementation keeps TimeRCD and the language model frozen, trains only the Hint Tuner, preserves all valid channel-by-time evidence, and never truncates a prompt.

## Registered formal protocol

- Student LLM: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.
- Also supported by the same model interface: `Qwen/Qwen3.5-9B`.
- Frozen multivariate TimeRCD encoder and anomaly head, loaded strictly from the supplied checkpoint.
- Full Phase D architecture: Step-Local, anomaly evidence, Joint-Local, 30 Fixed hints, 1,024 vocabulary prototypes.
- All ordinary cross-attention uses the CUDA FlashAttention backend; CPU exists only as a unit-test fallback.
- Teacher supervision is answer-only NLL. The 17 empty teacher answers are filtered.
- Split is by `base_sample_id`, 90/10, seed 42, with zero group overlap.
- Exactly 40 epochs, no early stopping; select the lowest validation answer-token NLL after all epochs.
- Formal inference covers 478new, SMD, SWaT, LEMMA-RCA, and VTA.
- MC and TF use final-label exact match; OE uses parseability. Results are emitted overall, by question type, and by dataset.
- Five Judges score every test set. GPT-5.4 is the only Judge used for baseline deltas; the other four are used for cross-Judge robustness.

The checked-in formal configuration records the actual world size and effective batch size. Do not change accumulation or world size after launch; a resumed run must use the same configuration.

## Components

- `src/models/MultiAXIS/`: configuration, prompt construction, frozen TimeRCD wrapper, Flash cross-attention, Hint Tuner, and dual-LLM model wrapper.
- `tools/multi_axis/build_manifests.py`: deterministic pairing, filtering, grouped split, source hashes, random-access indices, and the five evaluation manifests.
- `tools/multi_axis/train.py`: five-process manual data parallelism, frozen-encoder group cache, exact answer NLL, 40 epoch checkpoints, and crash resume.
- `tools/multi_axis/infer.py`: five-GPU, group-preserving, crash-resumable generation and deterministic merge.
- `tools/multi_axis/label_metrics.py`: exact MC/TF label metrics and OE parse rate.
- `tools/multi_axis/geval_runner.py`: crash-resilient five-Judge scoring with provider-specific probability handling.
- `tools/multi_axis/aggregate_geval.py`: ten metrics for every dataset × Judge plus micro/macro and cross-Judge statistics.
- `tools/multi_axis/audit_experiment.py`: fail-closed protocol, count, checkpoint, prediction, and Judge audit.
- `tools/multi_axis/report_experiment.py`: GPT-5.4 baseline deltas and the final experiment-analysis Markdown.

## Environment

Use Linux, five visible NVIDIA H100 GPUs, BF16, PyTorch 2.5 or newer, Transformers 4.57 or newer, and FlashAttention compatible with the installed Torch/CUDA ABI. On a preconfigured server, preserve its PyTorch build and install the additional packages selectively instead of allowing pip to replace Torch.

```bash
python -m pip install --upgrade "transformers>=4.57,<5" "openai>=1.99" "google-genai>=1.30" "httpx>=0.28" packaging psutil ninja
python -m pip install --no-build-isolation --no-deps "flash-attn==2.7.4.post1"
```

Before any formal work, verify `torch.cuda.device_count() == 5`, that all devices are H100, BF16 is supported, `flash_attn` imports, and PyTorch Flash SDPA dispatch succeeds. Export `PYTHONPATH` to the repository root and put Hugging Face caches on a filesystem with enough space. For an audited offline snapshot, set `MULTI_AXIS_MODEL_PATH` to its local directory; the registered model ID remains unchanged in the configuration and checkpoint metadata.

## Build immutable manifests

```bash
export PYTHONPATH="$PWD"
python tools/multi_axis/build_manifests.py \
  --data-root /path/to/multi-axis-assets \
  --output-dir /path/to/run/manifests \
  --seed 42 \
  --validation-fraction 0.10
```

This must report 67,820 aligned rows, 17 filtered empty answers, 67,803 retained rows, and zero train/validation group overlap. Evaluation counts must be 478, 200, 184, 12, and 200 with the registered per-type counts. Input SHA-256 hashes are retained unless `--skip-input-hashes` is explicitly used; the formal run must not use that flag.

## Train exactly 40 epochs

Release only the experiment's own verified reservation processes immediately before launch. Never terminate another user's process.

```bash
torchrun --standalone --nproc_per_node=5 tools/multi_axis/train.py \
  --config experiments/multi_axis/formal_deepseek_40epochs.json \
  --data-root /path/to/multi-axis-assets \
  --manifest-dir /path/to/run/manifests \
  --timercd-checkpoint /path/to/pretrain_checkpoint_best_multi.pth \
  --output-dir /path/to/run/training
```

To resume after an interruption, provide `--resume /path/to/run/training/last_train_state.pt`. A valid completion has `TRAINING_COMPLETE`, 40 records in `epochs.jsonl`, 40 hint checkpoints, and `best_checkpoint.json` pointing to the minimum validation answer NLL.

## Five-GPU inference

Read the selected checkpoint filename from `best_checkpoint.json`; do not select by test or Judge scores.

```bash
torchrun --standalone --nproc_per_node=5 tools/multi_axis/infer.py \
  --config experiments/multi_axis/formal_deepseek_40epochs.json \
  --data-root /path/to/multi-axis-assets \
  --manifest-dir /path/to/run/manifests \
  --timercd-checkpoint /path/to/pretrain_checkpoint_best_multi.pth \
  --hint-checkpoint /path/to/run/training/hint_epoch_XX.pt \
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
  --config experiments/multi_axis/formal_deepseek_40epochs.json \
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