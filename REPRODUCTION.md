# AXIS Table 1 baseline reproduction

This directory is the runnable baseline copied from `AXIS_original_codes` and completed for
architecture research. The formal target of this delivery is the **AXIS row of Table 1**. The
Table 2 ablation runners remain available, but Table 2 is outside the current result scope and
must not be inferred from partial files.

`paper140` reproduces the released `AXIS_test.py` coverage (70 series / 140 QA). `full284`
evaluates every released test QA (142 series / 284 QA). The formal Table 1 row is built from
`paper140`; `full284` is retained as a coverage and diagnostic set.

## Authoritative protocol

- Runtime: PyTorch 2.5.1, Transformers 4.45.2, NumPy 1.26.4. PyTorch 2.1/2.2 diagnostics are
  excluded because their generation behavior is not equivalent.
- Historical low-budget Phase II used the released Phase-I encoder, fresh Hint Tuner,
  frozen encoder/LLM, a 95/5 series split, seed 72, AdamW, LR 1e-4, weight decay
  1e-5, and three epochs. Author-compatible training should explicitly cover up to
  35 epochs with seed 42 and select only by full-validation loss; see
  AUTHOR_COMPATIBLE_TRAINING.md.
- Historical training used `train_phase2_memory_safe_v3` on 3 x A100 40GB. It processed 28,500
  synchronized training steps in 20,497.7 s (5.694 h).
- The three historical checkpoints were inferred on the same 1,500-series validation set (3,000 QA).
  Checkpoint selection is fail-closed and uses minimum mean teacher-forced validation loss.
  Epoch 3 is best overall and separately on MC, OE, and TF.
- Formal test generation matches the author's released AXIS_test.py: series-batched,
  beam size 5, max_new_tokens=1000, and no teacher-forced loss. Per-record generation
  is retained only as a controlled diagnostic and for validation-loss workflows.
- The historical checkpoint is `epoch_3_inference.pth`, SHA-256
  `8d562f8ff22c6709caaa2f7ba8208f3631a652d1932f001f39fa75ff8700fa4b`.
- Formal author-compatible G-Eval uses Gemini 2.5 Pro with the exact author prompt,
  rubrics, dimensions, and weights. PackyAPI currently returns no logprobs for this
  model, so the reproducible author-code path is one temperature-0 integer score.
  Strict 20-sample means are calibration diagnostics and are never mixed into the
  formal table.
- Training, inference, free metrics, and API/G-Eval are executed on GPU servers. A local process
  may provide a loopback-only SSH proxy and file transfer, but it does not run evaluation.
- Credentials are read into process scope from a protected file and are never stored in commands,
  manifests, predictions, scores, or documentation.

The historical table used DeepSeek-v4-pro and remains preserved as a judge-control
artifact. The author-compatible table uses Gemini 2.5 Pro. Judge scores, generations,
checkpoint hashes, prompt hashes, and the logprobs-availability flag are reported
separately.

## Core commands

Run these commands from `AXIS/baseline` on the GPU server.

```bash
# Author-compatible long-budget Phase-II training
torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_memory_safe_v3 \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --output experiments/reproduction/phase2_author_compatible_seed42 \
  --epochs 35 --seed 42 --lr 1e-4 --weight-decay 1e-5

# Historical three-epoch checkpoint-selection example
python -m tools.axis_repro.select_best_loss \
  experiments/reproduction/val_epoch_1/predictions.jsonl \
  experiments/reproduction/val_epoch_2/predictions.jsonl \
  experiments/reproduction/val_epoch_3/predictions.jsonl \
  --output experiments/reproduction/phase2_torch251_transfer/best_validation_loss.json

# Author-compatible test generation with the validation-selected checkpoint
torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.run_inference_cli \
  --checkpoint <validation-selected-checkpoint.pth> \
  --data data/AXIS_qa_test --subset paper140 --modes base \
  --batching series --skip-loss \
  --output experiments/reproduction/test_author_compatible_paper140

# Author-compatible Gemini scoring (key is supplied only via environment)
python -m tools.axis_repro.geval_gemini \
  --predictions experiments/reproduction/test_author_compatible_paper140/predictions.jsonl \
  --output experiments/reproduction/test_author_compatible_paper140/geval_gemini25pro.jsonl \
  --model gemini-2.5-pro --prompt-template author --scoring-mode author

# Fail-closed integrity check and Table 1 aggregation
python -m tools.axis_repro.audit_results \
  --predictions experiments/reproduction/test_author_compatible_paper140/predictions.jsonl \
  --scores experiments/reproduction/test_author_compatible_paper140/geval_gemini25pro.jsonl \
  --manifest experiments/reproduction/manifests/paper140.json --modes base
python -m tools.axis_repro.table_runner \
  --scores experiments/reproduction/test_author_compatible_paper140/geval_gemini25pro.jsonl \
  --output-prefix experiments/reproduction/test_author_compatible_paper140/table1
```

Final numerical results, hashes, audits, and paper-vs-reproduction deltas are recorded in
`REPRODUCTION_RESULTS.md`. Files marked `diagnostic`, `partial`, or belonging to the interrupted
validation G-Eval journals are never used in Table 1.
