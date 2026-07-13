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
- Phase II: released Phase-I encoder, fresh Hint Tuner, frozen encoder/LLM, 95/5 series split,
  seed 72, AdamW, LR `1e-4`, weight decay `1e-5`, three epochs.
- Formal training used `train_phase2_memory_safe_v3` on 3 x A100 40GB. It processed 28,500
  synchronized training steps in 20,497.7 s (5.694 h).
- All three checkpoints were inferred on the same 1,500-series validation set (3,000 QA).
  Checkpoint selection is fail-closed and uses minimum mean teacher-forced validation loss.
  Epoch 3 is best overall and separately on MC, OE, and TF.
- Generation matches the released per-record path: beam size 5 and `max_new_tokens=1000`.
  Series-batched generation is diagnostic only because it changes responses.
- The formal checkpoint is `epoch_3_inference.pth`, SHA-256
  `8d562f8ff22c6709caaa2f7ba8208f3631a652d1932f001f39fa75ff8700fa4b`.
- G-Eval uses the exact Appendix E.4 output format and Appendix E.2 dimensions/weights with the
  requested `deepseek-v4-pro` judge, high thinking, temperature 0, and `top_logprobs=20`.
  The score is the probability-weighted mean over 1-5 at the final score position. If all five
  score tokens are unavailable, exactly 20 valid temperature-1 samples are averaged.
- Formal judge calls use `max_tokens=4096`. The earlier 1,200-token limit often truncated high
  thinking before `**Score:**`; those truncated calls are diagnostic and are never interpreted as
  a need for 20-sample fallback.
- Training, inference, free metrics, and API/G-Eval are executed on GPU servers. A local process
  may provide a loopback-only SSH proxy and file transfer, but it does not run evaluation.
- Credentials are read into process scope from a protected file and are never stored in commands,
  manifests, predictions, scores, or documentation.

The paper used Gemini-2.5, while this reproduction uses DeepSeek-v4-pro. Score differences are
therefore not evidence of model non-reproduction by themselves; generations and checkpoint hashes
are reported separately from judge scores.

## Core commands

Run these commands from `AXIS/baseline` on the GPU server.

```bash
# Formal 3-GPU Phase-II training
torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_memory_safe_v3 \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --output experiments/reproduction/phase2_torch251

# Select the checkpoint from the three complete validation prediction files
python -m tools.axis_repro.select_best_loss \
  experiments/reproduction/val_epoch_1/predictions.jsonl \
  experiments/reproduction/val_epoch_2/predictions.jsonl \
  experiments/reproduction/val_epoch_3/predictions.jsonl \
  --output experiments/reproduction/phase2_torch251_transfer/best_validation_loss.json

# Per-record test generation with the selected trained checkpoint
torchrun --standalone --nproc_per_node=2 \
  -m tools.axis_repro.run_inference_cli \
  --checkpoint experiments/reproduction/phase2_torch251/epoch_3_inference.pth \
  --data data/AXIS_qa_test --subset full --modes base \
  --output experiments/reproduction/test_epoch_3_full284

# Strict Appendix-E G-Eval on the paper140 file
python -m tools.axis_repro.run_with_secret \
  --credential-file /secure/path/credential.md \
  python -m tools.axis_repro.geval_resilient \
  --predictions experiments/reproduction/test_epoch_3_paper140/predictions.jsonl \
  --output experiments/reproduction/test_epoch_3_paper140/geval.jsonl \
  --model deepseek-v4-pro --fallback-samples 20 --max-tokens 4096 \
  --primary-workers 8 --fallback-workers 1

# Fail-closed integrity check and Table 1 aggregation
python -m tools.axis_repro.audit_results \
  --predictions experiments/reproduction/test_epoch_3_paper140/predictions.jsonl \
  --scores experiments/reproduction/test_epoch_3_paper140/geval.jsonl \
  --manifest experiments/reproduction/manifests/paper140.json --modes base
python -m tools.axis_repro.table_runner \
  --scores experiments/reproduction/test_epoch_3_paper140/geval.jsonl \
  --output-prefix experiments/reproduction/test_epoch_3_paper140/table1
```

Final numerical results, hashes, audits, and paper-vs-reproduction deltas are recorded in
`REPRODUCTION_RESULTS.md`. Files marked `diagnostic`, `partial`, or belonging to the interrupted
validation G-Eval journals are never used in Table 1.
