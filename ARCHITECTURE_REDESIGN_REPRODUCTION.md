# AXIS architecture redesign reproduction

This branch extends the source-likelihood-ratio Phase-II objective with three
independent architecture factors. The formal model uses all three factors and is
trained from the same released Phase-I time-series encoder with a fresh Hint
Tuner. It is not warm-started from the loss-only Phase-II checkpoint.

## Formal architecture

1. **QK-Norm on prototype cross-attention only.** Q and K are split into heads,
   L2-normalized along the head dimension, and scored as
   `softmax(g * Q_hat K_hat^T) V`. V is never normalized. Each
   `MultiheadAttention` module owns one learnable scalar `g`, shared by all
   heads. Rank 0 computes `L = ceil(P97.5)` over Local hint lengths in the
   seed-72 28,500-series training split and initializes
   `g = log2(L^2 - L)`; the integer is broadcast to all three ranks and stored
   in checkpoint metadata.

2. **Continuous bypass with prototype sidecar.** For each selected encoder state
   `H_t`, the direct path is `C_t = W_c H_t`. The prototype path remains
   `P_t = Attn(W_local H_t, S_proto, S_proto)`. A vector gate and residual
   fusion produce
   `LN(C_t + sigmoid(W_g [C_t; P_t]) * P_t)`. `W_g` is zero-initialized with
   bias -2, so training starts from a continuous-path-dominant state while
   retaining a trainable semantic sidecar.

3. **Direct task soft prompt.** The 30 fixed/task positions are filled by one
   directly learned `[1, 30, d_llm]` parameter. The old fixed-query-to-prototype
   attention path is not instantiated in the formal model, preventing unused
   trainable parameters under DDP.

The LLM and Phase-I encoder remain frozen. Only the redesigned Perceiver/Hint
Tuner is optimized. The objective remains full-answer NLL plus source-specific
likelihood-ratio supervision with beta 1 and margin ln(2).

## Factorial ablation presets

The CLI option `--architecture-variant` supports all 2x2x2 combinations:

| Variant | QK-Norm | Continuous bypass + sidecar | Direct task prompt |
| --- | ---: | ---: | ---: |
| `loss_only` | 0 | 0 | 0 |
| `qk_only` | 1 | 0 | 0 |
| `bypass_only` | 0 | 1 | 0 |
| `task_prompt_only` | 0 | 0 | 1 |
| `qk_bypass` | 1 | 1 | 0 |
| `qk_task_prompt` | 1 | 0 | 1 |
| `bypass_task_prompt` | 0 | 1 | 1 |
| `full` | 1 | 1 | 1 |

The current authorized formal run is `full` only. The other presets are
implemented for later ablation runs and do not authorize extra GPU training.

## Formal three-GPU training

Run only on the approved GPU server, with GPUs 0, 1, and 2 when they are free:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_loss_redesign \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --counterfactual-index experiments/reproduction/loss_redesign/counterfactual_index.json \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/architecture_redesign/full/phase2 \
  --architecture-variant full \
  --epochs 3 --lr 1e-4 --weight-decay 1e-5 --seed 72 \
  --num-workers 2 --save-every 5000 --beta 1.0
```

Do not pass `--qk-norm-seq-len` for formal training: it must be calculated from
the fixed training split. Read the resulting integer from
`reproduction_meta.architecture.qk_norm_seq_len` and pass that exact value to
all validation and test inference commands.

## Selection and evaluation

1. Strip optimizer state from all three epoch checkpoints. The stripping helper
   preserves `reproduction_meta`.
2. Run complete 1,500-series / 3,000-QA teacher-forced validation for epochs
   Audit every stripped checkpoint before inference:
   ```bash
   python -m tools.axis_repro.audit_architecture_checkpoint \
     --checkpoint epoch_1_inference.pth --expected-variant full
   ```
   Repeat for epochs 2 and 3.

   1, 2, and 3 on three GPUs, always with:
   `--architecture-variant full --qk-norm-seq-len <training-L> --gate-bias -2`.
3. Audit each validation output and select minimum overall mean teacher-forced
   NLL without using test or judge scores.
4. Run per-record full284 inference using the selected checkpoint, filter the
   fixed paper140 manifest, and audit coverage.
5. Run strict Appendix-E DeepSeek-v4-pro G-Eval and Table 1 aggregation.

Training, validation, full284 inference, and every DeepSeek-v4-pro G-Eval API
request must run on the approved GPU server. The local CPU machine is used only
for code, unit tests, secure orchestration, and artifact transfer. Credentials
are injected into the remote judge subprocess from a protected file and must
never enter commands, logs, checkpoints, manifests, or the repository.
