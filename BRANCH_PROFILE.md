# Branch profile: multi-axis-VL-32B

This branch is derived exactly from `multi-axis-VL` commit
`ab2efb872b6b43e8afadc14d80a45c8b2ebd0d7e` and replaces only the frozen VLM
backbone contract with pinned `Qwen/Qwen3-VL-32B-Instruct` revision
`0cfaf48183f594c314753d30a4c4974bc75f3ccb`.

## Non-negotiable experiment identity

- Start the 32B Hint Tuner at Epoch 0. Never load, migrate, or partially reuse an
  8B Hint Tuner, optimizer state, scheduler state, or question-semantic cache.
- Freeze the complete 32B VLM and TimeRCD. Only the documented Step/Channel/
  question-conditioned Joint/Fixed Hint Tuner parameters are trainable.
- Use text hidden size 5120, vocabulary size 151,936, 64 text decoder layers,
  and 20 prototype-attention heads (`head_dim=256`).
- The three Hint Tuner cross-attention modules and native VLM attention must all
  resolve to FlashAttention-2; there is no eager fallback.
- Keep the grouped seed-42 90/10 split, 20 epochs, effective batch 32, 2048-token
  visual budget (`runtime_max_pixels=2,097,152`), and validation answer NLL
  checkpoint selection.

## Hardware is deliberately unresolved

The company allocation is unknown until connection through its jump host. The
included eight-GPU JSON is a conservative topology template, not a finalized
launch decision. Run `tools/multi_axis/hardware_preflight.py` and short formal
optimizer-step benchmarks on the allocated compute nodes first. The current
training implementation replicates the ~67 GB BF16 frozen VLM on every rank;
therefore an 80 GB nameplate alone is not evidence that training will fit.

Do not install dependencies, download weights, or train on the jump host. Keep
all credentials outside the repository and logs.
