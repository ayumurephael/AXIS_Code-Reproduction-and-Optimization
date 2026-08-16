# Four-GPU batch tuning decision

Date: 2026-08-05
Hardware: GPU indices 1-4 on the registered Port 2229 host, four NVIDIA H100 80 GB cards.
Fixed controls: DeepSeek-R1-0528-Qwen3-8B, seed 42, the same grouped 90/10 train/validation manifests, BF16, FlashAttention 2, and effective global batch size 32.

| Candidate | Optimizer steps | Result | Elapsed (s) | Samples/s | Max allocated (MiB) | Max reserved (MiB) |
|---|---:|---|---:|---:|---:|---:|
| micro=2, accumulation=4 | 10 | Completed | 132.8666 | 2.4084 | 33740.14 | 42816.00 |
| micro=4, accumulation=2 | 2 attempted; 1 completed | OOM on the shared-memory-constrained fourth visible rank | N/A | N/A | 38117.34 after step 1 | 44270.00 after step 1 |

The runtime audit resolved all 36 Qwen3Attention modules to flash_attention_2, imported both dense and variable-length FlashAttention kernels from flash-attn 2.7.4.post1, found two fail-closed Hint Tuner FlashCrossAttention modules, and recorded fallback_allowed=false.

On four completely free H100s, `micro_batch_size=2` and
`accumulation_steps=4` remains the fastest registered candidate at effective
batch 32.

The first prompt-v2 attempt had to fall back to `micro_batch_size=1` and
`accumulation_steps=8` while one selected H100 coexisted with a pre-existing
process owned by another account. That run measured a median 41.8 seconds per
optimizer step (about 22.17 hours per epoch), compared with 15.0 seconds and
about 7.95 hours per epoch for the previous `micro=2, accumulation=4` run.

The main prompt-v2 regression was redundant serialization: every value/hint
pair repeated its global index, `t=`, `value=`, punctuation, and a line break.
The compact representation retains the exact half-open interval, channel IDs,
ordered value/hint pairs, and a lossless offset-to-global-time mapping while
removing those repeated tokens. A later formal Channel-Local + Joint-Local run
demonstrated that the shared fourth rank can still encounter a larger dynamic
batch: after global step 1060 it required another 932 MiB while only about
840 MiB was free. The failure occurred during backward with finite loss and
gradients; it was a capacity failure, not divergence.

Therefore the checked-in shared-allocation formal profile is
`micro_batch_size=1`, `accumulation_steps=8`, preserving global batch 32. This
is the fail-closed profile whenever one selected H100 coexists with the
pre-existing process owned by another account. No process owned by another
account may be stopped. The original five-GPU, 40-epoch configuration is
unchanged.
