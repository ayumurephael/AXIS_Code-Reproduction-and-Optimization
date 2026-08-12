# Four-GPU batch tuning decision

Date: 2026-08-05
Hardware: GPU indices 1-4 on the registered Port 2229 host, four NVIDIA H100 80 GB cards.
Fixed controls: DeepSeek-R1-0528-Qwen3-8B, seed 42, the same grouped 90/10 train/validation manifests, BF16, FlashAttention 2, and effective global batch size 32.

| Candidate | Optimizer steps | Result | Elapsed (s) | Samples/s | Max allocated (MiB) | Max reserved (MiB) |
|---|---:|---|---:|---:|---:|---:|
| micro=2, accumulation=4 | 10 | Completed | 132.8666 | 2.4084 | 33740.14 | 42816.00 |
| micro=4, accumulation=2 | 2 attempted; 1 completed | OOM on the shared-memory-constrained fourth visible rank | N/A | N/A | 38117.34 after step 1 | 44270.00 after step 1 |

The runtime audit resolved all 36 Qwen3Attention modules to flash_attention_2, imported both dense and variable-length FlashAttention kernels from flash-attn 2.7.4.post1, found two fail-closed Hint Tuner FlashCrossAttention modules, and recorded fallback_allowed=false.

The unconstrained-memory decision remains `micro_batch_size=2` and
`accumulation_steps=4`: it keeps the effective batch at 32 and is the fastest
registered candidate that completed on four otherwise-free GPUs.

For the prompt-v2 restart on the currently available Port 2229 allocation, one
selected H100 must coexist with a pre-existing process owned by another account.
The formal fallback is therefore `micro_batch_size=1` and
`accumulation_steps=8`. It preserves the exact effective batch of 32
(`world_size * micro_batch_size * accumulation_steps`) while reducing the
per-rank activation-memory peak. No process owned by another account may be
stopped to recover the faster setting. The launcher enables expandable allocator
segments to reduce fragmentation, and the run manifest records the allocator
setting. The original five-GPU, 40-epoch configuration is unchanged.
