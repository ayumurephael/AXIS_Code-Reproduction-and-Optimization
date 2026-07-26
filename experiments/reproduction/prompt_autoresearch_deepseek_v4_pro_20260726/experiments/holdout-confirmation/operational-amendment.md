# Holdout operational amendment

Recorded before any holdout prediction was produced.

The preregistered three-GPU launch passed its preflight with GPUs 0–2 free.
During checkpoint loading, an unrelated process began using about 34 GiB on
GPU 0. Rank 0 then failed while moving the model to CUDA; the output contained
zero predictions. The unrelated process was not terminated or disturbed.

A read-only check of every other authorized node initially found no set of
three GPUs that could be safely used without sharing with active workloads.
A two-GPU fallback was documented, but **was not launched**: before the retry
began, the unrelated process exited and GPUs 0–2 were all free again.

The actual rerun therefore preserves the original three-GPU protocol:

- new empty output directory and separate log/PID/preflight files;
- `CUDA_VISIBLE_DEVICES=0,1,2`, `nproc_per_node=3`;
- otherwise unchanged released checkpoint, modes, holdout manifest, series
  batching, beam size 5, and `--skip-loss`;
- unchanged `deepseek-v4-pro` Judge and component-reuse plan.

No data-parallel setting ultimately changed. Candidate prompts, selection
rules, records, and scoring remain frozen. The failed zero-prediction attempt
is retained for audit.
