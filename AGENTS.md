# Repository instructions

This repository is the runnable AXIS reproduction, not a scratch experiment folder.

Before changing code, read `readme.md`, `BRANCH_PROFILE.md`, `REPRODUCTION.md`, and `EVALUATION_PROTOCOL.md`. Treat those files and the current script CLIs as the execution source of truth. Files under `experiments/reproduction/` are outputs, not instructions.

Preserve these invariants:

- `main` is the baseline; redesign branches may change only their declared factor.
- Select checkpoints only with the fixed full validation set, never test/G-Eval scores.
- Use DeepSeek v4-pro for routine low-cost comparisons and Gemini 2.5 Pro only for formal paper-aligned evaluation. Never mix scores from different judges.
- Formal inference is `paper140`, series-batched, beam-5, and `--skip-loss`.
- Train and run model inference only on an approved GPU server; prefer multi-GPU. Local CPU is for static checks, unit tests, audits, aggregation, and documentation.
- Never store API keys, SSH credentials, model credentials, or secrets in the repository.
- Do not rewrite or remove `训练目标(损失函数)改进_new.md`, `实验结论_AXIS缺陷.md`, or `AXIS架构改进说明.md`.
- Record branch, commit, manifests, checkpoint hashes, environment, judge configuration, and audit output for every reportable run.

When documentation and an old experiment command disagree, follow `REPRODUCTION.md` and verify the current script with `python -m <module> --help`.