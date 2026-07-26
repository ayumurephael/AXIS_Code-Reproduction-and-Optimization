# Internal holdout execution log

| Date (Asia/Shanghai) | Event |
|---|---|
| 2026-07-27 | Started from preregistered commit `b48c5cd`; candidate prompts, ranking rule, released checkpoint, generation settings, and holdout manifest were frozen. |
| 2026-07-27 | The first three-GPU launch passed preflight, but an unrelated process occupied GPU 0 during checkpoint loading. Rank 0 exited with CUDA OOM before producing any prediction. The zero-prediction log and preflight were retained. |
| 2026-07-27 | Before a documented two-GPU fallback was launched, GPUs 0–2 became free. Retried the unchanged original three-GPU protocol in a fresh output directory. |
| 2026-07-27 | Completed 144/144 predictions: 48 Baseline, 48 R3-02, and 48 R3-01. Prediction audit passed with 48 unique records, no duplicate keys, no empty responses, series batching, world size 3, and `skip_loss=true`. |
| 2026-07-27 | Constructed the preregistered selective Judge set: all 48 Baseline answers plus the 17 MC answers from each candidate, for 82 Judge inputs and 179 dimensions. Verified exact counts and unique keys before launch. |
| 2026-07-27 | Started formal `deepseek-v4-pro` scoring with eight primary workers, one exact-20 fallback worker, 4096 maximum output tokens, and at most 12 retry rounds. Partial scores were not used for candidate selection. |
| 2026-07-27 | Completed 179/179 dimensions: 178 final-score probability readouts and one exact-20 fallback. Baseline and both candidate-MC component audits passed; no fatal Judge log entry was present. |
| 2026-07-27 | Assembled 144 full predictions and 333 full scores with exact Baseline OE/TF reuse. Final audit passed with no missing/duplicate key, provider/model mismatch, invalid prompt hash, or unapproved method. |
| 2026-07-27 | Both candidates failed the confirmation gate and each introduced one MC correct→wrong decision. No candidate was locked; no holdout-driven tuning or paper140 candidate run was performed. |
