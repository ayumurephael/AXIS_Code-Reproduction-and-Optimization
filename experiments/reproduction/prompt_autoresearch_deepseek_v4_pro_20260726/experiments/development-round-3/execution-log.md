# Development round 3 execution log

Append-only operational record. Candidate routes and the component-reuse policy
were preregistered before implementation.

| Time (Asia/Shanghai) | Event |
|---|---|
| 2026-07-27 02:05 | Preregistered eight routes on the exposed development96 pool. Holdout and paper140 remained untouched. |
| 2026-07-27 02:12 | Implemented Round-3 routing and three new OE-only rules. Local prompt suite passed 28/28; repository-root discovery passed 39/39. |
| 2026-07-27 02:13 | Rendered and hashed 24 candidate/question-type prompt templates. Nine preregistered cross-round component hash-equivalence checks passed. |
| 2026-07-27 02:16 | GPU preflight confirmed GPUs 0–2 free and GPU 3 occupied by an unrelated service. Launched three-mode inference on GPUs 0–2 only. |
| 2026-07-27 02:20 | Formal inference completed 288/288. Remote and local artifacts cover 96 records × three modes with no duplicate or empty response; GPUs were released. |
| 2026-07-27 02:21 | Filtered exactly 87 new OE component predictions and started 261 `deepseek-v4-pro` Judge dimensions with system CA verification, eight primary workers, one exact-20 fallback worker, and journaled bounded retries. |
| 2026-07-27 02:35 | Judge completed 261/261 new OE dimension scores: 260 logprob readouts and one exact-20 fallback. Candidate audit passed with no fatal log events. |
| 2026-07-27 02:36 | Assembled eight complete routes from preregistered components. Combined audit passed for 864 predictions and 1,989 scores. R3-01/02 passed the full metric and paired-decision gate; R3-03/04 passed 10/10 metrics but failed the unresolved correct→wrong guard. |