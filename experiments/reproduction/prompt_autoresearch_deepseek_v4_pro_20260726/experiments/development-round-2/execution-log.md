# Development round 2 execution log

Append-only operational record. The preregistered prompts, data, scoring, and
selection gate are unchanged.

| Time (Asia/Shanghai) | Event |
|---|---|
| 2026-07-26 22:24 | Local targeted prompt suite passed 24/24; repository-root discovery passed 35/35. A child-directory discovery invocation produced 12 expected relative-import errors and was discarded because it did not run tests as a package. |
| 2026-07-26 22:26 | Remote source upload and SHA-256 comparison passed. The inference environment lacks optional `pytest`, so remote verification used standard-library `unittest` and passed 24/24 without installing or changing dependencies. |
| 2026-07-26 22:27 | GPU preflight confirmed GPUs 0–2 free and GPU 3 occupied by an unrelated service. Only GPUs 0–2 were selected. |
| 2026-07-26 22:28 | First launcher attempt exited with zero predictions. Root cause: the launcher supplied manifest key `record_ids`, while `run_inference.py` filters `record.series_file`; therefore the filtered work list was empty. The failed log was archived as `inference_manifest_key_failed.log`. |
| 2026-07-26 22:30 | Corrected only the launcher key to the manifest's `series_files`; no prompt, record list, model, or evaluation setting changed. Relaunched from an empty output directory. Formal inference entered generation on three GPUs. |
| 2026-07-26 22:32 | Cached Baseline merge audit passed: 96 predictions and 221 scores; 220 logprob readouts and one exact-20 fallback; model exactly `deepseek-v4-pro`. |
| 2026-07-26 22:46 | Formal inference completed 576/576 predictions. Remote and local audits both passed: 96 records, six modes × 96, no duplicate keys or empty responses. GPU memory was released after completion. |
| 2026-07-26 22:47 | Started 1,326 `deepseek-v4-pro` Judge dimension tasks on the same approved GPU server, with system CA verification, eight primary workers, one fallback worker, and exact-20 fallback only. |
| 2026-07-27 01:52 | Judge completed 1,326/1,326 dimension scores: 1,312 final-score logprob readouts and 14 exact-20 fallbacks. The process exited normally after a transient-network-safe journaled run. |
| 2026-07-27 01:54 | Remote candidate audit and local combined Baseline/candidate audit passed. The combined artifact contains 672 predictions and 1,547 scores with valid prompt hashes and exactly `deepseek-v4-pro`. A missing optional remote diagnostics module caused one aggregation wrapper exit after the core audit; local diagnostics already existed, and rerunning only the audit/metadata step succeeded without changing predictions or scores. |
| 2026-07-27 01:55 | No candidate passed the 10/10 gate. Best: `route_r2_04_oe_old_contract`, 7/10, worst -0.381, mean +0.205. Holdout remained untouched. |