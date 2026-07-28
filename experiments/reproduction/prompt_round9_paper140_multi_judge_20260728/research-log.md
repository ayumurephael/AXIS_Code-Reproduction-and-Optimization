# Research log

## 2026-07-28 — Protocol lock

The user confirmed evaluation of Baseline plus all three final Round 9 routes
on paper140. Exact extraction from the already-audited full284 prediction
artifact is authorized, so AXIS inference will not be rerun. The Judge matrix
is locked to DeepSeek v4-pro, Qwen3.5-397B-A17B, and
Qwen3-30B-A3B-Instruct-2507; Gemini is excluded.

Scores will be de-duplicated only for byte-identical per-record G-Eval inputs
within a Judge, then expanded back to four complete 335-dimension Table-I
conditions. Cross-Judge reuse and score pooling are prohibited.

## 2026-07-28 — Evaluation complete

The frozen paper140 extraction produced 560 condition rows, 201 unique
responses, and 457 unique score dimensions per Judge. DeepSeek v4-pro and
Qwen3-30B completed under the locked protocols. Qwen3.5 completed 456/457
dimensions under bounded logprob or deterministic readout. The user then
explicitly authorized an exact-20 sampling fallback for the sole unresolved
dimension, `series_000096:0 / base / open_ended accuracy`. Exactly 20 valid
integer scores were collected with `enable_thinking=false`; no other Qwen3.5
dimension was resampled.

All three score sets expanded to 1,340/1,340 dimensions and passed the formal
coverage audit. The experiment-level integrity audit also passed model,
provider, thinking-mode, prompt-hash, fallback-size, Qwen missing-mass-bound,
credential-scan, and artifact-hash checks. Relevant tests passed 19/19.

No route met the all-ten non-degradation criterion under all three Judges.
Round 9 TF+OE passed 10/10 with three strict improvements under Qwen3-30B,
but failed under DeepSeek because TF justification decreased and under
Qwen3.5 because TF correctness decreased. MC routing reduced MC Final and
Correctness under all three Judges. The paper140 subset contains no adopted
OE repair, so every OE score is exactly equal to that Judge's Baseline.
