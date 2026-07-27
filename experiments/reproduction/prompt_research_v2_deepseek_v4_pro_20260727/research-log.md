# Research log

## 2026-07-27 — Baseline discrepancy audit

The historical `4.25 / 4.32 / ...` Baseline and the later
`4.04 / 4.07 / ...` Baseline were not computed on the same evaluation set.
The historical table used the author-compatible `paper140` manifest
(140 QA, 70 series); the later table used the entire released directory
(284 QA, 142 series).

The current full-284 Baseline contains the exact same 140 paper records. For
those records, question, gold answer, prompt text, prompt hash, and generated
response are identical for 140/140 records across the two runs. Therefore the
large aggregate difference is not caused by a changed checkpoint or Baseline
prompt.

The additional 144 QA are materially harder for MC and TF. A secondary Judge
effect also exists: all 335 paper140 score prompts have identical SHA-256
hashes across the two Judge runs, but only 239/335 sampled raw integer ratings
match. The aggregate paper140 rescore remains close to the historical table.

## 2026-07-27 — Literature refresh

The updated primary-source review adds self-verification, self-refinement,
step-back prompting, self-consistency, and structured-output failure evidence
to the earlier work on RE2, symbol binding, prompt sensitivity, and
time-series prompt prefixes.

The most relevant conclusion for this released 7B checkpoint is not to add a
long universal protocol. The strongest unresolved AXIS failure is evidence
use: many low-scoring answers reject a real contextual anomaly because the
displayed values look smooth, even though `Per-Step Analysis` carries
sample-specific full-series context. Round 1 therefore tests short, balanced
evidence-use rules while retaining the trained scaffold and the
`Overall Summary Hints` token neighborhood.

## 2026-07-27 — Round 1 preregistration

Nine single-family prompt components are frozen in
`experiments/round-1/protocol.md`. They do not repeat the already rejected
Fixed-header replacement, long Evidence Contract, value/local interleaving,
EOS-plus-`Answer:` boundary, universal output schema, or RE2-only modes.

The 144 non-paper records are used as an exposed falsification pool. Unchanged
question families reuse canonical Baseline predictions and Judge scores
exactly. No candidate may reach `paper140` merely because it improves a small
24-QA screen.
