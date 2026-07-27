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

## 2026-07-27 — Round 1 screening24 and outer-loop synthesis

Two A100s on the authorized Port-2225 node produced 684 full-series batch
rows: 409 selected task-family components and 275 support rows. The selected
count exactly matched the preregistered prompt-change calculation. Formal
screening used 164 `deepseek-v4-pro` Judge dimensions: 162 probability-score
rows and two registered exact-20 fallbacks.

Both MC rules improved all three MC metrics, and the whole-statement TF rule
improved TF Final and Justification with effectively unchanged Correctness.
Four OE evidence-use rules raised Accuracy, but all lost Relevance on the
small screen. Failure inspection showed that direct anomaly decisions often
omitted requested boundary/method/subtype coverage. The Fixed-role postnote
and exact TF-prefix rule reduced every changed-family dimension.

The wide screen rejects only those two all-dimension failures. Seven modes
advance to validation72 under the frozen
`experiments/round-1/validation-protocol.md`.

## 2026-07-27 — Round 1 validation72 and outer-loop synthesis

All 398 preregistered validation score dimensions completed under
`deepseek-v4-pro`, including ten registered fallback rows. Both MC components
improve Correctness, Reasoning, and Final on validation72 and pooled
development96. Every OE component loses at least one of Accuracy,
Completeness, or Relevance. The TF whole-statement rule reverses its screening
gain and lowers all three TF metrics on validation72.

Case analysis shows that OE context rules are not selectively correcting
Baseline: they recover some missed anomalies but also overwrite correct
anomaly decisions with false-normal answers. TF losses are dominated by
Boolean polarity binding, where `False` precedes a rationale that says the
negative statement is true. Round 1 therefore retains the two MC components
only; it does not satisfy the required changed-and-improved OE condition.

## 2026-07-27 — `new.md` compatibility protocol launched

The user-supplied `new.md` was decomposed into five frozen interventions:
short compatibility rule, aligned Value-before-Local rows, Question-first,
evidence-last with Fixed-first, and the full document template. The document
itself says the full information-flow design should be paired with Phase-II
retraining, so these released-checkpoint results test compatibility only.
Prompt integrity tests passed before two-GPU screening24 inference began.
