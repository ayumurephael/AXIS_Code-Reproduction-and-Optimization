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

## 2026-07-27 — `new.md` supplemental compatibility results

All five `new.md` modes completed GPU inference and 275
`deepseek-v4-pro` Judge dimensions. None passed the screening24
all-ten-nonlower gate. P1 and P2 improved some MC dimensions, and P2 improved
OE Completeness, but both reduced OE Relevance substantially and changed some
real anomalies to false-normal. Question-first, evidence-last, and the full
template reduced every task family.

The result is scoped to the released checkpoint. Textual row alignment does
not create learned alignment, and question-first cannot alter the already
computed Local representations without the proposed Phase-II retraining.
According to the frozen protocol, no Local corruption/swap test is run because
there is no performance survivor.

## 2026-07-27 — Round 3 main holdout48 and outer-loop synthesis

Round 3 evaluated the two Round-1 MC survivors and three increasingly narrow
OE balanced-evidence routers. All 74 planned holdout Judge dimensions
completed. The two MC modes, despite gains on screening24, validation72, and
pooled development96, lost all three MC metrics on holdout48. The broad and
boundary OE routers lost `0.1333` on both Accuracy and Completeness; the
assessment router lost `0.0667` on each.

Failure inspection rejected the routing premise. Lexical matches for
“support/refute” and “boundary” included records whose Baseline already
covered every requested part; the extra rule introduced an open `<think>`
prefix, dropped boundary-specific conclusions, or added speculative trends.
The next OE round will target answer coverage and relevance without imposing
an anomaly-status prior or claiming that text decodes learned Local vectors.

## 2026-07-27 — Rounds 4–6 outer loop

Round 4 tested five OE coverage-only prompts and found no validation survivor.
Round 5 moved to a frozen-Baseline two-pass design; `verbatim_or_add` was the
closest mode but lost Completeness/Relevance on holdout48. Round 6 tested
three A/B selectors, but the released checkpoint frequently ignored the
selection meta-task. These results ruled out universal coverage reminders,
unconditional second-pass rewriting, and checkpoint self-selection.

## 2026-07-27 — Rounds 7–9 deterministic repair loop

Round 7 routed only malformed OE responses with an unclosed `<think>` and no
Answer boundary. It fixed `series_000111:0` but harmed `series_000141:0`.
Round 8 added explicit no-anomaly verdict preservation; it passed the exposed
views but selected four full284 repairs, three of which deleted requested
content and lowered Completeness/Relevance.

Round 9 added a 60% word-retention guard. It was designed after inspecting the
Round-8 full284 failures and is therefore labeled post-hoc. On full284 it
selects only `series_000111:0`.

## 2026-07-27 — Locked full284 formal result

The final component union uses:

- MC: `lit_r5_02_mc_status_then_shape`;
- TF: `lit_r4_01_tf_neg_re2`, activated only by explicit negative cues;
- OE: `v2_r5_02_oe_verbatim_or_add`, accepted only by the Round-9
  deterministic gate.

Authorized Port-2225 GPUs generated all new responses. Formal
`deepseek-v4-pro` scoring completed for the 277 unique changed dimensions;
unchanged answers reused the canonical Baseline score exactly. Three routes
pass the user's final criterion:

- `prompt_final_round9_mc_oe`: 10/10 non-lower, six strict gains;
- `prompt_final_round9_tf_oe`: 10/10 non-lower, six strict gains;
- `prompt_final_round9_joint`: 10/10 non-lower, nine strict gains.

The joint route changes 94 MC, 43 TF, and one OE response. A secondary paired
Judge repeat was launched for all 138 changed records (276 predictions,
554 dimensions) to quantify Judge variance without replacing the locked
primary result.

## 2026-07-28 — Paired Judge-repeat audit complete

The paired repeat completed 554 dimensions for 276 Baseline/candidate
predictions: 552 top-logprob expectations and two exact-20 fallbacks. All
rows used `deepseek-v4-pro`, provider `deepseek`, with nonempty prompt hashes.

MC and TF deltas remained positive. The only OE repair had OE Final `+0.0010`,
Accuracy `+0.0202`, Completeness tied at four decimals, and Relevance
`-0.0202`. Thus the MC+OE and TF+OE routes each had five strict gains and the
joint route had eight, but every route was 9/10 rather than 10/10 in this
secondary repeat.

The locked primary formal result remains the declared Table-I result. The
claim is narrowed to locked-primary engineering PASS; repeat-robust full
non-degradation is not established.
