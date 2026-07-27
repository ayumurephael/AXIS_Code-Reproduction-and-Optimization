# Rounds 5–9 outer-loop analysis

## Round 5 — conservative two-pass OE refinement

Five modes reused a frozen Baseline draft and asked the released checkpoint to
preserve its anomaly/normal verdict while filling requested information.
`v2_r5_02_oe_verbatim_or_add` was closest:

| View | OE Final Δ | Accuracy Δ | Completeness Δ | Relevance Δ |
|---|---:|---:|---:|---:|
| screening24 | +0.0571 | +0.0000 | +0.2857 | -0.1429 |
| validation72 | +0.0080 | +0.0727 | -0.0455 | -0.0051 |
| pooled development96 | +0.0199 | +0.0552 | +0.0345 | -0.0383 |
| holdout48 | -0.1585 | +0.2000 | -0.1500 | -0.5867 |

The verdict constraint reduced bidirectional anomaly flips but did not prevent
the second pass from removing boundary analysis, comparisons, methods, or
counterevidence. Exact response-score reuse was used whenever the second pass
returned the Baseline verbatim.

## Round 6 — checkpoint A/B selection

Three selectors compared Candidate A (Baseline) with Candidate B (Round-5
revision). On screening the first two selected A for every OE answer; the
content guard selected B once. On validation the content guard produced
OE Final `+0.0136`, Accuracy effectively tied at four decimals, Completeness
effectively tied, and Relevance `+0.0455`. On holdout its deltas were
`+0.0008 / +0.0667 / +0.0500 / -0.1333`.

Failure inspection showed that the fine-tuned checkpoint often ignored the
selection instruction and answered the original anomaly question. The sparse
B decisions were therefore not a reliable self-evaluation mechanism.

## Round 7 — malformed-answer boundary repair

The deterministic gate selected a Round-5 revision only when the Baseline had
an unclosed `<think>`, no closing tag, and no explicit Answer line, while the
candidate removed the tags and added an Answer boundary. It selected two
holdout OE responses:

- `series_000111:0`: improved Accuracy and Completeness;
- `series_000141:0`: retained a false anomaly conclusion and removed requested
  boundary analysis.

Holdout aggregate OE deltas were Final `-0.0425`, Accuracy `+0.0667`,
Completeness `-0.0167`, Relevance `-0.2000`.

## Round 8 — no-anomaly verdict guard

Round 8 additionally required both answers to explicitly preserve a
no-anomaly conclusion. It selected only `series_000111:0` on holdout and
passed there. On full284 it selected four responses. Three candidates retained
only 40.0%, 58.1%, and 48.2% of Baseline word tokens and lost requested
content. Full284 OE deltas were Final `-0.0030`, Accuracy `+0.0303`,
Completeness `-0.0303`, Relevance `-0.0099`.

## Round 9 — 60% content-retention guard

Round 9 retained every Round-8 condition and required candidate word count to
be at least 60% of Baseline word count. It was designed after inspecting the
Round-8 full284 failures and is therefore post-hoc. It selected only
`series_000111:0` on full284:

- OE Final `+0.0070706`;
- Accuracy `+0.0101007`;
- Completeness `+0.0101010`;
- Relevance `+0.00000001` (tie at four decimals).

## Final task-family combination

The exposed-split-qualified status-first MC component improves all three MC
metrics on full284. The explicit-negative-cue qualitative TF RE2 component
improves all three TF metrics. Combining either or both with Round-9 OE yields
three final routes:

| Route | All ten non-lower | Strict gains |
|---|---:|---:|
| `prompt_final_round9_mc_oe` | yes | 6 |
| `prompt_final_round9_tf_oe` | yes | 6 |
| `prompt_final_round9_joint` | yes | 9 |

The locked primary formal result uses `deepseek-v4-pro` only. A paired repeat
of the 138 changed Baseline/candidate records is a secondary Judge-variance
audit and does not replace the primary result.

### Paired Judge-repeat limitation

The secondary paired repeat re-judged all 138 changed Baseline/candidate
records in one run. MC and TF remained positive on every component metric.
The single OE repair improved Final and Accuracy, tied Completeness at four
decimals, but lowered Relevance by `-0.0202`. Consequently all three routes
were 9/10 non-lower rather than 10/10 in the repeat.

This does not overwrite the locked primary formal result, but it changes the
strength of the claim. The Round-9 routes are locked-primary engineering
passes, not Judge-repeat-robust passes. Because OE changes only one of 99
answers, a one-point or two-point Judge change on that record moves the OE
mean by about 0.01–0.02. Future confirmation needs either a larger set of
independently improved OE cases or an untouched set with repeated paired
judging.
