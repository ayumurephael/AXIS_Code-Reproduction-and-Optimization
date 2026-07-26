# Development round 3 analysis

## Outcome

All eight preregistered routed candidates were assembled from audited
question-type components. The combined fail-closed audit passed for 864
predictions and 1,989 `deepseek-v4-pro` dimension scores. Four candidates met
the numerical 10/10 Table-I gate:

| Rank | Mode | Nonnegative | Worst | Mean delta | Decision flips |
|---:|---|---:|---:|---:|---|
| 1 | `route_r3_04_mc_f1_tf_p0` | 10/10 | 0.000 | +0.202 | 1 correct→wrong, 18 wrong→correct |
| 2 | `route_r3_03_mc_f0_tf_p0` | 10/10 | 0.000 | +0.192 | 1 correct→wrong, 17 wrong→correct |
| 3 | `route_r3_02_mc_f1_safe` | 10/10 | 0.000 | +0.075 | 0 correct→wrong, 3 wrong→correct |
| 4 | `route_r3_01_mc_f0_safe` | 10/10 | 0.000 | +0.064 | 0 correct→wrong, 2 wrong→correct |

R3-03/R3-04 do not pass the paired-decision guard. Both inherit TF-P0's
`series_000132:1` regression: Baseline correctly identifies the real local
spike followed by sustained increase, whereas TF-P0 answers False. The
mechanism is diagnosed, but there is no evidence-based justification for
turning a correct real-anomaly decision into an incorrect one. It is therefore
an unresolved harmful increase, not an acceptable explained trade-off.

R3-01 and R3-02 satisfy the full gate and advance to holdout. They improve all
three MC metrics and have exact Baseline OE/TF components, giving seven exact
ties and no parsed correct-to-wrong closed-task decisions.

## New OE rules

No new OE rule passed:

| Component | OE Final | OE Accuracy | OE Completeness | OE Relevance |
|---|---:|---:|---:|---:|
| Q1 positive evidence | -0.221 | -0.378 | -0.165 | -0.103 |
| Q2 named-event test | -0.312 | -0.586 | -0.276 | -0.034 |
| Q3 qualitative synthesis | -0.247 | -0.552 | -0.241 | +0.103 |

The severe cases predicted before scoring remained severe. All Q1/Q2/Q3
responses still denied the sharp downward event in `series_000012:0`. Q2
occasionally copied its Task Rule into the answer, and Q1 independently
produced six “no further/additional evidence needed” statements even after the
earlier negative wording was removed. These results show a model-conditioned
OE limitation: added task text competes with the trained prompt and changes
content decisions, while exact Baseline OE remains the only robust component
on the exposed development pool.

## Decision

Direction: **confirm** the two gate-passing conservative routes. Freeze
`route_r3_02_mc_f1_safe` and `route_r3_01_mc_f0_safe` unchanged and evaluate
them once on the untouched 48-QA / 24-series holdout. Do not tune either
candidate from holdout outcomes.
