# Round 3 main holdout48 analysis

## Formal results

All 74 planned `deepseek-v4-pro` Judge dimensions completed. The two MC
components changed 17 MC responses each. The three OE routers reused the same
already-generated Round-1 component on two, two, and one matched OE records.
All other families and records reuse exact Baseline predictions and scores.

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3.8588 | 3.8235 | 3.9412 | 3.0533 | 2.7333 | 2.7333 | 3.8000 | 3.2750 | 3.2500 | 3.3125 |
| MC context-balanced | 3.7235 | 3.7059 | 3.7647 | 3.0533 | 2.7333 | 2.7333 | 3.8000 | 3.2750 | 3.2500 | 3.3125 |
| Δ | -0.1353 | -0.1176 | -0.1764 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| MC content-output | 3.5647 | 3.5294 | 3.6471 | 3.0533 | 2.7333 | 2.7333 | 3.8000 | 3.2750 | 3.2500 | 3.3125 |
| Δ | -0.2941 | -0.2941 | -0.2941 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| OE balanced-support | 3.8588 | 3.8235 | 3.9412 | 2.9600 | 2.6000 | 2.6000 | 3.8000 | 3.2750 | 3.2500 | 3.3125 |
| Δ | +0.0000 | +0.0000 | +0.0000 | -0.0933 | -0.1333 | -0.1333 | -0.0000 | +0.0000 | +0.0000 | +0.0000 |
| OE boundary-balanced | 3.8588 | 3.8235 | 3.9412 | 2.9600 | 2.6000 | 2.6000 | 3.8000 | 3.2750 | 3.2500 | 3.3125 |
| Δ | +0.0000 | +0.0000 | +0.0000 | -0.0933 | -0.1333 | -0.1333 | -0.0000 | +0.0000 | +0.0000 | +0.0000 |
| OE assessment-balanced | 3.8588 | 3.8235 | 3.9412 | 3.0067 | 2.6667 | 2.6666 | 3.8000 | 3.2750 | 3.2500 | 3.3125 |
| Δ | +0.0000 | +0.0000 | +0.0000 | -0.0467 | -0.0667 | -0.0667 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

No mode satisfies the all-ten-nonlower gate on holdout48, so none can be
combined into a formal 284-QA candidate.

## Failure mechanisms

The MC content/context instructions generalized on screening24,
validation72, and pooled development96 but reversed on holdout. The largest
failure is `series_000141:1`: Baseline correctly selects the normal option A,
whereas the context-balanced prompt selects D after reinterpreting ordinary
alternating values as repeated sudden drops and recoveries. Several unchanged
labels also lose one Judge point because the prompted explanations become
shorter and omit a detail present in the expected answer. The intervention is
therefore not a stable semantic decoder; it changes both the decision
threshold and explanation coverage.

The OE lexical route also overfit the speech-act surface. For
`series_000020:1`, Baseline already gives a complete 5/5/5 answer that both
refutes the alleged consecutive spikes and addresses their boundary
location. The routed response adds an open `<think>` prefix and weakens the
last boundary-specific conclusion, scoring 4/4/5. For
`series_000062:1`, Baseline directly explains how boundaries can hide an
outside-window anomaly. The routed response speculates about a longer-term
trend and loses one Accuracy and one Completeness point. Matching the words
“support/refute” and “boundary” does not establish that an extra instruction
is needed.

## Outer-loop conclusion

Question-text routing cannot rescue a prompt component merely because its
development gains concentrate in the same lexical subtype. The next main
round must:

1. retain the original AXIS token neighborhood and question-ending boundary;
2. avoid instructions that reinterpret Local embeddings as a natural-language
   anomaly oracle;
3. focus OE changes on coverage and relevance rather than anomaly-status
   priors;
4. require broad development evidence before consulting the remaining formal
   paper140 set.
