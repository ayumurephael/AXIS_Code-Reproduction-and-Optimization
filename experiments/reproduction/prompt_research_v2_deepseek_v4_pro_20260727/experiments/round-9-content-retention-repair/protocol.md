# Round 9 protocol: no-anomaly repair with content-retention guard

## Formal-set failure that motivates this extension

Round 8 selected four full284 OE revisions. Accuracy increased, but three
revisions deleted substantial requested evidence or comparison detail, causing
Completeness and Relevance regressions. Their candidate/Baseline word-count
ratios were approximately 0.40, 0.58, and 0.48. The one revision that improved
both Accuracy and Completeness retained 0.606 of the Baseline words.

## Frozen mode

`v2_r9_01_oe_no_anomaly_content_retention_repair` keeps every Round-8
condition and additionally requires the revision to retain at least 60% of the
Baseline word tokens (`\b\w+\b`). This is a deterministic deletion guard: a
second pass may remove at most 40% of the words. It does not inspect labels,
Judge scores, record IDs, or hidden representations.

Selected records use the existing Round-5 `verbatim_or_add` prompt. All other
responses and scores are exact Baseline reuse.

## Interpretation limit

This mode was designed after inspecting Round-8 outcomes on full284. Its
full284 result is therefore a post-hoc engineering result, not untouched-test
generalization evidence. This provenance must remain explicit in the final
report even if all ten displayed Table-I metrics pass.