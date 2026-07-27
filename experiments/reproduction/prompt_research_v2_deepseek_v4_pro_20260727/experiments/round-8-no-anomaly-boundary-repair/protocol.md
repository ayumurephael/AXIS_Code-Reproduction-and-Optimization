# Round 8 protocol: verdict-preserving no-anomaly boundary repair

## Observed Round-7 failure

The broad answer-boundary repair selected two holdout OE responses. It improved
`series_000111:0`, whose Baseline and revision both concluded no anomaly, but
hurt `series_000141:0`, whose malformed Baseline already contradicted the
question's normality premise and whose revision preserved that false-positive
conclusion while deleting requested boundary analysis.

## Frozen mode

`v2_r8_01_oe_no_anomaly_boundary_repair` retains all four Round-7 structural
conditions and adds one semantic preservation condition: both the Baseline and
revision must explicitly use a bounded `no`/`without ... anomaly/irregularity`
construction. The condition is a deterministic surface check, not an LLM
quality judgment. It prevents the repair prompt from replacing a malformed
answer whose existing conclusion asserts an anomaly.

Selected records use the already frozen Round-5
`v2_r5_02_oe_verbatim_or_add` second-pass prompt. Every unselected OE response
and every MC/TF response is copied byte-for-byte from Baseline with exact score
reuse.

## Gate

The new mode is evaluated alone on screening24, validation72, and holdout48.
It advances to the full284 joint route only if all displayed Table-I metrics
are non-lower on every split and OE is changed with a strict gain on at least
one split. No previously failed Round-7 mode is rerun as a candidate.