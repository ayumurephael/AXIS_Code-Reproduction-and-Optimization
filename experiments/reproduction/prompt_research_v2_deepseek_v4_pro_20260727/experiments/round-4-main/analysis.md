# Round 4 analysis: OE coverage rules

## Outcome

All five one-pass OE rules failed validation72. None advances to holdout48.
MC and TF are exact Baseline components in every row.

## screening24

| Mode | OE Final Δ | OE Acc. Δ | OE Comp. Δ | OE Rel. Δ |
|---|---:|---:|---:|---:|
| `v2_r4_01_oe_requested_parts` | -0.0907 | +0.1429 | +0.1430 | -0.6360 |
| `v2_r4_02_oe_hypothesis_check` | -0.3835 | -0.0000 | -0.4284 | -0.7786 |
| `v2_r4_03_oe_silent_checklist` | -0.0406 | +0.5714 | -0.1427 | -0.6357 |
| `v2_r4_04_oe_evidence_scope` | +0.0879 | +0.2857 | +0.1430 | -0.2071 |
| `v2_r4_05_oe_compact_facets` | -0.0328 | +0.5714 | +0.1431 | -0.9428 |

The small screen made `evidence_scope` look promising, so all five modes were
kept for validation under the frozen wide-screen policy.

## validation72

| Mode | OE Final Δ | OE Acc. Δ | OE Comp. Δ | OE Rel. Δ |
|---|---:|---:|---:|---:|
| `v2_r4_01_oe_requested_parts` | -0.2559 | -0.3114 | -0.2250 | -0.2273 |
| `v2_r4_02_oe_hypothesis_check` | -0.4441 | -0.7205 | -0.4998 | -0.0568 |
| `v2_r4_03_oe_silent_checklist` | -0.2192 | -0.1750 | -0.2273 | -0.2614 |
| `v2_r4_04_oe_evidence_scope` | -0.2817 | -0.4023 | -0.3636 | -0.0455 |
| `v2_r4_05_oe_compact_facets` | -0.1582 | -0.1273 | -0.0909 | -0.2727 |

Every candidate lowers every OE dimension on validation72. The screening gains
therefore do not generalize.

## Concrete failures

### A content-neutral rule still flips a correct anomaly decision

Record `series_000012:0` asks whether a sharp downward movement is anomalous.
The Baseline identifies the isolated drop to about `-1.44` as a true local
anomaly. `evidence_scope` instead says that no value stands out and concludes
that the window is normal. Its paired Accuracy/Completeness/Relevance deltas
are `-4/-1/-3`.

The rule contains no explicit normality prior, yet it changes which evidence
the checkpoint attends to. “Base the answer only on supplied evidence” is
therefore not a passive style constraint for this finetuned model.

### Coverage pressure can erase a correct learned-hint interpretation

On `series_000104:1`, the Baseline correctly identifies a localized downward
spike and explains its abruptness and recovery. Both `evidence_scope` and
`compact_facets` change the conclusion to “no clear evidence,” treating the
dip as regular noise. Each loses `-2/-3/-2` across
Accuracy/Completeness/Relevance.

This is the same bidirectional evidence failure seen in earlier rounds:
natural-language answer guidance competes with, rather than merely formats,
the learned Local/Fixed representation.

### Compactness omits a salient phase of the pattern

On `series_000005:0`, the Baseline describes both the decline and subsequent
recovery of the smooth V-shaped normal cycle. `compact_facets` describes only
an upward trend. The conclusion remains normal, but the missing first phase
reduces factual coverage and completeness.

## Synthesis

1. Direct coverage instructions do not reliably operate after the model has
   made a content decision; they can change that decision.
2. Shorter output is not automatically more relevant. Compression often
   drops a requested boundary, method, or phase of the trajectory.
3. The next intervention should preserve the Baseline content decision
   explicitly and treat coverage as a separate editing operation.
4. This motivates Round 5's frozen two-pass, verdict-preserving refinement
   protocol rather than another one-pass rule before `### Question`.
