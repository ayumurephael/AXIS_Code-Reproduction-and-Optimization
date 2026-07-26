# Development round 2 analysis

## Outcome

The six preregistered routed candidates completed 576 GPU predictions and
1,326 `deepseek-v4-pro` Judge dimension scores. The fail-closed audit passed:
1,312 scores used final-score token probabilities and 14 used the exact
20-sample fallback. No candidate met the locked 10/10 development gate, so the
internal holdout remains untouched.

The best candidate was `route_r2_04_oe_old_contract` at 7/10 nonnegative
dimensions (worst delta -0.381, mean delta +0.205). It improved all MC and TF
metrics, and OE Relevance, but reduced OE Final, Accuracy, and Completeness.

## Mechanistic result

Question-type routing successfully removed cross-task interference:

- the fixed-role MC component improved all three MC metrics;
- the direct-polarity TF component improved all three TF metrics;
- neither MC component caused a correct-to-wrong parsed MC decision;
- the remaining correct-to-wrong flip under the minimal routes was TF
  `series_000132:1`, a real anomaly that the added rule suppressed.

OE remains the bottleneck:

- the speech-act rule made responses longer but reduced all four OE metrics;
- its negative phrase “do not claim that no further evidence is needed”
  appeared to prime exactly that failure: five speech-act-only and eight
  Contract-plus-speech responses claimed no further evidence was needed;
- the historical Contract improved OE Relevance by +0.311, but two severe
  diagnostic reversals dominated Accuracy: `series_000012:0` denied the sharp
  downward movement named and visible in the input, and `series_000122:0`
  converted normal reversible variability into anomalies while citing steps
  outside the window;
- other losses came from unsupported numerical decoding
  (`series_000026:1`, `series_000080:0`), generic claims that contradicted
  visible sharp fluctuations (`series_000023:0`, `series_000137:0`), and
  incomplete answers to evidence/boundary questions (`series_000080:1`).

The failure is therefore not a lack of answer length. It is an OE instruction
that changes the model's diagnosis or invites unsupported specifics. Round 3
will preserve exact Baseline OE as two conservative controls and test only
short, positive, qualitative evidence obligations in the remaining routes.

## Decision

Direction: **deepen the routed design**. Keep the validated MC components,
retain exact Baseline behavior for conservative TF/OE routes, and isolate new
OE rules. Preregister eight candidates on the exposed 96-QA development pool.
Do not touch the 48-QA holdout unless at least two candidates satisfy the
locked 10/10 gate.
