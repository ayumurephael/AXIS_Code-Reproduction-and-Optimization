# Literature Round 5 prompt catalog

All modes retain the released AXIS scaffold. Only the `### Answering Rule` text below is inserted before `### Question`; components marked `base` are byte-identical to the released prompt.

## MC rule A — structured-signature guard

```text
Compare every option by its complete meaning. Treat alternating signs, ordinary peaks or troughs, isolated large or small values, and irregular-looking fluctuation as normal variability unless the supplied evidence supports the option's specific structured anomaly signature, such as a localized contrast, persistence, recovery, or boundary pattern. Select an anomalous option only when that defining signature is supported; otherwise select the normal option. Explain the decisive qualitative pattern without quoting exact values or step numbers unless asked.
```

Used by `lit_r5_01_mc_structured_guard` and `lit_r5_03_joint_structured_tf_neg_re2`.

## MC rule B — status then shape

```text
First decide anomaly status from the supplied Per-Step Analysis and Overall Summary Hints, independently of the option wording. Ordinary variance, alternating signs, isolated highs or lows, and irregular-looking fluctuation are not anomalies by themselves. Then compare only options consistent with that status and choose the one whose complete text best matches the observed shape, persistence, recovery, and boundary behavior. Explain the decisive qualitative evidence without quoting exact values or step numbers unless asked.
```

Used by `lit_r5_02_mc_status_then_shape` and `lit_r5_04_joint_status_tf_neg_re2`.

## Frozen explicit-negation TF rule

The lexical router activates only when the TF question contains one of: `no`, `not`, `without`, `absence`, `lack`, `neither`, `nor`, `cannot`, `can't`, `doesn't`, `isn't`, `aren't`, `wasn't`, `weren't`. Otherwise the entire prompt is byte-identical Baseline.

```text
Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.
```

The TF rule is used by the existing fallback `lit_r4_01_tf_neg_re2` and both Round-5 joint modes. OE is always exact Baseline.
