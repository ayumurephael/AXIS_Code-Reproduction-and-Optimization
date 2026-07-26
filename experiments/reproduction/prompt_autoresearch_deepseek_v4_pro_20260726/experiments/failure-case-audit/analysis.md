# Evidence Contract failure-case analysis

## Aggregate correction and comparison

The two controlled comparisons use `fixed_role` as the control.

| Treatment − Fixed role | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Old full Contract | -0.3586 | -0.3727 | -0.3256 | +0.1721 | +0.1818 | +0.0146 | +0.3445 | -0.3000 | -0.3571 | -0.2143 |
| Revised Contract | -0.3051 | -0.3262 | -0.2558 | -0.1669 | -0.1996 | -0.1091 | -0.1964 | -0.0524 | **+0.0714** | -0.2381 |

Thus the revised Contract does not reduce every individual metric: TF Correctness increases slightly, while the much larger TF Justification loss makes TF Final decline. MC and OE do decline in every reported dimension.

The old Contract's OE gain is heterogeneous: 31/55 records improve and 24/55 decline. Its mean OE response is 161.5 characters longer, but the response-length change and Final change have Pearson correlation -0.13. The gain is therefore not explained by verbosity alone.

## Representative failure cases

### MC: normal variability is converted into an anomaly

`series_000072:1`, window `[521,539)`, is labeled normal.

- Fixed role selects the correct normal option C and receives Final 5.000.
- Both Contracts select option B and call ordinary alternating negative values such as -1.34 and -0.92 “multiple irregular dips”; both receive Final 1.300.

The Contract says that large or small values alone are not anomalies, but it does not give an operational comparison rule for deciding whether a local extremum is unexpected. The model substitutes generic smoothness for the missing baseline and over-detects anomalies.

### MC: old positional wording causes evidence to leave the window

`series_000094:0` asks about `[640,674)` and is normal.

- Fixed role correctly selects B, Final 5.000.
- Old Contract selects A by citing a spike at **step 703**, outside the supplied window, Final 1.900.
- Revised Contract removes the out-of-window citation and returns to B, Final 5.000.

This is direct evidence that replacing “same row” with positional alignment can repair some failures. Across all MC records, explicit out-of-window step citations move from 0% under Fixed role to 4.65% under the old Contract and back to 0% under the revised Contract.

### MC: scale correction is insufficient when anomaly calibration is wrong

`series_000064:0`, `[478,494)`, is a smooth normal rise/decline.

- Fixed role selects normal option D, Final 4.000.
- Old Contract decodes -1.30 as **-13.0** and calls a transient drop, Final 1.300.
- Revised Contract no longer makes the -13.0 error, but still calls the same smooth change a transient anomaly and remains at Final 1.300.

The numeric rule fixes one surface error but does not repair the more important decision error: the prompt never defines how much local contrast, persistence, or latent evidence is required before a shape becomes anomalous.

### MC: correct option, post-hoc incorrect explanation

`series_000135:1` has a rapid rise followed by a slow decline.

- Fixed role selects A with a directionally correct explanation, Final 4.700.
- Old Contract also selects A but invents “13.0 to 17.6” and misdescribes the shape, Final 2.000.
- Revised Contract selects A but calls a decrease “the rapid rise” and the following increase “slow decline,” Final 2.700.

This is option anchoring: the model first commits to an option, then generates a superficially matching explanation without verifying direction and temporal order.

### TF: answer label and explanation become logically inconsistent

`series_000090:1` asks whether there is **no evidence** of anomalies in a normal window.

- Fixed role answers True with consistent evidence, Final 5.000.
- Old Contract corrupts a value into **-113**, answers False, and receives Final 2.000.
- Revised Contract repeatedly explains that there are no anomalies, yet begins and ends with **False**, Final 2.000.

The revised representation rule repairs the value interpretation but not proposition evaluation. The prompt lacks an explicit two-stage TF rule: decide the underlying anomaly claim, then apply the proposition's negation exactly once.

### TF: long reasoning produces a second, contradictory answer

`series_000051:0` is a smooth normal window and the proposition “there is evidence of anomaly” is False.

- Fixed role answers False once, Final 4.400.
- Old Contract first answers False correctly, then emits a second `Answer: True`, inventing an anomaly at **step 78** outside `[555,590)`, Final 1.600.
- Revised Contract answers False once and recovers to Final 4.400.

Under the old Contract, TF multiple-answer output rises from 0% to 4.76%. Extra TF text is harmful: for the revised Contract, response-length delta versus Final delta has Pearson correlation -0.44.

### OE: useful gain comes from identifying the actual transition

`series_000050:1` asks what subtle evidence to inspect in `[808,824)`.

- Fixed role incorrectly calls the positive-to-negative transition smooth and receives Final 1.300.
- Old Contract explicitly identifies the abrupt step 812→813 level shift, separates a possible regime shift from an isolated point anomaly, and receives Final 3.600.
- Revised Contract again calls the transition gradual and receives Final 1.000.

The useful behavior is not generic “Evidence Contract” prose. It is a concrete OE reasoning pattern: state the observed transition, explain why it may be meaningful, and preserve uncertainty when wider context is missing.

### OE: revised numeric instructions provoke sequence fabrication

`series_000085:1` contains a rapid rise followed by gradual decline.

- Old Contract recognizes the pattern and improves Fixed role's Final 1.350 to 2.600, despite some number corruption.
- Revised Contract invents steps 200–220 and 114–230 outside `[127,149)`, flips to “no anomaly,” and falls to Final 1.000.

`series_000127:1` is normal:

- Fixed role and old Contract both receive Final 4.650.
- Revised Contract invents steps 800, 1003 and 20001, values -14.49 and 19.98, converts the OE question into a new a/b multiple-choice task, and receives Final 1.000.

At population level, explicit out-of-window step citations in OE rise from 3.64% under Fixed role to 18.18% under the revised Contract. The OE unsupported-number fraction rises from 45.95% to 51.01%.

### OE: catastrophic generation collapse

`series_000125:1` has a clear end-of-window convex anomaly.

- Fixed role and old Contract both receive essentially 5.000.
- Revised Contract stops after 148 characters at `Answer: Within the window from step 4`, receiving 1.000 in Accuracy, Completeness, and Relevance.

This cannot be explained by a wrong anomaly threshold alone. It shows that the longer revised instruction changes generation dynamics for the released checkpoint and can trigger a distribution-shift failure.

## Failure taxonomy

1. **Cross-task instruction overload.** Representation semantics, anomaly definition, epistemic policy, option handling, and answer style are mixed into one long shared Contract. Different task families need different decision operations.
2. **No operational anomaly threshold.** “Unexpected relative to the global pattern” is conceptually correct but underspecified. The model treats sign changes, ordinary extrema, or smooth trends as anomalies without checking persistence, local contrast, recovery, and latent evidence together.
3. **No proposition-to-label rule for TF.** The model can reason “no anomaly” and still output False for a negated proposition.
4. **No evidence-to-option verification for MC.** Correct-looking option text anchors the response; the explanation is generated post hoc and can reverse rise/decline direction.
5. **Numeric decoding is made too salient.** Most questions need qualitative shape, not reconstructed physical values. Scaling language causes attention to move toward exact-number generation, where decimal shifts and fabricated values appear.
6. **Separate text blocks remain hard to bind.** Rewording alignment helps some cases but cannot force the old checkpoint to bind every latent token to the intended value. Forcing a new interleaved layout also failed in the previous experiment because the checkpoint was not trained on it.
7. **Long unconstrained answers permit self-revision.** Duplicate labels, contradictory final answers, and invented follow-up tasks reduce TF Justification and OE Relevance.
8. **Inference-prompt distribution shift.** The released checkpoint was trained on the short author prompt. Long semantic contracts can alter attention positions and generation modes enough to cause truncation or task-format collapse.

## First evidence-grounded candidate: Pareto-v1

This is the strongest initial candidate implied by the failure cases, but it is not yet a demonstrated improvement. It deliberately keeps the author's two-block layout and Fixed-token position, shortens shared evidence semantics, removes routine numeric conversion, and inserts a compact task-conditioned rule.

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Evidence Use
- Use only the evidence inside the half-open window [{start}, {end}). Never invent or cite a step outside this window.
- Judge qualitative shape, order, persistence, recovery, and local contrast. A sign change, smooth trend, ordinary peak or trough, or a large or small value is not anomalous by itself.
- Per-Step Analysis tokens align with the displayed values by sequence order. Use them only as local unexpectedness evidence. Shared Task-Control tokens guide the task but are not anomaly evidence.
- Do not convert or report exact numerical values unless the question explicitly requests a numerical value in the original scale.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Learned Task Guidance/Shared Task-Control Tokens:** {fixed_hint_tokens}

### Task Rule
{task_rule}

### Question
{question}
```

`{task_rule}` is selected by question type:

### MC rule

```text
Choose the one option whose complete claim matches the observed anomaly status, shape, temporal direction, position, and boundary relation. Reject any option that requires an event not present in the window. Answer once with the option letter and exact option text, then give two concise evidence-grounded sentences.
```

### TF rule

```text
Evaluate every clause of the proposition, including negation. First decide the underlying evidence claim, then map that decision to True or False exactly once. Output one label only and keep the explanation consistent with it; do not revise the label later.
```

### OE rule

```text
Begin with the diagnostic conclusion, then cover every component requested by the question. State the observed shape and location, explain why it supports or refutes an anomaly, and discuss boundary uncertainty only when relevant. Distinguish observed evidence from additional evidence that would be needed; do not invent values, steps, answer choices, or a new task.
```

## Proposed search, without repeatedly tuning on paper140

The full candidate test collection has 284 QA. `paper140` has already been used repeatedly; the remaining 144 QA comprise 72 untouched series with MC=51, OE=44, TF=49 and anomaly labels true=53/false=91.

Recommended series-level search:

1. Establish Baseline and Fixed-role scores once on all remaining 144 QA.
2. Use 12 series / 24 QA as a balanced screening set for 10–12 mechanistic variants.
3. Evaluate the best four variants on a disjoint 36 series / 72 QA validation set.
4. Evaluate the best two on the final disjoint 24 series / 48 QA internal holdout.
5. Run only the single locked winner once on `paper140` as the final comparability check.

Candidate factors should be orthogonal rather than a random wording sweep:

- Shared rules: none / boundary-only / boundary plus anomaly-calibration.
- Numeric policy: omit scaling rule / qualitative-only / conditional conversion.
- Task rule: none / MC only / TF only / OE only / all three task-conditioned.
- Answer length: unconstrained / concise one-answer boundary.
- OE uncertainty rule: absent / observed-versus-needed distinction.

Formal selection should use `deepseek-v4-pro` throughout. A suitable primary objective is the minimum of the three task-Final deltas versus Baseline, because it penalizes trade-offs; Macro Final is only a tie-breaker.

## Decisions that must be locked before new runs

1. Does “全面提升” mean all three Final scores above Baseline, or all ten reported dimensions above Baseline?
2. May the prompt contain the three short task-conditioned rules above, or must exactly the same text be used for MC/OE/TF?
3. Is the proposed approximate budget acceptable: 10–12 screened variants, four validation variants, two internal-holdout variants, and one final `paper140` run?
