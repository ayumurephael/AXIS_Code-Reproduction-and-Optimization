# Screening round 1 analysis

## Audit status

- Inference: 312/312 predictions, comprising 24 records for Baseline and each
  of the 12 prompt conditions.
- Judge: 715/715 dimension scores from `deepseek-v4-pro`.
- Readout: 712 `final_score_top_logprobs` rows and three preregistered
  `exact_sample_mean_20` fallback rows.
- Integrity: no duplicate, missing, or unexpected prediction/score keys; all
  715 rows have the expected provider/model and valid prompt SHA-256 identity.
- The same audit passed independently on the GPU node and after download.

## Preregistered ranking

| Rank | Mode | Nonnegative dimensions | Worst delta | Mean delta |
|---:|---|---:|---:|---:|
| 1 | `pareto_p01_boundary` | 9/10 | -0.286 | +0.253 |
| 2 | `pareto_p03_task_rule` | 8/10 | -0.143 | +0.205 |
| 3 | `pareto_p06_abc` | 8/10 | -0.286 | +0.287 |
| 4 | `pareto_p07_abd` | 7/10 | -0.143 | +0.038 |

The complete Table-I score and delta tables are in
`results/screening_summary.md`. No screening condition satisfies the strict
all-ten Pareto criterion. The four rows above advance under the locked
lexicographic rule; they also provide mechanism diversity (A only, C only,
ABC, and ABD), so the diversity replacement is not invoked.

## What worked

### A: boundary guard

P01 is the strongest worst-dimension candidate and changes no correct parsed
MC/TF decision into an incorrect one. It improves MC Final by 0.375, OE Final
by 0.129, and TF Final by 0.289. A single concrete boundary sentence appears
to ground the answer without imposing a new anomaly definition.

Two paired improvements illustrate the effect:

- `series_000024:0` (normal MC): Baseline calls ordinary mixed-sign
  fluctuations an erratic anomaly (D, Final 1.3); P01 selects the correct
  stable/normal option C (Final 5.0).
- `series_000132:1` (anomalous TF): P01 keeps the correct `True` decision and
  improves the paired Final by 1.0 instead of suppressing the real compound
  anomaly.

P01's only aggregate loss is OE Accuracy (-0.286). On
`series_000043:1`, both Baseline and P01 miss the local drop plus subsequent
level/trend change; P01 becomes less complete by reducing the answer to a
single smooth-trend interpretation. The boundary rule improves grounding but
does not itself improve anomaly-shape recognition.

### C: task-conditioned rule

P03 corrects five previously wrong MC/TF decisions while breaking two. It
improves MC and TF and raises OE Completeness, supporting task decomposition
as a useful mechanism. In particular:

- `series_000091:1` changes from the wrong brief upward peak (B) to the correct
  broad downward dip (C), improving paired Final by 3.7.
- `series_000112:0`, a negated normal TF statement, changes from wrong `False`
  to correct `True`, improving paired Final by 3.0.

The current wording is nevertheless too interventionist. On real anomalies
`series_000132:1` and `series_000058:1`, it changes correct `True` decisions to
`False` and describes pronounced spike/convex patterns as stable. “Decide the
underlying evidence claim first” makes the model re-litigate a proposition
whose described pattern is actually present, and can override useful learned
task-control behavior.

### ABC: best mean improvement, but over-corrected anomalies and OE methods

P06 has the best mean ten-metric delta (+0.287) and the best TF gains among
the advanced candidates, but loses OE Accuracy and marginally loses OE
Completeness. Its largest failures expose two distinct interactions:

- On `series_000132:1`, B plus C turns a correct `True` into `False`. The
  calibration list makes non-anomaly language salient and suppresses a real
  spike followed by a sustained level change.
- On OE method questions `series_000028:1` and `series_000083:0`, the model
  gives an early categorical “no anomaly” conclusion and omits the requested
  indicators/methods and uncertainty analysis. The paired Final drops by
  1.35 and 1.25. “Start from a diagnostic conclusion” is inappropriate when
  the question asks how to evaluate or what evidence would be needed.

### ABD: conservative control

P07 changes no correct parsed MC/TF decision into an incorrect one, but its TF
Final is 0.067 below Baseline and the mean improvement is small. It also has a
high unsupported-number fraction (0.489 overall versus 0.410 for Baseline),
showing that an instruction not to quote numbers does not reliably prevent
fabricated precision. On `series_000132:1` it retains the correct `True` label
but cites the wrong spike step and magnitude, reducing both TF dimensions.

## Failure mechanisms from non-advanced controls

- **B is not a neutral calibration rule.** P02 improves TF but lowers all
  three MC metrics by 0.25. On normal MC `series_000112:1`, it changes the
  correct D to anomalous option A by treating a smooth rise as a sudden spike.
  Listing anomaly-like features can prime those features even when the
  sentence says they are insufficient.
- **E must not be a shared OE brevity rule.** P09 loses every OE record and
  drops OE Relevance by 0.714. P05 shows the same trade-off: MC improves
  strongly while OE Relevance drops 0.714. “Shortest explanation” and
  categorical closure remove the conditional evidence and analytical methods
  requested by OE questions.
- **Lower unsupported-number rate is not sufficient.** P09 reduces the OE
  unsupported-number fraction from 0.435 to 0.119 but still loses all four OE
  metrics. Content selection and epistemic stance dominate formatting
  cleanliness.
- **Strict parse and answer-first rates are diagnostics, not selection
  metrics.** Several high-scoring candidates have low strict parse rates; the
  formal Judge scores and robust extracted decisions are the relevant
  outcomes.

## Outer-loop synthesis 1

Direction: **deepen**, without adding a new prompt before validation.

The evidence favors short, low-semantic-load constraints. A is the most
promising common rule; C is useful only when its per-task wording does not
force unnecessary re-diagnosis; B and E require question-type and speech-act
conditioning. The next prompt-generation cycle, if validation rejects the
current candidates, should test:

1. P01 plus an MC-only option-verification rule.
2. P01 plus a TF rule that maps the proposition to a label but explicitly
   preserves an observed pattern stated in the proposition.
3. P01 plus an OE speech-act router: diagnostic questions may conclude,
   whereas “how would you evaluate / what evidence would you seek” questions
   must give conditional evidence and methods without premature closure.
4. Closed-task-only one-answer formatting; no brevity rule for OE.
5. Positive evidence language rather than a long list of features that are
   “not anomalous by themselves.”

These are hypotheses for a later, separately preregistered batch. The current
four advanced candidates remain unchanged for validation so that the locked
12-to-4 funnel is honored.
