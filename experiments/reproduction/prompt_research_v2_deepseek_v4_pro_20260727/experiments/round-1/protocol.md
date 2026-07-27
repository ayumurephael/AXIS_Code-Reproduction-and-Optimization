# Round 1 protocol: short evidence-use and task-family rules

Frozen before candidate GPU inference or new Judge calls.

## Objective

Test whether a short rule can repair the dominant Baseline failure—ignoring
sample-specific `Per-Step Analysis` when raw values look smooth—without the
distribution shift of the rejected long Evidence Contracts or Fixed-header
replacement.

## Fixed scaffold

Every non-Baseline component preserves:

- the exact released opening;
- `Values` followed by `Per-Step Analysis` followed by
  `Overall Summary Hints`;
- all 30 Fixed tokens and their immediate textual prefix;
- the question at the end of the generation prefix;
- no `EOS + Answer:` prefill;
- no value/local interleaving;
- no numeric scale explanation;
- no repeated question;
- no exposed chain-of-thought request.

Only the current question family's short `### Answering Rule` is added after
the unchanged hint blocks.

## Frozen modes

### Open-ended components

1. `v2_r1_01_oe_context_balanced`

   > Use Per-Step Analysis as sample-specific context for whether the
   > displayed local behavior is expected in the complete time series. It can
   > support either normality or anomaly; do not decide from displayed
   > magnitude alone.

2. `v2_r1_02_oe_context_contrast`

   > Combine the window shape with Per-Step Analysis. Do not dismiss an
   > anomaly merely because the displayed values look smooth, and do not call
   > a value anomalous merely because it is extreme.

3. `v2_r1_03_oe_evidence_router`

   Only for OE questions that explicitly ask what evidence, features,
   indicators, or analytical checks should be used:

   > First state what the supplied window and Per-Step Analysis currently
   > support, then address the requested evidence or boundary qualification.
   > Do not replace the supplied-window assessment with a generic tutorial.

   Other OE questions are exact Baseline.

4. `v2_r1_04_oe_context_direct`

   > Use Per-Step Analysis as context for whether the window behavior is
   > expected. Start with a direct answer to the question, then give only the
   > window pattern and contextual evidence needed to support it.

5. `v2_r1_05_oe_fixed_postnote`

   > Overall Summary Hints are shared learned task guidance, not
   > sample-specific evidence. Use the Window and Per-Step Analysis to assess
   > this sample.

   This tests the user's desired Fixed-role clarification while preserving the
   trained `Overall Summary Hints` header immediately before the 30 soft
   tokens.

### Multiple-choice components

6. `v2_r1_06_mc_context_balanced`

   > Use Per-Step Analysis to judge whether each displayed local behavior is
   > expected in the complete time series. Choose the option whose full
   > description best matches both that context and the window shape; value
   > magnitude alone is not evidence of anomaly.

7. `v2_r1_07_mc_content_output`

   > Select the best-supported option by its complete meaning before mapping
   > it to a letter. Begin with that letter and option text, then give one
   > concise reason from the Window and Per-Step Analysis.

### True/false components

8. `v2_r1_08_tf_context_whole`

   > Judge the complete statement against both the Window and Per-Step
   > Analysis. That context can support either normality or anomaly; preserve
   > every negation and keep the reason consistent with one True or False
   > verdict.

9. `v2_r1_09_tf_prefix_context`

   > Judge the complete statement using the Window and Per-Step Analysis.
   > Begin with exactly True. or False., then give a concise reason consistent
   > with that verdict.

## Evaluation

- Development records: all 144 non-paper140 QA. They are already exposed and
  are used for falsification, not untouched-validation claims.
- Raw GPU generation: only records whose task-family component changes.
- Unchanged components: exact reuse of the canonical full284 Baseline
  predictions and scores with explicit provenance.
- Formal Judge: `deepseek-v4-pro` only, same rubric and registered score
  readout as the canonical Baseline.
- Primary family gate: every changed family's Table-I dimensions are
  nonnegative versus Baseline, and its worst delta is positive enough to
  survive paired Judge repetition.
- OE advancement is mandatory for a comprehensive candidate.

After these nine experiments, inspect the largest gains and losses, all
changed MC/TF decisions, contextual-anomaly denials, normal-window false
positives, answer-prefix compliance, unsupported numbers, and response length.
Perform an outer-loop synthesis before adding another mode.
