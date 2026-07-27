# Round 2 protocol: `new.md` compatibility ablation

Frozen before implementation, GPU generation, or Judge calls for these modes.

## Source and scope

- User-supplied source:
  `architecture_redesign_fixedhint_frozen/改进系列说明/prompt 改进/new.md`
- Source SHA-256:
  `aad3534881900a92cd687f40d6d182482b90bd7b08dda7b43b7f416946728158`
- Checkpoint: the released AXIS checkpoint; no retraining.
- Interpretation: this is an old-checkpoint compatibility experiment. A
  failure of structural P2–P5 does not falsify the document's proposal to
  retrain Phase II with the same new prefix.

The document explicitly distinguishes Table-I gains from causal Local-Hint
use. Local corruption/swap sensitivity will be run only for a performance
survivor; a writing-style gain alone is not labeled improved Local use.

## Frozen modes

### `v2_r2_01_new_short_rule` — P1

Keep the released opening, Values → Per-Step → `Overall Summary Hints` order,
30 Fixed tokens, question-ending layout, and released answer boundary. Add
only the document's old-checkpoint compatibility rule:

```text
### Evidence Rule
Use Window Values and Per-Step Analysis jointly. Per-Step token k corresponds
to Window value k in the listed order. Overall Summary Hints are shared task
controls, not sample evidence. Treat the question and options as hypotheses.
```

### `v2_r2_02_new_step_aligned` — P2

P1 plus replacement of the separate Values/Per-Step lists with one row per
value and Local token:

```text
Step {step:04d} | Value {scaled_value:+06d} | Aligned context <|local_hint|>
```

Value precedes Local. The trained `Overall Summary Hints` header remains after
the rows and the question remains last.

### `v2_r2_03_new_question_first` — P3

P2 plus moving the question before the aligned evidence, so each Local token's
causal LLM state can attend to the question. Fixed remains in its P2 position
for this isolated step.

### `v2_r2_04_new_evidence_last` — P4

Move the 30 Fixed tokens before Question under the document's
`Shared Task-Control Tokens` role, followed by Question, the short Evidence
Rule, and aligned sample evidence last. Retain the released generation
boundary: do not add an answer prefill in P4.

### `v2_r2_05_new_full_prompt` — complete document template

Use the full recommended information flow and wording:

1. new time-series anomaly analyst opening;
2. Shared Task-Control tokens and its two-sentence role description;
3. Question type and question;
4. the document's full Evidence-Use Rule;
5. aligned sample evidence;
6. only the active short MC/OE/TF output rule;
7. prefix ending literally in `### Final Answer\n\nAnswer:`.

This mode uses an inline `Answer:` boundary without inserting an EOS between
the prompt and generated answer. The four schematic example rows in `new.md`
are not included.

## Integrity assertions

For every prompt:

- Local placeholder count equals `end - start`;
- Fixed placeholder count remains 30;
- step is `04d`;
- scaled value preserves the released `(x * 100):.0f` rounding and is emitted
  as signed `+06d`;
- P2–P5 order is Value before Local on every row;
- the tokenizer must not truncate Question, the final Local token, output
  rule, or inline `Answer:` boundary;
- inference retains author series batching and beam-search parameters.

## Evaluation

1. Run all five modes on screening24 using the authorized GPU and released
   checkpoint.
2. Judge only with `deepseek-v4-pro` and the canonical G-Eval rubrics/readout.
3. Compare all ten Table-I metrics with the exact split Baseline.
4. Inspect prompt length/truncation, output formatting, response length,
   MC/TF decision changes, OE subpart coverage, and the largest paired gains
   and losses.
5. Perform an outer-loop synthesis after these five experiments. Advance
   broadly enough that a candidate capable of all-split ten-metric nondecline
   is not discarded on small-split noise.
6. Only a candidate that later changes and improves OE can form a final
   comprehensive route.
