# Round 7 protocol: deterministic OE answer-boundary repair

## Motivation

Round 6 showed that the released checkpoint is not a reliable A/B quality
selector. On holdout48 it selected three Round-5 revisions. Two revisions
removed requested analytical or boundary evidence and each lost one Relevance
point. The only broadly improved revision repaired a Baseline response that
contained an unclosed `<think>` block and no explicit answer boundary.

Round 7 therefore does not ask the checkpoint to judge answer quality. It
applies the existing frozen Round-5 `verbatim_or_add` refinement only when an
objective response-format failure is present, and accepts the revision only
when that failure is visibly repaired.

## Frozen route

`v2_r7_01_oe_unclosed_think_answer_repair` selects the Round-5 response only
when all of the following hold:

1. the Baseline OE response contains `<think>`;
2. it contains no `</think>`;
3. it has no line beginning with `Answer:` or `Final Answer:` (Markdown
   heading prefixes are allowed);
4. the revision contains no `<think>`/`</think>` and does contain such an
   explicit answer line.

Otherwise the Baseline response and its exact Judge scores are reused. MC and
TF are always byte-identical Baseline components during OE testing.

## Prompt intervention

For selected records the second pass is the already frozen Round-5
`v2_r5_02_oe_verbatim_or_add` prompt. No new response-generation wording is
introduced in this round. The new intervention is a deterministic task/output
protocol that limits when that prompt is allowed to replace the Baseline.

## Evaluation

The route is evaluated first on screening24, validation72, and holdout48 by
exactly reusing already generated responses and their locked
`deepseek-v4-pro` scores. A final joint route may proceed to full284 only if
all displayed Table-I metrics are non-lower on all three splits and OE changes
with at least one strict improvement.

This post-stopping extension is justified by the user's instruction to
continue after the earlier 30-mode search failed. It targets a new observed
mechanism and does not repeat any rejected prompt.