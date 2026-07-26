# Prompt candidate matrix

This matrix applies four ideation lenses: failure-boundary probing, simplicity, decomposition of the monolithic Contract, and reconciliation of the OE-versus-closed-task tension.

## Mechanistic factors

- **A — Boundary guard:** use only `[start,end)` and never cite an outside step.
- **B — Anomaly calibration:** smooth trends, sign changes, ordinary extrema, or large/small values are not anomalies without unexpected local contrast, persistence/recovery, and supporting latent evidence.
- **C — Task-conditioned decision rule:** a short, different rule for MC, TF, and OE.
- **D — Qualitative numeric guard:** do not convert or report exact values unless the question explicitly requires an original-scale number; there is no routine “divide by 100” instruction.
- **E — One-answer/concise guard:** answer once, do not revise the label, and avoid long self-correction.

All candidates retain the author's two-block Values/Local layout and the Fixed-role rename. No candidate interleaves values with latent tokens, removes Fixed tokens, or adds `EOS + Answer:` during screening.

## Twelve screening candidates

| ID | A | B | C | D | E | Mechanistic purpose |
|---|:---:|:---:|:---:|:---:|:---:|---|
| P00 | — | — | — | — | — | Fixed-role reference |
| P01 | ✓ | — | — | — | — | Isolate out-of-window prevention |
| P02 | — | ✓ | — | — | — | Isolate anomaly over-detection control |
| P03 | — | — | ✓ | — | — | Isolate task decomposition |
| P04 | — | — | — | ✓ | — | Isolate removal of routine numeric decoding |
| P05 | — | — | — | — | ✓ | Isolate answer consistency/length |
| P06 | ✓ | ✓ | ✓ | — | — | Minimal Pareto core |
| P07 | ✓ | ✓ | — | ✓ | — | Best common-text-only candidate |
| P08 | ✓ | — | ✓ | ✓ | ✓ | Boundary plus task/format control without anomaly rule |
| P09 | — | ✓ | ✓ | — | ✓ | Calibration plus task/format control |
| P10 | ✓ | ✓ | ✓ | ✓ | ✓ | Full Pareto-v1 |
| P11 | ✓ | ✓ | ✓ | — | ✓ | Pareto-v1 without numeric guard |

## Converged candidates before data

1. **P10 full Pareto-v1.** Directly addresses every observed failure mechanism while preserving the trained layout.
2. **P06 minimal Pareto core.** Simplicity control: tests whether the numeric and answer guards are unnecessary complexity.
3. **P09 task-consistency candidate.** Targets the strongest closed-task failures: TF label mapping and MC post-hoc rationale.
4. **P07 common-text-only candidate.** Required fallback if one identical prompt must serve all three task types.
5. **P11 no-numeric-guard candidate.** Tests whether explicitly mentioning numbers is itself harmful.

## Pre-data predictions

- A should reduce explicit outside-window citations but may have little effect on MC false positives.
- B should improve normal MC/TF cases such as `series_000072:1`, but may hurt OE cases where a sharp level shift is meaningful unless the OE rule preserves uncertainty.
- C should have the largest TF Justification benefit by preventing negation/label inconsistency.
- D should mainly improve OE Accuracy/Relevance by reducing fabricated values and task-format drift.
- E should improve TF Justification and prevent catastrophic self-revision, but overly strict brevity may reduce OE Completeness.
- The B×C interaction is expected to be essential: anomaly calibration must be translated differently for option selection, proposition truth, and diagnostic explanation.

## Two-sentence research pitch

AXIS prompt modifications currently trade OE evidence coverage against MC/TF decision reliability because one long shared Contract entangles representation, anomaly semantics, and answer behavior. We decompose those functions into short orthogonal guards and task-conditioned decision rules, then optimize the worst task-Final delta so a candidate cannot win by sacrificing another task family.
