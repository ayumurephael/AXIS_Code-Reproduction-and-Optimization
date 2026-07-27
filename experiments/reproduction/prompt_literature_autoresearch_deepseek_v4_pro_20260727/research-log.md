# Research log

## 2026-07-27 — Cycle 0: prior-state audit

Read the complete prompt design note, the preceding autoresearch state/log/findings, and its failure-case dossiers. The preceding search tested 35 modes. Its two frozen candidates failed the then-untouched 48-QA holdout, so no candidate paper140 run was made.

Key confound found: every prior “minimal routed” MC/TF candidate still changed the trained `Overall Summary Hints` label to `Learned Task Guidance/Shared Task-Control Tokens`. Thus the effect of a short task rule on the byte-stable released scaffold was never isolated. The 30 Fixed tokens were co-trained in the old textual neighborhood, so the rename is not behaviorally neutral even if its human interpretation is clearer.

## 2026-07-27 — Cycle 0: literature synthesis

Primary-source review covered time-series prompt prefixes, MC option-order and symbol-binding sensitivity, contextual calibration, output-format bias, lexical prompt sensitivity, prompt robustness, RE2 question re-reading, chain-of-thought scale dependence, and small-model prompt optimization.

The literature does not justify a single long universal contract. It supports:

- keeping task/domain context short and directly relevant;
- separating MC option semantics from letter symbols;
- treating rigid output schemas as a possible accuracy intervention, not a neutral formatter;
- testing lexical variants because sensitivity is instance- and model-dependent;
- testing RE2 independently because even its authors report mixed behavior when extra instructions disrupt learned patterns;
- avoiding exposed chain-of-thought as a default for this 7B checkpoint.

## 2026-07-27 — Cycle 1 preregistration

Implemented 12 Round-1 modes. Every mode preserves:

- the released opening;
- Values → Per-Step Analysis → `Overall Summary Hints`;
- all 30 Fixed tokens;
- the question-ending generation boundary, with no `Answer:` prefill;
- no Evidence Contract, no numeric rescaling instruction, and no evidence reordering.

The only manipulated factors are one short task-specific answering rule and/or exact question re-reading. CPU prompt tests pass; GPU inference and formal judging remain pending.
