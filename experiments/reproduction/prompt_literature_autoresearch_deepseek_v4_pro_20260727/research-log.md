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

## 2026-07-27 — Cycle 1 execution

The frozen commit and released checkpoint SHA were verified on an authorized three-A100 node. The first launch failed before model loading because the fresh clone contained an empty tracked test-data directory; relaunch used the already-audited absolute dataset path. The successful run produced 288/288 predictions and released all GPUs.

An all-route Judge attempt was stopped at 89/660 scores after detecting that byte-identical Base components could differ under repeated CUDA generation. Before aggregate analysis, an operational amendment froze canonical component reuse. Ninety-seven unique non-Baseline prediction components required 215 Judge dimensions; 27 matching completed scores were reused and 188 were finished under the same `deepseek-v4-pro` configuration.

TLS initially failed because the virtual environment did not trust the site CA chain. Strict verification was restored with the node's system CA bundle; verification was never disabled. The final audit passed for 312 predictions and 715 scores, all from `deepseek-v4-pro`, all using complete score-position top-logprobs.

## 2026-07-27 — Outer loop 1

Four MC-only routes passed 10/10 on the exposed screening split with zero correct→wrong decisions. MC RE2 was strongest: MC Final +0.812, Correctness +1.000, Reasoning +0.375. It fixed two Baseline MC errors.

TF RE2 improved all TF metrics by +0.111 and changed four wrong decisions to correct, but changed one correct positive-anomaly case to False. The response invented a narrow stable range and denied the actual spike-plus-sustained-rise pattern. Static TF minimal/clause rules were worse.

OE RE2 increased Accuracy and Completeness by +0.143 each but reduced Relevance by -0.143. It repeated or expanded the question and sometimes rendered scaled integers as apparent raw values. The direct OE rule reduced unsupported-number incidence and length, but lowered OE Final, Accuracy, and Relevance.

Round 2 advances three zero-regression MC mechanisms. A fourth route combines the best MC RE2 component with TF RE2 and exact Baseline OE; it is an explicit risk review, not an automatic pass. Its 72-QA gate requires all ten metrics nonnegative and zero closed-task correct→wrong transitions.
