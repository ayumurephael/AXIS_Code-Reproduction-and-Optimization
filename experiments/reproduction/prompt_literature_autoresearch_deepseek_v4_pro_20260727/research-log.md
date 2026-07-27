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


## 2026-07-27 ? Outer loop 2

Round 2 completed 288/288 GPU predictions and 204/204 new `deepseek-v4-pro` dimensions. The combined audit passed for 360 predictions and 830 scores. No route passed: semantic binding and pointwise each reached 9/10, while MC Reasoning remained negative; semantic binding and RE2 each introduced one MC correct?wrong transition. TF RE2 raised all three TF metrics substantially but the combined route had one MC and one TF correct?wrong entry.

Failure analysis separated decision, evidence, and reporting errors. MC pointwise kept zero correct?wrong and fixed one Baseline error, but ?brief reason? shortened recovery/contrast evidence and exact-number narration created `1.03`?`10.30`-type hallucinations. The TF correct?wrong entry began with a semantic True inside `<think>` but lacked an explicit verdict; the extractor encountered ?no sudden spikes? and returned False. The same response hallucinated ?21.70, so the repair needs both explicit verdict and qualitative evidence.

## 2026-07-27 ? Cycle 3 preregistration

Frozen seven Round-3 routes before new inference. Three MC components test qualitative pointwise, qualitative semantic binding, and salient-event/aftermath matching. Two TF components test RE2 plus explicit verdict, with and without a qualitative numeric guard. Two joint routes reuse exact components. OE remains byte-identical Baseline. The adaptive development96 union is reported pooled and by its 24/72 constituents; only strict passers can reach exposed holdout48.

## 2026-07-27 — Outer loop 3

Round 3 completed 480/480 GPU predictions and 336/336 new Judge dimensions. The assembled audit passed for 768 predictions and 1,768 scores: 1,754 top-logprob rows and 14 exact-20 fallbacks after canonical reuse.

Qualitative semantic MC is the first literature route to pass the full development gate. It improves pooled MC Final/Correctness/Reasoning by +0.256/+0.324/+0.097, is nonnegative on both constituent splits, and has no closed-task correct→wrong transition. It advances unchanged to exposed holdout48.

Universal qualitative TF RE2 improves pooled TF by +0.361/+0.379/+0.333 and fixes 14 parsed decisions, but still changes the positive spike-plus-sustained-rise case `series_000132:1` from True to False. The requested exact final verdict appears at the end in 0/33 cases; the model usually places `Answer:` at the beginning instead. The next cycle therefore tests question-surface routing and a natural prefix, not stronger universal formatting.

## 2026-07-27 — Cycle 4 preregistration

Frozen six routes. Four reuse existing components to test explicit-negation and broader non-anomaly routers with qualitative TF RE2, alone and with the successful semantic MC component. Two routes test an explicit-negation-only, no-reread TF prefix, alone and in the same joint candidate. Positive anomaly propositions and all OE records retain the exact Baseline prompt.

## 2026-07-27 — Outer loop 4

Round 4 added 96/96 GPU predictions for the no-reread prefix and 30/30 new `deepseek-v4-pro` dimensions; four other routes reused audited Round-3 components. The final audit passed for 672 predictions and 1,547 scores.

The broad non-anomaly router is rejected despite the highest pooled mean. It reduces screening TF Justification by 0.111 because ordinary words such as `normal` and `stable` over-route questions and compress multi-phase evidence. The explicit-negative router passes pooled and both constituent gates with zero closed-task correct→wrong transitions.

`lit_r4_06_joint_semantic_tf_neg_prefix` improves pooled MC by +0.256/+0.324/+0.097 and TF by +0.255/+0.242/+0.273, while retaining exact Baseline OE. All 15 routed TF responses have the requested unambiguous first-answer prefix. It advances with the MC-only `lit_r3_02_mc_semantic_qual` to the preregistered exposed holdout48 check.

## 2026-07-27 — Exposed holdout preregistration

Frozen exactly two candidates and the exact-component assembly rule before new holdout inference. Both share the semantic qualitative MC component. The joint candidate adds the narrow, no-reread TF prefix only when an explicit grammatical negation is present; all other TF and all OE records reuse exact Baseline. A candidate must retain all ten holdout metrics, have zero closed-task correct→wrong transitions, satisfy prefix compliance, and pass the provenance/Judge audit before it can be locked for the one permitted paper140 run.
## 2026-07-27 — Exposed holdout outcome

Completed 96/96 GPU predictions and 54/54 new `deepseek-v4-pro` dimensions. The assembled audit passed for 144 predictions and 333 scores. Both candidates failed: the shared semantic qualitative MC component reduced MC Final/Correctness/Reasoning by 0.076/0.059/0.118 and changed two Baseline-correct normal cases to wrong anomalous choices. The joint TF prefix component independently improved all three TF dimensions by +0.275/+0.250/+0.313 and produced four wrong→correct transitions, but one routed response missed the exact `Answer:` prefix. No candidate was locked.

Failure inspection showed that semantic binding repaired a genuine localized-oscillation case (`series_000022:1`) but overdiagnosed ordinary alternating/random fluctuation (`series_000139:0`) and ordinary troughs (`series_000141:1`). The missing factor is an anomaly threshold between semantic option matching and evidence classification.

## 2026-07-27 — Outer loop 5 and final-cycle preregistration

Frozen the final four new modes under the 30-mode cap. Two MC mechanisms test a conservative structured-signature guard and a status-before-shape decomposition. Their joint versions reuse the already frozen explicit-negation TF RE2 route; OE remains exact Baseline. The existing TF-only route `lit_r4_01_tf_neg_re2` is included as a zero-new-mode fallback because it preserves MC/OE exactly and passed every development gate. Development96, both 24/72 constituents, and exposed holdout48 must independently have all ten nonnegative deltas and zero correct→wrong transitions before a unique candidate may reach paper140.
