# Candidate matrix — Literature Round 1

All modes retain the released opening, evidence order, `Overall Summary Hints` title, 30 Fixed tokens, and generation boundary.

| Mode | MC component | TF component | OE component | Main hypothesis |
|---|---|---|---|---|
| `lit_r1_01_mc_semantic_bind` | semantic text → letter | Base | Base | symbol binding |
| `lit_r1_02_mc_pointwise` | independently test options | Base | Base | reduce holistic/position shortcut |
| `lit_r1_03_mc_re2` | repeat question | Base | Base | comprehension via RE2 |
| `lit_r1_04_tf_minimal` | Base | coherent polarity | Base | label/rationale consistency |
| `lit_r1_05_tf_clause` | Base | clause + negation check | Base | composite-statement errors |
| `lit_r1_06_tf_re2` | Base | repeat question | Base | comprehension via RE2 |
| `lit_r1_07_oe_direct` | Base | Base | direct requested answer | speech-act coverage |
| `lit_r1_08_oe_re2` | Base | Base | repeat question | comprehension via RE2 |
| `lit_r1_09_triplet_minimal` | semantic binding | coherent polarity | direct answer | composition of shortest winners |
| `lit_r1_10_triplet_re2` | repeat | repeat | repeat | task-general RE2 |
| `lit_r1_11_decoupled` | decide then report | decide then report | analyze then answer | decision/format separation |
| `lit_r1_12_mc_bind_re2` | semantic binding + repeat | Base | Base | complementary MC mechanisms |

## Exact inserted rules

- MC semantic binding: “Choose the option by its complete text before mapping it to A, B, C, or D. Report that option once and give a brief reason from the supplied evidence.”
- MC pointwise: “Test each option's complete claim independently against the supplied evidence. Choose the best-supported option by its text, then report its attached letter and a brief reason.”
- TF minimal: “Judge whether the complete statement as written is true. Report True or False once, followed by a brief reason that supports the same truth value.”
- TF clause: “Check every required clause and negation in the statement. It is false if any required clause is contradicted; report one truth label and a brief consistent reason.”
- OE direct: “Answer exactly what the question asks, using evidence from this window. Include the requested conclusion or assessment and the brief reason needed to support it.”
- RE2 suffix: the full question is repeated verbatim after `Read the question again:`.

The three decision/report-decoupled wordings are stored verbatim in `src/models/AXIS/prompt_stage_a.py` and will be rendered into the prompt catalog before execution.


# Candidate matrix ? Literature Round 3

| Mode | MC component | TF component | OE | Targeted failure |
|---|---|---|---|---|
| `lit_r3_01_mc_pointwise_qual` | pointwise + qualitative pattern/aftermath/contrast | Base | Base | reasoning loss with safe decisions |
| `lit_r3_02_mc_semantic_qual` | semantic binding + qualitative support | Base | Base | local-spike suppression |
| `lit_r3_03_mc_salient_aftermath` | salient local change + immediate aftermath | Base | Base | compound anomaly matching |
| `lit_r3_04_tf_re2_verdict` | Base | RE2 + explicit final verdict | Base | parser ambiguity |
| `lit_r3_05_tf_re2_qual_verdict` | Base | RE2 + qualitative guard + verdict | Base | parser + numeric hallucination |
| `lit_r3_06_joint_pointwise_tf_qual` | route 01 MC | route 05 TF | Base | compositional Pareto route |
| `lit_r3_07_joint_semantic_tf_qual` | route 02 MC | route 05 TF | Base | alternate compositional route |

All exact prompt strings and SHA-256 identities are frozen in `experiments/literature-round-3/prompt_catalog.md` and `.json`.
