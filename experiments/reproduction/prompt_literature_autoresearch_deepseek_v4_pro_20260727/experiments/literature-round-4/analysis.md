# Literature Round 4 analysis

## Integrity and scope

- Data: exposed `development96`, reported pooled and as screening24 / validation72 constituents.
- Four lexical-router routes reused already audited Round-3 components. Two no-reread prefix routes added 96/96 raw GPU predictions.
- The prefix source contributed 15 routed negative TF predictions and 30 new Judge dimensions.
- Final assembly: 672 predictions and 1,547 scores; the extended audit passed for every prompt hash, key, model/provider, and score method.
- Judge: 1,530 `final_score_top_logprobs` rows and 17 `exact_sample_mean_20` fallbacks, all from `deepseek-v4-pro`.
- Positive-anomaly TF, every OE record, and the non-target task components were exact Baseline reuse.
- Candidate `paper140` outputs remain ungenerated.

## Pooled and constituent result

| Route | Pooled nonnegative | Pooled mean Δ | Screening nonnegative | Validation nonnegative | Closed-task correct→wrong |
|---|---:|---:|---:|---:|---:|
| `lit_r4_06_joint_semantic_tf_neg_prefix` | 10/10 | +0.145 | 10/10 | 10/10 | 0 |
| `lit_r4_02_joint_semantic_tf_neg_re2` | 10/10 | +0.143 | 10/10 | 10/10 | 0 |
| `lit_r4_04_joint_semantic_tf_nonanomaly_re2` | 10/10 | +0.167 | 9/10 | 10/10 | 0 |
| `lit_r4_05_tf_neg_prefix` | 10/10 | +0.077 | 10/10 | 10/10 | 0 |
| `lit_r4_01_tf_neg_re2` | 10/10 | +0.076 | 10/10 | 10/10 | 0 |
| `lit_r4_03_tf_nonanomaly_re2` | 10/10 | +0.099 | 9/10 | 10/10 | 0 |

`lit_r4_06_joint_semantic_tf_neg_prefix` is the selected Round-4 route. Its pooled MC Final/Correctness/Reasoning deltas are +0.256/+0.324/+0.097; TF Final/Correctness/Justification deltas are +0.255/+0.242/+0.273; OE is exact Baseline. On screening24, MC improves +1.075/+1.375/+0.375 and TF improves +0.400/+0.444/+0.333. On validation72, MC improves +0.004/+0.001/+0.012 and TF improves +0.200/+0.167/+0.250. It has 14 parsed wrong→correct transitions and no correct→wrong transition on pooled development96.

The no-reread prefix is checkpoint-compatible: every one of the 15 routed predictions begins, after whitespace, with an unambiguous `Answer: True.` or `Answer: False.`. This fixes the Round-3 placement mismatch without imposing a format change on positive-anomaly TF questions.

## Failure cases and mechanism

### Broad non-anomaly routing is over-inclusive

The broad router adds words such as `normal`, `stable`, `consistent`, and `expected`. It looks better pooled, but fails the constituent gate: screening TF Justification falls by 0.111.

On `series_000043:0`, the Baseline correctly describes the full pattern as a gradual increase to a peak followed by a decline toward zero. The broadly routed response compresses this into a “smooth and gradual downward trend.” The True decision remains correct, but the justification loses an important phase of the trajectory and falls from 5/5 to 3/5. The broad lexical set therefore routes questions because of ordinary descriptive vocabulary rather than because their logical surface creates a polarity risk.

### Explicit negative routing confines the intervention

The narrow router triggers on explicit grammatical negation such as `no`, `not`, `absence`, and `without`. It routes `series_000043:0`, whose proposition has nested negative semantics, but does not route the positive anomaly proposition `series_000132:1`. The latter remains byte-identical Baseline and therefore preserves the real spike-plus-sustained-increase decision that universal TF RE2 lost in Round 3.

The remaining Judge losses are explanation compression, not decision reversals. For instance, the prefix response for `series_000043:0` still oversimplifies the two-phase curve, but the route's aggregate TF Justification rises and no Baseline-correct decision is lost. This is preferable to universal or broad routing because it protects the known positive-anomaly failure class.

### Why the natural prefix beats a final-string requirement

Round 3 requested an exact verdict at the end, but the model obeyed that placement on 0/33 TF records and instead naturally emitted `Answer:` first. Round 4 follows the checkpoint's preferred decoding position. The intervention is short, avoids repeating the question, and asks for qualitative support. It improves label observability without reallocating attention through RE2 or a long formatting contract.

## Outer-loop synthesis and advancement

1. Reject both broad non-anomaly routes despite their pooled mean: they fail a constituent TF Justification guard.
2. Retain exact Baseline OE; no OE intervention has passed the decision-and-style guards.
3. Advance exactly one Round-4 joint route, `lit_r4_06_joint_semantic_tf_neg_prefix`.
4. Also carry the already advanced MC-only `lit_r3_02_mc_semantic_qual` as the ablation that isolates whether the TF prefix adds robust value.
5. Freeze both prompts before the exposed holdout48 robustness check. Do not inspect or tune on holdout responses before the preregistered comparison is complete.
