# Preregistered protocol — Literature Round 4

Status: frozen before assembling routed development results, before any new Round-4 GPU inference, and before any new Judge call.

## Mechanism

Round 3 showed that TF RE2 improves many negation/polarity cases but suppresses one legitimate positive anomaly. A deterministic question-surface router now protects positive anomaly propositions with the exact released prompt and applies a TF intervention only to negative/non-anomaly propositions. Routing uses question text only; it never reads values, latent tokens, labels, responses, or Judge scores.

### Router A: explicit negative cues

Activate when the lower-cased proposition contains a token or phrase from:

`no`, `not`, `without`, `absence`, `lack`, `neither`, `nor`, `cannot`, `can't`, `doesn't`, `isn't`, `aren't`, `wasn't`, `weren't`.

### Router B: non-anomaly language

Activate on Router A or when the proposition explicitly characterizes the window as `normal`, `stable`, `consistent`, `regular`, `expected`, or `typical`. These are surface-semantic cues, not predicted labels.

## Six frozen routes

| Route | MC | TF | OE |
|---|---|---|---|
| `lit_r4_01_tf_neg_re2` | Baseline | qualitative RE2 only under Router A | Baseline |
| `lit_r4_02_joint_semantic_tf_neg_re2` | Round-3 semantic qualitative | route 01 TF | Baseline |
| `lit_r4_03_tf_nonanomaly_re2` | Baseline | qualitative RE2 only under Router B | Baseline |
| `lit_r4_04_joint_semantic_tf_nonanomaly_re2` | Round-3 semantic qualitative | route 03 TF | Baseline |
| `lit_r4_05_tf_neg_prefix` | Baseline | no-reread prefix rule only under Router A | Baseline |
| `lit_r4_06_joint_semantic_tf_neg_prefix` | Round-3 semantic qualitative | route 05 TF | Baseline |

The new no-reread TF text is:

> Preserve the proposition's polarity exactly as written. Decide whether the complete statement is supported, and begin with exactly "Answer: True." or "Answer: False." Then give a qualitative reason consistent with that label. Do not quote exact values or step numbers unless the question asks for them.

All non-activated task components are byte-identical Baseline. The prompt keeps the released opening, Values → Per-Step Analysis → `Overall Summary Hints`, all 30 Fixed tokens, and the released generation boundary. It adds no Evidence Contract, scale explanation, evidence reordering, fixed-token rename, or answer prefill.

## Development execution

- Data: exposed development96, reported pooled and as screening24 / validation72.
- Routes 01–04 are assembled from already-audited byte-identical Baseline, Round-3 semantic-MC, and qualitative-TF-RE2 components; no new model or Judge call is permitted for those components.
- Routes 05–06 require one canonical no-reread TF component only on Router-A records. Run it with the released checkpoint on authorized GPUs and score new dimensions only with `deepseek-v4-pro`.
- Canonically reuse every identical component and preserve prompt SHA-256 provenance.

## Gate and advancement

A route passes only if:

1. all ten pooled Table-I deltas are nonnegative;
2. screening24 and validation72 each have nonnegative MC/OE/TF Final deltas;
3. every split has zero MC and TF correct→wrong transitions;
4. every activated no-reread TF response begins with an unambiguous `Answer: True.` or `Answer: False.`;
5. all prediction/Judge/component audits pass.

Rank passers by nonnegative dimensions, worst delta, mean delta, and then closed-task wrong→correct. At most one Round-4 joint route and the already advanced `lit_r3_02_mc_semantic_qual` may enter exposed holdout48. No candidate `paper140` output may be generated before one unique candidate is locked after the exposed robustness check.
