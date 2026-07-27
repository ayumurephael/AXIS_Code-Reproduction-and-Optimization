# Preregistered protocol ? Literature Round 3

Status: frozen before Round-3 GPU inference or new Judge calls.

## Data status

Use the union of the exposed screening24 and validation72 partitions (`development96.json`): 96 QA records. Both pools informed candidate design and are adaptive development resources, not untouched confirmation. Report pooled results and both constituent splits separately. The exposed holdout48 is used only if a route passes all development gates. Candidate paper140 outputs remain ungenerated.

## Seven candidate routes

| Route | MC | TF | OE | Mechanism |
|---|---|---|---|---|
| `lit_r3_01_mc_pointwise_qual` | pointwise + qualitative support | Baseline | Baseline | repair reasoning without changing decisions |
| `lit_r3_02_mc_semantic_qual` | semantic binding + qualitative support | Baseline | Baseline | repair local-event suppression |
| `lit_r3_03_mc_salient_aftermath` | salient event + aftermath | Baseline | Baseline | match compound anomaly descriptions |
| `lit_r3_04_tf_re2_verdict` | Baseline | RE2 + final verdict | Baseline | isolate parser/report repair |
| `lit_r3_05_tf_re2_qual_verdict` | Baseline | RE2 + qualitative evidence + verdict | Baseline | repair parser and numeric hallucination |
| `lit_r3_06_joint_pointwise_tf_qual` | route 01 MC | route 05 TF | Baseline | component-wise joint candidate |
| `lit_r3_07_joint_semantic_tf_qual` | route 02 MC | route 05 TF | Baseline | alternate joint candidate |

All routes preserve the released opening, Values ? Per-Step Analysis ? `Overall Summary Hints`, all 30 Fixed tokens, the released generation boundary, and exact Baseline OE. They add no Evidence Contract and no numeric scaling/alignment explanation.

## Fixed execution

- Released checkpoint SHA-256: `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.
- No training or weight updates; no local GPU computation.
- Three authorized free GPUs; series batching, beam 5, `--skip-loss`.
- Formal Judge only `deepseek-v4-pro`, high reasoning, 4096 completion tokens, strict system-CA TLS verification.
- Final score-position top-logprobs; exact-20 fallback only if a complete 1?5 distribution is unavailable.
- Canonically reuse each byte-identical Baseline or candidate task component.

## Expected work

Raw GPU generation: five source modes ? 96 records = 480 predictions.

Unique new components, given development96 counts MC=34, TF=33, OE=29:

- 34 MC ? three MC components = 102 predictions / 204 dimensions;
- 33 TF ? two TF components = 66 predictions / 132 dimensions;
- total = 168 predictions / 336 new Judge dimensions.

Final route assembly:

- seven routes ? 96 = 672 candidate predictions;
- seven routes ? 221 = 1,547 candidate scores;
- plus 96 Baseline predictions / 221 Baseline scores for pooled comparison.

## Gate

A route passes only if:

1. pooled development96 has all ten Table-I deltas >= 0;
2. screening24 and validation72 each have nonnegative MC/OE/TF Final deltas;
3. every split has zero MC and TF correct?wrong transitions;
4. TF verdict candidates end in an unambiguous explicit verdict on every TF item;
5. prediction and Judge audits pass with no missing/duplicate keys, model/provider mismatch, invalid prompt hashes, or unapproved score methods.

Rank passers by nonnegative dimensions, worst delta, mean delta, then closed-task wrong?correct. Advance at most two mechanistically distinct routes to the exposed holdout48.

## Outer loop

After all seven routes, inspect every changed MC/TF decision, the largest Judge gains/losses, exact-number hallucinations, response length, and explicit-verdict compliance. If no route passes, begin another mechanism-driven cycle within the 30-mode cap without touching paper140.
