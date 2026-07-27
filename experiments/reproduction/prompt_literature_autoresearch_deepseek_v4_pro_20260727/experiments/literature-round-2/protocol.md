# Preregistered protocol — Literature Round 2

Status: frozen before validation GPU inference and new Judge calls.

## Data status

Use the previous 72-QA / 36-series validation manifest. This pool was exposed by the preceding prompt search and is internal validation, not untouched confirmation. Candidate paper140 outputs remain ungenerated.

## Four candidate routes

| Route | MC | TF | OE | Reason |
|---|---|---|---|---|
| `lit_r1_03_mc_re2` | RE2 | Baseline | Baseline | strongest safe screen |
| `lit_r1_02_mc_pointwise` | pointwise option claims | Baseline | Baseline | distinct MC verification |
| `lit_r1_01_mc_semantic_bind` | option semantics → letter | Baseline | Baseline | conservative symbol binding |
| `lit_r2_01_mc_tf_re2_safe` | RE2 | RE2 | Baseline | explicit TF risk review |

The combined route is a new named mode, but its task components are byte-identical to the separately registered MC RE2, TF RE2, and Baseline prompts.

## Fixed execution

- Released checkpoint SHA-256 must remain `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.
- No training or weight updates.
- Three authorized free GPUs when available; series batching, beam 5, `--skip-loss`.
- Formal Judge only `deepseek-v4-pro`, high reasoning, 4096 completion tokens, strict system-CA TLS verification.
- Score readout: final score-position top-logprobs; exact-20 fallback only when complete score probabilities are unavailable.
- Reuse the already-audited validation Baseline prediction/score for byte-identical Base components.
- Reuse one canonical new prediction/score for every byte-identical task component.

## Expected work

Raw GPU generation: 4 source modes × 72 records = 288 predictions.

Unique new components:

- 26 MC × three MC interventions = 78 predictions / 156 dimensions;
- 24 TF × one RE2 intervention = 24 predictions / 48 dimensions;
- total = 102 predictions / 204 Judge dimensions.

Final candidate assembly:

- 4 × 72 = 288 candidate predictions;
- 4 × 166 = 664 candidate scores;
- plus 72 Baseline predictions / 166 Baseline scores for comparison.

## Gate

A route passes only if:

1. all ten Table-I deltas versus the validation Baseline are >= 0;
2. every task-family Final delta is >= 0;
3. MC and TF have zero correct→wrong transitions;
4. prediction and Judge audits pass with no missing/duplicate keys, wrong model/provider, unapproved score method, or invalid prompt hashes.

Rank passers by nonnegative count, worst delta, then mean delta. Advance at most two mechanistically distinct routes to the exposed 48-QA robustness check.

The combined route is rejected if TF produces even one correct→wrong transition, regardless of net wrong→correct gains.

## Outer loop

After these four routes, update state/log/findings and inspect every changed closed-task decision plus the largest Judge wins/losses. If no route passes, begin a new mechanism-driven prompt cycle without touching paper140. If at least one passes, freeze the robustness-check candidates before generating 48-QA outputs.
