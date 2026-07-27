# Literature Round 2 validation analysis

## Outcome

No candidate passed the preregistered 10/10 and zero-regression gate, so no route advanced to the 48-QA robustness pool and paper140 remained untouched.

| Rank | Route | Nonnegative | Worst delta | Mean delta | Closed-task correct?wrong |
|---:|---|---:|---:|---:|---:|
| 1 | `lit_r1_01_mc_semantic_bind` | 9/10 | ?0.077 MC Reasoning | +0.010 | 1 |
| 2 | `lit_r1_02_mc_pointwise` | 9/10 | ?0.231 MC Reasoning | +0.003 | 0 |
| 3 | `lit_r2_01_mc_tf_re2_safe` | 8/10 | ?0.154 MC Reasoning | +0.128 | 2 |
| 4 | `lit_r1_03_mc_re2` | 8/10 | ?0.154 MC Reasoning | ?0.013 | 1 |

The audit passed for 360 predictions and 830 scores. All scores came from `deepseek-v4-pro`; the 204 new component scores were all `final_score_top_logprobs`. The reused Baseline contains 162 top-logprob rows and four previously audited exact-20 fallbacks.

## Failure cases

### MC semantic binding: symbol binding helped, evidence salience shifted

`series_000089:1` changed from the correct C to D. Baseline identified a pronounced upward spike, while the semantic-binding prompt reinterpreted positive/negative alternation as a non-anomalous oscillation and explicitly denied the spike. The same route corrected `series_000137:1`, so semantic binding is not uniformly harmful; it exchanges one option-symbol/meaning error for one local-event suppression error.

### MC pointwise: decisions were safe, reasoning became shorter and less reliable

This was the only MC route with zero correct?wrong transitions and it corrected `series_000137:1`. Its aggregate MC Correctness rose by +0.192, but Reasoning fell by ?0.231. The worst example, `series_000000:1`, retained the correct D but rendered an encoded `1.03`-like event as `10.30` and described that increase as a drop. Other losses omitted recovery, boundary, or competing-option evidence. The phrase ?brief reason? therefore acts as a content bottleneck, while explicit exact-number narration amplifies latent-token decoding errors.

### MC RE2: rereading can overwrite a valid first interpretation

`series_000103:1` changed from the correct C to A. The Baseline recognized an upward spike followed by sustained movement; RE2 denied both and claimed a narrow, normal range. RE2 fixed `series_000089:0`, but the paired failure shows that a second reading can reconstruct the evidence rather than merely verify the question.

### TF RE2: large Judge gains, one semantic/numeric and parser failure

TF RE2 improved TF Final/Correctness/Justification by +0.445/+0.298/+0.667 and corrected nine Baseline TF decisions. The single TF correct?wrong entry, `series_000109:1`, is partly a reporting/parser failure: the response begins `<think> True`, but never emits an answer boundary; the extractor encounters ?no sudden spikes? before its generic `true` fallback and records False. The response also hallucinates a smooth endpoint of ?21.70 instead of about ?1.77, lowering Judge justification. A final explicit verdict plus qualitative-only evidence is therefore a targeted repair, not a generic strict-format intervention.

## Outer-loop synthesis

1. Preserve MC pointwise comparison because it had zero decision regressions, but remove ?brief reason? and require qualitative shape plus continuation/recovery.
2. Test semantic binding with the same qualitative evidence obligation to determine whether the single C?D regression was caused by insufficient local-event support.
3. Test a distinct salient-event/aftermath matcher for compound anomaly options.
4. Preserve TF rereading, but add an explicit final verdict; separately add a qualitative evidence guard to suppress numeric hallucination.
5. Keep OE exactly Baseline. No Evidence Contract, fixed-token rename, answer prefill, or evidence reordering is introduced.
