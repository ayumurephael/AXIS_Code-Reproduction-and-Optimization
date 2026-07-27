# Exposed holdout robustness analysis

## Outcome

Both preregistered candidates failed and are rejected. The audit contains 144 predictions and 333 dimension scores: 330 `deepseek-v4-pro` score-position top-logprob rows and three exact-20 canonical-reuse rows.

| Mode | Nonnegative | Worst Δ | Mean Δ | MC Final Δ | MC Corr. Δ | MC Rsn. Δ | OE four Δ | TF Final Δ | TF Corr. Δ | TF Justif. Δ | C→W | W→C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lit_r3_02_mc_semantic_qual` | 7/10 | -0.118 | -0.025 | -0.076 | -0.059 | -0.118 | all 0 | 0 | 0 | 0 | 2 | 1 |
| `lit_r4_06_joint_semantic_tf_neg_prefix` | 7/10 | -0.118 | +0.058 | -0.076 | -0.059 | -0.118 | all 0 | +0.275 | +0.250 | +0.313 | 2 | 4 |

The joint route also missed its strict prefix gate on one of ten routed TF records (`series_000111:1`, beginning `False.` rather than `Answer: False.`). The semantic decision was still parseable; the rejection is already forced by the MC regressions.

## Concrete MC failure cases

### `series_000139:0` — ordinary fluctuation overdiagnosed

Gold and Baseline select D, no anomaly. The semantic candidate selects C, “cluster of irregular fluctuations,” because it treats alternating signs and a few larger swings as sufficient anomaly evidence. Its own explanation admits that there is no persistent shift or spike. The prompt successfully binds the option text but lacks a threshold for when irregularity becomes unexpected.

### `series_000141:1` — ordinary troughs relabeled as transient events

Gold and Baseline select A, normal consistent behavior. The candidate selects D and describes negative troughs as sudden drops followed by recovery. It converts level variation into event semantics without first establishing that the local contrast is unexpected relative to the learned context.

### `series_000022:1` — genuine gain that must be preserved

Baseline inconsistently selects D while arguing that no option fits. The candidate correctly selects B, localized rapid oscillations, and gains about +3.7 MC Final. A blanket “prefer normal” repair would destroy this benefit. The required repair is a structured anomaly threshold, not a normality prior.

## Mechanistic conclusion

“Choose by complete option text” improves symbol/semantic binding but allows anomaly-worded distractors to define the latent classification. The model then rationalizes normal alternating values as a cluster or isolated events. The next prompt must separate anomaly-status inference from option-shape matching, or require a structured signature beyond raw magnitude/irregularity. Long contracts and more numerical detail are specifically excluded because earlier cycles showed attention competition and numeric hallucination.
