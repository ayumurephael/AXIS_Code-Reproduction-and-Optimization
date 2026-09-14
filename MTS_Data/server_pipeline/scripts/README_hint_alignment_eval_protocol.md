# Hint Alignment Eval Protocol

This note defines the default smoke-set protocol for checking whether AXIS hint
embeddings are genuinely aligned to the LLM, instead of merely perturbing
generation.

## Required Default Eval Metrics

For future eval summaries, always keep:

- `overall_with_open_acc`
- `explicit_answer_rate`
- `judgment_true_f1`

Also keep, when available:

- `open_decision_anomalous_f1`
- `answer_first_rate`
- `finish_reason`
- `hit_max_new_tokens`

## Standard Smoke Ablation

Use one fixed smoke dataset and run:

- `no hint`
- `true hint, ct=16, scale=0.05`
- `true hint, ct=16, scale=0.10`
- `true hint, ct=16, scale=0.20`
- `shuffled hint, ct=16, scale=0.05`
- `shuffled hint, ct=16, scale=0.10`
- `shuffled hint, ct=16, scale=0.20`

## Interpretation Rule

We want:

- real-hint runs to improve answer quality relative to the no-hint baseline
- shuffled-hint runs to fail to provide the same improvement

If shuffled hints behave similarly to real hints, that suggests the model is
being perturbed by extra embedding mass rather than using semantically aligned
hint information.

## Training Policy For Now

Do not change the loss function for this protocol yet.

This is an evaluation-only alignment check. Keep the current training loss and
use the smoke ablation to decide whether hint injection settings or hint
alignment need to be revisited first.
