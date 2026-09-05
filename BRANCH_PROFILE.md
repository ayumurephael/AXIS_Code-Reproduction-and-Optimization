# Branch profile: multi-axis-VL-ablation

This branch is created from the exact trained Multi-AXIS-VL source commit
`303ca1d49303eff4ab161fafd6117c3f1c010b3a`. It adds inference-time evidence
interface interventions for the predeclared Epoch-17 ablation study and does
not change the checkpoint parameters or training objective.

## Fixed experiment identity

- Checkpoint: Epoch 17, SHA-256
  `3a1c6e912371635f5f8b2f736224d42b0f7d023eab044fd078ffae745d2937be`.
- Dataset: `eval_478new.jsonl`, 478 rows, SHA-256
  `a5f919294545cc96b0d3a834d9dd4b223c440a26f32edd40d8eb663571b75a26`.
- Generation: beam 5, `do_sample=false`, `max_new_tokens=1000`.
- Judge: `qwen3-30b-a3b-instruct-2507`, non-thinking, temperature 0,
  top-5 score-token logprobs with fail-closed probability-mass audit.
- The existing full `multi_axis` row is reused; only the ten ablated variants
  are newly inferred and scored.

## Allowed change

Only the visible evidence interface and corresponding soft-token injection may
change. Every intervention deletes both the component's prompt material and
its placeholder; it never substitutes a zero vector or mask token. The exact
variant matrix is defined in `src/models/MultiAXIS/ablation.py` and tested by
`tools/multi_axis/test_ablation.py`.

Canonical execution and audit commands are in `ABLATION_PROTOCOL.md`.
