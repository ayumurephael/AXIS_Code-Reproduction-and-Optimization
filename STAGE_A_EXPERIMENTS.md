# AXIS Prompt Stage-A Experiment Contract

This branch evaluates prompt-only changes with the released author checkpoint.
No training or parameter update is permitted in Stage A.

## Frozen evaluation protocol

- Git base: `origin/main@e8c1aee59bb98cda9b37445bf2eb23619f374d54`
- Checkpoint: released `axis_qa_by_pretrain_best_accelerate/model_optimizer.pth`
- Checkpoint SHA-256:
  `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`
- Evaluation subset: `paper140` (`seed=42`, 70 series, 140 QA records)
- Batching: all QA records belonging to one series in one batch
- Generation: deterministic beam search, `num_beams=5`,
  `max_new_tokens=1000`, `repetition_penalty=1.15`,
  `no_repeat_ngram_size=3`, `length_penalty=1`
- Teacher-forced loss: skipped
- Judge provider/model: DeepSeek only, `deepseek-v4-pro`
- Judge rubric and Table-I weights: released paper-compatible G-Eval rubric

Window values, local-hint placeholders, fixed-hint count, prompt block order,
and value scaling/rounding remain identical to the released prompt unless a
mode below explicitly names the changed factor.

## Formal mode matrix

| Order | Mode | Changed factor(s) |
|---:|---|---|
| 0 | `base` | None; released prompt and released inference boundary |
| 1a | `answer_boundary` | Append an explicit EOS token and tokenized literal `Answer:` to the generation prefix |
| 1b | `answer_boundary_wo_fixed` | Mode 1a plus removal of all fixed-hint placeholders |
| 2 | `task_protocol` | Add only the active question-type output protocol |
| 3 | `fixed_role` | Rename `Overall Summary Hints` to `Learned Task Guidance/Shared Task-Control Tokens` |
| 4 | `fixed_role_evidence_contract` | Mode 3 plus the exact Evidence Contract from the experiment specification |
| 5 | `combined_234` | Combine modes 2, 3, and 4; retain the released generation boundary |
| 6 | `answer_boundary_combined_234` | Add `EOS + Answer:` boundary alignment to `combined_234` |

The Evidence Contract is reproduced verbatim from the controlling Stage-A
specification. In particular, its “same row” wording is retained even though
Stage A deliberately keeps the released two-block value/local-hint layout.
This inconsistency must be treated as part of the tested prompt, not silently
corrected after seeing results.

## Formal inference command

```bash
torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.run_inference \
  --checkpoint <released-checkpoint> \
  --data <AXIS_qa_test> \
  --subset paper140 \
  --modes \
    base \
    answer_boundary \
    answer_boundary_wo_fixed \
    task_protocol \
    fixed_role \
    fixed_role_evidence_contract \
    combined_234 \
    answer_boundary_combined_234 \
  --batching series \
  --skip-loss \
  --output <prediction-directory>
```

Every formal run must preserve its prediction shards, merged predictions,
run manifest, exact code commit, environment inventory, checkpoint and prompt
specification hashes, G-Eval journals, aggregate tables, and integrity audit.
