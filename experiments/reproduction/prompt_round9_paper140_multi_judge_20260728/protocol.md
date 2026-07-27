# AXIS Round 9 paper140 Multi-Judge Evaluation Protocol

Status: locked before any new Judge request.

## Objective

Evaluate the released AXIS Baseline prompt and all three final Round 9 prompt
routes on the author-compatible `paper140` subset under three independent
G-Eval Judges:

1. `deepseek-v4-pro`
2. `qwen3.5-397b-a17b`
3. `qwen3-30b-a3b-instruct-2507`

Gemini is explicitly excluded by the user. Scores from different Judges are
reported as separate measurement scales and are never pooled.

## Frozen prompt conditions

1. `base`
2. `prompt_final_round9_mc_oe`
3. `prompt_final_round9_tf_oe`
4. `prompt_final_round9_joint`

The exact prompts, question-type routing, two-pass OE revision prompt, and
Round 9 deterministic adoption gate are the already-published definitions in
`../prompt_final_full284_deepseek_v4_pro_20260727/prompt_final.md`. No prompt,
response, route, or adoption threshold may be changed after this protocol is
locked.

## Frozen prediction input

- Dataset: `paper140`, 140 QA / 70 series.
- Manifest:
  `../manifests/paper140.json`.
- Source:
  `../prompt_final_full284_deepseek_v4_pro_20260727/artifacts/final_round9/variants/predictions.jsonl`.
- The four modes are filtered exactly by the 140 manifest record IDs.
- AXIS inference is not rerun. The user explicitly approved exact extraction
  from the audited full284 predictions.
- Checkpoint SHA-256:
  `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.

Before scoring, the extracted file must pass a fail-closed audit for 560
prediction rows, 140 per mode, with no missing, duplicate, or empty response.

## Exact-response de-duplication

Within each Judge, G-Eval inputs that have the same record ID, question,
reference answer, question type, and generated response are identical. Such
rows are scored once and expanded back to all four prompt modes by exact
response SHA-256. The expanded result must contain 1,340 dimension rows
(`335 × 4`) and retain provenance indicating whether a score was directly
judged or exactly reused.

This is a variance-control rule, not a score-selection rule. No score is reused
across different Judges.

## Judge configurations

### DeepSeek

- Model: `deepseek-v4-pro`
- Author AXIS G-Eval dimensions, weights, rubric, and layout.
- Primary readout: final score-token 1–5 logprob expectation.
- Fallback: exactly 20 valid sampled integer scores only when the complete
  score-token distribution is unavailable.
- Maximum completion tokens: 4096.

### Qwen3.5

- Model: `qwen3.5-397b-a17b`
- `enable_thinking=false` in both primary judgment and deterministic readout.
- Author AXIS G-Eval prompt/rubric.
- `top_logprobs=5`, seed 72.
- Missing-score-mass upper bound: `1e-6`.
- If the primary score token exceeds the bound, use the deterministic JSON
  A–E readout of the same model and same already-generated judgment.
- No sampling fallback; unresolved rows fail closed.

### Qwen3 30B

- Model: `qwen3-30b-a3b-instruct-2507`
- Non-thinking model; omit the `enable_thinking` extension (`auto`).
- Otherwise use the same Qwen score-token and deterministic-readout protocol
  as Qwen3.5.
- No sampling fallback; unresolved rows fail closed.

## Execution and security

- API runners and formal audits execute on an authorized GPU server. No local
  GPU work and no training are performed.
- Credentials are read from the user-authorized local instruction files,
  injected only into restricted remote temporary files or process
  environments, and removed after completion.
- Secrets, authorization headers, and sensitive server configuration must not
  appear in Git, logs, result JSONL, or the final report.

## Required outputs

For each Judge:

1. unique prediction inputs;
2. raw score journal;
3. expanded four-mode score file;
4. fail-closed audit JSON;
5. Table-I JSON and Markdown;
6. method/model/provider/prompt-hash counts and file SHA-256.

The final `prompt_round9_paper140_multi_judge.md` must include:

- all Baseline and Round 9 prompts and routing/adoption rules;
- one complete Table-I score table per Judge;
- absolute deltas from that Judge's own Baseline;
- evaluation configuration, coverage, methods, integrity checks, and
  limitations.

