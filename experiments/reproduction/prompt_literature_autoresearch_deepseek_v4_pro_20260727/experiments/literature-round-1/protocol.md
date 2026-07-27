# Preregistered protocol — Literature Round 1

Status: frozen before GPU inference and Judge calls.

## Objective

Falsify or advance 12 prompt-only candidates while preserving the released checkpoint and evaluation semantics.

## Data and leakage status

Round 1 reuses the previous 24-QA / 12-series selection split. This split and every other non-paper140 split have already been inspected in earlier research, so Round 1 is exploratory. Results cannot be described as untouched validation.

Candidate outputs on the official 140-QA test set remain ungenerated. They are reserved for one final locked candidate.

## Fixed controls

- Released AXIS checkpoint only; verify its SHA-256 before inference.
- No training or weight updates.
- GPU inference only on an authorized server from the private GPU information file.
- Formal G-Eval + Judge model: `deepseek-v4-pro` only.
- Same decoding, checkpoint, dataset serialization, Judge rubrics, retries, and aggregation as the existing Baseline.
- Baseline predictions/scores may be reused only when provenance and QA identity match exactly.
- All runs require prediction-count, score-count, task-count, duplicate, missing-ID, model-name, and fallback audits.

## Candidate controls

Each candidate is one registered mode in `candidate-matrix.md`. No wording may be changed after seeing its results without assigning a new mode ID. Base-routed task families must be byte-identical to the released prompt and may reuse proven Baseline outputs/scores.

Forbidden in Round 1:

- Fixed-token header rename;
- Evidence Contract;
- numeric rescaling instructions;
- local/Fixed token removal;
- evidence interleaving or reordering;
- `Answer:` or EOS generation prefill;
- chain-of-thought requirement;
- unregistered multi-pass voting.

## Ranking and advancement

For each candidate, compare all ten Table-I metrics against the Baseline.

Primary lexicographic rank:

1. number of dimensions with delta >= 0;
2. largest worst-dimension delta;
3. largest mean delta.

Decision guards:

- report MC and TF correct→wrong and wrong→correct transitions;
- a closed-task correct→wrong transition blocks automatic advancement unless a larger replicated wrong→correct gain and all family metrics are nonnegative justify explicit review;
- no task-family Final metric may fall by more than 0.05 in Round 1;
- formatting/parse diagnostics are secondary and cannot offset content-score regression.

Advance at most four modes. Preference goes to mechanistically distinct candidates rather than lexical duplicates.

## Subsequent gates

Round 2: internal 72-QA validation; require all ten deltas >= 0 and no closed-task correct→wrong transition.

Round 3: exposed 48-QA robustness check; same all-ten and transition gates. It is not untouched confirmation.

Final: lock exactly one candidate before paper140 inference. Success requires every paper140 Table-I metric >= the locked Baseline. All ten strictly higher is preferred but not required.

## Outer loop and stopping

After every 5–10 completed modes, update the research state/log/findings and classify failures by case mechanism. New modes must target an observed mechanism and receive new IDs.

Stop when:

- the final paper140 criterion is met; or
- no mode survives after at least three mechanistically distinct cycles and at most 30 new modes.

If no candidate survives, report a rigorous negative result and do not evaluate a candidate on paper140.
