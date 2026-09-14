# Denoised 9-Group Student-Answer Eval

This script is for full reruns of the original 9-group VL/text ablation
matrix, but with a quieter channel-hint setting:

- `max_channel_tokens=8`
- `channel_injection_scale=0.05`

Main runner:

- `mvaxis_run_denoised_student_answer_eval.py`

Why this is separate:

- The original raw-answer runner is the stable default workflow.
- The denoised rerun is a hypothesis test about whether channel-hint
  verbosity and injection strength are what caused poor formatting or
  wrong option selection on some datasets.

What it does:

1. Writes temporary LLM configs under the output root:
   - AXIS-hints enabled with denoised channel settings
   - AXIS-hints disabled for no-hint/raw-image variants
2. Replays the original 9 groups:
   - `all_hints`
   - `no_global_hints`
   - `no_channel_hints`
   - `score_image_note_all_hints`
   - `score_text_all_hints`
   - `raw_image`
   - `score_text_image_no_global_hints`
   - `score_text_image_no_channel_hints`
   - `score_text_nohints`
3. Runs teacher-aligned evaluation after generation finishes.

## Default Group Semantics Going Forward

For future evals, the default assumption is that every group includes an image.

Use this mapping:

- `all_hints`
  - original window image
  - AXIS global hints + channel hints
- `no_global_hints`
  - original window image
  - AXIS channel hints only
- `no_channel_hints`
  - original window image
  - AXIS global hints only
- `raw_image`
  - original window image only
- `score_text_all_hints`
  - original window image
  - anomaly-score text
  - AXIS global hints + channel hints
- `score_text_image_no_global_hints`
  - anomaly-score curve image
  - anomaly-score text
  - AXIS channel hints only
- `score_text_image_no_channel_hints`
  - anomaly-score curve image
  - anomaly-score text
  - AXIS global hints only
- `score_image_note_all_hints`
  - anomaly-score curve image
  - AXIS global hints + channel hints
- `score_text_nohints`
  - original window image
  - anomaly-score text
  - no AXIS hints

Historical note:

- older runs in this workspace are not guaranteed to match this convention
- earlier experiments often used text-only prompts for groups such as
  `all_hints` and `no_global_hints`
- do not compare those legacy outputs against new image-default runs without
  checking the saved run context first

Inputs expected:

- question JSONL
- teacher-answer JSONL
- base LLM config
- AXIS/TimeRCD config
- interval proposer / encoder checkpoint
- score-image manifest
- raw-image manifest

Recommended use:

- Structured100 full rerun
- hard_200 full rerun
- Any later dataset where we want the original 9-group comparison, but
  with reduced channel-hint noise

## Default Eval Metrics

For future denoised 9-group evals, always maintain these default required
metrics:

- `overall_with_open_acc`
- `explicit_answer_rate`
- `judgment_true_f1`

Keep these with the standard per-type accuracy breakdown:

- `choice_acc`
- `judgment_acc`
- `open_decision_match`

And when available, also maintain:

- `open_decision_anomalous_f1`
- `answer_first_rate`
- `finish_reason`
- `hit_max_new_tokens`
- `interval_max_auroc` for score-bearing runs only

Interpretation rule:

- `overall_with_open_acc` is the main top-line number.
- `explicit_answer_rate` is mandatory because formatting drift can mask the
  actual model behavior.
- `judgment_true_f1` is mandatory because it exposes the common failure mode
  where anomalous intervals get reported as normal.
- `interval_max_auroc` measures anomaly-score quality, not final LLM answer
  quality.

Unless an experiment explicitly asks for deeper analysis, do not expand the
default report with extra micro/macro F1 variants or additional AUROC/AUPRC
variants.

## Hint-Alignment Smoke Check

After the interface is ready, keep one fixed smoke dataset for a standard
hint-alignment ablation. Run:

- `no hint`
- `true hint, ct=16, scale=0.05`
- `true hint, ct=16, scale=0.10`
- `true hint, ct=16, scale=0.20`
- `shuffled hint, ct=16, scale=0.05`
- `shuffled hint, ct=16, scale=0.10`
- `shuffled hint, ct=16, scale=0.20`

Evaluation expectation:

- real hints should improve answer quality relative to the no-hint baseline
- shuffled hints should not deliver the same gain

Do not change the loss function for this protocol yet. Treat it as an
evaluation-only check for hint alignment and generation stability.
