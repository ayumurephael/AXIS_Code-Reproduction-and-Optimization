# Student Prompt + Channel-Hint Experiments

This script family is for small, controlled ablations around prompt
structure and channel-hint injection strength.

Main experiment runner:

- `mvaxis_run_student_prompt_channel_experiments.py`

What it is for:

- Compare `all_hints` and `score_image_note_all_hints` under:
  - fewer channel hint tokens, e.g. `max_channel_tokens=8`
  - weaker channel hint injection, e.g. `channel_injection_scale=0.05`
  - optional prompt-order experiment: question first, then hints and
    observed `data(str)` together

Why it is separate from the main raw-answer runner:

- The main runner is our default workflow and should stay stable.
- These ablations are hypothesis-driven and should remain easy to find,
  rerun, and delete without touching the main evaluation path.

Inputs expected by the experiment runner:

- question JSONL
- teacher-answer JSONL
- standard AXIS/TimeRCD config
- interval proposer checkpoint
- LLM config
- a pre-generated anomaly-score image manifest named
  `score_images_manifest.json` inside the chosen experiment output root

Recommended workflow:

1. Generate or copy the evaluation question set.
2. Pre-generate score images with
   `scripts/mvaxis_prepare_dataset_visuals.py` or
   `scripts/mvaxis_generate_anomaly_score_images.py`.
3. Place or copy the score manifest into the experiment output root.
4. Run `mvaxis_run_student_prompt_channel_experiments.py`.
5. Inspect:
   - `teacher_aligned_summary.txt`
   - `metrics_table.tsv`
   - per-run `qwen_raw_answers.html`

