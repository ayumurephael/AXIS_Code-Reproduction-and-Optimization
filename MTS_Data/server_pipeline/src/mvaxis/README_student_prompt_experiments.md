# Student Prompt Experiments

This directory contains the main student-answer prompt provider in
`student_answer_provider.py` and a separate experimental provider in
`student_answer_provider_experimental.py`.

Use the experimental provider when we want to test prompt-order and
format-control hypotheses without perturbing the default workflow.

Current experimental prompt styles:

- `default_strict`
  - Reuses the current strict student-answer template.
- `question_first_hints_data`
  - Places the question first, then presents hidden hint trace and
    `data(str)` together as a support bundle.

Why this file exists:

- We observed that some VL runs with channel hints had poor explicit
  answer formatting even when the content looked directionally useful.
- We want a clean place to iterate on prompt structure without mixing
  experiment-only logic into the main provider file.

Typical caller:

- `scripts/mvaxis_run_student_prompt_channel_experiments.py`

