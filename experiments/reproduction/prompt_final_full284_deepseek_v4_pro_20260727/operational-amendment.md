# Operational amendment: environment recovery and exact RE2 reuse

This amendment does not change the frozen candidates, data, metrics, Judge, or
success criterion.

## GPU environment recovery

Two launches stopped before model inference and before any prediction row was
written:

1. a legacy Baseline virtual environment resolved to Python 3.6 and could not
   parse the current source;
2. the corrected Python 3.10 environment was initially missing the explicit
   local `AXIS_MODEL_NAME`, so offline Transformers could not resolve the
   model directory.

Both failure logs and PID files are retained separately. The accepted launch
uses Python 3.10.14, torch 2.5.1+cu124, Transformers 4.45.2, the local
DeepSeek-R1-Distill-Qwen-7B mirror, and the frozen commit/checkpoint.

## Exact component reuse

The frozen protocol initially described `lit_r1_08_oe_re2` as requiring its
own OE inference. Code inspection gives a stronger exact-reuse result:

- `lit_r1_08_oe_re2/open_ended` maps to literature component `re2`;
- `lit_r1_10_triplet_re2/open_ended` maps to the same component `re2`;
- their rendered OE prompts are therefore identical for every record, and
  the injected window/local/fixed evidence is identical;
- `lit_r1_08_oe_re2` maps MC and TF to exact Baseline.

The raw GPU run consequently generates only `base` and
`lit_r1_10_triplet_re2` over 284 QA. The assembled
`lit_r1_08_oe_re2` uses the latter's OE responses/scores and Baseline MC/TF
responses/scores. Prompt-component equality and response hashes are checked
before assembly, and every reused row records its source mode.

This reduces GPU/API work and prevents repeated CUDA generation or Judge
variation from being misreported as a prompt effect.

