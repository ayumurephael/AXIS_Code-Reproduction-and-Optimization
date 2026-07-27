# Literature-guided round 2 prompt catalog

Rendered before inference with a fixed illustrative window. Runtime
values, steps, latent placeholders, and questions are substituted per sample.

## lit_r2_01_mc_tf_re2_safe / multiple_choice

SHA-256: `274997d12d8950b8f3d3c28b2d1f3ce71035dbb865a912fc6b2adf5e4da1c17c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic

            Read the question again:
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

## lit_r2_01_mc_tf_re2_safe / open_ended

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

## lit_r2_01_mc_tf_re2_safe / true_false

SHA-256: `e50bfa34b2bcc26387738fd2a19b7923caddefb09db9eaf4c284762a3f4e266f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```
