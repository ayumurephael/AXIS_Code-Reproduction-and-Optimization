# Development round 2 routed prompt catalog

Rendered before inference with a fixed illustrative window. Runtime 
values, steps, latent placeholders, and questions are substituted per sample.

## route_r2_01_minimal / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## route_r2_01_minimal / open_ended

SHA-256: `e97b4a198a9d956275c778eb9dc569c5786b0a17d9b98c31ca8cb4b4a3a78ffe`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## route_r2_01_minimal / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## route_r2_02_mc_stable / multiple_choice

SHA-256: `0295f82d90bd24c4994d62c5fbaaabf5a2e497e8fcb26d0c523c080cf1c3ba17`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## route_r2_02_mc_stable / open_ended

SHA-256: `e97b4a198a9d956275c778eb9dc569c5786b0a17d9b98c31ca8cb4b4a3a78ffe`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## route_r2_02_mc_stable / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## route_r2_03_tf_boundary / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## route_r2_03_tf_boundary / open_ended

SHA-256: `e97b4a198a9d956275c778eb9dc569c5786b0a17d9b98c31ca8cb4b4a3a78ffe`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## route_r2_03_tf_boundary / true_false

SHA-256: `e7d3cc6f957bfceb4bcce9aa6b45e77079884e2e5c74d952bd3543ff55335e99`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Use only evidence inside the half-open window [7, 10); do not invent or cite a step outside it. Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## route_r2_04_oe_old_contract / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## route_r2_04_oe_old_contract / open_ended

SHA-256: `85ff0be5d920d2ae5ab9ed0f8cef122e12924436a9dbcae6e53a8014f847ac28`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## route_r2_04_oe_old_contract / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## route_r2_05_oe_contract_coverage / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## route_r2_05_oe_contract_coverage / open_ended

SHA-256: `45edabaf3706ef6ae79d1e1c190af5e7860ed052597b1c80d2ecb2264fcbf319`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## route_r2_05_oe_contract_coverage / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## route_r2_06_full_routed / multiple_choice

SHA-256: `0295f82d90bd24c4994d62c5fbaaabf5a2e497e8fcb26d0c523c080cf1c3ba17`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## route_r2_06_full_routed / open_ended

SHA-256: `45edabaf3706ef6ae79d1e1c190af5e7860ed052597b1c80d2ecb2264fcbf319`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## route_r2_06_full_routed / true_false

SHA-256: `e7d3cc6f957bfceb4bcce9aa6b45e77079884e2e5c74d952bd3543ff55335e99`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Use only evidence inside the half-open window [7, 10); do not invent or cite a step outside it. Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```
