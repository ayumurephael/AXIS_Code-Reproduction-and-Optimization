# Screening round 1 prompt catalog

Rendered before inference with a fixed illustrative window. Runtime 
values, steps, latent placeholders, and questions are substituted per sample.

## fixed_role / multiple_choice

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

## fixed_role / open_ended

SHA-256: `22a815d718ac77ccb09e894aa1b62d3fa3c7edc9593bba5117d63385bf551949`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## fixed_role / true_false

SHA-256: `ef8867788295ce4777b8e80b26cbb7b3360e088eb6fc30a1235a75ff28b490f7`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p01_boundary / multiple_choice

SHA-256: `6f3678985ec7846f6c8e2352b32ff48471cb383f5c16062eb1c77501816243dd`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.

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

## pareto_p01_boundary / open_ended

SHA-256: `84660aee5bfa6338fed0f7ee3be07e323faa4accab49553662d05f11cb932318`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p01_boundary / true_false

SHA-256: `439b90f2b3c665bb4c22b77ef30173b6838c3141645b7b1148bca52d16b8bea1`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p02_calibration / multiple_choice

SHA-256: `e600afd18b1ea8c095156e58e4faaf621a96cdeffafe073804bd0fc0ae351f6b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

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

## pareto_p02_calibration / open_ended

SHA-256: `45a12c90c98dab3ed0209d3c5642d08a6ad467f9390ad6f59ed7d9d7ab9e74aa`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p02_calibration / true_false

SHA-256: `f05e7eaf748ab51189d9b0c8fd0795a1e3f986fb7036b70264679257779e9aad`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p03_task_rule / multiple_choice

SHA-256: `26bac720e9743601acdd3a9a261db5e8197041cb28ec05bd4fe0ff48d368dd91`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## pareto_p03_task_rule / open_ended

SHA-256: `880e2ae5fd9bddc9bf9537e8582c1ab34d8a852e545d532c48757ea5b62578b6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p03_task_rule / true_false

SHA-256: `2966b574a3060dfcf02e27f14ee8a3e39da50c5ed8bec5fba02b488b0426ba47`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p04_numeric_guard / multiple_choice

SHA-256: `f5946c9225802e32f13dae2e5f8242ef5fe70297873593d3e12c3426814f88f9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

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

## pareto_p04_numeric_guard / open_ended

SHA-256: `4e30ce8cdd836f6223e51d3f906e1ac982cb890f66a6269bf65123523dbf3968`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p04_numeric_guard / true_false

SHA-256: `6448c49f39fe49564d2d0449c60f782cb16a23fee2b737431003263bb1ce24d9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p05_single_answer / multiple_choice

SHA-256: `0567400d3b90dbf3ac2dd5e9fb143c2cd2a433645cdc1badff59ba10e5af63d9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## pareto_p05_single_answer / open_ended

SHA-256: `c1311b5567ff9c78c15f38ea3bc6af69aad8d290289cec82e087a3b8a5ec1f3a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p05_single_answer / true_false

SHA-256: `892e216fa2a14f987c1872b3e90fc2c9adfe085300cf8275950d72e0d596b80c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p06_abc / multiple_choice

SHA-256: `99ab4177e85ca491d6d6c78fb692e8545631ce2ac3a7b07ed3564dbb4e620341`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## pareto_p06_abc / open_ended

SHA-256: `575c6ecb693bc202ef75dcdf8a07708538d337f8013cf2b8a8a89450e264f109`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p06_abc / true_false

SHA-256: `6410c53349710811327402e8379d97b481c4c6b96c62f5d3f16bf4c5907cebe3`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p07_abd / multiple_choice

SHA-256: `d808065164a247810684eb1fcca04cf5b33477793986a6e7382d4297863912d0`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

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

## pareto_p07_abd / open_ended

SHA-256: `c07f585e8ecee409260e07c2ff2480de6a1fbb8cee201fee4821e3160aa88125`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p07_abd / true_false

SHA-256: `0766abfebc778fafef4b1e6f9bdab0b802da6b577c0014c7663a4e98e1cb988a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p08_acde / multiple_choice

SHA-256: `427bcccb202c909b218619fae7d15fd9714cdcd384cfa192a132a95dc66f082a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## pareto_p08_acde / open_ended

SHA-256: `359d73d8f2c2b233da7eca78d8b6a5d1adc4eca0e44ab76119ffbb0404ddf7e0`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p08_acde / true_false

SHA-256: `bf7aa998bc72aae1dffd860e0b0897ea5b41ef33e0b82d97b56623a893062eeb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p09_bce / multiple_choice

SHA-256: `fff5bcbeee210446d7f8c4a776dc098655844569148713a01987bf5494d0f9d5`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## pareto_p09_bce / open_ended

SHA-256: `3f0e8de389a4f7ec362367495dbc0900f2216650c3c7c6fe78683412f8d45c2a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p09_bce / true_false

SHA-256: `ce52a77289889154c3642bc79374dfa5759c53370ec783aa15f0d8f93ae498f9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p10_abcde / multiple_choice

SHA-256: `0a081ec3dc48eccc591d1d697c6ae659b8883fe7cd4dcf950707d06889ade015`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## pareto_p10_abcde / open_ended

SHA-256: `1cfefc5bc3a43bd24b0f35ff9a4ad8e78546de49f9874e6fb6325a5a2aee0bda`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p10_abcde / true_false

SHA-256: `282b2c1f38a42cf4213bddc78b1b4116417b159ba316688c810e9a1c1a9dc092`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

## pareto_p11_abce / multiple_choice

SHA-256: `7d66ad6fd4d365020e3452ee190eed79a22e3e880e4cc3a888be1351fb7bc908`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

## pareto_p11_abce / open_ended

SHA-256: `4dad0176a51f018d2660312b7210bb8548f1163b44869393002c59de4b5ef7d4`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

## pareto_p11_abce / true_false

SHA-256: `ef67696bd745bbedf43d8604a23442b24946193beb2ccd65ff0e412016446f02`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```
