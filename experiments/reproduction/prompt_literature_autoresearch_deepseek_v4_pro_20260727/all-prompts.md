# All literature-guided AXIS prompt templates

Rendered from the frozen implementation with representative evidence. Router modes include every distinct lexical branch used in experiments.

## `lit_r1_01_mc_semantic_bind`

### multiple_choice / default

SHA-256: `3f3c8df43cfa0dd3ef81b40b0e630652abe24ab0f583aa01b8f01d79d41c5553`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Report that option once and give a brief reason from the supplied evidence.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

## `lit_r1_02_mc_pointwise`

### multiple_choice / default

SHA-256: `3c39225d2d988dfd90e89db98ab9678e1c182518541aea18c3df796376c8e0ed`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Test each option's complete claim independently against the supplied evidence. Choose the best-supported option by its text, then report its attached letter and a brief reason.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

## `lit_r1_03_mc_re2`

### multiple_choice / default

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

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

## `lit_r1_04_tf_minimal`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

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
```

### open_ended / default

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

### true_false / default

SHA-256: `116ebc1425118aa96591965ffbab669e1a85028ca0018317b4fa52064a3b0ef4`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Judge whether the complete statement as written is true. Report True or False once, followed by a brief reason that supports the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_05_tf_clause`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

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
```

### open_ended / default

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

### true_false / default

SHA-256: `92f67a6bd48ed68399badacd7f4d64366209d161bf0aa5608a766ca095bdf5ee`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Check every required clause and negation in the statement. It is false if any required clause is contradicted; report one truth label and a brief consistent reason.

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_06_tf_re2`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

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
```

### open_ended / default

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

### true_false / default

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

## `lit_r1_07_oe_direct`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

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
```

### open_ended / default

SHA-256: `625bd1ab3fbe1a3ae2beb0dc55c0b5877598b0afb38a8467f6b070ff9c9e1b6b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Answer exactly what the question asks, using evidence from this window. Include the requested conclusion or assessment and the brief reason needed to support it.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

## `lit_r1_08_oe_re2`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

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
```

### open_ended / default

SHA-256: `df2259a7bf131b369aa922c9f7ef0f84a13cdf61f458d69b941a442b96a34df6`

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

            Read the question again:
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

## `lit_r1_09_triplet_minimal`

### multiple_choice / default

SHA-256: `3f3c8df43cfa0dd3ef81b40b0e630652abe24ab0f583aa01b8f01d79d41c5553`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Report that option once and give a brief reason from the supplied evidence.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `625bd1ab3fbe1a3ae2beb0dc55c0b5877598b0afb38a8467f6b070ff9c9e1b6b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Answer exactly what the question asks, using evidence from this window. Include the requested conclusion or assessment and the brief reason needed to support it.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `116ebc1425118aa96591965ffbab669e1a85028ca0018317b4fa52064a3b0ef4`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Judge whether the complete statement as written is true. Report True or False once, followed by a brief reason that supports the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_10_triplet_re2`

### multiple_choice / default

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

### open_ended / default

SHA-256: `df2259a7bf131b369aa922c9f7ef0f84a13cdf61f458d69b941a442b96a34df6`

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

            Read the question again:
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

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

## `lit_r1_11_decoupled`

### multiple_choice / default

SHA-256: `897db1dd97f1faa9067b7c8ddfe2503769657eee5de022cc9dbecef3917ab662`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            First determine which complete option text is best supported by the evidence. Then report its attached letter once and give one brief reason.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `ef58122ce0a933b7612a3c7464414d96395618f64d29da0210926fe75851e99c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            First determine the analysis requested by the question. Then answer it directly with a brief reason from the supplied evidence.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `4cbf2443f3f85689a459a8fa3b61298a1e9d8814bab165885de76d2f9e787da3`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            First determine whether the complete statement is supported. Then report True or False once and give one brief reason consistent with it.

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_12_mc_bind_re2`

### multiple_choice / default

SHA-256: `ed8e444eb1af3637b216a39e818f7b5e44e1992b13d606fafedd1c95bbc21c2f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Report that option once and give a brief reason from the supplied evidence.

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

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

## `lit_r2_01_mc_tf_re2_safe`

### multiple_choice / default

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

### open_ended / default

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

### true_false / default

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

## `lit_r3_01_mc_pointwise_qual`

### multiple_choice / default

SHA-256: `e2d3e1cdbb25cc08a76562fd1f75810af9c08766295891f10ec17f43a4f51b7b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Test each option's complete claim independently against the supplied evidence, then choose the best-supported option by its text. Explain which observed pattern and continuation or recovery support it and which part of the strongest alternative is absent. Use qualitative evidence; do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

## `lit_r3_02_mc_semantic_qual`

### multiple_choice / default

SHA-256: `e5bb143cb12cde0a99fc5008b78a1160b36a6b11ede6e14fa0dfd252a35f0a5c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Explain the observed qualitative pattern and its persistence or recovery that distinguish the selected option from the strongest alternative. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

## `lit_r3_03_mc_salient_aftermath`

### multiple_choice / default

SHA-256: `5643e4407a55395cf1e13b46af951b4e4fed3ca27133b43fb723db5575934c28`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Match the options to the most salient local change and what happens immediately afterward. Select the option whose complete description matches both, then explain the local change and its continuation or recovery using qualitative evidence. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

## `lit_r3_04_tf_re2_verdict`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

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
```

### open_ended / default

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

### true_false / default

SHA-256: `ecd906c41dad15c8cee77b1644fc8927c7a9993c257a8f8eeca6e0fd6ac9629f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r3_05_tf_re2_qual_verdict`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

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
```

### open_ended / default

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

### true_false / default

SHA-256: `52ee71d54d97723c619bf6baecb1bbd2f012fa12f4b47d44c104244fbae888b7`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r3_06_joint_pointwise_tf_qual`

### multiple_choice / default

SHA-256: `e2d3e1cdbb25cc08a76562fd1f75810af9c08766295891f10ec17f43a4f51b7b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Test each option's complete claim independently against the supplied evidence, then choose the best-supported option by its text. Explain which observed pattern and continuation or recovery support it and which part of the strongest alternative is absent. Use qualitative evidence; do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `52ee71d54d97723c619bf6baecb1bbd2f012fa12f4b47d44c104244fbae888b7`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r3_07_joint_semantic_tf_qual`

### multiple_choice / default

SHA-256: `e5bb143cb12cde0a99fc5008b78a1160b36a6b11ede6e14fa0dfd252a35f0a5c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Explain the observed qualitative pattern and its persistence or recovery that distinguish the selected option from the strongest alternative. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `52ee71d54d97723c619bf6baecb1bbd2f012fa12f4b47d44c104244fbae888b7`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r4_01_tf_neg_re2`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

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
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```

## `lit_r4_02_joint_semantic_tf_neg_re2`

### multiple_choice / default

SHA-256: `e5bb143cb12cde0a99fc5008b78a1160b36a6b11ede6e14fa0dfd252a35f0a5c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Explain the observed qualitative pattern and its persistence or recovery that distinguish the selected option from the strongest alternative. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```

## `lit_r4_03_tf_nonanomaly_re2`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

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
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```

### true_false / normality_word

SHA-256: `40cba28623f4ea577cb2b16706b81c44d635f50e05eb21a345f4e4306bec3b00`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window shows stable normal behavior.

            Read the question again:
            True or False: The window shows stable normal behavior.
```

## `lit_r4_04_joint_semantic_tf_nonanomaly_re2`

### multiple_choice / default

SHA-256: `e5bb143cb12cde0a99fc5008b78a1160b36a6b11ede6e14fa0dfd252a35f0a5c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Explain the observed qualitative pattern and its persistence or recovery that distinguish the selected option from the strongest alternative. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```

### true_false / normality_word

SHA-256: `40cba28623f4ea577cb2b16706b81c44d635f50e05eb21a345f4e4306bec3b00`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window shows stable normal behavior.

            Read the question again:
            True or False: The window shows stable normal behavior.
```

## `lit_r4_05_tf_neg_prefix`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

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
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

### true_false / explicit_negative

SHA-256: `7cfdc00dd06d34be8a8d7cd577003f16da543871fb049a28d934f2e0fa9be71a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Preserve the proposition's polarity exactly as written. Decide whether the complete statement is supported, and begin with exactly "Answer: True." or "Answer: False." Then give a qualitative reason consistent with that label. Do not quote exact values or step numbers unless the question asks for them.

            ### Question
            True or False: There is no evidence of an anomaly.
```

## `lit_r4_06_joint_semantic_tf_neg_prefix`

### multiple_choice / default

SHA-256: `e5bb143cb12cde0a99fc5008b78a1160b36a6b11ede6e14fa0dfd252a35f0a5c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Explain the observed qualitative pattern and its persistence or recovery that distinguish the selected option from the strongest alternative. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

### true_false / explicit_negative

SHA-256: `7cfdc00dd06d34be8a8d7cd577003f16da543871fb049a28d934f2e0fa9be71a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Preserve the proposition's polarity exactly as written. Decide whether the complete statement is supported, and begin with exactly "Answer: True." or "Answer: False." Then give a qualitative reason consistent with that label. Do not quote exact values or step numbers unless the question asks for them.

            ### Question
            True or False: There is no evidence of an anomaly.
```

## `lit_r5_01_mc_structured_guard`

### multiple_choice / default

SHA-256: `bb5c687deb0a81584c681aba304103a61dc2fbda4b4f57b85f22d3f79e606ab1`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Compare every option by its complete meaning. Treat alternating signs, ordinary peaks or troughs, isolated large or small values, and irregular-looking fluctuation as normal variability unless the supplied evidence supports the option's specific structured anomaly signature, such as a localized contrast, persistence, recovery, or boundary pattern. Select an anomalous option only when that defining signature is supported; otherwise select the normal option. Explain the decisive qualitative pattern without quoting exact values or step numbers unless asked.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

### true_false / explicit_negative

SHA-256: `f352829131025c30cf5824a43d9cf510a9229cd3f2a3022e41376120fa26f4c0`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: There is no evidence of an anomaly.
```

## `lit_r5_02_mc_status_then_shape`

### multiple_choice / default

SHA-256: `36ec916e9cb9004c6f91c7601978c35839103319a490740a3c4e24ff817f9eca`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            First decide anomaly status from the supplied Per-Step Analysis and Overall Summary Hints, independently of the option wording. Ordinary variance, alternating signs, isolated highs or lows, and irregular-looking fluctuation are not anomalies by themselves. Then compare only options consistent with that status and choose the one whose complete text best matches the observed shape, persistence, recovery, and boundary behavior. Explain the decisive qualitative evidence without quoting exact values or step numbers unless asked.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

### true_false / explicit_negative

SHA-256: `f352829131025c30cf5824a43d9cf510a9229cd3f2a3022e41376120fa26f4c0`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: There is no evidence of an anomaly.
```

## `lit_r5_03_joint_structured_tf_neg_re2`

### multiple_choice / default

SHA-256: `bb5c687deb0a81584c681aba304103a61dc2fbda4b4f57b85f22d3f79e606ab1`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Compare every option by its complete meaning. Treat alternating signs, ordinary peaks or troughs, isolated large or small values, and irregular-looking fluctuation as normal variability unless the supplied evidence supports the option's specific structured anomaly signature, such as a localized contrast, persistence, recovery, or boundary pattern. Select an anomalous option only when that defining signature is supported; otherwise select the normal option. Explain the decisive qualitative pattern without quoting exact values or step numbers unless asked.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```

## `lit_r5_04_joint_status_tf_neg_re2`

### multiple_choice / default

SHA-256: `36ec916e9cb9004c6f91c7601978c35839103319a490740a3c4e24ff817f9eca`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            First decide anomaly status from the supplied Per-Step Analysis and Overall Summary Hints, independently of the option wording. Ordinary variance, alternating signs, isolated highs or lows, and irregular-looking fluctuation are not anomalies by themselves. Then compare only options consistent with that status and choose the one whose complete text best matches the observed shape, persistence, recovery, and boundary behavior. Explain the decisive qualitative evidence without quoting exact values or step numbers unless asked.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

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

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

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
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```
