from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.AXIS.ts_encoder_bi_bias import TimeSeriesEncoder
from src.models.MultiAXIS.attention import FlashCrossAttention
from src.models.MultiAXIS.config import MultiAxisConfig, TrainingConfig
from src.models.MultiAXIS.data import (
    normalize_and_serialize,
    teacher_answer,
    teacher_model_answer,
)
from src.models.MultiAXIS.model import (
    MultiAxisForConditionalGeneration,
    MultiAxisHintTuner,
    answer_only_causal_nll,
    rms_unit,
)
from src.models.MultiAXIS.prompting import HINT_TOKENS, MultiAxisPromptBuilder
from src.models.MultiAXIS.response_contracts import (
    OUTPUT_CONTRACTS,
    canonicalize_teacher_answer,
    parse_response_label,
    response_contract_error,
)
from tools.multi_axis.build_manifests import (
    align_eval_rows,
    load_training_recovery,
    normalized_question,
    question_reference_answer,
    sha256_file,
    teacher_reference_answer,
)
from tools.multi_axis.geval_runner import bounded_distribution
from tools.multi_axis.label_metrics import (
    canonical_label,
    open_parseable,
    parse_prediction,
    summarize as summarize_label_metrics,
    target_label as metric_target_label,
)
from tools.multi_axis.train import compute_timercd_cached
from tools.multi_axis.audit_teacher_targets import audit_split


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"
        self.chat_template = "native-test-template"

    def add_special_tokens(self, payload):
        added = 0
        for token in payload.get("additional_special_tokens", []):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def encode(self, text, add_special_tokens=True):
        pieces = re.findall(r"<[^>]+>|\S+", text)
        ids = []
        if add_special_tokens:
            ids.append(self.vocab["<bos>"])
        for piece in pieces:
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
            ids.append(self.vocab[piece])
        return ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        pieces = ["<bos>"]
        for message in messages:
            pieces.extend([f"<{message['role']}>", message["content"]])
            if message["role"] != "assistant":
                pieces.append("<end>")
        if add_generation_prompt:
            pieces.append("<assistant>")
        elif messages and messages[-1]["role"] == "assistant":
            pieces.append("<end>")
        return "\n".join(pieces)


def test_prompt_placeholder_counts_and_order():
    tokenizer = FakeTokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": list(HINT_TOKENS)})
    builder = MultiAxisPromptBuilder(
        tokenizer, fixed_tokens=30, max_context_tokens=32768
    )
    values = [[-12, -8, 15], [3, 4, 29]]
    text = builder.build_text(
        "Which channel changes?", 10, 13, values, "MC", ["ch_0", "ch_1"]
    )
    assert text.index("### Question") < text.index("### Task-prior hints")
    assert text.index("### Task-prior hints") < text.index("### Channel-aligned")
    assert text.index("### Channel-aligned") < text.index("### Joint-Local")
    assert "t=10, value=-12 <STEP_HINT>" in text
    assert text.count("### Output Contract") == 1
    assert not text.endswith("Answer:")
    tokenized = builder.tokenize(
        ["Which channel changes?"],
        [(10, 13)],
        [values],
        [["ch_0", "ch_1"]],
        ["MC"],
        ["Answer: A\n\nch_0 changes most."],
    )
    assert tokenized.step_positions[0].numel() == 6
    assert tokenized.joint_positions[0].numel() == 1
    assert tokenized.fixed_positions[0].numel() == 30
    assert torch.all(tokenized.labels[0, : tokenized.prompt_lengths[0]] == -100)
    assert (tokenized.labels[0] != -100).sum() > 0


def test_formal_teacher_target_does_not_fall_back_to_short_label():
    row = {"model_answer": "  ", "windows_0_answer": "short fallback", "answer": "A"}
    assert teacher_model_answer(row) == ""
    assert teacher_answer(row) == "short fallback"


@pytest.mark.parametrize(
    (
        "world_size",
        "micro_batch_size",
        "accumulation_steps",
        "epochs",
        "effective_batch_size",
    ),
    [
        (5, 1, 6, 40, 30),
        (4, 1, 8, 20, 32),
        (4, 2, 4, 20, 32),
        (4, 4, 2, 20, 32),
    ],
)
def test_supported_formal_distributed_profiles(
    world_size,
    micro_batch_size,
    accumulation_steps,
    epochs,
    effective_batch_size,
):
    config = TrainingConfig(
        expected_world_size=world_size,
        micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps,
        epochs=epochs,
    )
    config.validate()
    assert config.effective_batch_size == effective_batch_size


def test_unsupported_formal_distributed_profile_fails_closed():
    with pytest.raises(ValueError, match="Unsupported formal distributed profile"):
        TrainingConfig(
            expected_world_size=4,
            micro_batch_size=2,
            accumulation_steps=8,
            epochs=20,
        ).validate()


def test_checked_in_four_and_five_gpu_configs():
    root = Path(__file__).resolve().parents[2]
    five_gpu = MultiAxisConfig.load_json(
        root / "experiments/multi_axis/formal_deepseek_40epochs.json"
    )
    four_gpu = MultiAxisConfig.load_json(
        root / "experiments/multi_axis/formal_deepseek_20epochs_4gpu.json"
    )
    four_gpu_micro4 = MultiAxisConfig.load_json(
        root / "experiments/multi_axis/benchmark_deepseek_20epochs_4gpu_micro4.json"
    )
    assert (
        five_gpu.training.epochs,
        five_gpu.training.expected_world_size,
        five_gpu.training.micro_batch_size,
        five_gpu.training.accumulation_steps,
        five_gpu.training.effective_batch_size,
    ) == (40, 5, 1, 6, 30)
    assert (
        four_gpu.training.epochs,
        four_gpu.training.expected_world_size,
        four_gpu.training.micro_batch_size,
        four_gpu.training.accumulation_steps,
        four_gpu.training.effective_batch_size,
    ) == (20, 4, 2, 4, 32)
    assert (
        four_gpu_micro4.training.epochs,
        four_gpu_micro4.training.expected_world_size,
        four_gpu_micro4.training.micro_batch_size,
        four_gpu_micro4.training.accumulation_steps,
        four_gpu_micro4.training.effective_batch_size,
    ) == (20, 4, 4, 2, 32)


def test_batch_timercd_cache_reuses_base_series_and_preserves_padding():
    class FakeTimeRCD:
        def __init__(self):
            self.calls = 0

        def __call__(self, series, time_mask, channel_mask):
            self.calls += 1
            local = series.unsqueeze(-1).repeat(1, 1, 1, 3)
            logits = torch.stack((-series, series), dim=-1)
            return local, logits

    fake_timercd = FakeTimeRCD()
    model = SimpleNamespace(timercd=fake_timercd)
    first_inputs = {
        "normalized_series": torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            ]
        ),
        "time_mask": torch.ones(2, 3, dtype=torch.bool),
        "channel_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    cache = {}
    first_local, first_logits = compute_timercd_cached(
        model, first_inputs, cache, ["base-a", "base-a"], [3, 3], [2, 2]
    )
    assert fake_timercd.calls == 1
    assert first_local.shape == (2, 3, 2, 3)
    assert first_logits.shape == (2, 3, 2, 2)
    assert torch.equal(first_local[0], first_local[1])

    second_inputs = {
        "normalized_series": torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 0.0], [8.0, 0.0], [0.0, 0.0]],
            ]
        ),
        "time_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "channel_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
    }
    second_local, second_logits = compute_timercd_cached(
        model, second_inputs, cache, ["base-a", "base-b"], [3, 2], [2, 1]
    )
    assert fake_timercd.calls == 2
    assert torch.equal(second_local[0], first_local[0])
    assert torch.count_nonzero(second_local[1, 2:]) == 0
    assert torch.count_nonzero(second_local[1, :, 1:]) == 0
    assert torch.count_nonzero(second_logits[1, 2:]) == 0


def test_attention_backend_audit_rejects_eager_resolution():
    dummy = SimpleNamespace(
        config=SimpleNamespace(
            llm=SimpleNamespace(attention_implementation="flash_attention_2"),
            hints=SimpleNamespace(require_flash_attention=True),
        ),
        llm=SimpleNamespace(
            config=SimpleNamespace(
                _attn_implementation="eager",
                _attn_implementation_internal="eager",
            )
        ),
    )
    with pytest.raises(RuntimeError, match="did not resolve to flash_attention_2"):
        MultiAxisForConditionalGeneration.attention_backend_audit(dummy)


def test_channelwise_normalization_and_serialization():
    values = np.asarray([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]], dtype=np.float32)
    normalized, window = normalize_and_serialize(
        values, (0, 3), epsilon=1e-5, scale=100
    )
    assert normalized.shape == values.shape
    assert abs(float(normalized[:, 0].mean())) < 1e-6
    assert window[1] == [0, 0, 0]
    assert window[0][0] < 0 < window[0][-1]


def test_dynamic_timercd_encoder_accepts_runtime_channel_count():
    encoder = TimeSeriesEncoder(
        d_model=16,
        d_proj=4,
        patch_size=4,
        num_layers=1,
        num_heads=4,
        d_ff_dropout=0.0,
        num_features=20,
        activation="gelu",
    )
    series = torch.randn(2, 7, 3)
    time_mask = torch.tensor([[1] * 7, [1] * 5 + [0] * 2], dtype=torch.bool)
    channel_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    output = encoder(series, time_mask, channel_mask)
    assert output.shape == (2, 7, 3, 4)
    assert torch.count_nonzero(output[1, 5:]) == 0
    assert torch.count_nonzero(output[1, :, 2]) == 0


def test_flash_cross_attention_cpu_reference_path_and_mask():
    layer = FlashCrossAttention(embed_dim=16, num_heads=4, require_flash=True)
    query = torch.randn(2, 3, 16, requires_grad=True)
    memory = torch.randn(2, 5, 16, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    output = layer(query, memory, memory, mask)
    assert output.shape == (2, 3, 16)
    output.sum().backward()
    assert query.grad is not None and memory.grad is not None


def test_flash_cross_attention_rejects_unsupported_head_dimension():
    with pytest.raises(ValueError, match=r"head_dim=512"):
        FlashCrossAttention(embed_dim=512, num_heads=1, require_flash=True)


def test_flash_cross_attention_allows_head_dimension_256():
    layer = FlashCrossAttention(embed_dim=512, num_heads=2, require_flash=True)
    assert layer.head_dim == 256


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-only FlashAttention regression"
)
def test_flash_cross_attention_cuda_varlen_mask():
    pytest.importorskip("flash_attn")
    layer = (
        FlashCrossAttention(embed_dim=256, num_heads=4, require_flash=True)
        .cuda()
        .to(torch.bfloat16)
    )
    query = torch.randn(
        2, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    memory = torch.randn(
        2, 40, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    dense_output = layer(query, memory, memory)
    mask = torch.ones(2, 40, device="cuda", dtype=torch.bool)
    mask[1, 31:] = False
    output = layer(query, memory, memory, mask)
    assert dense_output.shape == output.shape == (2, 1, 256)
    assert torch.isfinite(dense_output).all() and torch.isfinite(output).all()
    (dense_output.float().sum() + output.float().sum()).backward()
    assert query.grad is not None and memory.grad is not None


def test_hint_tuner_shapes_and_zero_initialized_anomaly_gate():
    config = SimpleNamespace(
        num_prototypes=7,
        prototype_heads=4,
        fixed_tokens=5,
        joint_heads=2,
        representation_epsilon=1e-6,
        require_flash_attention=True,
    )
    tuner = MultiAxisHintTuner(
        vocab_size=13, llm_hidden_size=16, d_proj=8, config=config
    )
    local = torch.randn(2, 5, 3, 8)
    logits_a = torch.randn(2, 5, 3, 2)
    logits_b = logits_a * 10
    words = torch.randn(13, 16)
    first = tuner(local, logits_a, [(1, 4), (0, 2)], [3, 2], words)
    second = tuner(local, logits_b, [(1, 4), (0, 2)], [3, 2], words)
    assert [item.shape for item in first[0]] == [(9, 16), (4, 16)]
    assert first[1].shape == (2, 1, 16)
    assert first[2].shape == (2, 5, 16)
    for left, right in zip(first[0], second[0]):
        assert torch.allclose(left, right, atol=1e-5)
    with torch.no_grad():
        tuner.anomaly_direction.fill_(0.25)
    third = tuner(local, logits_b, [(1, 4), (0, 2)], [3, 2], words)
    assert not torch.allclose(first[0][0], third[0][0])


def test_rms_unit_uses_requested_epsilon():
    value = torch.tensor([[3.0, 4.0]])
    expected = value / torch.sqrt(torch.tensor((9.0 + 16.0) / 2.0 + 1e-6))
    assert torch.allclose(rms_unit(value, 1e-6), expected)


def test_sparse_answer_nll_matches_dense_causal_loss_and_gradient():
    torch.manual_seed(7)
    output_embeddings = torch.nn.Linear(7, 11, bias=False)
    output_embeddings.requires_grad_(False)
    dense_hidden = torch.randn(2, 5, 7, requires_grad=True)
    sparse_hidden = dense_hidden.detach().clone().requires_grad_(True)
    labels = torch.tensor(
        [[-100, 2, 3, -100, 5], [-100, -100, 4, 1, -100]],
        dtype=torch.long,
    )
    dense_logits = output_embeddings(dense_hidden)
    dense_loss = torch.nn.functional.cross_entropy(
        dense_logits[:, :-1, :].float().reshape(-1, 11),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    sparse = answer_only_causal_nll(sparse_hidden, labels, output_embeddings)
    assert sparse.supervised_token_count == 5
    assert torch.allclose(sparse.loss, dense_loss, atol=1e-7, rtol=1e-6)
    dense_loss.backward()
    sparse.loss.backward()
    assert torch.allclose(sparse_hidden.grad, dense_hidden.grad, atol=1e-7, rtol=1e-6)


def test_frozen_llm_train_mode_enables_checkpointing_but_not_dropout():
    dummy = SimpleNamespace(
        config=SimpleNamespace(llm=SimpleNamespace(gradient_checkpointing=True)),
        llm=torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.5)),
    )
    MultiAxisForConditionalGeneration._set_llm_runtime_mode(dummy, True)
    assert dummy.llm.training and dummy.llm[0].training
    assert not dummy.llm[1].training
    MultiAxisForConditionalGeneration._set_llm_runtime_mode(dummy, False)
    assert not dummy.llm.training and not dummy.llm[0].training


def test_label_parsing_contract():
    assert parse_prediction("Answer: C\nAnalysis:\n...", "MC") == "C"
    assert parse_prediction("Yes.\n\nThe proposition is supported.", "TF") == "yes"
    assert parse_prediction("Answer: False\nAnalysis:\n...", "TF") is None
    assert parse_prediction("Reasoning first.\nAnswer: C", "MC") is None
    assert parse_prediction(" Answer: C\n\nEvidence.", "MC") is None
    assert parse_prediction("\nYes.\n\nEvidence.", "TF") is None
    assert canonical_label("True") == "yes"
    assert open_parseable(
        "Decision:\nThis interval is normal.\n\n"
        "Main evidence:\nThe channels stay stable.\n\n"
        "Interpretation:\nThe interval is consistent with context."
    )
    assert not open_parseable("  ")


def test_output_contracts_and_teacher_normalization():
    assert "Answer: D" in OUTPUT_CONTRACTS["MC"]
    assert "True/False" in OUTPUT_CONTRACTS["TF"]
    assert OUTPUT_CONTRACTS["OE"].count("### Output Contract") == 1
    normalized = canonicalize_teacher_answer(
        "The second option matches.\n\nAnswer: B", "MC"
    )
    assert normalized.startswith("Answer: B\n\n")
    assert response_contract_error(normalized, "MC") is None
    assert response_contract_error("\n" + normalized, "MC") == "mc_first_line"
    assert parse_response_label("Answer: B  \n\nEvidence.", "MC") == "B"
    assert parse_response_label("No. \n\nEvidence.", "TF") == "no"
    assert parse_response_label("No. explanation", "TF") is None

    inline_no = canonicalize_teacher_answer(
        "No. The proposition is not supported by the interval.", "TF"
    )
    assert inline_no == "No.\n\nThe proposition is not supported by the interval."
    inline_yes = canonicalize_teacher_answer(
        "Yes, the interval is anomalous, but spike is not the best label.\n\n"
        "The evidence instead supports an outlier.",
        "TF",
    )
    assert inline_yes.startswith("Yes.\n\nthe interval is anomalous")
    assert response_contract_error(inline_yes, "TF") is None


def test_teacher_tf_normalization_uses_only_the_leading_label():
    normalized = canonicalize_teacher_answer(
        "Yes. The evidence rejects the broad claim. Answer: False", "TF"
    )
    assert normalized.startswith("Yes.\n\n")
    assert "Answer: False" in normalized
    with pytest.raises(ValueError, match="no recoverable Yes/No label"):
        canonicalize_teacher_answer(
            "The evidence is inconclusive. Answer: True", "TF"
        )


def test_inference_group_shards_are_disjoint_and_complete():
    from tools.multi_axis.infer import dataset_indices

    dataset = type(
        "GroupedDataset",
        (),
        {"groups": [[0, 1], [2], [3, 4, 5], [6], [7, 8], [9]]},
    )()
    shard_zero = dataset_indices(dataset, 0, 1, shard_count=2, shard_indices=(0,))
    shard_one = dataset_indices(dataset, 0, 1, shard_count=2, shard_indices=(1,))
    assert shard_zero == [0, 1, 3, 4, 5, 7, 8]
    assert shard_one == [2, 6, 9]
    assert set(shard_zero).isdisjoint(shard_one)
    assert sorted(shard_zero + shard_one) == list(range(10))

    rank_zero = dataset_indices(dataset, 0, 2, shard_count=1, shard_indices=(0,))
    rank_one = dataset_indices(dataset, 1, 2, shard_count=1, shard_indices=(0,))
    assert set(rank_zero).isdisjoint(rank_one)
    assert sorted(rank_zero + rank_one) == list(range(10))


def test_label_metric_summary_respects_selected_datasets():
    rows = [
        {
            "dataset": "478new",
            "question_group": "MC",
            "predicted_label": "A",
            "target_label": "A",
            "exact_match": True,
            "parseable": None,
            "success": True,
        }
    ]
    summary = summarize_label_metrics(rows, ("478new",))
    assert set(summary["by_dataset"]) == {"478new"}
    assert summary["by_dataset"]["478new"]["mc_exact_match_accuracy"] == 1.0


def test_label_metric_tf_target_uses_embedded_model_answer_only():
    reference = {
        "teacher_short_answer": (
            "Question: Is the claim true?\n\n"
            "model_answer: No.\n\n"
            "The explanation mentions Yes only as a rejected alternative."
        )
    }
    assert metric_target_label(reference, "TF") == "no"


def test_teacher_target_audit_reads_question_offsets_and_normalizes(tmp_path):
    data_root = tmp_path / "data"
    question_path = data_root / "question.jsonl"
    question_path.parent.mkdir(parents=True)
    rows = [
        {
            "sample_id": "mc-1",
            "question": "Which option?",
            "is_anomalous": True,
        },
        {
            "sample_id": "tf-1",
            "question": "Is the claim true?",
            "is_anomalous": False,
        },
    ]
    encoded = [
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    ]
    question_path.write_bytes(b"".join(encoded))
    manifest = tmp_path / "train.jsonl"
    offset = 0
    records = []
    for row, raw, group, teacher in zip(
        rows,
        encoded,
        ("MC", "TF"),
        ("Answer: B Supporting evidence.", "No. The claim is contradicted."),
    ):
        records.append(
            {
                "question_path": "question.jsonl",
                "question_offset": offset,
                "question_length": len(raw),
                "sample_id": row["sample_id"],
                "question": row["question"],
                "question_group": group,
                "teacher_answer": teacher,
            }
        )
        offset += len(raw)
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    result = audit_split(manifest, data_root)
    assert result["passed"] is True, json.dumps(result, ensure_ascii=False)
    assert result["examples"] == 2
    assert result["canonicalized_total"] == 2


def test_bounded_logprob_distribution():
    entry = {
        "top_logprobs": [
            {"token": str(score), "logprob": -abs(score - 4.0)} for score in range(1, 6)
        ]
    }
    result = bounded_distribution(
        entry, {str(i): i for i in range(1, 6)}, requested_topk=5, bound=1e-6
    )
    assert result is not None
    score, probabilities, metadata = result
    assert 1 <= score <= 5
    assert abs(sum(probabilities.values()) - 1.0) < 1e-8
    assert metadata["missing_mass_upper_bound"] == 0.0


def test_phase_a_disables_anomaly_and_joint_paths():
    config = SimpleNamespace(
        num_prototypes=7,
        prototype_heads=4,
        fixed_tokens=5,
        joint_heads=2,
        representation_epsilon=1e-6,
        require_flash_attention=True,
        ablation_phase="A",
    )
    tuner = MultiAxisHintTuner(
        vocab_size=13, llm_hidden_size=16, d_proj=8, config=config
    )
    assert not tuner.anomaly_direction.requires_grad
    assert not tuner.joint_query.requires_grad
    assert all(
        not parameter.requires_grad for parameter in tuner.joint_pool.parameters()
    )

    local = torch.randn(1, 4, 2, 8)
    words = torch.randn(13, 16)
    with torch.no_grad():
        tuner.anomaly_direction.fill_(10.0)
    low = tuner(local, torch.zeros(1, 4, 2, 2), [(0, 3)], [2], words)
    high = tuner(local, torch.randn(1, 4, 2, 2) * 100, [(0, 3)], [2], words)
    assert low[1] is None and high[1] is None
    assert torch.allclose(low[0][0], high[0][0], atol=1e-5)


def test_prompt_can_remove_joint_placeholder_for_ablation():
    tokenizer = FakeTokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": list(HINT_TOKENS)})
    builder = MultiAxisPromptBuilder(
        tokenizer, fixed_tokens=3, max_context_tokens=1024, include_joint=False
    )
    values = [[1, 2], [3, 4]]
    text = builder.build_text(
        "Is the interval anomalous?", 0, 2, values, "TF", ["ch_0", "ch_1"]
    )
    assert "### Joint-Local multivariate evidence" not in text
    tokenized = builder.tokenize(
        ["Is the interval anomalous?"],
        [(0, 2)],
        [values],
        [["ch_0", "ch_1"]],
        ["TF"],
        ["Yes.\n\nThe interval contains a coordinated change."],
    )
    assert tokenized.step_positions[0].numel() == 4
    assert tokenized.joint_positions[0].numel() == 0
    assert tokenized.fixed_positions[0].numel() == 3


def test_eval_index_alignment_handles_bias_neutralized_question_rewrite():
    question_item = {
        "row": {"question": "Neutral rewrite", "windows": [{"answer": "Yes"}]}
    }
    teacher_item = {"row": {"question": "Older wording", "windows_0_answer": " yes "}}
    matches, used, strategy = align_eval_rows(
        "SMD", {"alignment": "index", "expected": 1}, [question_item], [teacher_item]
    )
    assert matches == [(question_item, teacher_item)]
    assert used == {0} and strategy == "index"
    assert question_reference_answer(question_item["row"]) == "yes"
    assert teacher_reference_answer(teacher_item["row"]) == "yes"

    text_matches, _, _ = align_eval_rows(
        "478new",
        {"alignment": "question_text", "expected": 1},
        [question_item],
        [teacher_item],
    )
    assert text_matches == []


def test_audited_training_recovery_bundle(tmp_path):
    recovery_root = tmp_path / "derived" / "training_recovery"
    recovery_root.mkdir(parents=True)
    questions_path = recovery_root / "recovered_questions.jsonl"
    row = {
        "question": "  Which Channel? ",
        "_multi_axis_recovery": {"source_shard": "shard-a", "source_index": 7},
    }
    questions_path.write_text(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "format": "multi-axis-training-recovery-v1",
        "recovered_question_rows": 1,
        "recovered_output": {
            "size": questions_path.stat().st_size,
            "sha256": sha256_file(questions_path),
        },
    }
    manifest_path = recovery_root / "recovery_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    lookup, loaded_manifest, source_files = load_training_recovery(
        tmp_path, hash_inputs=True
    )
    assert normalized_question(row) == "which channel?"
    assert set(lookup) == {("shard-a", 7)}
    assert loaded_manifest == manifest
    assert all(item["sha256"] for item in source_files)

    questions_path.write_text(
        questions_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="size"):
        load_training_recovery(tmp_path, hash_inputs=True)
