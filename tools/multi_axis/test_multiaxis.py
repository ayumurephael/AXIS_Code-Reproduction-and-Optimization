from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models.AXIS.ts_encoder_bi_bias import TimeSeriesEncoder
from src.models.MultiAXIS.attention import FlashCrossAttention
from src.models.MultiAXIS.data import (
    normalize_and_serialize,
    teacher_answer,
    teacher_model_answer,
)
from src.models.MultiAXIS.model import MultiAxisHintTuner, rms_unit
from src.models.MultiAXIS.prompting import HINT_TOKENS, MultiAxisPromptBuilder
from tools.multi_axis.build_manifests import (
    align_eval_rows,
    load_training_recovery,
    normalized_question,
    question_reference_answer,
    sha256_file,
    teacher_reference_answer,
)
from tools.multi_axis.geval_runner import bounded_distribution
from tools.multi_axis.label_metrics import canonical_label, open_parseable, parse_prediction


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
        self.pad_token_id = 0
        self.eos_token_id = 1

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


def test_prompt_placeholder_counts_and_order():
    tokenizer = FakeTokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": list(HINT_TOKENS)})
    builder = MultiAxisPromptBuilder(tokenizer, fixed_tokens=30, max_context_tokens=32768)
    values = [[-12, -8, 15], [3, 4, 29]]
    text = builder.build_text("Which channel changes?", 10, 13, values)
    assert text.index("### Question") < text.index("### Task-prior hints")
    assert text.index("### Task-prior hints") < text.index("### Channel-aligned")
    assert text.index("### Channel-aligned") < text.index("### Joint-Local")
    assert "-12 <STEP_HINT> -8 <STEP_HINT> 15 <STEP_HINT>" in text
    tokenized = builder.tokenize(["Which channel changes?"], [(10, 13)], [values], ["Answer: A"])
    assert tokenized.step_positions[0].numel() == 6
    assert tokenized.joint_positions[0].numel() == 1
    assert tokenized.fixed_positions[0].numel() == 30
    assert torch.all(tokenized.labels[0, : tokenized.prompt_lengths[0]] == -100)
    assert (tokenized.labels[0] != -100).sum() > 0


def test_formal_teacher_target_does_not_fall_back_to_short_label():
    row = {"model_answer": "  ", "windows_0_answer": "short fallback", "answer": "A"}
    assert teacher_model_answer(row) == ""
    assert teacher_answer(row) == "short fallback"


def test_channelwise_normalization_and_serialization():
    values = np.asarray([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]], dtype=np.float32)
    normalized, window = normalize_and_serialize(values, (0, 3), epsilon=1e-5, scale=100)
    assert normalized.shape == values.shape
    assert abs(float(normalized[:, 0].mean())) < 1e-6
    assert window[1] == [0, 0, 0]
    assert window[0][0] < 0 < window[0][-1]


def test_dynamic_timercd_encoder_accepts_runtime_channel_count():
    encoder = TimeSeriesEncoder(d_model=16, d_proj=4, patch_size=4, num_layers=1, num_heads=4, d_ff_dropout=0.0, num_features=20, activation="gelu")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only FlashAttention regression")
def test_flash_cross_attention_cuda_varlen_mask():
    pytest.importorskip("flash_attn")
    layer = FlashCrossAttention(embed_dim=256, num_heads=4, require_flash=True).cuda().to(torch.bfloat16)
    query = torch.randn(2, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    memory = torch.randn(2, 40, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dense_output = layer(query, memory, memory)
    mask = torch.ones(2, 40, device="cuda", dtype=torch.bool)
    mask[1, 31:] = False
    output = layer(query, memory, memory, mask)
    assert dense_output.shape == output.shape == (2, 1, 256)
    assert torch.isfinite(dense_output).all() and torch.isfinite(output).all()
    (dense_output.float().sum() + output.float().sum()).backward()
    assert query.grad is not None and memory.grad is not None


def test_hint_tuner_shapes_and_zero_initialized_anomaly_gate():
    config = SimpleNamespace(num_prototypes=7, prototype_heads=4, fixed_tokens=5, joint_heads=2, representation_epsilon=1e-6, require_flash_attention=True)
    tuner = MultiAxisHintTuner(vocab_size=13, llm_hidden_size=16, d_proj=8, config=config)
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


def test_label_parsing_contract():
    assert parse_prediction("Answer: C\nAnalysis:\n...", "MC") == "C"
    assert parse_prediction("Answer: False\nAnalysis:\n...", "TF") == "no"
    assert canonical_label("True") == "yes"
    assert open_parseable("Decision:\nThis interval is normal.")
    assert not open_parseable("  ")


def test_bounded_logprob_distribution():
    entry = {"top_logprobs": [{"token": str(score), "logprob": -abs(score - 4.0)} for score in range(1, 6)]}
    result = bounded_distribution(entry, {str(i): i for i in range(1, 6)}, requested_topk=5, bound=1e-6)
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
    assert all(not parameter.requires_grad for parameter in tuner.joint_pool.parameters())

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
        tokenizer, fixed_tokens=3, max_context_tokens=256, include_joint=False
    )
    values = [[1, 2], [3, 4]]
    text = builder.build_text("Is the interval anomalous?", 0, 2, values)
    assert "### Joint-Local multivariate evidence" not in text
    tokenized = builder.tokenize(
        ["Is the interval anomalous?"], [(0, 2)], [values], ["Answer: Yes"]
    )
    assert tokenized.step_positions[0].numel() == 4
    assert tokenized.joint_positions[0].numel() == 0
    assert tokenized.fixed_positions[0].numel() == 3

def test_eval_index_alignment_handles_bias_neutralized_question_rewrite():
    question_item = {"row": {"question": "Neutral rewrite", "windows": [{"answer": "Yes"}]}}
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
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
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

    lookup, loaded_manifest, source_files = load_training_recovery(tmp_path, hash_inputs=True)
    assert normalized_question(row) == "which channel?"
    assert set(lookup) == {("shard-a", 7)}
    assert loaded_manifest == manifest
    assert all(item["sha256"] for item in source_files)

    questions_path.write_text(questions_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="size"):
        load_training_recovery(tmp_path, hash_inputs=True)
