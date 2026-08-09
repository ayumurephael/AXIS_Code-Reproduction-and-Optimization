from __future__ import annotations

import json
import inspect
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
from src.models.MultiAXIS.prompting import HINT_TOKENS, MultiAxisPromptBuilder, TokenizedPrompts
from tools.multi_axis.build_manifests import (
    align_eval_rows,
    load_training_recovery,
    normalized_question,
    question_reference_answer,
    sha256_file,
    teacher_reference_answer,
)
from tools.multi_axis.geval_runner import bounded_distribution
from tools.multi_axis.distributed import validate_topology_records
from tools.multi_axis.label_metrics import (
    canonical_label,
    open_parseable,
    parse_prediction,
)
from tools.multi_axis.train import compute_timercd_cached
from tools.multi_axis.render_images import render_normalized_series


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"

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


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(merge_size=2)
        self.image_token = "<image_pad>"
        self.last_messages = None
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": list(HINT_TOKENS)
                + ["<vision_start>", "<image_pad>", "<vision_end>", "<end>"]
            }
        )

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        self.last_messages = messages
        pieces = ["<bos>"]
        for message in messages:
            role = message["role"]
            pieces.append(f"<{role}>")
            content = message["content"]
            if isinstance(content, str):
                pieces.append(content)
            else:
                for item in content:
                    if item["type"] == "image":
                        pieces.extend(["<vision_start>", "<image_pad>", "<vision_end>"])
                    else:
                        pieces.append(item["text"])
            if role != "assistant":
                pieces.append("<end>")
        if add_generation_prompt:
            pieces.append("<assistant>")
        elif messages and messages[-1]["role"] == "assistant":
            pieces.append("<end>")
        return " ".join(pieces)

    def replace_image_token(self, image_inputs, image_idx):
        count = int(image_inputs["image_grid_thw"][image_idx].prod().item() // 4)
        return " ".join(["<image_pad>"] * count)

    def __call__(self, *, text, padding, return_tensors, images=None, **kwargs):
        assert return_tensors == "pt"
        del kwargs
        grids = torch.tensor([[1, 4, 4]] * len(images), dtype=torch.long) if images else None
        rows = []
        for index, value in enumerate(text):
            replacement = self.replace_image_token({"image_grid_thw": grids}, index) if images else None
            expanded = value.replace(self.image_token, replacement, 1) if images else value
            rows.append(self.tokenizer.encode(expanded, add_special_tokens=False))
        width = max(map(len, rows))
        input_ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            start = 0 if self.tokenizer.padding_side == "right" else width - len(row)
            input_ids[index, start : start + len(row)] = torch.tensor(row)
            attention_mask[index, start : start + len(row)] = 1
        output = {"input_ids": input_ids, "attention_mask": attention_mask}
        if images:
            output.update(
                {
                    "pixel_values": torch.zeros(len(images) * 16, 12),
                    "image_grid_thw": grids,
                    "mm_token_type_ids": torch.zeros_like(input_ids),
                }
            )
        return output


def test_prompt_placeholder_counts_and_order():
    processor = FakeProcessor()
    builder = MultiAxisPromptBuilder(
        processor, fixed_tokens=30, max_context_tokens=32768, use_images=False
    )
    values = [[-12, -8, 15], [3, 4, 29]]
    text = builder.build_text("Which channel changes?", 10, 13, values)
    assert text.index("### Question") < text.index("### Fixed Hint")
    assert text.index("### Fixed Hint") < text.index("### Target Window")
    assert text.index("### Target Window") < text.index("### Joint-Local Hint")
    assert text.index("### Joint-Local Hint") < text.index("### Output Contract")
    assert "t=10, value=-12 <STEP_HINT>" in text
    tokenized = builder.tokenize(
        ["Which channel changes?"],
        [(10, 13)],
        [values],
        [["ch_0", "ch_1"]],
        ["MC"],
        [None],
        ["Answer: A"],
    )
    assert tokenized.step_positions[0].numel() == 6
    assert tokenized.joint_positions[0].numel() == 1
    assert tokenized.fixed_positions[0].numel() == 30
    assert torch.all(tokenized.labels[0, : tokenized.prompt_lengths[0]] == -100)
    assert (tokenized.labels[0] != -100).sum() > 0


def test_native_vlm_image_first_prefix_mask_and_pixel_metadata(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "window.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    processor = FakeProcessor()
    builder = MultiAxisPromptBuilder(
        processor, fixed_tokens=2, max_context_tokens=2048, use_images=True
    )
    tokenized = builder.tokenize(
        ["Which channel changes?"],
        [(3, 5)],
        [[[10, 11], [20, 21]]],
        [["ch_0", "ch_1"]],
        ["MC"],
        [str(image_path)],
        ["Answer: B\n\nBecause ch_1 changes."],
    )
    user_content = processor.last_messages[1]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "text"
    assert str(image_path) not in user_content[1]["text"]
    assert tokenized.visual_token_counts == [4]
    assert tokenized.original_image_sizes == [(32, 32)]
    assert {"pixel_values", "image_grid_thw", "mm_token_type_ids"}.issubset(
        tokenized.model_inputs
    )
    assert tokenized.padding_side == "right"
    assert torch.all(tokenized.labels[0, : tokenized.prompt_lengths[0]] == -100)


def test_generation_uses_left_padding_and_image_off_has_no_visual_section():
    processor = FakeProcessor()
    builder = MultiAxisPromptBuilder(
        processor, fixed_tokens=2, max_context_tokens=2048, use_images=False
    )
    tokenized = builder.tokenize(
        ["Short?", "This is a substantially longer question for padding?"],
        [(0, 2), (0, 2)],
        [[[1, 2]], [[3, 4]]],
        [["ch_0"], ["ch_0"]],
        ["TF", "TF"],
        [None, None],
        answers=None,
    )
    assert tokenized.padding_side == "left"
    assert tokenized.attention_mask[0, 0].item() == 0
    assert tokenized.attention_mask[0, -1].item() == 1
    assert tokenized.visual_token_counts == [0, 0]
    text = builder.build_text("Short?", 0, 2, [[1, 2]], ["ch_0"], "TF")
    assert "### Visual Evidence" not in text


def test_formal_teacher_target_does_not_fall_back_to_short_label():
    row = {"model_answer": "  ", "windows_0_answer": "short fallback", "answer": "A"}
    assert teacher_model_answer(row) == ""
    assert teacher_answer(row) == "short fallback"


@pytest.mark.parametrize(
    (
        "world_size",
        "expected_nodes",
        "micro_batch_size",
        "accumulation_steps",
        "epochs",
        "effective_batch_size",
    ),
    [
        (5, 1, 1, 6, 40, 30),
        (4, 1, 1, 8, 20, 32),
        (4, 1, 2, 4, 20, 32),
        (4, 1, 4, 2, 20, 32),
        (8, 2, 1, 4, 25, 32),
        (8, 2, 2, 2, 25, 32),
        (8, 2, 4, 1, 25, 32),
        (32, 4, 1, 1, 25, 32),
    ],
)
def test_supported_formal_distributed_profiles(
    world_size,
    expected_nodes,
    micro_batch_size,
    accumulation_steps,
    epochs,
    effective_batch_size,
):
    config = TrainingConfig(
        expected_world_size=world_size,
        expected_nodes=expected_nodes,
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
    vlm = MultiAxisConfig.load_json(
        root / "experiments/multi_axis/formal_qwen3_vl_25epochs_8gpu.json"
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
    assert (
        vlm.training.epochs,
        vlm.training.expected_world_size,
        vlm.training.expected_nodes,
        vlm.training.micro_batch_size,
        vlm.training.accumulation_steps,
        vlm.training.effective_batch_size,
        vlm.vision.enabled,
        vlm.vision.min_pixels,
        vlm.vision.max_pixels,
    ) == (25, 8, 2, 2, 2, 32, True, 65_536, 16_777_216)


def test_two_node_four_gpu_topology_validation():
    records = [
        {
            "rank": rank,
            "local_rank": rank % 4,
            "hostname": "g40" if rank < 4 else "g44",
            "gpu_name": "NVIDIA H800 80GB HBM3",
            "local_cuda_device_count": 4,
        }
        for rank in range(8)
    ]
    audit = validate_topology_records(
        records,
        world_size=8,
        expected_nodes=2,
        required_gpu_substring="H800",
    )
    assert audit["node_count"] == 2
    assert audit["gpus_per_node"] == 4
    assert [node["hostname"] for node in audit["nodes"]] == ["g40", "g44"]


def test_two_node_topology_rejects_duplicate_local_device():
    records = [
        {
            "rank": rank,
            "local_rank": rank % 4,
            "hostname": "g40" if rank < 4 else "g44",
            "gpu_name": "NVIDIA H800 80GB HBM3",
            "local_cuda_device_count": 4,
        }
        for rank in range(8)
    ]
    records[1]["local_rank"] = 0
    with pytest.raises(RuntimeError, match="Duplicate distributed device assignment"):
        validate_topology_records(records, world_size=8, expected_nodes=2)


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


def test_vlm_renderer_consumes_normalized_array_without_mutation(tmp_path):
    values = np.asarray(
        [[-1.0, 0.5], [0.0, 0.25], [1.0, 0.0], [0.5, -0.25]],
        dtype=np.float32,
    )
    before = values.copy()
    output = tmp_path / "render.png"
    audit = render_normalized_series(values, (1, 3), ["ch_0", "ch_1"], output)
    assert np.array_equal(values, before)
    assert output.is_file() and audit["sha256"]
    assert audit["width"] > 0 and audit["height"] > 0
    assert audit["font_family"] == "Times New Roman"
    assert len(audit["font_bundle_sha256"]) == 64


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


def test_embedding_hook_preserves_input_ids_and_only_replaces_full_prompt():
    embedding = torch.nn.Embedding(20, 4)

    class FakeLLM:
        def get_input_embeddings(self):
            return embedding

    input_ids = torch.tensor([[3, 4, 5, 6]])
    tokenized = TokenizedPrompts(
        model_inputs={
            "input_ids": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        },
        labels=None,
        prompt_lengths=[4],
        step_positions=[torch.tensor([1])],
        joint_positions=[torch.tensor([2])],
        fixed_positions=[torch.tensor([0])],
        visual_token_counts=[0],
        original_image_sizes=[None],
        padding_side="right",
    )
    dummy = SimpleNamespace(
        config=SimpleNamespace(vision=SimpleNamespace(enabled=False)), llm=FakeLLM()
    )
    fixed = torch.full((1, 1, 4), 10.0, requires_grad=True)
    step = [torch.full((1, 4), 20.0, requires_grad=True)]
    joint = torch.full((1, 1, 4), 30.0, requires_grad=True)
    model_inputs, labels, hook = MultiAxisForConditionalGeneration._model_inputs_and_hint_hook(
        dummy, tokenized, step, joint, fixed, torch.device("cpu")
    )
    assert labels is None
    assert torch.equal(model_inputs["input_ids"], input_ids)
    handle = embedding.register_forward_hook(hook)
    try:
        replaced = embedding(model_inputs["input_ids"])
        beamed = embedding(model_inputs["input_ids"].repeat_interleave(5, dim=0))
        incremental = embedding(model_inputs["input_ids"][:, -1:])
    finally:
        handle.remove()
    assert torch.equal(model_inputs["input_ids"], input_ids)
    assert torch.all(replaced[0, 0] == 10)
    assert torch.all(replaced[0, 1] == 20)
    assert torch.all(replaced[0, 2] == 30)
    assert beamed.shape[0] == 5 and torch.all(beamed[:, 0] == 10)
    assert torch.all(beamed[:, 1] == 20) and torch.all(beamed[:, 2] == 30)
    assert torch.allclose(incremental, embedding(input_ids[:, -1:]))
    replaced.sum().backward()
    assert fixed.grad is not None and step[0].grad is not None and joint.grad is not None


def test_generation_bridge_preserves_mm_token_types_for_visual_prefill():
    class FakeVisualModel(torch.nn.Module):
        def forward(self, **kwargs):
            return kwargs

    class FakeLLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeVisualModel()

        def prepare_inputs_for_generation(self, input_ids, pixel_values=None, **kwargs):
            del kwargs
            return {"input_ids": input_ids, "pixel_values": pixel_values}

    wrapper = object.__new__(MultiAxisForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(vision=SimpleNamespace(enabled=True))
    wrapper.llm = FakeLLM()
    model_inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "pixel_values": torch.ones(1, 3, 4, 4),
        "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
    }
    with wrapper._native_generation_metadata_context(model_inputs) as audit:
        signature = inspect.signature(wrapper.llm.prepare_inputs_for_generation)
        assert "mm_token_type_ids" in signature.parameters
        prepared = wrapper.llm.prepare_inputs_for_generation(**model_inputs)
        forwarded = wrapper.llm.model(**prepared)
        assert torch.equal(
            forwarded["mm_token_type_ids"], model_inputs["mm_token_type_ids"]
        )
    assert audit == {"required": True, "prefill_calls": 1, "metadata_present": True}
    assert wrapper.last_generation_metadata_audit == audit


def test_frozen_vlm_stays_in_eval_mode_during_hint_training():
    dummy = SimpleNamespace(
        config=SimpleNamespace(llm=SimpleNamespace(gradient_checkpointing=True)),
        llm=torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.5)),
    )
    MultiAxisForConditionalGeneration._set_llm_runtime_mode(dummy, True)
    assert not dummy.llm.training and not dummy.llm[0].training
    assert not dummy.llm[1].training
    MultiAxisForConditionalGeneration._set_llm_runtime_mode(dummy, False)
    assert not dummy.llm.training and not dummy.llm[0].training


def test_label_parsing_contract():
    assert parse_prediction("Answer: C\nAnalysis:\n...", "MC") == "C"
    assert parse_prediction("Answer: False\nAnalysis:\n...", "TF") == "no"
    assert canonical_label("True") == "yes"
    assert open_parseable("Decision:\nThis interval is normal.")
    assert not open_parseable("  ")


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
    processor = FakeProcessor()
    builder = MultiAxisPromptBuilder(
        processor,
        fixed_tokens=3,
        max_context_tokens=256,
        include_joint=False,
        use_images=False,
    )
    values = [[1, 2], [3, 4]]
    text = builder.build_text("Is the interval anomalous?", 0, 2, values)
    assert "### Joint-Local multivariate evidence" not in text
    tokenized = builder.tokenize(
        ["Is the interval anomalous?"],
        [(0, 2)],
        [values],
        [["ch_0", "ch_1"]],
        ["TF"],
        [None],
        ["Answer: Yes"],
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
