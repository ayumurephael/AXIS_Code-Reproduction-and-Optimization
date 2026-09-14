from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.answer_schema import normalize_answer_label
from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.llm_client import LocalHFChatClient, LocalHFVLChatClient, create_llm_client
from src.mvaxis.proposal import propose_interval, sample_with_interval
from src.mvaxis.semantic_alignment import extract_teacher_answer
from src.mvaxis.student_answer_provider import build_student_answer_messages, parse_student_answer
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_jsonl_maybe_limited(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    resolved = _resolve(path)
    if limit is None:
        return read_jsonl(resolved)
    rows: List[Dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
            if len(rows) >= int(limit):
                break
    return rows


def _dist_barrier() -> None:
    if _dist_enabled() and dist.is_initialized():
        dist.barrier()


def _materialize_distributed_jsonl_pair_shards(
    data_path: str | Path,
    teacher_path: str | Path,
    *,
    limit: Optional[int],
    shard_root: Path,
) -> Dict[str, Any]:
    data_resolved = _resolve(data_path)
    teacher_resolved = _resolve(teacher_path)
    world_size = _dist_world_size()
    rank = _dist_rank()
    limit_tag = "all" if limit is None else str(int(limit))
    shard_dir = shard_root / f"{data_resolved.stem}__{teacher_resolved.stem}__limit_{limit_tag}__ws_{world_size}"
    metadata_path = shard_dir / "metadata.json"
    local_data_path = shard_dir / f"data_rank{rank:02d}.jsonl"
    local_teacher_path = shard_dir / f"teacher_rank{rank:02d}.jsonl"

    if _dist_is_main():
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_data_paths = [shard_dir / f"data_rank{idx:02d}.jsonl" for idx in range(world_size)]
        shard_teacher_paths = [shard_dir / f"teacher_rank{idx:02d}.jsonl" for idx in range(world_size)]
        if not (
            metadata_path.exists()
            and all(path.exists() for path in shard_data_paths)
            and all(path.exists() for path in shard_teacher_paths)
        ):
            _rank0_print(
                {
                    "stage": "distributed_shard_materialize_start",
                    "data_path": str(data_resolved),
                    "teacher_path": str(teacher_resolved),
                    "shard_dir": str(shard_dir),
                    "limit": None if limit is None else int(limit),
                    "world_size": world_size,
                }
            )
            shard_counts = [0 for _ in range(world_size)]
            with data_resolved.open("r", encoding="utf-8") as data_handle, teacher_resolved.open(
                "r", encoding="utf-8"
            ) as teacher_handle:
                with (
                    shard_data_paths[0].open("w", encoding="utf-8") as _dummy_data0,
                    shard_teacher_paths[0].open("w", encoding="utf-8") as _dummy_teacher0,
                ):
                    pass
                data_writers = [path.open("w", encoding="utf-8") for path in shard_data_paths]
                teacher_writers = [path.open("w", encoding="utf-8") for path in shard_teacher_paths]
                try:
                    total = 0
                    for row_idx, pair in enumerate(zip_longest(data_handle, teacher_handle, fillvalue=None)):
                        data_line, teacher_line = pair
                        if data_line is None or teacher_line is None:
                            raise RuntimeError(
                                f"Data/teacher length mismatch while building distributed shards at row {row_idx}: "
                                f"data_missing={data_line is None}, teacher_missing={teacher_line is None}"
                            )
                        if limit is not None and total >= int(limit):
                            break
                        shard_idx = total % world_size
                        data_writers[shard_idx].write(data_line if data_line.endswith("\n") else data_line + "\n")
                        teacher_writers[shard_idx].write(
                            teacher_line if teacher_line.endswith("\n") else teacher_line + "\n"
                        )
                        shard_counts[shard_idx] += 1
                        total += 1
                finally:
                    for handle in data_writers:
                        handle.close()
                    for handle in teacher_writers:
                        handle.close()
            save_json(
                {
                    "data_path": str(data_resolved),
                    "teacher_path": str(teacher_resolved),
                    "limit": None if limit is None else int(limit),
                    "world_size": world_size,
                    "total_num_samples": int(sum(shard_counts)),
                    "per_rank_counts": shard_counts,
                },
                metadata_path,
            )
            _rank0_print(
                {
                    "stage": "distributed_shard_materialize_done",
                    "shard_dir": str(shard_dir),
                    "total_num_samples": int(sum(shard_counts)),
                    "per_rank_counts": shard_counts,
                }
            )
    _dist_barrier()
    metadata = load_json(metadata_path)
    return {
        "total_num_samples": int(metadata["total_num_samples"]),
        "local_data_path": local_data_path,
        "local_teacher_path": local_teacher_path,
        "shard_dir": shard_dir,
        "per_rank_counts": list(metadata.get("per_rank_counts") or []),
    }


def _answer_label_vocab_flags(items: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "judgment": int(any(item.get("answer_type") == "judgment" and item.get("answer_label") for item in items)),
        "choice": int(any(item.get("answer_type") == "choice" and item.get("answer_label") for item in items)),
    }


def _build_answer_label_vocabs_from_flags(flags: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    vocabs: Dict[str, Dict[str, int]] = {}
    if int(flags.get("judgment", 0)):
        vocabs["judgment"] = {"False": 0, "True": 1}
    if int(flags.get("choice", 0)):
        vocabs["choice"] = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    return vocabs


def _dist_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _dist_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _dist_enabled() -> bool:
    return _dist_world_size() > 1


def _init_distributed() -> None:
    if not _dist_enabled() or dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=_dist_rank(), world_size=_dist_world_size())
    if torch.cuda.is_available():
        torch.cuda.set_device(_dist_local_rank())


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _rank0_print(payload: Dict[str, Any]) -> None:
    if _dist_is_main():
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def _dist_print(payload: Dict[str, Any]) -> None:
    enriched = {"rank": _dist_rank(), "world_size": _dist_world_size(), **payload}
    print(json.dumps(enriched, ensure_ascii=False), flush=True)


def _distributed_shard(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not _dist_enabled():
        return items
    rank = _dist_rank()
    world_size = _dist_world_size()
    if not items:
        return items
    # Keep every rank on the same number of steps so manual all_reduce calls
    # stay aligned even when the dataset size is not divisible by world_size.
    padded_size = int(math.ceil(len(items) / world_size) * world_size)
    if padded_size > len(items):
        items = items + items[: padded_size - len(items)]
    return items[rank::world_size]


def _average_gradients(parameters: List[torch.nn.Parameter]) -> None:
    if not _dist_enabled():
        return
    world_size = float(_dist_world_size())
    for param in parameters:
        if not param.requires_grad:
            continue
        if param.grad is None:
            grad = torch.zeros_like(param, memory_format=torch.preserve_format)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            param.grad = grad.div_(world_size)
        else:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def _reduce_scalar_dict(values: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not _dist_enabled():
        return values
    keys = list(values.keys())
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(tensor[idx].item()) for idx, key in enumerate(keys)}


def _sample_keys(row: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key in ("sample_id", "question_pair_id", "source_window_sample_id", "base_sample_id"):
        value = row.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                keys.append(text)
    seen = set()
    ordered: List[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _question_text(row: Dict[str, Any]) -> str:
    question = row.get("question")
    if question is None:
        windows = row.get("windows") or []
        if windows and isinstance(windows[0], dict):
            question = windows[0].get("question")
    return "" if question is None else str(question).strip()


def _question_type(row: Dict[str, Any]) -> str:
    value = (
        row.get("question_answer_type")
        or (row.get("target_output") or {}).get("fact_check", {}).get("question_answer_type")
        or row.get("question_type")
        or "open"
    )
    return str(value).strip().lower()


def _row_num_channels(row: Dict[str, Any]) -> int:
    series = row.get("series") or {}
    shape = series.get("shape") or []
    if len(shape) >= 2:
        return int(shape[1])
    values = series.get("values") or []
    if values and isinstance(values[0], list):
        return int(len(values[0]))
    channels = row.get("channels") or []
    if channels:
        return int(len(channels))
    raise ValueError("Unable to infer num_channels from training row.")


def _load_encoder_for_channel_count(
    checkpoint_path: str | Path,
    config: Dict[str, Any],
    *,
    num_channels: int,
) -> Tuple[AXISMultivariateIntervalProposer, Dict[str, Any]]:
    channel_config = copy.deepcopy(config)
    channel_config.setdefault("data", {})
    channel_config["data"]["num_channels"] = int(num_channels)
    threshold = float((channel_config.get("interval_proposal") or {}).get("threshold", 0.5))
    return AXISMultivariateIntervalProposer.load_shape_compatible_checkpoint(
        str(checkpoint_path),
        channel_config,
        threshold=threshold,
        num_channels_override=int(num_channels),
        skip_global_hint_head=False,
    )


def _configure_encoder_for_hint_training(
    encoder: AXISMultivariateIntervalProposer,
    *,
    include_global_hints: bool,
) -> None:
    encoder.eval()
    for param in encoder.ts_encoder.parameters():
        param.requires_grad = False
    for param in encoder.anomaly_head.parameters():
        param.requires_grad = False
    encoder.global_hint_head.train(mode=include_global_hints)
    for param in encoder.global_hint_head.parameters():
        param.requires_grad = include_global_hints


def _question_frame(row: Dict[str, Any]) -> Optional[str]:
    value = row.get("question_frame_override")
    if value is not None:
        text = str(value).strip()
        if text:
            return text
    artifacts = row.get("question_generation_artifacts") or {}
    for key in ("frame", "frame_type", "polarity"):
        value = artifacts.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return None


def _focus_key(row: Dict[str, Any]) -> Optional[str]:
    artifacts = row.get("question_generation_artifacts") or {}
    for key in ("focus_type", "focus", "semantic_focus"):
        value = artifacts.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    for key in ("question_family", "source_question_type", "question_answer_type"):
        value = row.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return None


def _build_pair_lookup(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        pair_id = row.get("question_pair_id")
        focus_key = _focus_key(row)
        sample_id = row.get("sample_id")
        if not pair_id or not focus_key or not sample_id:
            continue
        grouped.setdefault((str(pair_id), str(focus_key)), []).append(row)
    lookup: Dict[str, str] = {}
    for items in grouped.values():
        anomaly_items = [item for item in items if _question_frame(item) == "anomaly_frame"]
        normal_items = [item for item in items if _question_frame(item) == "normal_frame"]
        if not anomaly_items or not normal_items:
            continue
        for anomaly_item in anomaly_items:
            for normal_item in normal_items:
                lookup[str(anomaly_item["sample_id"])] = str(normal_item["sample_id"])
                lookup[str(normal_item["sample_id"])] = str(anomaly_item["sample_id"])
    return lookup


def _permute_sample(sample: Dict[str, Any], permutation: List[int]) -> Dict[str, Any]:
    permuted = json.loads(json.dumps(sample))
    series = permuted.get("series") or {}
    series["values"] = [[row[idx] for idx in permutation] for row in series.get("values") or []]
    series["labels"] = [[row[idx] for idx in permutation] for row in series.get("labels") or []]
    permuted["series"] = series
    channels = permuted.get("channels") or []
    permuted["channels"] = [channels[idx] for idx in permutation]
    normal_values = permuted.get("normal_series")
    if normal_values is not None:
        permuted["normal_series"] = [[row[idx] for idx in permutation] for row in normal_values]
    original_data = permuted.get("original_data") or {}
    if original_data.get("normal_series") is not None:
        original_data["normal_series"] = [[row[idx] for idx in permutation] for row in original_data["normal_series"]]
    if original_data.get("time_series") is not None:
        original_data["time_series"] = [[row[idx] for idx in permutation] for row in original_data["time_series"]]
    permuted["original_data"] = original_data
    return permuted


def _masked_time_ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_logits = logits[mask]
    valid_labels = labels[mask]
    if valid_logits.numel() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(valid_logits, valid_labels)


def _question_attention_map(
    diagnostics: Dict[str, Any],
    num_question_queries: int,
    time_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    channel_attention = diagnostics.get("channel_attention")
    if channel_attention is None or channel_attention.ndim != 4:
        return None
    question_attention = channel_attention[:, :, -int(num_question_queries) :, :]
    valid = time_mask.bool().unsqueeze(-1).unsqueeze(-1)
    denom = valid.to(question_attention.dtype).sum().clamp_min(1.0)
    return (question_attention * valid).sum(dim=1) / denom


def _questions_compatible(sample: Dict[str, Any], teacher_row: Dict[str, Any]) -> bool:
    sample_question = _question_text(sample).strip().lower()
    teacher_question = _question_text(teacher_row).strip().lower()
    if sample_question and teacher_question:
        return sample_question == teacher_question
    sample_qid = str(sample.get("question_id") or "").strip()
    teacher_qid = str(teacher_row.get("question_id") or "").strip()
    if sample_qid and teacher_qid:
        return sample_qid == teacher_qid
    return True


def _teacher_lookup(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for key in _sample_keys(row):
            lookup[key] = row
    return lookup


def _teacher_record_for_sample(
    sample: Dict[str, Any],
    primary_rows: List[Dict[str, Any]],
    primary_lookup: Dict[str, Dict[str, Any]],
    override_lookup: Dict[str, Dict[str, Any]],
    index: int,
) -> Tuple[Dict[str, Any], str]:
    for key in _sample_keys(sample):
        candidate = override_lookup.get(key)
        if candidate is not None and _questions_compatible(sample, candidate):
            return candidate, "override"
    for key in _sample_keys(sample):
        candidate = primary_lookup.get(key)
        if candidate is not None:
            return candidate, "primary_lookup"
    if 0 <= index < len(primary_rows):
        return primary_rows[index], "primary_index_fallback"
    return sample, "sample_fallback"


def _teacher_answer_label(sample: Dict[str, Any], teacher_answer: str) -> Optional[str]:
    parsed = parse_student_answer(
        teacher_answer,
        question_type=_question_type(sample),
        question=_question_text(sample),
    )
    return normalize_answer_label(parsed.get("answer_label"))


def _canonical_answer_supervision(
    sample: Dict[str, Any],
    teacher_answer: str,
) -> Tuple[Optional[str], Optional[str]]:
    question_type = _question_type(sample)
    parsed = parse_student_answer(
        teacher_answer,
        question_type=question_type,
        question=_question_text(sample),
    )
    label = normalize_answer_label(parsed.get("answer_label"))
    if question_type == "choice":
        if label in {"A", "B", "C", "D", "E"}:
            return "choice", label
        return "choice", None
    if question_type == "judgment":
        if label is None:
            return "judgment", None
        lowered = str(label).strip().lower()
        if lowered in {"yes", "true", "y", "t", "1"}:
            return "judgment", "True"
        if lowered in {"no", "false", "n", "f", "0"}:
            return "judgment", "False"
        return "judgment", None
    return None, None


def _load_score_image_manifest(path: str | None) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    resolved = _resolve(path)
    payload = load_json(resolved)
    records = payload.get("records") if isinstance(payload, dict) else payload
    mapping: Dict[str, Dict[str, Any]] = {}
    for record in records or []:
        if "index" in record:
            mapping[f"index:{int(record['index'])}"] = record
        for key in _sample_keys(record):
            mapping[f"sample:{key}"] = record
    return mapping


def _load_image_manifest(path: str | None) -> Dict[str, Dict[str, Any]]:
    return _load_score_image_manifest(path)


def _score_image_for_row(mapping: Dict[str, Dict[str, Any]], idx: int, row: Dict[str, Any]) -> Optional[str]:
    record = mapping.get(f"index:{idx}")
    if record is None:
        for key in _sample_keys(row):
            record = mapping.get(f"sample:{key}")
            if record is not None:
                break
    if record is None:
        return None
    value = record.get("path") or record.get("image_path")
    return None if value is None else str(value)


def _image_path_for_row(mapping: Dict[str, Dict[str, Any]], idx: int, row: Dict[str, Any]) -> Optional[str]:
    return _score_image_for_row(mapping, idx, row)


def _messages_to_prompt(client: LocalHFChatClient, messages: List[Dict[str, str]]) -> str:
    return client._render_messages(messages)


def _inject_axis_tokens_text_prompt(
    prompt: str,
    client: LocalHFChatClient,
    global_count: int,
    channel_count: int,
) -> str:
    global_tokens = client.GLOBAL_HINT_TOKEN * int(global_count)
    channel_tokens = client.CHANNEL_HINT_TOKEN * int(channel_count)
    if client.GLOBAL_HINT_TOKEN_PLACEHOLDER in prompt or client.CHANNEL_HINT_TOKEN_PLACEHOLDER in prompt:
        prompt = prompt.replace(client.GLOBAL_HINT_TOKEN_PLACEHOLDER, global_tokens, 1)
        prompt = prompt.replace(client.CHANNEL_HINT_TOKEN_PLACEHOLDER, channel_tokens, 1)
        return prompt
    if client.HINT_TOKEN_PLACEHOLDER in prompt:
        return prompt.replace(client.HINT_TOKEN_PLACEHOLDER, global_tokens + channel_tokens, 1)
    return prompt + "\n\n" + global_tokens + channel_tokens + "\n"


def _sample_interval(
    sample: Dict[str, Any],
    encoder: AXISMultivariateIntervalProposer,
    config: Dict[str, Any],
    interval_source: str,
    window_size: int,
    stride: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if interval_source == "target":
        interval = sample.get("target_interval") or {}
        start = int(interval.get("start", 0))
        end = int(interval.get("end", start + 1))
        return sample_with_interval(sample, start, end), {
            "start": start,
            "end": end,
            "source": "target_interval_for_teacher_forced_nexttoken_training",
        }
    proposal = propose_interval(
        sample,
        encoder,
        hint_tuner=None,
        window_size=int(window_size),
        stride=int(stride),
        top_k_channels=int(config["model"].get("top_k_channels", 3)),
    )
    if proposal.get("start") is None:
        return sample, proposal
    return sample_with_interval(sample, int(proposal["start"]), int(proposal["end"])), proposal


def _hint_bundle(
    encoder: AXISMultivariateIntervalProposer,
    sample: Dict[str, Any],
    client: LocalHFChatClient | LocalHFVLChatClient,
    *,
    include_global_hints: bool,
) -> Dict[str, Any]:
    interval = sample.get("target_interval") or {}
    start = int(interval.get("start", 0))
    end = int(interval.get("end", start + 1))
    axis_cfg = client.axis_hints_config
    if include_global_hints:
        return encoder.make_embedding_hint_tensors(
            sample,
            start,
            end,
            question_text=_question_text(sample),
            llm_question_embedder=client,
            query_mode=axis_cfg.get("global_query_mode"),
            query_mix_alpha=axis_cfg.get("global_query_mix_alpha"),
            global_sampling_strategy=axis_cfg.get("global_sampling_strategy", "uniform"),
            max_question_tokens=axis_cfg.get("max_question_tokens", 128),
            top_k_channels=axis_cfg.get("top_k_channels"),
            max_channel_tokens=axis_cfg.get("max_channel_tokens", axis_cfg.get("max_local_tokens")),
            max_global_tokens=axis_cfg.get(
                "max_global_tokens",
                axis_cfg.get("num_global_tokens", axis_cfg.get("num_fixed_tokens")),
            ),
        )
    return encoder.make_channel_only_embedding_hints(
        sample,
        start,
        end,
        top_k_channels=axis_cfg.get("top_k_channels"),
        max_channel_tokens=axis_cfg.get("max_channel_tokens", axis_cfg.get("max_local_tokens")),
    )


def _tokenizer_for_client(client: LocalHFChatClient | LocalHFVLChatClient):
    if hasattr(client, "tokenizer"):
        return client.tokenizer
    return client.processor.tokenizer


def _is_vl_client(client: Any) -> bool:
    return isinstance(client, LocalHFVLChatClient)


def _model_device(client: LocalHFChatClient | LocalHFVLChatClient):
    if isinstance(getattr(client.model, "hf_device_map", None), dict):
        return client.device
    return client.model.device if hasattr(client.model, "device") else client.device


def _trim_target_text(tokenizer: Any, text: str, max_target_tokens: int) -> str:
    target = text.strip()
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        target += eos_token
    if int(max_target_tokens) <= 0:
        return target
    encoded = tokenizer(target, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][:, : int(max_target_tokens)]
    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _project_hint_embeddings(
    client: LocalHFChatClient | LocalHFVLChatClient,
    hint_tensors: Dict[str, Any],
    embedding_device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if client.perceiver is None:
        raise RuntimeError("axis_hints.enabled must be true")
    if next(client.perceiver.parameters()).device != embedding_device or next(client.perceiver.parameters()).dtype != torch.float32:
        client.perceiver.to(device=embedding_device, dtype=torch.float32)
    word_embeddings = client.model.get_input_embeddings().weight.detach()
    if word_embeddings.device != embedding_device:
        word_embeddings = word_embeddings.to(device=embedding_device)
    autocast_dtype = word_embeddings.dtype if word_embeddings.dtype in (torch.float16, torch.bfloat16) else torch.float16
    with torch.autocast(device_type=embedding_device.type, dtype=autocast_dtype, enabled=embedding_device.type == "cuda"):
        source_embeddings = client.perceiver.get_source_embeddings(word_embeddings)
    source_embeddings = source_embeddings.to(device=embedding_device, dtype=torch.float32)
    projected_global = None
    projected_channel = None
    pooled_global = None
    global_tensor = hint_tensors.get("global")
    channel_tensor = hint_tensors.get("channel")
    if global_tensor is not None:
        global_tensor = global_tensor.to(device=embedding_device, dtype=torch.float32)
        projected_global = client.perceiver.process_local_embeddings(global_tensor, source_embeddings)
        pooled_global = projected_global.mean(dim=0)
    if channel_tensor is not None:
        channel_tensor = channel_tensor.to(device=embedding_device, dtype=torch.float32)
        projected_channel = client.perceiver.process_local_embeddings(channel_tensor, source_embeddings)
    return projected_global, projected_channel, pooled_global


def _inject_projected_hints(
    client: LocalHFChatClient | LocalHFVLChatClient,
    input_ids: torch.Tensor,
    input_embeddings: torch.Tensor,
    projected_global: Optional[torch.Tensor],
    projected_channel: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, int]]:
    tokenizer = _tokenizer_for_client(client)
    global_positions = (input_ids[0] == client.global_hint_token_id).nonzero(as_tuple=True)[0]
    channel_positions = (input_ids[0] == client.channel_hint_token_id).nonzero(as_tuple=True)[0]
    neutral_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    neutral_id = int(neutral_ids[0]) if neutral_ids else int(eos_token_id)
    neutral_embedding = client.model.get_input_embeddings().weight[neutral_id].detach().to(
        device=input_embeddings.device,
        dtype=input_embeddings.dtype,
    )

    if len(global_positions) > 0 and projected_global is not None:
        scale = float(client.axis_hints_config.get("global_injection_scale", client.axis_hints_config.get("injection_scale", 1.0)))
        scale = max(0.0, min(1.0, scale))
        injected = (1.0 - scale) * neutral_embedding + scale * projected_global[: len(global_positions)].to(input_embeddings.dtype)
        input_embeddings[0, global_positions] = injected
    if len(channel_positions) > 0 and projected_channel is not None:
        scale = float(client.axis_hints_config.get("channel_injection_scale", client.axis_hints_config.get("injection_scale", 1.0)))
        scale = max(0.0, min(1.0, scale))
        injected = (1.0 - scale) * neutral_embedding + scale * projected_channel[: len(channel_positions)].to(input_embeddings.dtype)
        input_embeddings[0, channel_positions] = injected
    return input_embeddings, {
        "global_hint_tokens": int(len(global_positions)),
        "channel_hint_tokens": int(len(channel_positions)),
    }


def _answer_auxiliary_loss(
    pooled_global: Optional[torch.Tensor],
    answer_heads: torch.nn.ModuleDict,
    answer_type: Optional[str],
    answer_label: Optional[str],
    answer_label_vocabs: Dict[str, Dict[str, int]],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if (
        pooled_global is None
        or answer_type not in answer_heads
        or answer_type not in answer_label_vocabs
        or answer_label is None
        or answer_label not in answer_label_vocabs[answer_type]
    ):
        device = pooled_global.device if pooled_global is not None else next(answer_heads.parameters()).device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {
            "answer_loss": 0.0,
            "answer_acc": 0.0,
            "answer_type": answer_type,
            "answer_label": answer_label,
            "answer_pred": None,
            "answer_supervised": False,
        }
    label_vocab = answer_label_vocabs[answer_type]
    logits = answer_heads[answer_type](pooled_global.unsqueeze(0))
    target = torch.tensor([label_vocab[answer_label]], device=pooled_global.device, dtype=torch.long)
    loss = F.cross_entropy(logits, target)
    inv_vocab = {idx: label for label, idx in label_vocab.items()}
    pred = inv_vocab.get(int(logits.argmax(dim=-1).item()))
    return loss, {
        "answer_loss": float(loss.detach().cpu()),
        "answer_acc": float((logits.argmax(dim=-1) == target).float().mean().detach().cpu()),
        "answer_type": answer_type,
        "answer_label": answer_label,
        "answer_pred": pred,
        "answer_supervised": True,
    }


def _global_hint_auxiliary_losses(
    *,
    encoder: AXISMultivariateIntervalProposer,
    sample: Dict[str, Any],
    paired_sample: Optional[Dict[str, Any]],
    client: LocalHFChatClient | LocalHFVLChatClient,
    global_anomaly_head: Optional[torch.nn.Module],
    global_anom_loss_weight: float,
    perm_loss_weight: float,
    pair_loss_weight: float,
    enable_perm_loss: bool,
    enable_pair_loss: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    axis_cfg = client.axis_hints_config
    query_mode = axis_cfg.get("global_query_mode")
    query_mix_alpha = axis_cfg.get("global_query_mix_alpha")
    max_question_tokens = int(axis_cfg.get("max_question_tokens", 128))
    forward = encoder.global_hint_forward_tensors(
        sample,
        question_text=_question_text(sample),
        llm_question_embedder=client,
        query_mode=query_mode,
        query_mix_alpha=query_mix_alpha,
        max_question_tokens=max_question_tokens,
    )
    global_states = forward["global_states"]
    time_mask = forward["time_mask"].bool()
    time_labels = forward["time_labels"].long()

    zero = global_states.new_zeros(())
    loss_global = zero
    if global_anomaly_head is not None and float(global_anom_loss_weight) > 0.0:
        logits_global = global_anomaly_head(global_states)
        loss_global = _masked_time_ce(logits_global[0], time_labels[0], time_mask[0])

    loss_perm = zero
    if enable_perm_loss and float(perm_loss_weight) > 0.0:
        channel_count = int(sample["series"]["shape"][1])
        permutation = torch.randperm(channel_count).tolist()
        permuted_sample = _permute_sample(sample, permutation)
        forward_perm = encoder.global_hint_forward_tensors(
            permuted_sample,
            question_text=_question_text(sample),
            llm_question_embedder=client,
            query_mode=query_mode,
            query_mix_alpha=query_mix_alpha,
            max_question_tokens=max_question_tokens,
        )
        common_mask = time_mask & forward_perm["time_mask"].bool()
        if common_mask.any():
            loss_perm = F.mse_loss(
                global_states[common_mask],
                forward_perm["global_states"][common_mask],
            )

    loss_pair = zero
    if enable_pair_loss and paired_sample is not None and float(pair_loss_weight) > 0.0 and forward["query_mode_used"] != "data_only":
        pair_forward = encoder.global_hint_forward_tensors(
            paired_sample,
            question_text=_question_text(paired_sample),
            llm_question_embedder=client,
            query_mode=query_mode,
            query_mix_alpha=query_mix_alpha,
            max_question_tokens=max_question_tokens,
        )
        num_question_queries = int(encoder.global_hint_head.question_adapter.num_question_queries)
        current_map = _question_attention_map(forward["diagnostics"], num_question_queries, forward["time_mask"][0])
        pair_map = _question_attention_map(pair_forward["diagnostics"], num_question_queries, pair_forward["time_mask"][0])
        if current_map is not None and pair_map is not None and current_map.shape == pair_map.shape:
            loss_pair = F.mse_loss(current_map, pair_map)

    total = (
        float(global_anom_loss_weight) * loss_global
        + float(perm_loss_weight if enable_perm_loss else 0.0) * loss_perm
        + float(pair_loss_weight if enable_pair_loss else 0.0) * loss_pair
    )
    return total, {
        "global_aux_loss": float(total.detach().cpu()),
        "global_anom_loss": float(loss_global.detach().cpu()),
        "perm_loss": float(loss_perm.detach().cpu()),
        "pair_loss": float(loss_pair.detach().cpu()),
        "time_anomaly_rate": float(time_labels[0][time_mask[0]].float().mean().detach().cpu()) if time_mask[0].any() else 0.0,
        "query_mode_requested": forward["query_mode_requested"],
        "query_mode_used": forward["query_mode_used"],
    }


def _causal_lm_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _forward_causal_lm_loss(
    model: torch.nn.Module,
    *,
    forward_kwargs: Dict[str, Any],
    full_labels: torch.Tensor,
    short_labels: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    try:
        outputs = model(**forward_kwargs)
        loss = getattr(outputs, "loss", None)
        if loss is not None:
            return loss
        logits = getattr(outputs, "logits", None)
        if logits is not None and logits.shape[1] == short_labels.shape[1]:
            return _causal_lm_loss_from_logits(logits, short_labels)
    except (TypeError, ValueError):
        pass

    fallback_kwargs = dict(forward_kwargs)
    fallback_kwargs.pop("labels", None)
    fallback_kwargs.pop("logits_to_keep", None)
    outputs = model(**fallback_kwargs)
    return _causal_lm_loss_from_logits(outputs.logits, full_labels)


def _loss_for_item_text(
    *,
    client: LocalHFChatClient,
    prompt: str,
    teacher_answer: str,
    hint_bundle: Dict[str, Any],
    answer_heads: torch.nn.ModuleDict,
    answer_type: Optional[str],
    answer_label: Optional[str],
    answer_loss_weight: float,
    answer_label_vocabs: Dict[str, Dict[str, int]],
    max_prompt_tokens: int,
    max_target_tokens: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    tokenizer = _tokenizer_for_client(client)
    teacher_text = _trim_target_text(tokenizer, teacher_answer, int(max_target_tokens))
    prompt_ids = tokenizer(prompt, return_tensors="pt")
    target_ids = tokenizer(teacher_text, return_tensors="pt", add_special_tokens=False)
    if int(max_prompt_tokens) > 0 and prompt_ids["input_ids"].shape[-1] > int(max_prompt_tokens):
        for key in list(prompt_ids.keys()):
            prompt_ids[key] = prompt_ids[key][:, -int(max_prompt_tokens) :]
    input_ids = torch.cat([prompt_ids["input_ids"], target_ids["input_ids"]], dim=1).to(client.device)
    prompt_attention = prompt_ids.get("attention_mask", torch.ones_like(prompt_ids["input_ids"]))
    target_attention = target_ids.get("attention_mask", torch.ones_like(target_ids["input_ids"]))
    attention_mask = torch.cat([prompt_attention, target_attention], dim=1).to(client.device)
    labels = input_ids.clone()
    prompt_len = int(prompt_ids["input_ids"].shape[-1])
    labels[:, :prompt_len] = -100
    target_token_count = int((labels != -100).sum().detach().cpu())
    logits_to_keep = max(2, target_token_count + 1)
    labels_for_loss = labels[:, -logits_to_keep:].contiguous()

    prepared = client._prepare_axis_hint_tensors(hint_bundle["embeddings"])
    input_embeddings = client.model.get_input_embeddings()(input_ids).detach()
    projected_global, projected_channel, pooled_global = _project_hint_embeddings(client, prepared, input_embeddings.device)
    input_embeddings, token_stats = _inject_projected_hints(
        client,
        input_ids,
        input_embeddings,
        projected_global,
        projected_channel,
    )
    forward_kwargs = {
        "inputs_embeds": input_embeddings,
        "attention_mask": attention_mask,
        "labels": labels_for_loss,
        "logits_to_keep": logits_to_keep,
        "use_cache": False,
    }
    ntp_loss = _forward_causal_lm_loss(
        client.model,
        forward_kwargs=forward_kwargs,
        full_labels=labels,
        short_labels=labels_for_loss,
        logits_to_keep=logits_to_keep,
    )
    answer_loss, answer_stats = _answer_auxiliary_loss(
        pooled_global,
        answer_heads,
        answer_type,
        answer_label,
        answer_label_vocabs,
    )
    total_loss = ntp_loss + float(answer_loss_weight) * answer_loss
    return total_loss, {
        "ntp_loss": float(ntp_loss.detach().cpu()),
        "answer_loss": float(answer_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
        "target_tokens": target_token_count,
        **token_stats,
        **answer_stats,
    }


def _loss_for_item_vl(
    *,
    client: LocalHFVLChatClient,
    messages: List[Dict[str, str]],
    image_paths: List[str],
    teacher_answer: str,
    hint_bundle: Dict[str, Any],
    answer_heads: torch.nn.ModuleDict,
    answer_type: Optional[str],
    answer_label: Optional[str],
    answer_loss_weight: float,
    answer_label_vocabs: Dict[str, Dict[str, int]],
    max_prompt_tokens: int,
    max_target_tokens: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    tokenizer = _tokenizer_for_client(client)
    teacher_text = _trim_target_text(tokenizer, teacher_answer, int(max_target_tokens))
    prepared = client._prepare_axis_hint_tensors(hint_bundle["embeddings"])
    axis_messages = client._inject_axis_hint_tokens(messages, prepared)
    prompt_vl_messages = client._to_vl_messages(axis_messages, image_paths)
    full_vl_messages = client._to_vl_messages(axis_messages + [{"role": "assistant", "content": teacher_text}], image_paths)
    prompt_text = client.processor.apply_chat_template(prompt_vl_messages, tokenize=False, add_generation_prompt=True)
    full_text = client.processor.apply_chat_template(full_vl_messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = client.process_vision_info(prompt_vl_messages)
    prompt_inputs = client.processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    full_inputs = client.processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    if int(max_prompt_tokens) > 0 and prompt_len > int(max_prompt_tokens):
        keep = int(max_prompt_tokens)
        drop = prompt_len - keep
        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            if key in prompt_inputs:
                prompt_inputs[key] = prompt_inputs[key][:, -keep:]
            if key in full_inputs:
                full_inputs[key] = full_inputs[key][:, drop:]
        prompt_len = keep
    del prompt_inputs
    device = _model_device(client)
    full_inputs = full_inputs.to(device)
    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs.get("attention_mask")
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    target_token_count = int((labels != -100).sum().detach().cpu())
    logits_to_keep = max(2, target_token_count + 1)
    labels_for_loss = labels[:, -logits_to_keep:].contiguous()
    input_embeddings = client.model.get_input_embeddings()(input_ids).detach()
    projected_global, projected_channel, pooled_global = _project_hint_embeddings(client, prepared, input_embeddings.device)
    input_embeddings, token_stats = _inject_projected_hints(
        client,
        input_ids,
        input_embeddings,
        projected_global,
        projected_channel,
    )
    forward_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "inputs_embeds": input_embeddings,
        "attention_mask": attention_mask,
        "labels": labels_for_loss,
        "logits_to_keep": logits_to_keep,
        "use_cache": False,
    }
    for optional_key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
        if optional_key in full_inputs:
            forward_kwargs[optional_key] = full_inputs[optional_key]
    ntp_loss = _forward_causal_lm_loss(
        client.model,
        forward_kwargs=forward_kwargs,
        full_labels=labels,
        short_labels=labels_for_loss,
        logits_to_keep=logits_to_keep,
    )
    answer_loss, answer_stats = _answer_auxiliary_loss(
        pooled_global,
        answer_heads,
        answer_type,
        answer_label,
        answer_label_vocabs,
    )
    total_loss = ntp_loss + float(answer_loss_weight) * answer_loss
    return total_loss, {
        "ntp_loss": float(ntp_loss.detach().cpu()),
        "answer_loss": float(answer_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
        "target_tokens": target_token_count,
        "num_images": len(image_paths),
        **token_stats,
        **answer_stats,
    }


def _build_answer_label_vocabs(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    vocabs: Dict[str, Dict[str, int]] = {}
    if any(item.get("answer_type") == "judgment" and item.get("answer_label") for item in items):
        vocabs["judgment"] = {"False": 0, "True": 1}
    if any(item.get("answer_type") == "choice" and item.get("answer_label") for item in items):
        vocabs["choice"] = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    return vocabs


def _load_bridge_warm_start(
    checkpoint_path: Path,
    client: LocalHFChatClient | LocalHFVLChatClient,
    encoder: AXISMultivariateIntervalProposer,
    *,
    global_anomaly_head: Optional[torch.nn.Module] = None,
    load_global_hint_head: bool = True,
) -> Dict[str, bool]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loaded = {
        "perceiver": False,
        "global_hint_head": False,
        "global_anomaly_head": False,
    }
    perceiver_state = checkpoint.get("perceiver")
    if perceiver_state:
        client.perceiver.load_state_dict(perceiver_state, strict=False)
        loaded["perceiver"] = True
    global_state = checkpoint.get("global_hint_head")
    if load_global_hint_head and global_state:
        encoder.global_hint_head.load_state_dict(global_state, strict=False)
        loaded["global_hint_head"] = True
    anomaly_state = checkpoint.get("global_anomaly_head")
    if global_anomaly_head is not None and anomaly_state:
        global_anomaly_head.load_state_dict(anomaly_state, strict=False)
        loaded["global_anomaly_head"] = True
    return loaded


def _percent_milestones(count: int, step_percent: int) -> List[Tuple[int, int]]:
    if count <= 0 or step_percent <= 0:
        return []
    milestones: List[Tuple[int, int]] = []
    seen = set()
    for pct in range(step_percent, 101, step_percent):
        index = max(1, math.ceil(count * pct / 100.0))
        if index not in seen:
            seen.add(index)
            milestones.append((index, pct))
    if milestones and milestones[-1][0] != count:
        milestones.append((count, 100))
    return milestones


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/question_final_530/questions_600.jsonl")
    parser.add_argument("--teacher", default="outputs/teacheranswer_question_final_530/teacher_gpt55.jsonl")
    parser.add_argument(
        "--teacher-overrides",
        default="mvaxis_answer_audit/results/teacher_gpt55_all555_reference_fixed/repair_results.jsonl",
    )
    parser.add_argument("--config", default="configs/torch_fixed60_qa600_timercd.json")
    parser.add_argument("--llm-config", default="configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256.json")
    parser.add_argument("--interval-proposer-checkpoint", default=None)
    parser.add_argument("--init-bridge-checkpoint", default=None)
    parser.add_argument("--interval-source", choices=["target", "proposal"], default="target")
    parser.add_argument("--image-manifest", default=None)
    parser.add_argument("--image-kind", choices=["score", "highlight", "raw"], default="score")
    parser.add_argument("--score-image-manifest", default=None)
    parser.add_argument(
        "--disable-global-hints",
        action="store_true",
        help="Disable global hint tokens and train only with channel/local hint embeddings.",
    )
    parser.add_argument(
        "--include-anomaly-score-text",
        action="store_true",
        help="Include anomaly-score text in the student prompt during next-token training.",
    )
    parser.add_argument(
        "--no-anomaly-score-image-note",
        action="store_true",
        help="Disable the anomaly-score image note in the student prompt, even for VL clients.",
    )
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--answer-loss-weight", type=float, default=0.2)
    parser.add_argument("--global-anom-loss-weight", type=float, default=0.2)
    parser.add_argument("--perm-loss-weight", type=float, default=0.05)
    parser.add_argument("--pair-loss-weight", type=float, default=0.05)
    parser.add_argument("--disable-perm-loss", action="store_true")
    parser.add_argument("--disable-pair-loss", action="store_true")
    parser.add_argument("--disable-global-anomaly-head", action="store_true")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-window-rows", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=0)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    parser.add_argument(
        "--preserve-device-map-in-distributed",
        action="store_true",
        help="When running distributed, keep llm_config device_map/max_memory/offload so each worker can use model sharding across its visible GPUs.",
    )
    parser.add_argument("--output-checkpoint", default="checkpoints/qwen25_vl7b_nexttoken_answer_bridge.pt")
    parser.add_argument("--output-encoder-checkpoint", default=None)
    parser.add_argument("--report", default="outputs/runs/qwen25_vl7b_nexttoken_answer_bridge/report.json")
    parser.add_argument("--run-note", default="")
    parser.add_argument("--progress-percent-step", type=int, default=10)
    args = parser.parse_args()

    _init_distributed()
    rank = _dist_rank()
    world_size = _dist_world_size()
    local_rank = _dist_local_rank()

    config = load_json(_resolve(args.config))
    llm_config = load_json(_resolve(args.llm_config))
    llm_config["use_cache"] = False
    llm_config.setdefault("axis_hints", {})["enabled"] = True
    if _dist_enabled():
        if args.preserve_device_map_in_distributed:
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        config.setdefault("model", {})["device"] = device_str
        llm_config["device"] = device_str
        if not args.preserve_device_map_in_distributed:
            llm_config.pop("device_map", None)
            llm_config.pop("max_memory", None)
            llm_config.pop("offload_folder", None)

    total_num_samples = 0
    shard_metadata: Dict[str, Any] = {}
    if _dist_enabled():
        shard_root = _resolve(args.report).parent / "_dist_jsonl_shards"
        shard_metadata = _materialize_distributed_jsonl_pair_shards(
            args.data,
            args.teacher,
            limit=args.limit,
            shard_root=shard_root,
        )
        rows = _read_jsonl_maybe_limited(shard_metadata["local_data_path"])
        teacher_rows = _read_jsonl_maybe_limited(shard_metadata["local_teacher_path"])
        total_num_samples = int(shard_metadata["total_num_samples"])
    else:
        rows = _read_jsonl_maybe_limited(args.data, args.limit)
        teacher_rows = _read_jsonl_maybe_limited(args.teacher, args.limit)
        total_num_samples = len(rows)
    override_rows: List[Dict[str, Any]] = []
    if args.teacher_overrides:
        override_path = _resolve(args.teacher_overrides)
        if override_path.exists():
            override_rows = _read_jsonl_maybe_limited(override_path, args.limit)

    primary_lookup = _teacher_lookup(teacher_rows)
    override_lookup = _teacher_lookup(override_rows)
    image_manifest_path = args.image_manifest or args.score_image_manifest
    image_kind = str(args.image_kind)
    include_global_hints = not bool(args.disable_global_hints)
    image_manifest = _load_image_manifest(image_manifest_path)

    training_items: List[Dict[str, Any]] = []
    teacher_sources: Dict[str, int] = {}
    missing_images = 0
    for idx, row in enumerate(rows):
        teacher_row, teacher_source = _teacher_record_for_sample(
            row,
            teacher_rows,
            primary_lookup,
            override_lookup,
            idx,
        )
        teacher_answer = extract_teacher_answer(teacher_row)
        answer_type, answer_label = _canonical_answer_supervision(row, teacher_answer)
        image_path = _image_path_for_row(image_manifest, idx, row)
        teacher_sources[teacher_source] = teacher_sources.get(teacher_source, 0) + 1
        if image_manifest and not image_path:
            missing_images += 1
        training_items.append(
            {
                "index": idx,
                "row": row,
                "teacher_answer": teacher_answer,
                "answer_type": answer_type,
                "answer_label": answer_label,
                "teacher_source": teacher_source,
                "image_path": image_path,
            }
        )
    pair_lookup = _build_pair_lookup(rows)
    sample_lookup = {
        str(item["row"].get("sample_id")): item["row"]
        for item in training_items
        if item["row"].get("sample_id") is not None
    }

    local_vocab_flags = _answer_label_vocab_flags(training_items)
    if _dist_enabled():
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        flag_tensor = torch.tensor(
            [float(local_vocab_flags["judgment"]), float(local_vocab_flags["choice"])],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)
        answer_label_vocabs = _build_answer_label_vocabs_from_flags(
            {
                "judgment": int(flag_tensor[0].item() > 0),
                "choice": int(flag_tensor[1].item() > 0),
            }
        )
    else:
        answer_label_vocabs = _build_answer_label_vocabs_from_flags(local_vocab_flags)
    if not answer_label_vocabs:
        raise RuntimeError("No supervised answer labels could be parsed from the teacher answers.")
    if not _dist_enabled():
        training_items = _distributed_shard(training_items)
    _dist_print(
        {
            "stage": "distributed_shard_ready",
            "total_num_samples": total_num_samples,
            "local_num_samples": len(training_items),
            "shard_dir": str(shard_metadata.get("shard_dir", "")),
        }
    )

    checkpoint_path = args.interval_proposer_checkpoint or (config.get("train_interval_proposal") or {}).get("checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError("Missing interval proposer checkpoint. Set --interval-proposer-checkpoint or config.train_interval_proposal.checkpoint_path.")
    ckpt = relative_to_root(ROOT, checkpoint_path)
    if not training_items:
        raise RuntimeError("No training items available after sharding.")
    reference_num_channels = _row_num_channels(training_items[0]["row"])
    _dist_print(
        {
            "stage": "encoder_load_start",
            "checkpoint_path": str(ckpt),
            "reference_num_channels": reference_num_channels,
        }
    )
    encoder, encoder_load_info = _load_encoder_for_channel_count(
        ckpt,
        config,
        num_channels=reference_num_channels,
    )
    _configure_encoder_for_hint_training(
        encoder,
        include_global_hints=include_global_hints,
    )
    encoder_cache: Dict[int, AXISMultivariateIntervalProposer] = {
        reference_num_channels: encoder,
    }
    encoder_load_infos: Dict[int, Dict[str, Any]] = {
        reference_num_channels: encoder_load_info,
    }
    shared_global_hint_head = encoder.global_hint_head
    _dist_print(
        {
            "stage": "encoder_load_done",
            "checkpoint_path": str(ckpt),
            "reference_num_channels": reference_num_channels,
            "loaded_keys": encoder_load_info.get("loaded_keys"),
            "skipped_shape_mismatch_keys": len(encoder_load_info.get("skipped_shape_mismatch_keys") or []),
            "skipped_global_hint_head_keys": len(encoder_load_info.get("skipped_global_hint_head_keys") or []),
        }
    )
    global_hint_d_proj = int(
        getattr(
            encoder.global_hint_head,
            "d_proj",
            config.get("model", {}).get("ts_encoder", {}).get("d_proj", 256),
        )
    )
    global_anomaly_head: Optional[torch.nn.Module] = None
    if include_global_hints:
        global_anomaly_head = torch.nn.Linear(global_hint_d_proj, 2).to(encoder.device, dtype=torch.float32)
        if bool(args.disable_global_anomaly_head):
            global_anomaly_head.eval()
            for param in global_anomaly_head.parameters():
                param.requires_grad = False
        else:
            global_anomaly_head.train()

    def _encoder_for_sample(sample: Dict[str, Any]) -> AXISMultivariateIntervalProposer:
        channel_count = _row_num_channels(sample)
        cached = encoder_cache.get(channel_count)
        if cached is not None:
            return cached
        loaded_encoder, load_info = _load_encoder_for_channel_count(
            ckpt,
            config,
            num_channels=channel_count,
        )
        loaded_encoder.global_hint_head = shared_global_hint_head
        _configure_encoder_for_hint_training(
            loaded_encoder,
            include_global_hints=include_global_hints,
        )
        encoder_cache[channel_count] = loaded_encoder
        encoder_load_infos[channel_count] = load_info
        _dist_print(
            {
                "stage": "encoder_cache_add",
                "checkpoint_path": str(ckpt),
                "num_channels": channel_count,
                "loaded_keys": load_info.get("loaded_keys"),
                "skipped_shape_mismatch_keys": len(load_info.get("skipped_shape_mismatch_keys") or []),
                "skipped_global_hint_head_keys": len(load_info.get("skipped_global_hint_head_keys") or []),
            }
        )
        return loaded_encoder

    stagger_seconds = float(os.environ.get("MVAXIS_CLIENT_INIT_STAGGER_SEC", "0") or 0)
    if _dist_enabled() and stagger_seconds > 0:
        delay = float(_dist_rank()) * stagger_seconds
        _dist_print({"stage": "client_init_stagger", "sleep_seconds": delay})
        time.sleep(delay)
    _dist_print({"stage": "client_init_start", "provider": llm_config.get("provider"), "model": llm_config.get("model")})
    client = create_llm_client(llm_config)
    _dist_print({"stage": "client_init_done", "provider": llm_config.get("provider"), "model": llm_config.get("model")})
    if not isinstance(client, (LocalHFChatClient, LocalHFVLChatClient)):
        raise RuntimeError("This training script supports only local_hf and local_hf_vl providers.")
    client.model.eval()
    client.model.config.use_cache = False
    if hasattr(client.model, "gradient_checkpointing_enable"):
        try:
            client.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            client.model.gradient_checkpointing_enable()
    for param in client.model.parameters():
        param.requires_grad = False
    if client.perceiver is None:
        raise RuntimeError("LLM config must enable axis_hints for next-token hint training.")
    client.perceiver.train()
    for param in client.perceiver.parameters():
        param.requires_grad = True

    if args.init_bridge_checkpoint:
        warm_start_path = _resolve(args.init_bridge_checkpoint)
        _dist_print({"stage": "warm_start_load_start", "checkpoint_path": str(warm_start_path)})
        warm_loaded = _load_bridge_warm_start(
            warm_start_path,
            client,
            encoder,
            global_anomaly_head=global_anomaly_head,
            load_global_hint_head=include_global_hints,
        )
        _dist_print({"stage": "warm_start_load_done", "checkpoint_path": str(warm_start_path), "warm_loaded": warm_loaded})
    else:
        warm_loaded = {"perceiver": False, "global_hint_head": False}

    if _is_vl_client(client) and image_manifest_path:
        unavailable = [item["row"].get("sample_id") for item in training_items if not item.get("image_path")]
        if unavailable:
            preview = unavailable[:5]
            raise RuntimeError(
                f"Missing {image_kind} image paths for {len(unavailable)} samples. Examples: {preview}. "
                "Regenerate the manifest or verify sample-id/index alignment."
            )

    hidden_size = int(getattr(client.model.config, "hidden_size", 0) or client.model.config.text_config.hidden_size)
    aux_device = _model_device(client)
    answer_heads = torch.nn.ModuleDict()
    if "judgment" in answer_label_vocabs:
        answer_heads["judgment"] = torch.nn.Linear(hidden_size, len(answer_label_vocabs["judgment"]))
    if "choice" in answer_label_vocabs:
        answer_heads["choice"] = torch.nn.Linear(hidden_size, len(answer_label_vocabs["choice"]))
    answer_heads = answer_heads.to(aux_device, dtype=torch.float32)
    answer_heads.train()

    trainable = list(client.perceiver.parameters()) + list(answer_heads.parameters())
    if include_global_hints:
        trainable += list(shared_global_hint_head.parameters())
    if global_anomaly_head is not None and not bool(args.disable_global_anomaly_head):
        trainable += list(global_anomaly_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    accum = max(1, int(args.gradient_accumulation_steps))
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()
    global_step = 0

    _rank0_print(
        {
            "stage": "train_setup",
            "num_samples": total_num_samples,
            "local_num_samples": len(training_items),
            "rank": rank,
            "world_size": world_size,
            "run_note": str(args.run_note),
            "image_kind": image_kind,
            "image_manifest_path": image_manifest_path,
            "include_global_hints": include_global_hints,
            "include_anomaly_score_text": bool(args.include_anomaly_score_text),
            "include_anomaly_score_image_note": (
                _is_vl_client(client)
                and image_kind == "score"
                and not bool(args.no_anomaly_score_image_note)
            ),
            "include_raw_image_note": _is_vl_client(client) and image_kind in {"highlight", "raw"},
            "answer_label_vocabs": answer_label_vocabs,
            "pair_lookup_size": len(pair_lookup),
            "global_anom_loss_weight": float(args.global_anom_loss_weight),
            "perm_loss_weight": float(0.0 if args.disable_perm_loss else args.perm_loss_weight),
            "pair_loss_weight": float(0.0 if args.disable_pair_loss else args.pair_loss_weight),
            "teacher_sources": teacher_sources,
            "image_manifest_loaded": bool(image_manifest),
            "missing_images": missing_images,
            "llm_model": llm_config.get("model"),
            "provider": llm_config.get("provider"),
            "warm_start": warm_loaded,
            "preserve_device_map_in_distributed": bool(args.preserve_device_map_in_distributed),
        }
    )

    for epoch in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_ntp = 0.0
        epoch_answer = 0.0
        epoch_global_aux = 0.0
        epoch_global_anom = 0.0
        epoch_perm = 0.0
        epoch_pair = 0.0
        answer_supervised = 0
        milestones = _percent_milestones(len(training_items), int(args.progress_percent_step))
        next_milestone_idx = 0

        for idx, item in enumerate(training_items):
            raw_sample = item["row"]
            item_encoder = _encoder_for_sample(raw_sample)
            paired_sample = None
            paired_sample_id = pair_lookup.get(str(raw_sample.get("sample_id")))
            if paired_sample_id:
                candidate_pair = sample_lookup.get(str(paired_sample_id))
                if candidate_pair is not None and _row_num_channels(candidate_pair) == _row_num_channels(raw_sample):
                    paired_sample = candidate_pair
            sample, proposal = _sample_interval(
                raw_sample,
                item_encoder,
                config,
                str(args.interval_source),
                int(args.window_size),
                int(args.stride),
            )
            hint_bundle = _hint_bundle(
                item_encoder,
                sample,
                client,
                include_global_hints=include_global_hints,
            )
            messages = build_student_answer_messages(
                sample,
                embedding_hint_trace=hint_bundle["trace"],
                use_axis_hints=True,
                include_global_hints=include_global_hints,
                include_channel_hints=True,
                include_evidence_card=False,
                include_anomaly_score_text=bool(args.include_anomaly_score_text),
                include_anomaly_score_image_note=(
                    _is_vl_client(client)
                    and image_kind == "score"
                    and not bool(args.no_anomaly_score_image_note)
                ),
                include_raw_image_note=_is_vl_client(client) and image_kind in {"highlight", "raw"},
                max_window_rows=None if int(args.max_window_rows) <= 0 else int(args.max_window_rows),
            )
            if _is_vl_client(client):
                image_paths = [item["image_path"]] if item.get("image_path") else []
                loss, stats = _loss_for_item_vl(
                    client=client,
                    messages=messages,
                    image_paths=image_paths,
                    teacher_answer=item["teacher_answer"],
                    hint_bundle=hint_bundle,
                    answer_heads=answer_heads,
                    answer_type=item["answer_type"],
                    answer_label=item["answer_label"],
                    answer_loss_weight=float(args.answer_loss_weight),
                    answer_label_vocabs=answer_label_vocabs,
                    max_prompt_tokens=int(args.max_prompt_tokens),
                    max_target_tokens=int(args.max_target_tokens),
                )
            else:
                prompt = _messages_to_prompt(client, messages)
                global_embeddings = hint_bundle["embeddings"].get("global")
                prompt = _inject_axis_tokens_text_prompt(
                    prompt,
                    client,
                    0 if global_embeddings is None else int(global_embeddings.shape[0]),
                    int(hint_bundle["embeddings"]["channel"].shape[0]),
                )
                loss, stats = _loss_for_item_text(
                    client=client,
                    prompt=prompt,
                    teacher_answer=item["teacher_answer"],
                    hint_bundle=hint_bundle,
                    answer_heads=answer_heads,
                    answer_type=item["answer_type"],
                    answer_label=item["answer_label"],
                    answer_loss_weight=float(args.answer_loss_weight),
                    answer_label_vocabs=answer_label_vocabs,
                    max_prompt_tokens=int(args.max_prompt_tokens),
                    max_target_tokens=int(args.max_target_tokens),
                )
            global_aux_loss = loss.new_zeros(())
            global_aux_stats = {
                "global_aux_loss": 0.0,
                "global_anom_loss": 0.0,
                "perm_loss": 0.0,
                "pair_loss": 0.0,
                "time_anomaly_rate": 0.0,
                "query_mode_requested": None,
                "query_mode_used": None,
            }
            if include_global_hints:
                global_aux_loss, global_aux_stats = _global_hint_auxiliary_losses(
                    encoder=item_encoder,
                    sample=sample,
                    paired_sample=paired_sample,
                    client=client,
                    global_anomaly_head=global_anomaly_head,
                    global_anom_loss_weight=float(args.global_anom_loss_weight),
                    perm_loss_weight=float(args.perm_loss_weight),
                    pair_loss_weight=float(args.pair_loss_weight),
                    enable_perm_loss=not bool(args.disable_perm_loss),
                    enable_pair_loss=not bool(args.disable_pair_loss),
                )
            total_loss = loss + global_aux_loss

            (total_loss / accum).backward()
            epoch_loss += float(total_loss.detach().cpu())
            epoch_ntp += float(stats["ntp_loss"])
            epoch_answer += float(stats["answer_loss"])
            epoch_global_aux += float(global_aux_stats["global_aux_loss"])
            epoch_global_anom += float(global_aux_stats["global_anom_loss"])
            epoch_perm += float(global_aux_stats["perm_loss"])
            epoch_pair += float(global_aux_stats["pair_loss"])
            answer_supervised += int(bool(stats.get("answer_supervised")))

            if (idx + 1) % accum == 0 or idx + 1 == len(training_items):
                _average_gradients(trainable)
                torch.nn.utils.clip_grad_norm_(trainable, float(args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            record = {
                "epoch": epoch + 1,
                "index": idx,
                "sample_id": raw_sample.get("sample_id"),
                "teacher_source": item["teacher_source"],
                "proposal_start": proposal.get("start"),
                "proposal_end": proposal.get("end"),
                "paired_sample_id": paired_sample_id,
                **stats,
                **global_aux_stats,
                "lm_total_loss": float(stats["total_loss"]),
                "total_loss": float(total_loss.detach().cpu()),
            }
            history.append(record)

            while next_milestone_idx < len(milestones) and (idx + 1) >= milestones[next_milestone_idx][0]:
                _, pct = milestones[next_milestone_idx]
                _rank0_print(
                    {
                        "stage": "epoch_progress",
                        "epoch": epoch + 1,
                        "progress_percent": pct,
                        "progress": f"{idx + 1}/{len(training_items)}",
                        "total_loss": record["total_loss"],
                        "ntp_loss": record["ntp_loss"],
                        "answer_loss": record["answer_loss"],
                        "answer_type": record.get("answer_type"),
                        "answer_label": record.get("answer_label"),
                        "answer_pred": record.get("answer_pred"),
                        "global_hint_tokens": record.get("global_hint_tokens"),
                        "channel_hint_tokens": record.get("channel_hint_tokens"),
                    }
                )
                next_milestone_idx += 1

        reduced = _reduce_scalar_dict(
            {
                "epoch_loss": epoch_loss,
                "epoch_ntp": epoch_ntp,
                "epoch_answer": epoch_answer,
                "epoch_global_aux": epoch_global_aux,
                "epoch_global_anom": epoch_global_anom,
                "epoch_perm": epoch_perm,
                "epoch_pair": epoch_pair,
                "answer_supervised": float(answer_supervised),
                "local_count": float(len(training_items)),
                "global_step": float(global_step),
            },
            device=aux_device,
        )
        denom = max(1.0, float(reduced["local_count"]))
        epoch_report = {
            "stage": "epoch_summary",
            "epoch": epoch + 1,
            "mean_total_loss": float(reduced["epoch_loss"]) / denom,
            "mean_ntp_loss": float(reduced["epoch_ntp"]) / denom,
            "mean_answer_loss": float(reduced["epoch_answer"]) / denom,
            "mean_global_aux_loss": float(reduced["epoch_global_aux"]) / denom,
            "mean_global_anom_loss": float(reduced["epoch_global_anom"]) / denom,
            "mean_perm_loss": float(reduced["epoch_perm"]) / denom,
            "mean_pair_loss": float(reduced["epoch_pair"]) / denom,
            "answer_supervised_items": int(round(float(reduced["answer_supervised"]))),
            "global_steps": int(round(float(reduced["global_step"]))),
        }
        _rank0_print(epoch_report)

    reduced_final = _reduce_scalar_dict(
        {
            "hist_total": sum(x["total_loss"] for x in history),
            "hist_ntp": sum(x["ntp_loss"] for x in history),
            "hist_answer": sum(x["answer_loss"] for x in history),
            "hist_global_aux": sum(x.get("global_aux_loss", 0.0) for x in history),
            "hist_global_anom": sum(x.get("global_anom_loss", 0.0) for x in history),
            "hist_perm": sum(x.get("perm_loss", 0.0) for x in history),
            "hist_pair": sum(x.get("pair_loss", 0.0) for x in history),
            "hist_count": float(len(history)),
            "global_step": float(global_step),
        },
        device=aux_device,
    )
    if _dist_enabled():
        dist.barrier()
    if _dist_is_main():
        output_path = _resolve(args.output_checkpoint)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoder_path = _resolve(args.output_encoder_checkpoint) if args.output_encoder_checkpoint else output_path.with_name(output_path.stem + "_encoder.pt")
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "perceiver": client.perceiver.state_dict(),
                "global_hint_head": shared_global_hint_head.state_dict(),
                "global_anomaly_head": None if global_anomaly_head is None else global_anomaly_head.state_dict(),
                "answer_heads": answer_heads.state_dict(),
                "answer_label_vocabs": answer_label_vocabs,
                "axis_hints": llm_config.get("axis_hints"),
                "train_args": vars(args),
                "run_note": str(args.run_note),
                "reference_encoder_num_channels": int(reference_num_channels),
                "seen_encoder_channel_counts": sorted(int(key) for key in encoder_cache.keys()),
                "loss_definition": {
                    "next_token_prediction": "teacher-forced causal LM cross entropy on teacher answer tokens",
                    "answer_auxiliary": "type-aware cross entropy on parsed teacher answer label using separate judgment/choice heads",
                    "global_anomaly": "per-time binary cross entropy on global hint states through a Linear(d_proj,2) head",
                    "permutation_consistency": "MSE between global states before and after random channel permutation",
                    "pair_consistency": "MSE between question-conditioned channel attention maps for anomaly/normal paired questions",
                    "total_loss": (
                        f"ntp_loss + {float(args.answer_loss_weight)} * answer_loss"
                        f" + {float(args.global_anom_loss_weight)} * global_anom_loss"
                        f" + {0.0 if args.disable_perm_loss else float(args.perm_loss_weight)} * perm_loss"
                        f" + {0.0 if args.disable_pair_loss else float(args.pair_loss_weight)} * pair_loss"
                    ),
                },
            },
            output_path,
        )
        encoder_cache[reference_num_channels].save(str(encoder_path))
        hist_denom = max(1.0, float(reduced_final["hist_count"]))
        report = {
            "checkpoint_path": str(output_path),
            "encoder_checkpoint_path": str(encoder_path),
            "reference_encoder_num_channels": int(reference_num_channels),
            "seen_encoder_channel_counts": sorted(int(key) for key in encoder_cache.keys()),
            "encoder_load_infos": {
                str(key): {
                    "loaded_keys": value.get("loaded_keys"),
                    "skipped_shape_mismatch_keys": len(value.get("skipped_shape_mismatch_keys") or []),
                    "skipped_global_hint_head_keys": len(value.get("skipped_global_hint_head_keys") or []),
                }
                for key, value in sorted(encoder_load_infos.items())
            },
            "num_samples": total_num_samples,
            "local_num_samples": len(training_items),
            "epochs": int(args.epochs),
            "global_steps": int(round(float(reduced_final["global_step"]))),
            "world_size": world_size,
            "run_note": str(args.run_note),
            "answer_loss_weight": float(args.answer_loss_weight),
            "global_anom_loss_weight": float(args.global_anom_loss_weight),
            "perm_loss_weight": float(0.0 if args.disable_perm_loss else args.perm_loss_weight),
            "pair_loss_weight": float(0.0 if args.disable_pair_loss else args.pair_loss_weight),
            "answer_label_vocabs": answer_label_vocabs,
            "teacher_sources": teacher_sources,
            "image_manifest_loaded": bool(image_manifest),
            "image_kind": image_kind,
            "include_global_hints": include_global_hints,
            "warm_start": warm_loaded,
            "mean_total_loss": float(reduced_final["hist_total"]) / hist_denom,
            "mean_ntp_loss": float(reduced_final["hist_ntp"]) / hist_denom,
            "mean_answer_loss": float(reduced_final["hist_answer"]) / hist_denom,
            "mean_global_aux_loss": float(reduced_final["hist_global_aux"]) / hist_denom,
            "mean_global_anom_loss": float(reduced_final["hist_global_anom"]) / hist_denom,
            "mean_perm_loss": float(reduced_final["hist_perm"]) / hist_denom,
            "mean_pair_loss": float(reduced_final["hist_pair"]) / hist_denom,
            "elapsed_seconds": time.perf_counter() - started,
            "history_tail": history[-10:],
        }
        report_path = _resolve(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(report, report_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if _dist_enabled():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
