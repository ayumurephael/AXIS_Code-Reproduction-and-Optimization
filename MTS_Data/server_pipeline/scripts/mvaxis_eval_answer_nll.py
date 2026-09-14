from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.llm_client import LocalHFChatClient, LocalHFVLChatClient, create_llm_client
from src.mvaxis.proposal import sample_with_interval
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


_MESSAGE_BLOCK_RE = re.compile(r"(?ms)^([A-Z_]+):\n(.*?)(?=\n\n[A-Z_]+:\n|\Z)")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _parse_prompt_preview(prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for match in _MESSAGE_BLOCK_RE.finditer(str(prompt or "").strip()):
        role = match.group(1).strip().lower()
        content = match.group(2)
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("Failed to parse prompt preview into chat messages.")
    return messages


def _load_encoder_for_hints(config_path: Path, checkpoint_path: str | None) -> AXISMultivariateIntervalProposer:
    config = load_json(config_path)
    checkpoint = checkpoint_path or (config.get("train_interval_proposal") or {}).get("checkpoint_path")
    if not checkpoint:
        raise ValueError("Missing interval proposer checkpoint for hint reconstruction.")
    resolved = Path(relative_to_root(ROOT, str(checkpoint)))
    try:
        return AXISMultivariateIntervalProposer.load(str(resolved), config)
    except Exception:
        model, _ = AXISMultivariateIntervalProposer.load_timercd_checkpoint(
            str(resolved),
            config,
            threshold=float((config.get("interval_proposal") or {}).get("threshold", 0.5)),
        )
        return model


def _filter_embedding_hints(
    hint_bundle: Dict[str, Any],
    *,
    include_global_hints: bool,
    include_channel_hints: bool,
) -> Dict[str, Any]:
    embeddings = hint_bundle.get("embeddings") or {}
    filtered_embeddings: Dict[str, Any] = {}
    if include_global_hints and embeddings.get("global") is not None:
        filtered_embeddings["global"] = embeddings["global"]
    if include_channel_hints and embeddings.get("channel") is not None:
        filtered_embeddings["channel"] = embeddings["channel"]
    if not filtered_embeddings:
        raise ValueError("At least one of global or channel embedding hints must remain enabled.")
    return {"embeddings": filtered_embeddings, "trace": hint_bundle.get("trace")}


def _prompt_row(sample: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    prompt_interval = record.get("prompt_interval") or []
    if isinstance(prompt_interval, Sequence) and len(prompt_interval) == 2:
        start = int(prompt_interval[0])
        end = int(prompt_interval[1])
        target = sample.get("target_interval") or {}
        if int(target.get("start", start)) != start or int(target.get("end", end)) != end:
            return sample_with_interval(sample, start, end)
    return sample


def _image_paths_for_record(workflow: Dict[str, Any], record: Dict[str, Any]) -> List[str]:
    if not bool(workflow.get("image_consumed_by_vl_qwen")):
        return []
    if bool(workflow.get("include_anomaly_score_image_note")) and record.get("score_image_path"):
        return [str(record["score_image_path"])]
    if bool(workflow.get("include_raw_image_note")) and record.get("raw_image_path"):
        return [str(record["raw_image_path"])]
    return []


def _answer_text(record: Dict[str, Any], sample: Dict[str, Any], teacher_lookup: Dict[str, str], answer_source: str) -> str:
    if answer_source == "student":
        return str(record.get("raw_response") or "")
    if answer_source == "teacher":
        sample_id = str(sample.get("sample_id") or "")
        return str(teacher_lookup.get(sample_id) or "")
    raise ValueError(f"Unsupported answer source: {answer_source}")


def _teacher_lookup(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for row in rows:
        teacher = row.get("teacher_answer_llm") or {}
        answer = teacher.get("answer") if isinstance(teacher, dict) else row.get("answer")
        sample_id = row.get("sample_id")
        if sample_id and answer:
            lookup[str(sample_id)] = str(answer)
    return lookup


def _append_eos(text: str, eos_token: Optional[str]) -> str:
    answer = str(text or "").strip()
    if eos_token:
        answer += eos_token
    return answer


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def _inject_text_axis_embeddings(
    *,
    client: LocalHFChatClient,
    input_ids: torch.Tensor,
    input_embeddings: torch.Tensor,
    axis_embeddings: Dict[str, Any],
) -> torch.Tensor:
    hint_tensors = client._prepare_axis_hint_tensors(axis_embeddings)
    embedding_device = input_embeddings.device
    if next(client.perceiver.parameters()).device != embedding_device or next(client.perceiver.parameters()).dtype != torch.float32:
        client.perceiver.to(device=embedding_device, dtype=torch.float32)
    word_embeddings = client.model.get_input_embeddings().weight.detach().to(device=embedding_device, dtype=torch.float32)
    source_embeddings = client.perceiver.get_source_embeddings(word_embeddings)

    global_tensor = hint_tensors.get("global")
    channel_tensor = hint_tensors.get("channel")
    projected_global = None
    projected_channel = None
    if global_tensor is not None:
        projected_global = client.perceiver.process_local_embeddings(global_tensor.to(device=embedding_device, dtype=torch.float32), source_embeddings)
    if channel_tensor is not None:
        projected_channel = client.perceiver.process_local_embeddings(channel_tensor.to(device=embedding_device, dtype=torch.float32), source_embeddings)

    global_positions = (input_ids[0] == client.global_hint_token_id).nonzero(as_tuple=True)[0]
    channel_positions = (input_ids[0] == client.channel_hint_token_id).nonzero(as_tuple=True)[0]
    neutral_ids = client.tokenizer("\n", add_special_tokens=False)["input_ids"]
    neutral_id = int(neutral_ids[0]) if neutral_ids else int(client.tokenizer.eos_token_id)
    neutral_embedding = client.model.get_input_embeddings().weight[neutral_id].detach().to(
        device=embedding_device,
        dtype=input_embeddings.dtype,
    )

    if len(global_positions) > 0 and projected_global is not None:
        scale = float(client.axis_hints_config.get("global_injection_scale", client.axis_hints_config.get("injection_scale", 1.0)))
        scale = max(0.0, min(1.0, scale))
        projected = projected_global[: len(global_positions)].to(input_embeddings.dtype)
        input_embeddings[0, global_positions] = (1.0 - scale) * neutral_embedding + scale * projected
    if len(channel_positions) > 0 and projected_channel is not None:
        scale = float(client.axis_hints_config.get("channel_injection_scale", client.axis_hints_config.get("injection_scale", 1.0)))
        scale = max(0.0, min(1.0, scale))
        projected = projected_channel[: len(channel_positions)].to(input_embeddings.dtype)
        input_embeddings[0, channel_positions] = (1.0 - scale) * neutral_embedding + scale * projected
    return input_embeddings


def _inject_vl_axis_embeddings(
    *,
    client: LocalHFVLChatClient,
    input_ids: torch.Tensor,
    input_embeddings: torch.Tensor,
    axis_embeddings: Dict[str, Any],
) -> torch.Tensor:
    hint_tensors = client._prepare_axis_hint_tensors(axis_embeddings)
    embedding_device = input_embeddings.device
    if next(client.perceiver.parameters()).device != embedding_device or next(client.perceiver.parameters()).dtype != torch.float32:
        client.perceiver.to(device=embedding_device, dtype=torch.float32)
    word_embeddings = client.model.get_input_embeddings().weight.detach().to(device=embedding_device, dtype=torch.float32)
    source_embeddings = client.perceiver.get_source_embeddings(word_embeddings)

    global_tensor = hint_tensors.get("global")
    channel_tensor = hint_tensors.get("channel")
    projected_global = None
    projected_channel = None
    if global_tensor is not None:
        projected_global = client.perceiver.process_local_embeddings(global_tensor.to(device=embedding_device, dtype=torch.float32), source_embeddings)
    if channel_tensor is not None:
        projected_channel = client.perceiver.process_local_embeddings(channel_tensor.to(device=embedding_device, dtype=torch.float32), source_embeddings)

    global_positions = (input_ids[0] == client.global_hint_token_id).nonzero(as_tuple=True)[0]
    channel_positions = (input_ids[0] == client.channel_hint_token_id).nonzero(as_tuple=True)[0]
    neutral_ids = client.processor.tokenizer("\n", add_special_tokens=False)["input_ids"]
    neutral_id = int(neutral_ids[0]) if neutral_ids else int(client.processor.tokenizer.eos_token_id)
    neutral_embedding = client.model.get_input_embeddings().weight[neutral_id].detach().to(
        device=embedding_device,
        dtype=input_embeddings.dtype,
    )

    if len(global_positions) > 0 and projected_global is not None:
        scale = float(client.axis_hints_config.get("global_injection_scale", client.axis_hints_config.get("injection_scale", 1.0)))
        scale = max(0.0, min(1.0, scale))
        projected = projected_global[: len(global_positions)].to(input_embeddings.dtype)
        input_embeddings[0, global_positions] = (1.0 - scale) * neutral_embedding + scale * projected
    if len(channel_positions) > 0 and projected_channel is not None:
        scale = float(client.axis_hints_config.get("channel_injection_scale", client.axis_hints_config.get("injection_scale", 1.0)))
        scale = max(0.0, min(1.0, scale))
        projected = projected_channel[: len(channel_positions)].to(input_embeddings.dtype)
        input_embeddings[0, channel_positions] = (1.0 - scale) * neutral_embedding + scale * projected
    return input_embeddings


def _score_text_answer(
    *,
    client: LocalHFChatClient,
    messages: List[Dict[str, str]],
    axis_embeddings: Optional[Dict[str, Any]],
    answer_text: str,
) -> Dict[str, Any]:
    if axis_embeddings is not None:
        hint_tensors = client._prepare_axis_hint_tensors(axis_embeddings)
        global_tensor = hint_tensors.get("global")
        channel_tensor = hint_tensors.get("channel")
        global_tokens = client.GLOBAL_HINT_TOKEN * int(0 if global_tensor is None else global_tensor.shape[0])
        channel_tokens = client.CHANNEL_HINT_TOKEN * int(0 if channel_tensor is None else channel_tensor.shape[0])
        axis_messages = [dict(m) for m in messages]
        content = axis_messages[-1].get("content", "")
        if client.GLOBAL_HINT_TOKEN_PLACEHOLDER in content or client.CHANNEL_HINT_TOKEN_PLACEHOLDER in content:
            content = content.replace(client.GLOBAL_HINT_TOKEN_PLACEHOLDER, global_tokens, 1)
            content = content.replace(client.CHANNEL_HINT_TOKEN_PLACEHOLDER, channel_tokens, 1)
            axis_messages[-1]["content"] = content
        elif client.HINT_TOKEN_PLACEHOLDER in content:
            axis_messages[-1]["content"] = content.replace(client.HINT_TOKEN_PLACEHOLDER, global_tokens + channel_tokens, 1)
        else:
            axis_messages[-1]["content"] = content + "\n\n" + global_tokens + channel_tokens + "\n"
    else:
        axis_messages = [dict(m) for m in messages]
        content = axis_messages[-1].get("content", "")
        content = content.replace(client.GLOBAL_HINT_TOKEN_PLACEHOLDER, "", 1)
        content = content.replace(client.CHANNEL_HINT_TOKEN_PLACEHOLDER, "", 1)
        content = content.replace(client.HINT_TOKEN_PLACEHOLDER, "", 1)
        axis_messages[-1]["content"] = content

    prompt = client._render_messages(axis_messages)
    target_text = _append_eos(answer_text, client.tokenizer.eos_token)
    prompt_ids = client.tokenizer(prompt, return_tensors="pt").to(client.device)
    target_ids = client.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).to(client.device)
    input_ids = torch.cat([prompt_ids["input_ids"], target_ids["input_ids"]], dim=1)
    attention_mask = torch.cat([prompt_ids["attention_mask"], target_ids["attention_mask"]], dim=1)
    labels = input_ids.clone()
    prompt_len = int(prompt_ids["input_ids"].shape[-1])
    labels[:, :prompt_len] = -100
    target_tokens = int((labels != -100).sum().item())

    if axis_embeddings is None:
        with torch.no_grad():
            outputs = client.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
    else:
        input_embeddings = client.model.get_input_embeddings()(input_ids).detach()
        input_embeddings = _inject_text_axis_embeddings(
            client=client,
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            axis_embeddings=axis_embeddings,
        )
        with torch.no_grad():
            outputs = client.model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
    mean_nll = float(outputs.loss.detach().cpu().item())
    total_nll = mean_nll * target_tokens
    del outputs
    if axis_embeddings is not None:
        del input_embeddings
    del input_ids, attention_mask, labels, prompt_ids, target_ids
    _clear_cuda_cache()
    return {
        "mean_nll": mean_nll,
        "total_nll": total_nll,
        "target_tokens": target_tokens,
    }


def _score_vl_answer(
    *,
    client: LocalHFVLChatClient,
    messages: List[Dict[str, str]],
    image_paths: List[str],
    axis_embeddings: Optional[Dict[str, Any]],
    answer_text: str,
) -> Dict[str, Any]:
    if axis_embeddings is not None:
        hint_tensors = client._prepare_axis_hint_tensors(axis_embeddings)
        axis_messages = client._inject_axis_hint_tokens(messages, hint_tensors)
    else:
        axis_messages = [dict(m) for m in messages]
        content = axis_messages[-1].get("content", "")
        content = content.replace(client.GLOBAL_HINT_TOKEN_PLACEHOLDER, "", 1)
        content = content.replace(client.CHANNEL_HINT_TOKEN_PLACEHOLDER, "", 1)
        axis_messages[-1]["content"] = content
    vl_messages = client._to_vl_messages(axis_messages, image_paths)
    prompt_text = client.processor.apply_chat_template(vl_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = client.process_vision_info(vl_messages)
    prompt_inputs = client.processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = client.model.device if not isinstance(getattr(client.model, "hf_device_map", None), dict) else client.device
    prompt_inputs = prompt_inputs.to(device)
    target_text = _append_eos(answer_text, client.processor.tokenizer.eos_token)
    target_ids = client.processor.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).to(device)

    prompt_input_ids = prompt_inputs["input_ids"]
    prompt_attention_mask = prompt_inputs.get("attention_mask", torch.ones_like(prompt_input_ids))
    target_input_ids = target_ids["input_ids"]
    target_attention_mask = target_ids["attention_mask"]
    target_tokens = int(target_input_ids.shape[-1])

    prompt_embeddings = client.model.get_input_embeddings()(prompt_input_ids).detach()
    if axis_embeddings is not None:
        prompt_embeddings = _inject_vl_axis_embeddings(
            client=client,
            input_ids=prompt_input_ids,
            input_embeddings=prompt_embeddings,
            axis_embeddings=axis_embeddings,
        )
    prompt_forward_kwargs: Dict[str, Any] = {
        "input_ids": prompt_input_ids,
        "inputs_embeds": prompt_embeddings,
        "attention_mask": prompt_attention_mask,
        "use_cache": True,
        "return_dict": True,
    }
    for optional_key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
        if optional_key in prompt_inputs:
            prompt_forward_kwargs[optional_key] = prompt_inputs[optional_key]
    with torch.no_grad():
        prompt_outputs = client.model(**prompt_forward_kwargs)

    next_logits = prompt_outputs.logits[:, -1, :].float()
    past_key_values = prompt_outputs.past_key_values
    attention_mask = prompt_attention_mask
    total_nll = 0.0
    chunk_size = 32
    processed = 0
    while processed < target_tokens:
        end = min(processed + chunk_size, target_tokens)
        chunk = target_input_ids[:, processed:end]
        chunk_len = int(chunk.shape[-1])
        total_nll += float(F.cross_entropy(next_logits, chunk[:, 0], reduction="sum").detach().cpu().item())

        step_attention = torch.ones(
            (attention_mask.shape[0], chunk_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_attention_mask = torch.cat([attention_mask, step_attention], dim=1)
        cache_position = torch.arange(
            attention_mask.shape[-1],
            attention_mask.shape[-1] + chunk_len,
            device=chunk.device,
        )
        with torch.no_grad():
            chunk_outputs = client.model(
                input_ids=chunk,
                attention_mask=extended_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        if chunk_len > 1:
            logits_for_rest = chunk_outputs.logits[:, :-1, :].reshape(-1, chunk_outputs.logits.shape[-1]).float()
            rest_targets = chunk[:, 1:].reshape(-1)
            total_nll += float(F.cross_entropy(logits_for_rest, rest_targets, reduction="sum").detach().cpu().item())
            del logits_for_rest, rest_targets

        next_logits = chunk_outputs.logits[:, -1, :].float()
        past_key_values = chunk_outputs.past_key_values
        attention_mask = extended_attention_mask
        processed = end
        del chunk_outputs, chunk, step_attention, extended_attention_mask, cache_position
        _clear_cuda_cache()

    mean_nll = total_nll / max(1, target_tokens)
    del prompt_outputs, next_logits, prompt_inputs, prompt_input_ids, prompt_attention_mask
    del prompt_embeddings, target_ids, target_input_ids, target_attention_mask, attention_mask, past_key_values
    _clear_cuda_cache()
    return {
        "mean_nll": mean_nll,
        "total_nll": total_nll,
        "target_tokens": target_tokens,
    }


def _score_answer(
    *,
    client: Any,
    messages: List[Dict[str, str]],
    image_paths: List[str],
    axis_embeddings: Optional[Dict[str, Any]],
    answer_text: str,
) -> Dict[str, Any]:
    if isinstance(client, LocalHFVLChatClient):
        return _score_vl_answer(
            client=client,
            messages=messages,
            image_paths=image_paths,
            axis_embeddings=axis_embeddings,
            answer_text=answer_text,
        )
    if isinstance(client, LocalHFChatClient):
        return _score_text_answer(
            client=client,
            messages=messages,
            axis_embeddings=axis_embeddings,
            answer_text=answer_text,
        )
    raise TypeError(f"Unsupported client type: {type(client).__name__}")


def _safe_ppl(mean_nll: float) -> float:
    return float(math.exp(min(20.0, max(-20.0, float(mean_nll)))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--llm-config-base", required=True)
    parser.add_argument("--answer-source", choices=["student", "teacher"], default="student")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-step", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_root = _resolve(args.run_root)
    llm_config_base = load_json(_resolve(args.llm_config_base))
    output_path = _resolve(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_lookup = _teacher_lookup(read_jsonl(_resolve("outputs/quetiongeneration_1_teacher_answer/teacher_gpt55.jsonl")))
    client_cache: Dict[str, Any] = {}
    encoder_cache: Dict[str, AXISMultivariateIntervalProposer] = {}
    data_cache: Dict[str, List[Dict[str, Any]]] = {}
    summaries: List[Dict[str, Any]] = []
    for group in args.groups:
        group_dir = run_root / group
        report = load_json(group_dir / "qwen_raw_report.json")
        workflow = dict(report.get("student_workflow") or {})
        data_path = _resolve(report.get("data_path"))
        data_key = str(data_path)
        if data_key not in data_cache:
            data_cache[data_key] = read_jsonl(data_path)
        data_rows = data_cache[data_key]
        if args.limit is not None:
            data_rows = data_rows[: int(args.limit)]
        run_config = _resolve(workflow.get("config") or "configs/torch_fixed60_qa600_timercd.json")
        interval_checkpoint = workflow.get("interval_proposer_checkpoint")
        records = read_jsonl(group_dir / "qwen_raw_answers.jsonl")
        if args.limit is not None:
            records = records[: int(args.limit)]

        llm_config = dict(llm_config_base)
        llm_config["provider"] = str((report.get("llm_config") or {}).get("provider") or llm_config.get("provider"))
        llm_config["model"] = str((report.get("llm_config") or {}).get("model") or llm_config.get("model"))
        llm_config["max_tokens"] = int((report.get("llm_config") or {}).get("max_tokens") or llm_config.get("max_tokens", 1024))
        axis_hints = dict(llm_config.get("axis_hints") or {})
        first_meta = (records[0].get("llm_raw_metadata") or {}) if records else {}
        if first_meta.get("perceiver_checkpoint"):
            axis_hints["checkpoint_path"] = str(first_meta["perceiver_checkpoint"])
        axis_hints["enabled"] = bool((report.get("llm_config") or {}).get("axis_hints_enabled", axis_hints.get("enabled", False)))
        llm_config["axis_hints"] = axis_hints

        client_key = json.dumps(llm_config, sort_keys=True)
        if client_key not in client_cache:
            client_cache[client_key] = create_llm_client(llm_config)
        client = client_cache[client_key]
        encoder = None
        if axis_hints.get("enabled"):
            encoder_key = json.dumps({"config": str(run_config), "checkpoint": str(interval_checkpoint)}, sort_keys=True)
            if encoder_key not in encoder_cache:
                encoder_cache[encoder_key] = _load_encoder_for_hints(run_config, interval_checkpoint)
            encoder = encoder_cache[encoder_key]

        total_nll = 0.0
        total_tokens = 0
        per_item: List[Dict[str, Any]] = []
        for pos, record in enumerate(records):
            idx = int(record["index"])
            sample = data_rows[idx]
            prompt_messages = _parse_prompt_preview(str(record.get("prompt") or ""))
            prompt_row = _prompt_row(sample, record)
            axis_embeddings = None
            if bool(record.get("axis_embedding_hints_used")) and encoder is not None:
                prompt_interval = prompt_row.get("target_interval") or {}
                start = int(prompt_interval.get("start", 0))
                end = int(prompt_interval.get("end", start + 1))
                hint_bundle = encoder.make_embedding_hints(
                    prompt_row,
                    start,
                    end,
                    top_k_channels=axis_hints.get("top_k_channels"),
                    max_channel_tokens=axis_hints.get("max_channel_tokens", axis_hints.get("max_local_tokens")),
                    max_global_tokens=axis_hints.get("max_global_tokens", axis_hints.get("num_global_tokens", axis_hints.get("num_fixed_tokens"))),
                )
                hint_bundle = _filter_embedding_hints(
                    hint_bundle,
                    include_global_hints=not bool(workflow.get("hide_global_hints")),
                    include_channel_hints=not bool(workflow.get("hide_channel_hints")),
                )
                axis_embeddings = hint_bundle["embeddings"]
            answer_text = _answer_text(record, sample, teacher_lookup, args.answer_source)
            if not answer_text.strip():
                continue
            image_paths = _image_paths_for_record(workflow, record)
            score = _score_answer(
                client=client,
                messages=prompt_messages,
                image_paths=image_paths,
                axis_embeddings=axis_embeddings,
                answer_text=answer_text,
            )
            total_nll += float(score["total_nll"])
            total_tokens += int(score["target_tokens"])
            per_item.append(
                {
                    "index": idx,
                    "target_tokens": int(score["target_tokens"]),
                    "mean_nll": float(score["mean_nll"]),
                    "total_nll": float(score["total_nll"]),
                }
            )
            if int(args.progress_step) > 0 and ((pos + 1) % int(args.progress_step) == 0 or pos + 1 == len(records)):
                print(f"[nll:{group}] {pos + 1}/{len(records)}", flush=True)

        mean_nll = total_nll / max(1, total_tokens)
        summary = {
            "group": group,
            "answer_source": args.answer_source,
            "num_samples": len(per_item),
            "total_target_tokens": total_tokens,
            "total_nll": total_nll,
            "mean_nll": mean_nll,
            "perplexity": _safe_ppl(mean_nll),
            "provider": llm_config.get("provider"),
            "model": llm_config.get("model"),
        }
        summaries.append(summary)
        save_json(
            {
                "summary": summary,
                "per_item": per_item,
            },
            output_path.parent / f"{output_path.stem}.{group}.json",
        )

    save_json({"groups": summaries}, output_path)
    print(json.dumps({"groups": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
