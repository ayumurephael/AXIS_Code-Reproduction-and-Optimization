from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.answer_schema import normalize_answer_label, target_answer_label
from src.mvaxis.anomaly_score_context import aggregate_anomaly_scores, attach_anomaly_score_context, score_stats
from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.data_schema import ModelInput, convert_to_model_input
from src.mvaxis.html_reports import write_answer_generation_html
from src.mvaxis.llm_client import create_llm_client
from src.mvaxis.proposal import sample_with_interval
from src.mvaxis.question_provider import apply_question_spec, fixed_question_specs
from src.mvaxis.randomhint import build_random_embedding_hints
from src.mvaxis.student_answer_provider import (
    build_student_answer_messages,
    parse_student_answer,
    prompt_from_messages,
)
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json, write_jsonl


_QUESTION_BY_ID = {spec.question_id: spec for spec in fixed_question_specs()}


def _normalize_student_eval_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    if "evidence_card" not in normalized or normalized.get("evidence_card") is None:
        normalized["evidence_card"] = {}
    return normalized


def _ensure_teacher_reasoning(row: Dict[str, Any]) -> Dict[str, Any]:
    target = row.get("target_output") or {}
    if target.get("reasoning_process"):
        return row
    question_id = row.get("question_id") or row.get("question_type")
    spec = _QUESTION_BY_ID.get(str(question_id))
    if spec is None:
        return row
    rebuilt = dict(row)
    apply_question_spec(rebuilt, spec)
    return rebuilt


def _load_rows_and_inputs(path: Path, limit: int | None) -> Tuple[List[Dict[str, Any]], List[ModelInput]]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    rows = [_normalize_student_eval_row(_ensure_teacher_reasoning(row)) for row in rows]
    return rows, [convert_to_model_input(row) for row in rows]


def _row_num_channels(row: Dict[str, Any]) -> int | None:
    series = row.get("series") or {}
    shape = series.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        try:
            return int(shape[1])
        except Exception:
            pass
    channels = row.get("channels")
    if isinstance(channels, list) and channels:
        return int(len(channels))
    values = series.get("values")
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, list):
            return int(len(first))
        return 1
    return None


def _load_llm_config(
    path: Path,
    model_path: str | None,
    max_tokens: int | None,
    *,
    use_axis_embedding_hints: bool,
) -> Dict[str, Any]:
    config = load_json(path)
    if model_path:
        config["model"] = model_path
    if max_tokens is not None:
        config["max_tokens"] = int(max_tokens)
    axis_hints = dict(config.get("axis_hints") or {})
    axis_hints["enabled"] = bool(use_axis_embedding_hints)
    checkpoint_path = axis_hints.get("checkpoint_path")
    if checkpoint_path:
        resolved = Path(str(checkpoint_path))
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        if not resolved.exists():
            axis_hints["checkpoint_path"] = None
    config["axis_hints"] = axis_hints
    return config


def _load_encoder_for_hints(
    config_path: Path,
    checkpoint_path: str | None,
    *,
    num_channels_override: int | None = None,
) -> AXISMultivariateIntervalProposer:
    config = load_json(config_path)
    if num_channels_override is not None:
        config = dict(config)
        data_cfg = dict(config.get("data") or {})
        data_cfg["num_channels"] = int(num_channels_override)
        config["data"] = data_cfg
    checkpoint = checkpoint_path or (config.get("train_interval_proposal") or {}).get("checkpoint_path")
    if not checkpoint:
        raise ValueError("--interval-proposer-checkpoint is required when using AXIS embedding hints")
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


def _load_score_image_manifest(path: str | None) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    payload = load_json(resolved)
    records = payload.get("records") if isinstance(payload, dict) else payload
    mapping: Dict[str, Dict[str, Any]] = {}
    for record in records or []:
        if "index" in record:
            mapping[f"index:{int(record['index'])}"] = record
        if record.get("sample_id"):
            mapping[f"sample_id:{record['sample_id']}"] = record
    return mapping


def _resolve_local_artifact_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    raw = str(path_value)
    candidates = [Path(raw)]
    remap_prefixes = [
        ("D:\\multiaxis\\outputs\\ts_data\\", "E:\\multiaxis\\data\\ts_data\\"),
        ("X:\\multiaxis\\outputs\\ts_data\\", "E:\\multiaxis\\data\\ts_data\\"),
        ("X:\\codex\\multiaxis\\outputs\\ts_data\\", "E:\\multiaxis\\data\\ts_data\\"),
    ]
    normalized = raw.replace("/", "\\")
    for src, dst in remap_prefixes:
        if normalized.startswith(src):
            candidates.append(Path(normalized.replace(src, dst, 1)))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return raw


def _score_image_for_row(mapping: Dict[str, Dict[str, Any]], idx: int, row: Dict[str, Any]) -> str | None:
    if not mapping:
        return None
    record = mapping.get(f"index:{idx}") or mapping.get(f"sample_id:{row.get('sample_id')}")
    if not record:
        return None
    return _resolve_local_artifact_path(record.get("path") or record.get("image_path"))


def _raw_image_for_row(mapping: Dict[str, Dict[str, Any]], idx: int, row: Dict[str, Any]) -> str | None:
    if mapping:
        record = mapping.get(f"index:{idx}") or mapping.get(f"sample_id:{row.get('sample_id')}")
        if record:
            return _resolve_local_artifact_path(
                record.get("path") or record.get("raw_image_path") or record.get("image_path")
            )
    artifacts = row.get("question_generation_artifacts") or {}
    return _resolve_local_artifact_path(artifacts.get("raw_image_path") or artifacts.get("image_path"))


def _canonical_label(label: Any) -> str | None:
    normalized = normalize_answer_label(label)
    if normalized is None:
        return None
    text = str(normalized).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "true":
        return "true"
    if lowered == "false":
        return "false"
    return lowered


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
    filtered_bundle = {"embeddings": filtered_embeddings, "trace": hint_bundle.get("trace")}
    random_metadata = hint_bundle.get("random_metadata")
    if isinstance(random_metadata, dict):
        filtered_random_metadata: Dict[str, Any] = {}
        if "global" in filtered_embeddings and random_metadata.get("global") is not None:
            filtered_random_metadata["global"] = random_metadata["global"]
        if "channel" in filtered_embeddings and random_metadata.get("channel") is not None:
            filtered_random_metadata["channel"] = random_metadata["channel"]
        if filtered_random_metadata:
            filtered_bundle["random_metadata"] = filtered_random_metadata
    return filtered_bundle


def _student_answer_metrics(rows: List[Dict[str, Any]], parsed_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparable = 0
    correct = 0
    parsed_labels = 0
    open_answers = 0
    for row, parsed in zip(rows, parsed_outputs):
        target_label = _canonical_label(target_answer_label(row.get("target_output") or {}))
        pred_label = _canonical_label((parsed or {}).get("answer_label"))
        if (parsed or {}).get("question_answer_type") == "open":
            open_answers += 1
        if pred_label is not None:
            parsed_labels += 1
        if target_label is None:
            continue
        comparable += 1
        correct += int(pred_label == target_label)
    return {
        "num_samples": len(parsed_outputs),
        "num_label_comparable": comparable,
        "num_open_answers": open_answers,
        "answer_label_parse_rate": parsed_labels / max(1, len(parsed_outputs)),
        "answer_label_accuracy": correct / max(1, comparable),
    }


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _case_text(record: Dict[str, Any]) -> str:
    parsed = record.get("parsed_student_answer") or {}
    target = record.get("target_output") or {}
    target_fact = target.get("fact_check") or {}
    lines = [
        f"CASE {record['index']}",
        f"sample_id: {record['sample_id']}",
        f"question_type: {record['question_type']}",
        f"question: {record['question']}",
        f"target_fact: {json.dumps(target_fact, ensure_ascii=False)}",
        f"teacher_reasoning_process: {json.dumps(target.get('reasoning_process', []), ensure_ascii=False)}",
        f"teacher_answer: {target.get('final_answer')}",
        f"student_answer_label: {parsed.get('answer_label')}",
        f"student_answer_type: {parsed.get('question_answer_type')}",
        f"student_final_answer: {parsed.get('final_answer')}",
        f"raw_response: {record.get('raw_response')}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--llm-config", default="configs/local_qwen25_7b_cuda_remote_fast_text_cache.json")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/runs/raw_qwen_answers")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--prompt-examples", type=int, default=2)
    parser.add_argument("--raw-examples", type=int, default=5)
    parser.add_argument("--max-window-rows", type=int, default=0, help="0 means include the full target window.")
    parser.add_argument("--digits", type=int, default=3)
    hint_group = parser.add_mutually_exclusive_group()
    hint_group.add_argument("--use-axis-embedding-hints", action="store_true")
    hint_group.add_argument("--use-random-hints", action="store_true")
    parser.add_argument("--random-hint-seed", type=int, default=72)
    parser.add_argument("--config", default=None)
    parser.add_argument("--interval-proposer-checkpoint", default=None)
    parser.add_argument("--interval-source", choices=["target", "proposal"], default="target")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--hide-global-hints", action="store_true")
    parser.add_argument("--hide-channel-hints", action="store_true")
    parser.add_argument("--hide-evidence-card", action="store_true")
    parser.add_argument("--include-anomaly-score-text", action="store_true")
    parser.add_argument("--include-anomaly-score-image-note", action="store_true")
    parser.add_argument("--include-raw-image-note", action="store_true")
    parser.add_argument("--anomaly-score-aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--score-image-manifest", default=None)
    parser.add_argument("--raw-image-manifest", default=None)
    parser.add_argument(
        "--progress-percent-step",
        type=int,
        default=5,
        help="Print Qwen answer-generation progress every N percent. Set <=0 to disable.",
    )
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    llm_config_path = Path(args.llm_config)
    if not llm_config_path.is_absolute():
        llm_config_path = ROOT / llm_config_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_jsonl_path = output_dir / "qwen_raw_answers.jsonl"

    rows, inputs = _load_rows_and_inputs(data_path, args.limit)
    hint_mode = "random" if args.use_random_hints else ("axis" if args.use_axis_embedding_hints else "none")
    axis_mode_enabled = hint_mode in {"axis", "random"}
    llm_config = _load_llm_config(
        llm_config_path,
        args.model_path,
        args.max_tokens,
        use_axis_embedding_hints=axis_mode_enabled,
    )
    encoder = None
    encoder_cache: Dict[int | None, AXISMultivariateIntervalProposer] = {}
    run_config = None
    config_path = None
    needs_encoder = bool(
        axis_mode_enabled
        or args.include_anomaly_score_text
    )
    if needs_encoder:
        if not args.config:
            raise ValueError("--config is required with AXIS embedding hints or anomaly-score text")
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        run_config = load_json(config_path)
    client = None if args.preview_only else create_llm_client(llm_config)
    client_supports_images = bool(client is not None and hasattr(client, "complete_with_images"))
    score_image_manifest = _load_score_image_manifest(args.score_image_manifest)
    raw_image_manifest = _load_score_image_manifest(args.raw_image_manifest)

    records: List[Dict[str, Any]] = []
    parsed_outputs: List[Dict[str, Any]] = []
    completed_indices: set[int] = set()
    if answers_jsonl_path.exists() and not args.preview_only:
        existing_records = read_jsonl(answers_jsonl_path)
        for record in existing_records:
            idx_value = record.get("index")
            if isinstance(idx_value, int):
                completed_indices.add(idx_value)
        records = existing_records
        parsed_outputs = [
            (record.get("parsed_student_answer") or record.get("parsed_response") or {})
            for record in existing_records
        ]
    prompt_examples: List[Dict[str, Any]] = []
    max_window_rows = None if int(args.max_window_rows) <= 0 else int(args.max_window_rows)
    total_items = min(len(rows), len(inputs))
    progress_step = max(1, min(100, int(args.progress_percent_step)))
    progress_enabled = bool(not args.preview_only and int(args.progress_percent_step) > 0 and total_items > 0)
    completed_count = len(completed_indices)
    next_progress_pct = ((completed_count // progress_step) + 1) * progress_step if progress_enabled else progress_step
    if progress_enabled:
        pct = int(completed_count * 100 / total_items)
        print(f"[qwen-progress] {pct}% ({completed_count}/{total_items}) started", flush=True)
    for idx, (row, model_input) in enumerate(zip(rows, inputs)):
        if idx in completed_indices:
            continue
        axis_mode = axis_mode_enabled
        random_hint_mode = hint_mode == "random"
        hint_bundle = None
        prompt_row = row
        proposal = None
        prompt_interval = list(model_input.interval)
        anomaly_score_stats = None
        score_image_path = _score_image_for_row(score_image_manifest, idx, row)
        raw_image_path = _raw_image_for_row(raw_image_manifest, idx, row)
        row_num_channels = _row_num_channels(row)
        if needs_encoder:
            if config_path is None:
                raise RuntimeError("config_path was not loaded for encoder-backed evaluation")
            encoder = encoder_cache.get(row_num_channels)
            if encoder is None:
                encoder = _load_encoder_for_hints(
                    config_path,
                    args.interval_proposer_checkpoint,
                    num_channels_override=row_num_channels,
                )
                encoder_cache[row_num_channels] = encoder
        if axis_mode:
            if encoder is None:
                raise RuntimeError("encoder was not loaded for AXIS embedding hints")
            if args.interval_source == "proposal":
                if run_config is None:
                    raise RuntimeError("run_config was not loaded for proposal interval mode")
                proposal_cfg = run_config.get("interval_proposal") or {}
                window_size = int(args.window_size or proposal_cfg.get("window_size", 32))
                stride = int(args.stride or proposal_cfg.get("stride", 8))
                top_k_channels = int(run_config.get("model", {}).get("top_k_channels", 3))
                proposal = encoder.propose_interval(row, window_size, stride, top_k_channels)
                if proposal.get("start") is not None and proposal.get("end") is not None:
                    prompt_interval = [int(proposal["start"]), int(proposal["end"])]
                    prompt_row = sample_with_interval(row, prompt_interval[0], prompt_interval[1])
            start, end = prompt_interval
            axis_cfg = llm_config.get("axis_hints") or {}
            hint_bundle = encoder.make_embedding_hints(
                prompt_row,
                start,
                end,
                top_k_channels=axis_cfg.get("top_k_channels"),
                max_channel_tokens=axis_cfg.get("max_channel_tokens", axis_cfg.get("max_local_tokens")),
                max_global_tokens=axis_cfg.get(
                    "max_global_tokens",
                    axis_cfg.get("num_global_tokens", axis_cfg.get("num_fixed_tokens")),
                ),
            )
            if random_hint_mode:
                hint_bundle = build_random_embedding_hints(
                    hint_bundle,
                    base_seed=int(args.random_hint_seed),
                    context=f"{data_path.name}:{idx}:{start}:{end}",
                )
            hint_bundle = _filter_embedding_hints(
                hint_bundle,
                include_global_hints=not bool(args.hide_global_hints),
                include_channel_hints=not bool(args.hide_channel_hints),
            )
        if args.include_anomaly_score_text:
            if encoder is None:
                raise RuntimeError("encoder is required for anomaly score context")
            channel_probs = encoder.anomaly_probabilities(prompt_row)
            time_scores = aggregate_anomaly_scores(
                channel_probs,
                prompt_row.get("channels") or [],
                aggregation=str(args.anomaly_score_aggregation),
            )
            anomaly_score_stats = score_stats(time_scores, int(prompt_interval[0]), int(prompt_interval[1]))
            prompt_row = attach_anomaly_score_context(
                prompt_row,
                time_scores,
                aggregation=str(args.anomaly_score_aggregation),
                image_path=score_image_path,
                copy_sample=True,
            )
        if args.include_raw_image_note:
            prompt_row = dict(prompt_row)
            visual_context = dict(prompt_row.get("visual_context") or {})
            visual_context["raw_image_path"] = raw_image_path
            prompt_row["visual_context"] = visual_context
        messages = build_student_answer_messages(
            prompt_row,
            embedding_hint_trace=hint_bundle["trace"] if hint_bundle else None,
            use_axis_hints=axis_mode,
            include_global_hints=not bool(args.hide_global_hints),
            include_channel_hints=not bool(args.hide_channel_hints),
            include_evidence_card=False,
            include_anomaly_score_text=bool(args.include_anomaly_score_text),
            include_anomaly_score_image_note=bool(args.include_anomaly_score_image_note),
            include_raw_image_note=bool(args.include_raw_image_note),
            max_window_rows=max_window_rows,
            digits=int(args.digits),
        )
        prompt = prompt_from_messages(messages)
        if len(prompt_examples) < max(0, int(args.prompt_examples)):
            prompt_examples.append(
                {
                    "index": idx,
                    "question": model_input.question,
                    "target_output": model_input.target_output,
                    "prompt": prompt,
                    "axis_embedding_hints_used": axis_mode,
                    "hint_mode": hint_mode,
                    "random_embedding_hints_used": random_hint_mode,
                    "embedding_hint_trace": hint_bundle["trace"] if hint_bundle else None,
                    "random_hint_metadata": hint_bundle.get("random_metadata") if hint_bundle else None,
                    "global_embedding_shape": list(hint_bundle["embeddings"]["global"].shape) if hint_bundle and "global" in hint_bundle["embeddings"] else None,
                    "channel_embedding_shape": list(hint_bundle["embeddings"]["channel"].shape) if hint_bundle and "channel" in hint_bundle["embeddings"] else None,
                    "interval_source": args.interval_source,
                    "prompt_interval": prompt_interval,
                    "proposal": proposal,
                    "anomaly_score_stats": anomaly_score_stats,
                    "score_image_path": score_image_path,
                    "raw_image_path": raw_image_path,
                    "image_not_consumed_by_text_qwen": bool(
                        (args.include_anomaly_score_image_note or args.include_raw_image_note)
                        and not client_supports_images
                    ),
                }
            )
        if args.preview_only:
            continue
        image_path_for_vl = None
        if bool(args.include_anomaly_score_image_note) and score_image_path:
            image_path_for_vl = score_image_path
        elif bool(args.include_raw_image_note) and raw_image_path:
            image_path_for_vl = raw_image_path
        if (
            axis_mode
            and image_path_for_vl
            and hasattr(client, "complete_with_images_and_axis_hints")
        ):
            response = client.complete_with_images_and_axis_hints(
                messages,
                [image_path_for_vl],
                hint_bundle["embeddings"],
            )
        elif image_path_for_vl and client_supports_images and not axis_mode:
            response = client.complete_with_images(messages, [image_path_for_vl])
        elif axis_mode:
            response = client.complete_with_axis_hints(messages, hint_bundle["embeddings"])
        else:
            response = client.complete(messages)
        parsed = parse_student_answer(
            response.content,
            question_type=model_input.question_type,
            question=model_input.question,
        )
        parsed_outputs.append(parsed)
        records.append(
            {
                "index": idx,
                "sample_id": f"{data_path.name}:{idx}",
                "question_type": model_input.question_type,
                "question": model_input.question,
                "prompt": prompt,
                "raw_response": response.content,
                "parsed_response": parsed,
                "parsed_student_answer": parsed,
                "target_output": model_input.target_output,
                "latency_seconds": response.latency_seconds,
                "axis_embedding_hints_used": axis_mode,
                "hint_mode": hint_mode,
                "random_embedding_hints_used": random_hint_mode,
                "embedding_hint_trace": hint_bundle["trace"] if hint_bundle else None,
                "random_hint_metadata": hint_bundle.get("random_metadata") if hint_bundle else None,
                "global_embedding_shape": list(hint_bundle["embeddings"]["global"].shape) if hint_bundle and "global" in hint_bundle["embeddings"] else None,
                "channel_embedding_shape": list(hint_bundle["embeddings"]["channel"].shape) if hint_bundle and "channel" in hint_bundle["embeddings"] else None,
                "interval_source": args.interval_source,
                "prompt_interval": prompt_interval,
                "proposal": proposal,
                "anomaly_score_stats": anomaly_score_stats,
                "score_image_path": score_image_path,
                "raw_image_path": raw_image_path,
                "image_not_consumed_by_text_qwen": bool(
                    (args.include_anomaly_score_image_note or args.include_raw_image_note)
                    and not client_supports_images
                ),
                "llm_raw_metadata": response.raw,
            }
        )
        _append_jsonl(answers_jsonl_path, records[-1])
        if progress_enabled:
            completed_count += 1
            completed = completed_count
            pct = int(completed * 100 / total_items)
            if completed == total_items or pct >= next_progress_pct:
                print(
                    "[qwen-progress] "
                    f"{min(100, pct)}% ({completed}/{total_items}) "
                    f"last_latency={response.latency_seconds:.2f}s",
                    flush=True,
                )
                while next_progress_pct <= pct:
                    next_progress_pct += progress_step

    if args.preview_only:
        save_json(
            {
                "examples": prompt_examples,
                "llm_config": {
                    "provider": llm_config.get("provider"),
                    "model": llm_config.get("model"),
                    "axis_hints": llm_config.get("axis_hints"),
                },
            },
            output_dir / "qwen_prompt_examples.json",
        )
        print(json.dumps({"preview_only": True, "prompt_examples": str(output_dir / "qwen_prompt_examples.json")}, ensure_ascii=False, indent=2))
        return

    metrics = _student_answer_metrics(rows[: len(parsed_outputs)], parsed_outputs)
    metrics["data_path"] = str(data_path)
    metrics["llm_config"] = {
        "provider": llm_config.get("provider"),
        "model": llm_config.get("model"),
        "axis_hints_enabled": bool((llm_config.get("axis_hints") or {}).get("enabled")),
        "max_tokens": llm_config.get("max_tokens"),
    }
    metrics["student_workflow"] = {
        "hint_mode": hint_mode,
        "use_axis_embedding_hints": bool(args.use_axis_embedding_hints),
        "use_random_hints": bool(args.use_random_hints),
        "random_hint_seed": int(args.random_hint_seed),
        "interval_source": args.interval_source,
        "hide_global_hints": bool(args.hide_global_hints),
        "hide_channel_hints": bool(args.hide_channel_hints),
        "hide_evidence_card": True,
        "evidence_card_in_prompt": False,
        "include_anomaly_score_text": bool(args.include_anomaly_score_text),
        "include_anomaly_score_image_note": bool(args.include_anomaly_score_image_note),
        "include_raw_image_note": bool(args.include_raw_image_note),
        "anomaly_score_aggregation": str(args.anomaly_score_aggregation),
        "score_image_manifest": args.score_image_manifest,
        "raw_image_manifest": args.raw_image_manifest,
        "image_not_consumed_by_text_qwen": bool(
            (args.include_anomaly_score_image_note or args.include_raw_image_note)
            and not (client_supports_images or hasattr(client, "complete_with_images_and_axis_hints"))
        ),
        "image_consumed_by_vl_qwen": bool(
            (args.include_anomaly_score_image_note or args.include_raw_image_note)
            and (client_supports_images or hasattr(client, "complete_with_images_and_axis_hints"))
        ),
        "interval_proposer_checkpoint": args.interval_proposer_checkpoint,
        "config": args.config,
    }

    write_jsonl(records, answers_jsonl_path)
    html_path = write_answer_generation_html(answers_jsonl_path, output_dir / "qwen_raw_answers.html")
    metrics["html_report"] = str(html_path)
    save_json(metrics, output_dir / "qwen_raw_report.json")
    save_json({"examples": prompt_examples}, output_dir / "qwen_prompt_examples.json")
    save_json({"examples": records[: max(0, int(args.raw_examples))]}, output_dir / "qwen_raw_examples.json")
    (output_dir / "qwen_case_study.txt").write_text(
        "\n\n" + ("=" * 80) + "\n\n".join(_case_text(r) for r in records[: max(0, int(args.raw_examples))]),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
