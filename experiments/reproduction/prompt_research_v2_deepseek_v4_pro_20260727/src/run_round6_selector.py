"""GPU-only selection between frozen Baseline and Round-5 OE answers."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.axis_repro.common import load_axis_records, read_jsonl  # noqa: E402
from tools.axis_repro.io_utils import append_jsonl  # noqa: E402
from tools.axis_repro.model_utils import (  # noqa: E402
    build_model,
    load_axis_checkpoint,
    sha256_file,
)


ROUND6_MODES = (
    "v2_r6_01_oe_missing_part_selector",
    "v2_r6_02_oe_table_i_selector",
    "v2_r6_03_oe_content_guard_selector",
)
ROUND5_SOURCE_MODE = "v2_r5_02_oe_verbatim_or_add"

SELECTOR_RULES = {
    "v2_r6_01_oe_missing_part_selector": (
        "Select Candidate B only if it answers an explicit part of the "
        "Original Question that Candidate A omits, while retaining every "
        "conclusion, evidence item, and requested qualification from A. "
        "Otherwise select A, including any tie or uncertainty."
    ),
    "v2_r6_02_oe_table_i_selector": (
        "Compare the candidates on factual accuracy, completeness with respect "
        "to every requested part, and relevance. Select B only if it is no "
        "worse than A on all three and clearly better on at least one. "
        "Otherwise select A, including any tie or uncertainty."
    ),
    "v2_r6_03_oe_content_guard_selector": (
        "Select Candidate B only if it preserves every substantive conclusion, "
        "evidence claim, number, boundary qualification, and method from "
        "Candidate A while removing only markup, repetition, or unrelated "
        "material. Otherwise select A, including any tie or uncertainty."
    ),
}

SELECTION_PATTERNS = (
    re.compile(r"\bSelection\s*:\s*([AB])\b", flags=re.I),
    re.compile(r"\bAnswer\s*:\s*([AB])\b", flags=re.I),
    re.compile(r"^\s*([AB])(?:[\s.)]|$)", flags=re.I | re.M),
)


def selector_question(
    question: str,
    baseline_response: str,
    candidate_response: str,
    mode: str,
) -> str:
    try:
        rule = SELECTOR_RULES[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown Round-6 mode: {mode!r}") from exc
    return (
        "### Original Question\n"
        f"{question}\n\n"
        "### Candidate A — Baseline Draft\n"
        f"{baseline_response.strip()}\n\n"
        "### Candidate B — Conservative Revision\n"
        f"{candidate_response.strip()}\n\n"
        "### Selection Task\n"
        f"{rule}\n\n"
        "Output exactly Selection: A or Selection: B. Do not write a new answer."
    )


def parse_selection(response: str) -> tuple[str, bool]:
    choices = []
    for pattern in SELECTION_PATTERNS:
        choices.extend(match.upper() for match in pattern.findall(response))
    unique = set(choices)
    if len(unique) == 1:
        return unique.pop(), True
    return "A", False


def distributed_info() -> tuple[int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    local = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1:
        torch.distributed.init_process_group(
            "nccl",
            timeout=timedelta(hours=2),
        )
    return rank, world, local


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--data", required=True)
    value.add_argument("--series-manifest", required=True)
    value.add_argument("--series-key", required=True)
    value.add_argument("--baseline-predictions", required=True)
    value.add_argument("--candidate-predictions", required=True)
    value.add_argument("--output", required=True)
    value.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(ROUND6_MODES),
        default=list(ROUND6_MODES),
    )
    return value


def load_source(
    path: str,
    mode: str,
    allowed: set[str],
) -> dict[str, dict]:
    rows = [
        row
        for row in read_jsonl(path)
        if row["mode"] == mode and row["series_file"] in allowed
    ]
    result = {row["record_id"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate {mode} source records")
    return result


def main() -> None:
    args = parser().parse_args()
    rank, world, local = distributed_info()
    manifest = json.loads(
        Path(args.series_manifest).read_text(encoding="utf-8")
    )
    allowed = set(manifest[args.series_key])
    records = [
        row
        for row in load_axis_records(args.data, subset="full")
        if row.series_file in allowed
    ]
    baseline = load_source(args.baseline_predictions, "base", allowed)
    candidate = load_source(
        args.candidate_predictions,
        ROUND5_SOURCE_MODE,
        allowed,
    )
    oe_ids = {
        row.record_id for row in records if row.question_type == "open_ended"
    }
    if not oe_ids <= set(candidate):
        raise RuntimeError(
            f"Missing Round-5 candidates: {sorted(oe_ids - set(candidate))[:5]}"
        )
    if {row.record_id for row in records} != set(baseline):
        raise RuntimeError("Baseline prediction IDs do not match the split")

    grouped = collections.OrderedDict()
    for record in records:
        grouped.setdefault(record.series_file, []).append(record)
    all_groups = [
        group
        for group in grouped.values()
        if any(row.question_type == "open_ended" for row in group)
    ]
    groups = all_groups[rank::world]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shard = output / f"rank{rank}.jsonl"
    done = {
        (row["record_id"], row["mode"])
        for row in read_jsonl(shard)
    } if shard.exists() else set()

    model = build_model()
    load_axis_checkpoint(model, args.checkpoint)
    model.to(torch.device("cuda", local)).eval()

    for group in groups:
        first = group[0]
        series = torch.tensor(
            first.time_series,
            dtype=torch.float32,
            device=local,
        ).unsqueeze(0).repeat(len(group), 1)
        mask = torch.ones_like(series, dtype=torch.bool)
        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=torch.float16,
        ):
            embeddings = model.ts_pretrain_model(series, mask=mask)
            answers = [row.answer for row in group]
            question_types = [row.question_type for row in group]
            starts = [row.start_index for row in group]
            ends = [row.end_index for row in group]
            for mode in args.modes:
                group_keys = [(row.record_id, mode) for row in group]
                if all(key in done for key in group_keys):
                    continue
                if any(key in done for key in group_keys):
                    raise RuntimeError(
                        "Partially completed series/mode batch cannot resume"
                    )
                questions = [
                    selector_question(
                        row.question,
                        baseline[row.record_id]["response"],
                        candidate[row.record_id]["response"],
                        mode,
                    )
                    if row.question_type == "open_ended"
                    else row.question
                    for row in group
                ]
                selector_responses = model.axis.generate(
                    embeddings,
                    series,
                    questions,
                    answers,
                    starts,
                    ends,
                    ablation_mode="base",
                    question_types=question_types,
                )
                for row, selector_response in zip(group, selector_responses):
                    selected_component = row.question_type == "open_ended"
                    if selected_component:
                        selection, valid = parse_selection(selector_response)
                        source = (
                            candidate[row.record_id]
                            if selection == "B"
                            else baseline[row.record_id]
                        )
                        source_mode = (
                            ROUND5_SOURCE_MODE if selection == "B" else "base"
                        )
                    else:
                        selection, valid = "A", True
                        source = baseline[row.record_id]
                        source_mode = "base"
                    payload = asdict(row)
                    payload.update(
                        {
                            "mode": mode,
                            "response": source["response"],
                            "loss": None,
                            "selected_component": selected_component,
                            "support_series_batch_row": not selected_component,
                            "selector_response": selector_response,
                            "selector_choice": selection,
                            "selector_parse_valid": valid,
                            "selected_source_mode": source_mode,
                            "selected_response_sha256": hashlib.sha256(
                                source["response"].encode("utf-8")
                            ).hexdigest(),
                            "selector_rule": SELECTOR_RULES[mode],
                        }
                    )
                    append_jsonl(shard, payload)
                    done.add((row.record_id, mode))

    if world > 1:
        torch.distributed.barrier(device_ids=[local])
        torch.distributed.destroy_process_group()
    if rank == 0:
        rows = []
        for index in range(world):
            rows.extend(read_jsonl(output / f"rank{index}.jsonl"))
        rows.sort(key=lambda row: (row["record_id"], row["mode"]))
        merged = output / "predictions.jsonl"
        merged.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        selected = [row for row in rows if row["selected_component"]]
        choice_counts = {
            mode: dict(
                collections.Counter(
                    row["selector_choice"]
                    for row in selected
                    if row["mode"] == mode
                )
            )
            for mode in args.modes
        }
        run_manifest = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "series_key": args.series_key,
            "world_size": world,
            "modes": args.modes,
            "rows": len(rows),
            "selected_component_rows": len(selected),
            "series_count": len(all_groups),
            "selector_choice_counts": choice_counts,
            "invalid_selector_rows": sum(
                not row["selector_parse_valid"] for row in selected
            ),
            "runner_sha256": sha256_file(Path(__file__)),
        }
        (output / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2),
            encoding="utf-8",
        )
        print(
            f"wrote {len(rows)} rows ({len(selected)} selected) "
            f"to {merged}"
        )


if __name__ == "__main__":
    main()
