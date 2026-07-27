"""GPU-only two-pass OE refinement using frozen Baseline drafts."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
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

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round5_selection import (  # noqa: E402,E501
    ROUND5_MODES,
)


REVISION_RULES = {
    "v2_r5_01_oe_preserve_cover": (
        "Revise the draft only where needed so it directly answers every part "
        "of the original question. Keep its anomaly/normal conclusion "
        "unchanged. Do not add unsupported exact numbers, step locations, or "
        "causes. Output one concise final answer."
    ),
    "v2_r5_02_oe_verbatim_or_add": (
        "If the draft already directly answers every requested part, return it "
        "unchanged. Otherwise add only the missing requested information. "
        "Preserve its anomaly/normal verdict and existing evidence claims; do "
        "not add speculation or prompt commentary."
    ),
    "v2_r5_03_oe_conservative_edit": (
        "Perform a conservative edit for completeness and relevance only. "
        "Preserve the draft's anomaly/normal verdict. Cover each requested part "
        "once using only evidence already stated in the draft; remove "
        "repetition and unsupported details. Output only the revised answer."
    ),
    "v2_r5_04_oe_evidence_repair": (
        "Preserve the draft's anomaly/normal verdict. Use the supplied time-"
        "series evidence to repair unsupported details and add any missing "
        "answer to the original question. Do not invent exact values, steps, "
        "or causes. Output one evidence-grounded final answer."
    ),
    "v2_r5_05_oe_minimal_facets": (
        "Make the smallest possible revision while preserving the draft's "
        "anomaly/normal conclusion. Include the direct conclusion, decisive "
        "evidence, and only the boundary, subtype, method, counterevidence, or "
        "other qualification specifically requested by the original question. "
        "Output one compact answer without metadata."
    ),
}


def refinement_question(
    question: str,
    draft: str,
    mode: str,
) -> str:
    try:
        rule = REVISION_RULES[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown Round-5 mode: {mode!r}") from exc
    return (
        "### Original Question\n"
        f"{question}\n\n"
        "### Existing Draft Answer\n"
        f"{draft.strip()}\n\n"
        "### Revision Task\n"
        f"{rule}"
    )


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
    value.add_argument("--series-key", default="screening_series")
    value.add_argument("--baseline-predictions", required=True)
    value.add_argument("--output", required=True)
    value.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(ROUND5_MODES),
        default=list(ROUND5_MODES),
    )
    return value


def main() -> None:
    args = parser().parse_args()
    rank, world, local = distributed_info()
    records = load_axis_records(args.data, subset="full")
    manifest = json.loads(
        Path(args.series_manifest).read_text(encoding="utf-8")
    )
    allowed = set(manifest[args.series_key])
    records = [row for row in records if row.series_file in allowed]

    baseline_rows = [
        row
        for row in read_jsonl(args.baseline_predictions)
        if row["mode"] == "base" and row["series_file"] in allowed
    ]
    baseline = {row["record_id"]: row["response"] for row in baseline_rows}
    expected_ids = {row.record_id for row in records}
    if set(baseline) != expected_ids:
        missing = sorted(expected_ids - set(baseline))
        extra = sorted(set(baseline) - expected_ids)
        raise RuntimeError(
            f"Baseline draft mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

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
                        "A partially completed series/mode batch cannot be "
                        "resumed without changing batch semantics."
                    )
                questions = [
                    refinement_question(
                        row.question,
                        baseline[row.record_id],
                        mode,
                    )
                    if row.question_type == "open_ended"
                    else row.question
                    for row in group
                ]
                responses = model.axis.generate(
                    embeddings,
                    series,
                    questions,
                    answers,
                    starts,
                    ends,
                    ablation_mode="base",
                    question_types=question_types,
                )
                if len(responses) != len(group):
                    raise RuntimeError("Response count differs from series batch")
                for row, response in zip(group, responses):
                    selected = row.question_type == "open_ended"
                    draft = baseline[row.record_id]
                    payload = asdict(row)
                    payload.update(
                        {
                            "mode": mode,
                            "response": response,
                            "loss": None,
                            "selected_component": selected,
                            "support_series_batch_row": not selected,
                            "first_pass_source_mode": "base",
                            "first_pass_response_sha256": hashlib.sha256(
                                draft.encode("utf-8")
                            ).hexdigest(),
                            "revision_rule": REVISION_RULES[mode],
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
        selected_rows = [row for row in rows if row["selected_component"]]
        run_manifest = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "data": str(Path(args.data).resolve()),
            "series_manifest": str(Path(args.series_manifest).resolve()),
            "series_key": args.series_key,
            "baseline_predictions": str(
                Path(args.baseline_predictions).resolve()
            ),
            "baseline_predictions_sha256": sha256_file(
                args.baseline_predictions
            ),
            "world_size": world,
            "modes": args.modes,
            "rows": len(rows),
            "selected_component_rows": len(selected_rows),
            "support_series_batch_rows": len(rows) - len(selected_rows),
            "series_count": len(all_groups),
            "batching": "full series",
            "passes": 2,
            "first_pass": "frozen canonical Baseline response",
            "second_pass_prompt_rules": {
                mode: REVISION_RULES[mode] for mode in args.modes
            },
            "runner_sha256": sha256_file(Path(__file__)),
        }
        (output / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2),
            encoding="utf-8",
        )
        print(
            f"wrote {len(rows)} rows ({len(selected_rows)} selected) "
            f"to {merged}"
        )


if __name__ == "__main__":
    main()
