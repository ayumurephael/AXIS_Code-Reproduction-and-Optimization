"""GPU-only Round-1 inference with exact task-family component selection.

Each selected series is still generated as the author's full two-QA series
batch. Rows whose prompt is byte-identical to Baseline are retained as support
rows but are never used as candidate components.
"""

from __future__ import annotations

import argparse
import collections
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

from src.models.AXIS.prompt_stage_a import (  # noqa: E402
    V2_R1_MODES,
    build_question_prompt,
    mode_manifest,
)
from tools.axis_repro.common import load_axis_records, read_jsonl  # noqa: E402
from tools.axis_repro.io_utils import append_jsonl  # noqa: E402
from tools.axis_repro.model_utils import (  # noqa: E402
    build_model,
    load_axis_checkpoint,
    sha256_file,
)

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round1_selection import (  # noqa: E402,E501
    ACTIVE_FAMILY,
    prompt_changed,
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
    value.add_argument("--series-key", default="all_series")
    value.add_argument("--output", required=True)
    value.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(V2_R1_MODES),
        default=list(V2_R1_MODES),
    )
    value.add_argument("--skip-loss", action="store_true")
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

    grouped = collections.OrderedDict()
    for record in records:
        grouped.setdefault(record.series_file, []).append(record)
    groups = list(grouped.values())[rank::world]

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
            questions = [row.question for row in group]
            answers = [row.answer for row in group]
            question_types = [row.question_type for row in group]
            starts = [row.start_index for row in group]
            ends = [row.end_index for row in group]
            for mode in args.modes:
                selected = [
                    row.question_type == ACTIVE_FAMILY[mode]
                    and prompt_changed(row, mode)
                    for row in group
                ]
                if not any(selected):
                    continue
                group_keys = [(row.record_id, mode) for row in group]
                if all(key in done for key in group_keys):
                    continue
                if any(key in done for key in group_keys):
                    raise RuntimeError(
                        "A partially completed series/mode batch cannot be "
                        "resumed without changing batch semantics."
                    )

                loss = None
                if not args.skip_loss:
                    loss = model.axis(
                        embeddings,
                        series,
                        questions,
                        answers,
                        starts,
                        ends,
                        ablation_mode=mode,
                        question_types=question_types,
                    )
                responses = model.axis.generate(
                    embeddings,
                    series,
                    questions,
                    answers,
                    starts,
                    ends,
                    ablation_mode=mode,
                    question_types=question_types,
                )
                if len(responses) != len(group):
                    raise RuntimeError("Response count differs from series batch")
                for row, response, is_selected in zip(
                    group,
                    responses,
                    selected,
                ):
                    payload = asdict(row)
                    payload.update(
                        {
                            "mode": mode,
                            "response": response,
                            "loss": None if loss is None else float(loss),
                            "selected_component": is_selected,
                            "support_series_batch_row": not is_selected,
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
        prompt_spec = REPO / "src/models/AXIS/prompt_stage_a.py"
        run_manifest = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "data": str(Path(args.data).resolve()),
            "series_manifest": str(Path(args.series_manifest).resolve()),
            "series_key": args.series_key,
            "world_size": world,
            "modes": args.modes,
            "rows": len(rows),
            "selected_component_rows": len(selected_rows),
            "support_series_batch_rows": len(rows) - len(selected_rows),
            "series_count": len(grouped),
            "question_type_counts": dict(
                collections.Counter(row.question_type for row in records)
            ),
            "batching": "full series",
            "skip_loss": args.skip_loss,
            "prompt_spec": str(prompt_spec),
            "prompt_spec_sha256": sha256_file(prompt_spec),
            "mode_definitions": mode_manifest(args.modes),
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
