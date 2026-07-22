"""Calibrate TF single-token readout against actual AXIS generation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.models.AXIS.dataset import AXISAnomalyQADataset
from .common import extract_true_false
from .loss_redesign import _pad_with_mask
from .model_utils import build_model, freeze_for_phase2, load_axis_checkpoint
from .question_type_objectives import canonical_question_type, parse_tf_target, select_tf_verbalizers


class CalibrationDataset(AXISAnomalyQADataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["series_file"] = self.series_files[index].name
        return item


def collate(batch):
    sequences = []
    questions, answers, starts, ends, record_ids = [], [], [], [], []
    for item in batch:
        sequence = torch.as_tensor(item["time_series"], dtype=torch.float32)
        for window_index, window in enumerate(item["analysis_data"]):
            if canonical_question_type(window.get("question_type", "unknown")) != "true_false":
                continue
            if parse_tf_target(window.get("answer", "")) is None:
                continue
            sequences.append(sequence)
            questions.append(window["question"])
            answers.append(window["answer"])
            starts.append(int(window["window_range"]["start"]))
            ends.append(int(window["window_range"]["end"]))
            record_ids.append(f'{item["series_file"]}:{window_index}')
    if not sequences:
        return None
    padded, masks = _pad_with_mask(sequences)
    return {
        "padded_sequences": padded,
        "attention_masks": masks,
        "questions": questions,
        "answers": answers,
        "start_indices": starts,
        "end_indices": ends,
        "record_ids": record_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/anomaly_llava_training_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--architecture-variant", default="loss_only")
    parser.add_argument("--qk-norm-seq-len", type=int)
    parser.add_argument("--gate-bias", type=float, default=-2.0)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.distributed.init_process_group("nccl")

    dataset = CalibrationDataset(
        args.data, split="val", train_ratio=0.95, seed=args.seed
    )
    if args.max_series is not None:
        dataset.series_files = dataset.series_files[: args.max_series]
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate)
    model = build_model(args.architecture_variant, args.qk_norm_seq_len, args.gate_bias)
    load_axis_checkpoint(model, args.checkpoint, strict=True)
    freeze_for_phase2(model)
    model.to(local).eval()
    ddp = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False)
    verbalizers = select_tf_verbalizers(ddp.module.axis.tokenizer)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rank_path = output / f"rank_{rank}.jsonl"
    with rank_path.open("w", encoding="utf-8") as handle:
        for batch in loader:
            if batch is None:
                continue
            series = batch["padded_sequences"].to(local, dtype=torch.float32)
            mask = batch["attention_masks"].to(local)
            windows = [
                series[index, start:end].detach().cpu()
                for index, (start, end) in enumerate(zip(batch["start_indices"], batch["end_indices"]))
            ]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                local_embeddings = ddp.module.ts_pretrain_model(series, mask=mask)
                logits = ddp.module.axis.tf_logits(
                    local_embeddings,
                    series,
                    batch["questions"],
                    batch["start_indices"],
                    batch["end_indices"],
                    windows,
                    verbalizers.false_id,
                    verbalizers.true_id,
                    fixed_hint_frozen=True,
                )
                generated = ddp.module.axis.generate(
                    local_embeddings,
                    series,
                    batch["questions"],
                    batch["answers"],
                    batch["start_indices"],
                    batch["end_indices"],
                    max_new_tokens=args.max_new_tokens,
                )
            readouts = logits.argmax(dim=-1).tolist()
            for record_id, answer, response, readout in zip(
                batch["record_ids"], batch["answers"], generated, readouts
            ):
                generated_label = extract_true_false(response)
                row = {
                    "record_id": record_id,
                    "readout": bool(readout),
                    "generated": generated_label,
                    "reference": bool(parse_tf_target(answer)),
                    "agreement": generated_label is not None and bool(readout) == generated_label,
                    "generation_parseable": generated_label is not None,
                    "response_prefix": response[:300],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    torch.distributed.barrier()
    if rank == 0:
        rows = []
        for rank_index in range(world):
            for line in (output / f"rank_{rank_index}.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        keys = [row["record_id"] for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError("TF calibration contains duplicate records")
        parseable = sum(row["generation_parseable"] for row in rows)
        agreement = sum(row["agreement"] for row in rows)
        report = {
            "rows": len(rows),
            "parseable_generation": parseable,
            "parseable_rate": parseable / len(rows) if rows else 0.0,
            "readout_generation_agreement": agreement,
            "agreement_rate_on_all": agreement / len(rows) if rows else 0.0,
            "agreement_rate_on_parseable": agreement / parseable if parseable else 0.0,
            "max_new_tokens": args.max_new_tokens,
            "verbalizers": verbalizers.to_metadata(),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
