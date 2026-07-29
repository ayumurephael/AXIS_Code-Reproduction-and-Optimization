"""No-update replay for diagnosing the Phase-II gradient failure mechanism.

The script reconstructs the exact five-rank DistributedSampler batches on one
GPU, but never constructs or steps an optimizer.  Per-rank objective sums are
backpropagated sequentially with the global valid-row denominator, which is
algebraically equivalent to DDP's averaged gradient and avoids a batch-of-five
activation-memory spike.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DistributedSampler

from src.models.AXIS.AXIS_test import collate_fn
from src.models.AXIS.dataset import AXISAnomalyQADataset

from tools.axis_repro.loss_e2e import ERROR_ANSWER
from tools.axis_repro.loss_e2e_40epoch_runtime import (
    axis_objective_forward,
    install_fixed_hint_runtime,
    load_phase2_40epoch_checkpoint,
)
from tools.axis_repro.model_utils import build_model, freeze_for_phase2, sha256_file
from tools.axis_repro.prompt_boundary import install_simplified_prompt_runtime
from tools.axis_repro.train_loss_e2e_ddp import (
    METRIC_KEYS,
    _ensure_frozen_llm_storage_dtype,
    _gradient_l2_norm,
    _summary,
)
from tools.axis_repro.numerical_stability import top_gradient_diagnostics


@dataclass
class TensorStats:
    elements: int = 0
    square_sum: float = 0.0
    abs_max: float = 0.0
    nonfinite: int = 0

    def update(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().float().reshape(-1)
        finite = torch.isfinite(values)
        finite_values = values[finite]
        self.nonfinite += int(values.numel() - finite_values.numel())
        self.elements += int(finite_values.numel())
        if finite_values.numel():
            self.square_sum += float(finite_values.square().sum().item())
            self.abs_max = max(
                self.abs_max,
                float(finite_values.abs().max().item()),
            )

    def to_dict(self) -> dict:
        return {
            "finite_elements": self.elements,
            "nonfinite_elements": self.nonfinite,
            "rms": (
                None
                if self.elements == 0
                else math.sqrt(self.square_sum / self.elements)
            ),
            "abs_max": None if self.elements == 0 else self.abs_max,
        }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeat as LABEL=/absolute/path/epoch_NN.pth",
    )
    result.add_argument("--data", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--epoch", type=int, default=5)
    result.add_argument("--start-step", type=int, default=655)
    result.add_argument("--accumulation-steps", type=int, default=32)
    result.add_argument("--world-size", type=int, default=5)
    result.add_argument("--seed", type=int, default=72)
    result.add_argument("--alpha", type=float, default=0.40)
    result.add_argument("--loss-chunk-size", type=int, default=64)
    return result


def parse_checkpoints(values: Sequence[str]) -> list[tuple[str, Path]]:
    rows = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint must be LABEL=/absolute/path")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path).resolve()
        if not label or label in labels:
            raise ValueError("checkpoint labels must be nonempty and unique")
        if not path.is_file():
            raise FileNotFoundError(path)
        labels.add(label)
        rows.append((label, path))
    return rows


def sampler_indices(
    dataset,
    *,
    world_size: int,
    rank: int,
    seed: int,
    epoch: int,
) -> list[int]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    sampler.set_epoch(epoch)
    return list(iter(sampler))


def add_metrics(target: dict[str, float], source: Mapping[str, torch.Tensor]) -> None:
    for key in METRIC_KEYS:
        target[key] += float(source[key].detach().double().item())


def tensor_from_hook_input(inputs) -> torch.Tensor:
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        raise RuntimeError("diagnostic hook did not receive a tensor")
    return inputs[0]


def install_scale_hooks(base) -> tuple[dict[str, TensorStats], list]:
    stats = {
        "local_word_proj_input": TensorStats(),
        "local_word_proj_output": TensorStats(),
        "q_proj_input": TensorStats(),
        "q_proj_output": TensorStats(),
        "k_proj_input": TensorStats(),
        "k_proj_output": TensorStats(),
    }
    perceiver = base.axis.perceiver
    handles = [
        perceiver.local_word_proj.register_forward_pre_hook(
            lambda _module, inputs: stats["local_word_proj_input"].update(
                tensor_from_hook_input(inputs)
            )
        ),
        perceiver.local_word_proj.register_forward_hook(
            lambda _module, _inputs, output: stats["local_word_proj_output"].update(
                output
            )
        ),
        perceiver.local_attention.q_proj.register_forward_pre_hook(
            lambda _module, inputs: stats["q_proj_input"].update(
                tensor_from_hook_input(inputs)
            )
        ),
        perceiver.local_attention.q_proj.register_forward_hook(
            lambda _module, _inputs, output: stats["q_proj_output"].update(output)
        ),
        perceiver.local_attention.k_proj.register_forward_pre_hook(
            lambda _module, inputs: stats["k_proj_input"].update(
                tensor_from_hook_input(inputs)
            )
        ),
        perceiver.local_attention.k_proj.register_forward_hook(
            lambda _module, _inputs, output: stats["k_proj_output"].update(output)
        ),
    ]
    return stats, handles


def replay_checkpoint(
    *,
    label: str,
    checkpoint: Path,
    dataset,
    rank_indices: Sequence[Sequence[int]],
    device: int,
    start_step: int,
    accumulation_steps: int,
    world_size: int,
    seed: int,
    alpha: float,
    loss_chunk_size: int,
) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base = build_model()
    install_simplified_prompt_runtime(base.axis)
    payload = load_phase2_40epoch_checkpoint(base, checkpoint)
    freeze_for_phase2(base)
    install_fixed_hint_runtime(base.axis)
    base.axis.perceiver.fix_prompt_embeddings.requires_grad_(False)
    base.to(device)
    _ensure_frozen_llm_storage_dtype(base.axis.model, torch.bfloat16)
    base.ts_pretrain_model.eval()
    base.axis.model.train()
    base.axis.model.gradient_checkpointing_enable()
    base.axis.model.enable_input_require_grads()
    base.axis.model.config.use_cache = False
    base.axis.model.get_input_embeddings().register_forward_hook(
        lambda _module, _inputs, output: output.clone()
    )

    scale_stats, hook_handles = install_scale_hooks(base)
    local_encoder_stats = TensorStats()
    window_metrics = {key: 0.0 for key in METRIC_KEYS}
    gradient_curve = []
    milestones = {1, 2, 4, 8, 16, accumulation_steps}
    sample_files = []
    for parameter in base.parameters():
        parameter.grad = None

    try:
        for offset in range(accumulation_steps):
            step = start_step + offset
            if step < 1 or step > len(rank_indices[0]):
                raise ValueError("diagnostic replay step is outside the epoch")
            indices = [rank_indices[rank][step - 1] for rank in range(world_size)]
            items = [dataset[index] for index in indices]
            batches = [collate_fn([item]) for item in items]
            global_valid_rows = sum(
                str(answer).strip() != ERROR_ANSWER
                for batch in batches
                for answer in batch["answers"]
            )
            if global_valid_rows <= 0:
                raise RuntimeError("diagnostic global batch has no valid QA rows")
            batch_metrics = {key: 0.0 for key in METRIC_KEYS}
            if offset in {0, accumulation_steps - 1}:
                sample_files.extend(
                    Path(dataset.series_files[index]).name for index in indices
                )

            for batch in batches:
                time_series = batch["padded_sequences"].to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                attention_masks = batch["attention_masks"].to(
                    device,
                    non_blocking=True,
                )
                valid_rows = [
                    answer.strip() != ERROR_ANSWER for answer in batch["answers"]
                ]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    local_embeddings = base.ts_pretrain_model(
                        time_series,
                        mask=attention_masks,
                    )
                for row, (start, end) in enumerate(
                    zip(batch["start_indices"], batch["end_indices"])
                ):
                    local_encoder_stats.update(
                        local_embeddings[row, int(start) : int(end)]
                    )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = axis_objective_forward(
                        axis=base.axis,
                        local_embeddings=local_embeddings,
                        time_series=time_series,
                        questions=batch["questions"],
                        answers=batch["answers"],
                        start_indices=batch["start_indices"],
                        end_indices=batch["end_indices"],
                        question_types=batch["question_types"],
                        segment_alpha=alpha,
                        valid_rows=valid_rows,
                        loss_chunk_size=loss_chunk_size,
                    )
                add_metrics(batch_metrics, outputs)
                loss = outputs["objective_sum"] / global_valid_rows / accumulation_steps
                loss.backward()
                del outputs, loss, local_embeddings, time_series, attention_masks

            for key, value in batch_metrics.items():
                window_metrics[key] += value
            completed = offset + 1
            if completed in milestones:
                cumulative_norm = float(_gradient_l2_norm(base.parameters()))
                gradient_curve.append(
                    {
                        "global_batches_accumulated": completed,
                        "equivalent_mean_gradient_l2_norm": (
                            cumulative_norm * accumulation_steps / completed
                        ),
                        "batch_metrics": _summary(batch_metrics),
                    }
                )

        pre_clip_norm = float(_gradient_l2_norm(base.parameters()))
        top_gradients = top_gradient_diagnostics(
            base.named_parameters(),
            limit=12,
        )
        clip_return = float(
            torch.nn.utils.clip_grad_norm_(base.parameters(), max_norm=1.0)
        )
        post_clip_norm = float(_gradient_l2_norm(base.parameters()))
        return {
            "label": label,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "checkpoint_global_step": int(payload.get("global_step", -1)),
            "precision": "BF16 autocast; trainable parameters and gradients FP32",
            "grad_scaler_used": False,
            "optimizer_constructed": False,
            "optimizer_steps_applied": 0,
            "replay_world_size": world_size,
            "replay_start_step": start_step,
            "replay_end_step": start_step + accumulation_steps - 1,
            "accumulation_steps": accumulation_steps,
            "sample_files_first_and_last_batches": sample_files,
            "window_metrics": _summary(window_metrics),
            "gradient_curve": gradient_curve,
            "mean_gradient_l2_norm_before_clip": pre_clip_norm,
            "clip_grad_norm_return_value": clip_return,
            "gradient_l2_norm_after_clip": post_clip_norm,
            "top_gradients_before_clip": top_gradients,
            "activation_stats": {
                "phase1_local_embeddings_valid_window": (local_encoder_stats.to_dict()),
                **{name: stats.to_dict() for name, stats in scale_stats.items()},
            },
        }
    finally:
        for handle in hook_handles:
            handle.remove()
        for parameter in base.parameters():
            parameter.grad = None
        del base
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation-steps must be positive")
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive")
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    checkpoints = parse_checkpoints(args.checkpoint)
    dataset = AXISAnomalyQADataset(
        args.data,
        split="train",
        train_ratio=0.95,
        seed=args.seed,
    )
    rank_indices = [
        sampler_indices(
            dataset,
            world_size=args.world_size,
            rank=rank,
            seed=args.seed,
            epoch=args.epoch,
        )
        for rank in range(args.world_size)
    ]
    results = [
        replay_checkpoint(
            label=label,
            checkpoint=checkpoint,
            dataset=dataset,
            rank_indices=rank_indices,
            device=device,
            start_step=args.start_step,
            accumulation_steps=args.accumulation_steps,
            world_size=args.world_size,
            seed=args.seed,
            alpha=args.alpha,
            loss_chunk_size=args.loss_chunk_size,
        )
        for label, checkpoint in checkpoints
    ]
    payload = {
        "schema_version": 1,
        "experiment": "phase2_gradient_failure_no_update_replay",
        "invariants": {
            "checkpoint_mutation": False,
            "optimizer_steps": 0,
            "sampling": "exact DistributedSampler(seed=72, epoch=5, five ranks)",
            "gradient_equivalence": (
                "sequential per-rank objective sums divided by the global "
                "valid-row denominator"
            ),
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "labels": [x[0] for x in checkpoints]}))


if __name__ == "__main__":
    main()
