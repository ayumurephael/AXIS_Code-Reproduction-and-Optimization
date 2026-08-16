from __future__ import annotations

import argparse
import copy
import contextlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler, Subset

from src.models.MultiAXIS.config import DEEPSEEK_MODEL_ID, MultiAxisConfig
from src.models.MultiAXIS.data import ManifestDataset, collate_multiaxis
from src.models.MultiAXIS.model import MultiAxisForConditionalGeneration


NEW_ARCHITECTURE_PARAMETER_PREFIXES = (
    "channel_query",
    "channel_pool.",
    "question_projection.",
)


def is_new_architecture_parameter(name: str) -> bool:
    return any(
        name == prefix or name.startswith(prefix)
        for prefix in NEW_ARCHITECTURE_PARAMETER_PREFIXES
    )


def set_new_module_warmup(
    model: MultiAxisForConditionalGeneration,
    enabled: bool,
    formal_trainable_names: Sequence[str],
) -> List[torch.nn.Parameter]:
    allowed = set(formal_trainable_names)
    active: List[torch.nn.Parameter] = []
    for name, parameter in model.hint_tuner.named_parameters():
        requires_grad = name in allowed and (
            not enabled or is_new_architecture_parameter(name)
        )
        parameter.requires_grad_(requires_grad)
        if requires_grad:
            active.append(parameter)
    if enabled and not active:
        raise RuntimeError("New-module warm-up selected no trainable parameters")
    return active


def migrate_legacy_training_state(
    model: MultiAxisForConditionalGeneration,
    optimizer: AdamW,
    scheduler: LambdaLR,
    resume: Dict,
) -> Dict:
    legacy_state = resume["hint_tuner_state_dict"]
    current_module_state = model.hint_tuner.state_dict()
    loadable_state = dict(legacy_state)
    expanded_parameters = {}
    for name, old_value in legacy_state.items():
        current_value = current_module_state.get(name)
        if current_value is None or tuple(old_value.shape) == tuple(current_value.shape):
            continue
        if (
            name == "prototype_mapping"
            and old_value.ndim == 2
            and current_value.shape[0] == old_value.shape[0]
            and current_value.shape[1] == old_value.shape[1] + 1
        ):
            expanded = current_value.detach().clone()
            expanded[:, : old_value.shape[1]].copy_(old_value)
            loadable_state[name] = expanded
            expanded_parameters[name] = {
                "old_shape": list(old_value.shape),
                "new_shape": list(current_value.shape),
                "new_columns_initialized": 1,
            }
            continue
        raise RuntimeError(
            f"Unsupported legacy parameter shape migration for {name}: "
            f"{tuple(old_value.shape)} -> {tuple(current_value.shape)}"
        )
    incompatibility = model.hint_tuner.load_state_dict(loadable_state, strict=False)
    expected_missing = {
        "channel_query",
        "channel_pool.q_proj.weight",
        "channel_pool.k_proj.weight",
        "channel_pool.v_proj.weight",
        "channel_pool.out_proj.weight",
        "question_projection.weight",
    }
    missing = set(incompatibility.missing_keys)
    unexpected = set(incompatibility.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise RuntimeError(
            "Legacy architecture state mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if int(torch.count_nonzero(model.hint_tuner.question_projection.weight)) != 0:
        raise AssertionError("A_q must remain exactly zero after legacy migration")

    old_optimizer = resume["optimizer_state_dict"]
    if len(old_optimizer.get("param_groups", [])) != 1:
        raise RuntimeError("Legacy optimizer must contain exactly one parameter group")
    old_ids = list(old_optimizer["param_groups"][0]["params"])
    old_names = list(legacy_state.keys())
    if len(old_ids) != len(old_names):
        raise RuntimeError(
            "Cannot map legacy optimizer by name: parameter/state counts differ"
        )
    old_id_by_name = dict(zip(old_names, old_ids))

    current = optimizer.state_dict()
    if len(current["param_groups"]) != 1:
        raise RuntimeError("Current optimizer must contain exactly one parameter group")
    current_names = [
        name
        for name, parameter in model.hint_tuner.named_parameters()
        if parameter.requires_grad
    ]
    current_ids = list(current["param_groups"][0]["params"])
    if len(current_names) != len(current_ids):
        raise RuntimeError("Current optimizer parameter-name mapping is inconsistent")
    migrated_state = {}
    restored_names = []
    fresh_names = []
    for name, current_id in zip(current_names, current_ids):
        old_id = old_id_by_name.get(name)
        if old_id is None:
            fresh_names.append(name)
            continue
        if old_id in old_optimizer["state"]:
            migrated_entry = copy.deepcopy(old_optimizer["state"][old_id])
            old_parameter = legacy_state[name]
            current_parameter = dict(model.hint_tuner.named_parameters())[name]
            if tuple(old_parameter.shape) != tuple(current_parameter.shape):
                for field, value in list(migrated_entry.items()):
                    if not torch.is_tensor(value) or tuple(value.shape) != tuple(
                        old_parameter.shape
                    ):
                        continue
                    expanded = value.new_zeros(current_parameter.shape, device="cpu")
                    expanded[:, : old_parameter.shape[1]].copy_(value)
                    migrated_entry[field] = expanded
            migrated_state[current_id] = migrated_entry
        restored_names.append(name)
    migrated_group = copy.deepcopy(old_optimizer["param_groups"][0])
    migrated_group["params"] = current_ids
    optimizer.load_state_dict(
        {"state": migrated_state, "param_groups": [migrated_group]}
    )
    scheduler.load_state_dict(resume["scheduler_state_dict"])
    return {
        "schema": "multi-axis-legacy-architecture-migration-v1",
        "missing_initialized_parameters": sorted(missing),
        "restored_optimizer_parameter_names": restored_names,
        "fresh_optimizer_parameter_names": fresh_names,
        "question_projection_nonzero_count": 0,
        "expanded_legacy_parameters": expanded_parameters,
    }


class GroupedDistributedSampler(Sampler[int]):
    """Shards whole base_sample_id groups, balances ranks, and pads equally for collectives."""

    def __init__(
        self, groups: Sequence[Sequence[int]], rank: int, world_size: int, seed: int
    ):
        self.groups = [list(group) for group in groups]
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self._length = self._assign()[1]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._length = self._assign()[1]

    def _assign(self):
        groups = list(self.groups)
        random.Random(self.seed + self.epoch).shuffle(groups)
        assignments: List[List[List[int]]] = [[] for _ in range(self.world_size)]
        sizes = [0] * self.world_size
        for group in groups:
            target = min(range(self.world_size), key=lambda rank: (sizes[rank], rank))
            assignments[target].append(group)
            sizes[target] += len(group)
        max_size = max(sizes)
        indices = [index for group in assignments[self.rank] for index in group]
        if not indices:
            raise RuntimeError(f"Rank {self.rank} received no training examples")
        padded = list(indices)
        cursor = 0
        while len(padded) < max_size:
            padded.append(indices[cursor % len(indices)])
            cursor += 1
        return padded, max_size

    def __iter__(self) -> Iterator[int]:
        return iter(self._assign()[0])

    def __len__(self) -> int:
        return self._length


def distributed_setup():
    if "RANK" not in os.environ:
        raise RuntimeError("Formal training must be launched with torchrun")
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def optimizer_schedule(total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, round(total_steps * warmup_ratio))

    def factor(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor, warmup_steps


def move_model_inputs(batch: Dict, device: torch.device) -> Dict:
    return {
        "normalized_series": batch["normalized_series"].to(device, non_blocking=True),
        "time_mask": batch["time_mask"].to(device, non_blocking=True),
        "channel_mask": batch["channel_mask"].to(device, non_blocking=True),
        "questions": batch["questions"],
        "answers": batch["answers"],
        "intervals": batch["intervals"],
        "channel_counts": batch["channel_counts"],
        "window_values": batch["window_values"],
        "channel_means": batch["channel_means"],
        "channel_stds": batch["channel_stds"],
        "channel_ids": batch["channel_ids"],
        "question_groups": batch["question_groups"],
    }


def average_trainable_gradients(
    parameters: Iterable[torch.nn.Parameter], world_size: int
) -> None:
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError(
                "A declared Hint Tuner parameter did not receive a gradient"
            )
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cpu_hint_state(model) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.hint_tuner.state_dict().items()
    }


def append_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def git_revision() -> Dict[str, str]:
    def run(*args):
        return subprocess.check_output(args, text=True).strip()

    try:
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "branch", "--show-current"),
            "dirty": bool(run("git", "status", "--porcelain")),
        }
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def environment_manifest(
    config: MultiAxisConfig,
    rank: int,
    world_size: int,
    model,
    attention_audit: Dict,
    benchmark_optimizer_steps: int,
) -> Dict:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_revision(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "world_size": world_size,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(world_size)],
        "config": config.to_dict(),
        "pretrained_source": getattr(model, "pretrained_source", config.llm.model_name),
        "actual_effective_batch_size": config.training.micro_batch_size
        * config.training.accumulation_steps
        * world_size,
        "benchmark_optimizer_steps": benchmark_optimizer_steps,
        "pytorch_allocator_configuration": (
            os.environ.get("PYTORCH_ALLOC_CONF")
            or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            or ""
        ),
        "answer_loss_implementation": "causal_answer_positions_sparse_logits",
        "llm_gradient_checkpointing": config.llm.gradient_checkpointing,
        "flash_attention_audit": attention_audit,
        "timercd_sha256": model.timercd.checkpoint_sha256,
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "trainable_parameter_names": model.trainable_parameter_names(),
        "question_semantic_cache": model.question_semantic_cache_manifest(),
    }


def compute_timercd_cached(
    model,
    inputs: Dict,
    cache: Dict,
    base_sample_ids: Sequence[str],
    time_counts: Sequence[int],
    channel_counts: Sequence[int],
):
    """Encode each distinct base series once and assemble a padded micro-batch."""
    batch_size = int(inputs["normalized_series"].shape[0])
    if not (
        len(base_sample_ids) == len(time_counts) == len(channel_counts) == batch_size
    ):
        raise ValueError("TimeRCD cache metadata does not match the micro-batch size")

    entries = cache.setdefault("entries", {})
    active_ids = []
    for index, base_sample_id in enumerate(base_sample_ids):
        key = str(base_sample_id)
        active_ids.append(key)
        steps = int(time_counts[index])
        channels = int(channel_counts[index])
        if key in entries:
            entry = entries[key]
            if entry["time_count"] != steps or entry["channel_count"] != channels:
                raise RuntimeError(
                    f"Base sample {key!r} changed shape inside the TimeRCD cache"
                )
            continue

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if inputs["normalized_series"].is_cuda
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast:
            local, logits = model.timercd(
                inputs["normalized_series"][index : index + 1, :steps, :channels],
                inputs["time_mask"][index : index + 1, :steps],
                inputs["channel_mask"][index : index + 1, :channels],
            )
        entries[key] = {
            "local": local[0].detach(),
            "logits": logits[0].detach(),
            "time_count": steps,
            "channel_count": channels,
        }

    first = entries[active_ids[0]]
    max_steps = int(inputs["normalized_series"].shape[1])
    max_channels = int(inputs["normalized_series"].shape[2])
    local_batch = first["local"].new_zeros(
        (batch_size, max_steps, max_channels, first["local"].shape[-1])
    )
    logits_batch = first["logits"].new_zeros(
        (batch_size, max_steps, max_channels, first["logits"].shape[-1])
    )
    for index, key in enumerate(active_ids):
        entry = entries[key]
        steps = entry["time_count"]
        channels = entry["channel_count"]
        local_batch[index, :steps, :channels] = entry["local"]
        logits_batch[index, :steps, :channels] = entry["logits"]

    # Preserve enough neighboring groups for batches that straddle group boundaries,
    # while bounding cached GPU tensors over a full epoch.
    max_cache_entries = max(2, len(set(active_ids)))
    for key in list(entries):
        if len(entries) <= max_cache_entries:
            break
        if key not in active_ids:
            del entries[key]
    return local_batch, logits_batch


def validate(model, loader: DataLoader, device: torch.device):
    model.eval()
    local_nll = torch.zeros((), device=device, dtype=torch.float64)
    local_tokens = torch.zeros((), device=device, dtype=torch.float64)
    local_examples = torch.zeros((), device=device, dtype=torch.float64)
    local_canonicalized = torch.zeros((), device=device, dtype=torch.float64)
    cache: Dict = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prototype = model.build_prototype_bank()
        for batch in loader:
            inputs = move_model_inputs(batch, device)
            encoded = compute_timercd_cached(
                model,
                inputs,
                cache,
                batch["base_sample_ids"],
                batch["time_counts"],
                batch["channel_counts"],
            )
            outputs = model(
                **inputs, prototype_override=prototype, timercd_override=encoded
            )
            tokens = outputs.supervised_token_count
            local_nll += outputs.loss.double() * tokens
            local_tokens += tokens
            local_examples += len(batch["answers"])
            local_canonicalized += sum(batch["answer_was_canonicalized"])
    totals = torch.stack([local_nll, local_tokens, local_examples, local_canonicalized])
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {
        "answer_nll": (totals[0] / totals[1]).item(),
        "answer_tokens": int(totals[1].item()),
        "examples": int(totals[2].item()),
        "canonicalized_teacher_targets": int(totals[3].item()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Formal distributed Multi-AXIS Phase-II training"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--timercd-checkpoint", required=True)
    parser.add_argument("--question-semantic-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--migrate-legacy-architecture",
        action="store_true",
        help="Migrate a pre-Channel-Local training state by name with zero A_q.",
    )
    parser.add_argument(
        "--new-module-warmup-epochs",
        type=int,
        default=0,
        help="After a legacy migration, train only new Channel/Question modules first.",
    )
    parser.add_argument(
        "--benchmark-optimizer-steps",
        type=int,
        default=0,
        help="Run only N optimizer steps and write throughput/memory audit artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = MultiAxisConfig.load_json(args.config)
    if args.benchmark_optimizer_steps < 0:
        raise ValueError("--benchmark-optimizer-steps must be non-negative")
    if args.benchmark_optimizer_steps and args.resume:
        raise ValueError("A benchmark run cannot resume a formal training state")
    if args.migrate_legacy_architecture and not args.resume:
        raise ValueError("Legacy architecture migration requires --resume")
    if args.new_module_warmup_epochs < 0:
        raise ValueError("--new-module-warmup-epochs must be non-negative")
    if args.new_module_warmup_epochs and not args.migrate_legacy_architecture:
        raise ValueError("New-module warm-up is only valid for a legacy migration")
    if config.llm.model_name != DEEPSEEK_MODEL_ID:
        raise ValueError("The confirmed formal run must use DeepSeek-R1-0528-Qwen3-8B")
    rank, world_size, local_rank, device = distributed_setup()
    if world_size != config.training.expected_world_size:
        raise RuntimeError(
            f"Expected {config.training.expected_world_size} GPUs, got {world_size}"
        )
    # Every rank must start from bit-identical Hint Tuner and special-token weights.
    seed_everything(config.training.seed, 0)
    output_dir = Path(args.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    dataset_kwargs = dict(
        data_root=args.data_root,
        window_epsilon=config.hints.window_epsilon,
        window_scale=config.hints.window_scale,
        normalize_teacher_targets=config.training.normalize_teacher_targets,
    )
    train_dataset = ManifestDataset(
        Path(args.manifest_dir) / "train.jsonl", **dataset_kwargs
    )
    validation_dataset = ManifestDataset(
        Path(args.manifest_dir) / "validation.jsonl", **dataset_kwargs
    )
    train_sampler = GroupedDistributedSampler(
        train_dataset.groups, rank, world_size, config.training.seed
    )
    validation_indices = [
        index
        for group_number, group in enumerate(validation_dataset.groups)
        if group_number % world_size == rank
        for index in group
    ]
    validation_subset = Subset(validation_dataset, validation_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.micro_batch_size,
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=config.training.num_workers > 0,
        collate_fn=collate_multiaxis,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=1,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=config.training.num_workers > 0,
        collate_fn=collate_multiaxis,
    )

    hf_token = os.environ.get("HF_TOKEN") or None
    model = MultiAxisForConditionalGeneration.from_pretrained(
        config, token=hf_token, device=device
    )
    model.configure_question_semantic_cache(args.question_semantic_cache_dir)
    model.load_timercd_checkpoint(args.timercd_checkpoint)
    model.timercd.to(device)
    model.hint_tuner.to(device)
    model.assert_freeze_contract()
    model.train()
    attention_audit = model.attention_backend_audit()
    if rank == 0:
        print(
            "FLASH_ATTENTION_AUDIT "
            + json.dumps(attention_audit, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    dist.barrier()

    formal_trainable = [
        (name, parameter)
        for name, parameter in model.hint_tuner.named_parameters()
        if parameter.requires_grad
    ]
    formal_trainable_names = [name for name, _ in formal_trainable]
    trainable = [parameter for _, parameter in formal_trainable]
    for parameter in trainable:
        dist.broadcast(parameter.data, src=0)
    optimizer = AdamW(
        trainable,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / config.training.accumulation_steps)
    total_steps = steps_per_epoch * config.training.epochs
    schedule_function, warmup_steps = optimizer_schedule(
        total_steps, config.training.warmup_ratio
    )
    scheduler = LambdaLR(optimizer, lr_lambda=schedule_function)
    start_epoch = 1
    global_step = 0
    best_nll = float("inf")
    best_epoch = None
    migration_audit = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.migrate_legacy_architecture:
            migration_audit = migrate_legacy_training_state(
                model, optimizer, scheduler, resume
            )
        else:
            model.hint_tuner.load_state_dict(
                resume["hint_tuner_state_dict"], strict=True
            )
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume["completed_epoch"]) + 1
        global_step = int(resume["global_step"])
        best_nll = float(resume["best_nll"])
        best_epoch = resume.get("best_epoch")
        if migration_audit is not None:
            migration_audit.update(
                {
                    "source": str(Path(args.resume).resolve()),
                    "completed_epoch": int(resume["completed_epoch"]),
                    "global_step": global_step,
                    "new_module_warmup_epochs": args.new_module_warmup_epochs,
                }
            )
            if rank == 0:
                (output_dir / "architecture_transition.json").write_text(
                    json.dumps(migration_audit, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    if rank == 0:
        manifest = environment_manifest(
            config,
            rank,
            world_size,
            model,
            attention_audit,
            args.benchmark_optimizer_steps,
        )
        manifest.update(
            {
                "train_examples": len(train_dataset),
                "validation_examples": len(validation_dataset),
                "steps_per_epoch": steps_per_epoch,
                "total_optimizer_steps": total_steps,
                "warmup_steps": warmup_steps,
                "teacher_target_normalization": {
                    "enabled": config.training.normalize_teacher_targets,
                    "contract": "mc-answer-a-d_tf-yes-no_oe-three-sections-v1",
                },
            }
        )
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    dist.barrier()
    benchmark_start = None
    if args.benchmark_optimizer_steps:
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        benchmark_start = time.perf_counter()

    for epoch in range(start_epoch, config.training.epochs + 1):
        warmup_active = bool(
            args.migrate_legacy_architecture
            and epoch < start_epoch + args.new_module_warmup_epochs
        )
        active_trainable = set_new_module_warmup(
            model, warmup_active, formal_trainable_names
        )
        train_sampler.set_epoch(epoch)
        model.train()
        loader_iterator = iter(train_loader)
        remaining = len(train_loader)
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        epoch_examples = 0
        epoch_canonicalized = 0
        cache: Dict = {}
        while remaining:
            group_size = min(config.training.accumulation_steps, remaining)
            optimizer.zero_grad(set_to_none=True)
            if model.hint_tuner.prototype_mapping.requires_grad:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prototype_graph = model.build_prototype_bank()
                prototype_leaf = prototype_graph.detach().requires_grad_(True)
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prototype_graph = model.build_prototype_bank()
                prototype_leaf = prototype_graph.detach()
            group_loss = 0.0
            for _ in range(group_size):
                batch = next(loader_iterator)
                inputs = move_model_inputs(batch, device)
                encoded = compute_timercd_cached(
                    model,
                    inputs,
                    cache,
                    batch["base_sample_ids"],
                    batch["time_counts"],
                    batch["channel_counts"],
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        **inputs,
                        prototype_override=prototype_leaf,
                        timercd_override=encoded,
                    )
                    scaled_loss = outputs.loss / group_size
                if not torch.isfinite(outputs.loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch={epoch}, step={global_step}"
                    )
                scaled_loss.backward()
                tokens = outputs.supervised_token_count
                epoch_loss_sum += outputs.loss.detach().item() * tokens
                epoch_tokens += tokens
                epoch_examples += len(batch["answers"])
                epoch_canonicalized += sum(batch["answer_was_canonicalized"])
                group_loss += outputs.loss.detach().item()
            if prototype_graph.requires_grad:
                if prototype_leaf.grad is None:
                    raise RuntimeError("Cached prototype bank did not receive a gradient")
                prototype_graph.backward(prototype_leaf.grad)
            average_trainable_gradients(active_trainable, world_size)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                active_trainable, config.training.gradient_clip_norm
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            remaining -= group_size
            if rank == 0 and (args.benchmark_optimizer_steps or global_step % 20 == 0):
                append_jsonl(
                    output_dir / "train_steps.jsonl",
                    {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss_mean_microbatch": group_loss / group_size,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "gradient_norm": float(gradient_norm),
                        "max_memory_allocated_mib": torch.cuda.max_memory_allocated(
                            device
                        )
                        / (1024**2),
                        "max_memory_reserved_mib": torch.cuda.max_memory_reserved(
                            device
                        )
                        / (1024**2),
                    },
                )

            if (
                args.benchmark_optimizer_steps
                and global_step >= args.benchmark_optimizer_steps
            ):
                dist.barrier()
                if benchmark_start is None:
                    raise AssertionError("Benchmark timer was not initialized")
                local_benchmark = torch.tensor(
                    [
                        time.perf_counter() - benchmark_start,
                        float(torch.cuda.max_memory_allocated(device)),
                        float(torch.cuda.max_memory_reserved(device)),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(local_benchmark, op=dist.ReduceOp.MAX)
                elapsed_seconds = float(local_benchmark[0].item())
                summary = {
                    "status": "complete",
                    "optimizer_steps": global_step,
                    "micro_batch_size": config.training.micro_batch_size,
                    "accumulation_steps": config.training.accumulation_steps,
                    "world_size": world_size,
                    "effective_batch_size": config.training.effective_batch_size,
                    "elapsed_seconds_max_rank": elapsed_seconds,
                    "optimizer_steps_per_second": global_step / elapsed_seconds,
                    "training_examples_per_second": (
                        global_step
                        * config.training.effective_batch_size
                        / elapsed_seconds
                    ),
                    "max_memory_allocated_mib_max_rank": float(
                        local_benchmark[1].item() / (1024**2)
                    ),
                    "max_memory_reserved_mib_max_rank": float(
                        local_benchmark[2].item() / (1024**2)
                    ),
                    "flash_attention_audit": attention_audit,
                }
                if rank == 0:
                    payload = json.dumps(summary, ensure_ascii=False, indent=2)
                    (output_dir / "benchmark_summary.json").write_text(
                        payload, encoding="utf-8"
                    )
                    (output_dir / "BENCHMARK_COMPLETE").write_text(
                        payload, encoding="utf-8"
                    )
                    print(
                        "BENCHMARK_COMPLETE "
                        + json.dumps(summary, ensure_ascii=False, sort_keys=True),
                        flush=True,
                    )
                dist.barrier()
                dist.destroy_process_group()
                return

        train_totals = torch.tensor(
            [epoch_loss_sum, epoch_tokens, epoch_examples, epoch_canonicalized],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
        validation = validate(model, validation_loader, device)
        train_nll = (train_totals[0] / train_totals[1]).item()
        if rank == 0:
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_answer_nll": train_nll,
                "train_answer_tokens": int(train_totals[1].item()),
                "train_examples_with_sampler_padding": int(train_totals[2].item()),
                "train_canonicalized_teacher_targets_with_sampler_padding": int(
                    train_totals[3].item()
                ),
                "validation_answer_nll": validation["answer_nll"],
                "validation_answer_tokens": validation["answer_tokens"],
                "validation_examples": validation["examples"],
                "validation_canonicalized_teacher_targets": validation[
                    "canonicalized_teacher_targets"
                ],
                "learning_rate": scheduler.get_last_lr()[0],
                "new_module_warmup_active": warmup_active,
            }
            append_jsonl(output_dir / "epochs.jsonl", epoch_record)
            checkpoint_path = output_dir / f"hint_epoch_{epoch:02d}.pt"
            checkpoint = model.hint_checkpoint_payload(
                epoch,
                global_step,
                {
                    "validation_answer_nll": validation["answer_nll"],
                    "train_answer_nll": train_nll,
                },
            )
            checkpoint["hint_tuner_state_dict"] = cpu_hint_state(model)
            atomic_torch_save(checkpoint, checkpoint_path)
            if validation["answer_nll"] < best_nll:
                best_nll = validation["answer_nll"]
                best_epoch = epoch
            best_record = {
                "selection_metric": "validation_answer_nll",
                "best_epoch": best_epoch,
                "best_value": best_nll,
                "checkpoint": f"hint_epoch_{best_epoch:02d}.pt",
                "completed_epochs": epoch,
                "early_stopping": False,
            }
            (output_dir / "best_checkpoint.json").write_text(
                json.dumps(best_record, indent=2), encoding="utf-8"
            )
            train_state = {
                "format": "multi-axis-training-state-v2-channel-question",
                "architecture": "step-channel-question-conditioned-joint-v1",
                "completed_epoch": epoch,
                "global_step": global_step,
                "best_nll": best_nll,
                "best_epoch": best_epoch,
                "hint_tuner_state_dict": cpu_hint_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config.to_dict(),
            }
            atomic_torch_save(train_state, output_dir / "last_train_state.pt")
        dist.barrier()

    if rank == 0:
        (output_dir / "TRAINING_COMPLETE").write_text(
            json.dumps(
                {
                    "epochs": config.training.epochs,
                    "best_epoch": best_epoch,
                    "best_validation_answer_nll": best_nll,
                }
            ),
            encoding="utf-8",
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
