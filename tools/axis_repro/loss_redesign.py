from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset

from src.models.AXIS.dataset import AXISAnomalyQADataset
from .question_type_objectives import (
    SUPERVISION_CACHE_VERSION,
    canonical_question_type,
)


COUNTERFACTUAL_INDEX_VERSION = 3
STATE_VERBALIZER_CANDIDATES = (
    (" 0", " 1"),
    ("0", "1"),
)


@dataclass(frozen=True)
class StateVerbalizers:
    normal_text: str
    anomalous_text: str
    normal_id: int
    anomalous_id: int
    question: str

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def build_state_question(normal_text: str, anomalous_text: str) -> str:
    normal_label = normal_text.strip()
    anomalous_label = anomalous_text.strip()
    lines = [
        "Classify the target interval using only the time-series evidence above.",
        "",
        "NORMAL means that the target window contains no anomaly.",
        "ANOMALOUS means that the target window contains an anomaly.",
        "",
    ]
    if normal_label.lower() == "normal" and anomalous_label.lower() == "anomalous":
        lines.append("Return exactly one label: NORMAL or ANOMALOUS.")
    else:
        lines.extend([
            "Output exactly one label:",
            f"{normal_label} = normal",
            f"{anomalous_label} = anomalous",
        ])
    lines.append("Label:")
    return "\n".join(lines)


def select_state_verbalizers(tokenizer) -> StateVerbalizers:
    """Select the first distinct numeric 0/1 pair that is one token for this tokenizer."""
    for normal_text, anomalous_text in STATE_VERBALIZER_CANDIDATES:
        normal_ids = tokenizer.encode(normal_text, add_special_tokens=False)
        anomalous_ids = tokenizer.encode(anomalous_text, add_special_tokens=False)
        if len(normal_ids) == 1 and len(anomalous_ids) == 1 and normal_ids[0] != anomalous_ids[0]:
            selected = StateVerbalizers(
                normal_text=normal_text,
                anomalous_text=anomalous_text,
                normal_id=int(normal_ids[0]),
                anomalous_id=int(anomalous_ids[0]),
                question=build_state_question(normal_text, anomalous_text),
            )
            if (normal_text.strip(), anomalous_text.strip()) != ("0", "1"):
                raise RuntimeError("The formal objective requires numeric 0/1 labels")
            return selected
    raise RuntimeError("Neither space-prefixed nor bare numeric 0/1 labels are distinct single tokens")


def format_axis_question_prompt(
    *,
    question: str,
    start_index: int,
    end_index: int,
    window_values: torch.Tensor,
    num_fixed_tokens: int,
    include_local: bool,
    include_window: bool,
    include_fixed: bool = True,
    missing_window_text: str = "(not provided)",
) -> str:
    num_local_tokens = end_index - start_index
    local_tokens = "<|local_hint|>" * num_local_tokens if include_local else ""
    fixed_tokens = "<|fixed_hint|>" * num_fixed_tokens if include_fixed else ""
    if include_window:
        values = window_values.detach().cpu().reshape(-1).tolist()
        window_text = ", ".join(f"{(float(value) * 100):.0f}" for value in values)
    else:
        window_text = missing_window_text
    return f"""
            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps {start_index} to {end_index}
            - **Values (scaled by 100):** {window_text}

            ### Contextual Hints
            - **Per-Step Analysis:** {local_tokens}
            - **Overall Summary Hints:** {fixed_tokens}

            ### Question
            {question}
            """


def scheduled_beta(
    step: int,
    total_steps: int,
    *,
    target_beta: float = 0.1,
    warmup_ratio: float = 0.1,
) -> float:
    """Linearly increase beta from zero over the first warmup_ratio of updates."""
    if step < 0 or total_steps <= 0:
        raise ValueError("step must be non-negative and total_steps must be positive")
    if target_beta < 0 or not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("invalid beta schedule")
    if target_beta == 0.0 or warmup_ratio == 0.0:
        return float(target_beta)
    warmup_steps = max(2, int(math.ceil(total_steps * warmup_ratio)))
    progress = min(1.0, step / float(warmup_steps - 1))
    return float(target_beta * progress)


def compute_state_loss(state_logits: torch.Tensor, state_targets: torch.Tensor) -> torch.Tensor:
    if state_logits.ndim != 2 or state_logits.shape[1] != 2:
        raise ValueError("state logits must have shape [N, 2]")
    if state_targets.shape != state_logits.shape[:1]:
        raise ValueError("state targets must have shape [N]")
    if state_logits.shape[0] == 0:
        return state_logits.new_zeros(())
    return F.cross_entropy(state_logits.float(), state_targets.long())


def memory_safe_token_nll_sums(
    lm_head,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 64,
    checkpoint_chunks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row full-answer NLL sums/counts without retaining full-vocab logits."""
    batch_size = hidden.shape[0]
    full_sums = hidden.new_zeros(batch_size, dtype=torch.float32)
    full_counts = torch.zeros(batch_size, dtype=torch.long, device=hidden.device)
    for start in range(0, hidden.shape[1], chunk_size):
        chunk_hidden = hidden[:, start:start + chunk_size]
        chunk_targets = targets[:, start:start + chunk_size]

        def token_losses(current_hidden, current_targets):
            logits = lm_head(current_hidden).float()
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                current_targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(current_targets.shape)

        if checkpoint_chunks and torch.is_grad_enabled():
            nll = checkpoint(token_losses, chunk_hidden, chunk_targets, use_reentrant=False)
        else:
            nll = token_losses(chunk_hidden, chunk_targets)
        valid = chunk_targets.ne(-100)
        full_sums = full_sums + (nll * valid).sum(dim=1)
        full_counts = full_counts + valid.sum(dim=1)
    return full_sums, full_counts


@torch.no_grad()
def stable_uniform_from_key(seed: int, key: str) -> float:
    """Return a deterministic U[0,1) variate independent of batching and device."""
    digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def retrieve_consistent_donors(
    anchors: torch.Tensor,
    donor_normal: torch.Tensor,
    donor_abnormal: torch.Tensor,
    donor_valid: torch.Tensor,
    *,
    tau: float = 0.25,
    top_m: int = 8,
    random_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniformly sample a donor from the nearest compatible Top-M candidates.

    Returns selected donor indices, found flags, selected compatibility ratios,
    zero-based sampled ranks, and the number of all compatible donors per anchor.
    """
    anchors = anchors.float()
    donor_normal = donor_normal.float()
    donor_abnormal = donor_abnormal.float()
    donor_valid = donor_valid.bool()
    if top_m <= 0:
        raise ValueError("top_m must be positive")
    if anchors.ndim != 2 or donor_normal.shape != donor_abnormal.shape:
        raise ValueError("retrieval tensors must be flattened matrices")
    if donor_normal.shape[1] != anchors.shape[1] or donor_valid.numel() != donor_normal.shape[0]:
        raise ValueError("retrieval tensors have incompatible shapes")
    if donor_normal.shape[0] == 0:
        raise ValueError("at least one donor is required")
    random_values = random_values.to(device=anchors.device, dtype=torch.float64)
    if random_values.shape != anchors.shape[:1]:
        raise ValueError("random_values must have shape [num_anchors]")
    if not bool(torch.isfinite(random_values).all()) or not bool(
        ((random_values >= 0.0) & (random_values < 1.0)).all()
    ):
        raise ValueError("random_values must be finite values in [0, 1)")
    dimensions = anchors.shape[1]
    distance2 = (
        anchors.square().sum(dim=1, keepdim=True)
        + donor_normal.square().sum(dim=1).unsqueeze(0)
        - 2.0 * anchors @ donor_normal.T
    ).clamp_min_(0.0) / dimensions
    effect2 = (donor_abnormal - donor_normal).square().mean(dim=1)
    pair_valid = (
        donor_valid.unsqueeze(0)
        & effect2.gt(0).unsqueeze(0)
        & (distance2 <= tau**2 * effect2.unsqueeze(0))
    )
    eligible_count = pair_valid.sum(dim=1)
    pool_size = eligible_count.clamp(max=min(top_m, donor_normal.shape[0]))
    found = pool_size.gt(0)
    cost = distance2.masked_fill(~pair_valid, float("inf"))
    # Stable sorting makes exact-distance ties deterministic by donor order.
    _sorted_cost, sorted_index = torch.sort(cost, dim=1, stable=True)
    top_index = sorted_index[:, :min(top_m, donor_normal.shape[0])]
    sampled_rank = torch.floor(random_values * pool_size.clamp_min(1).to(torch.float64)).long()
    sampled_rank = torch.minimum(sampled_rank, pool_size.clamp_min(1) - 1)
    selected_index = top_index.gather(1, sampled_rank.unsqueeze(1)).squeeze(1)
    selected_cost = distance2.gather(1, selected_index.unsqueeze(1)).squeeze(1)
    selected_effect = effect2[selected_index]
    ratio = torch.sqrt(
        selected_cost / selected_effect.clamp_min(torch.finfo(effect2.dtype).tiny)
    )
    ratio = ratio.masked_fill(~found, float("inf"))
    sampled_rank = sampled_rank.masked_fill(~found, -1)
    return selected_index, found, ratio, sampled_rank, eligible_count


class CounterfactualAXISDataset(AXISAnomalyQADataset):
    """Phase-II dataset with one coherent patched sequence per state pair."""

    def __init__(
        self,
        dataset_dir: str,
        counterfactual_index: str,
        *,
        split: str = "train",
        train_ratio: float = 0.95,
        seed: int = 72,
        cache_size: int = 1000,
        supervision_cache: str | None = None,
    ) -> None:
        super().__init__(dataset_dir, split=split, train_ratio=train_ratio, seed=seed, cache_size=cache_size)
        self.counterfactual_index_path = Path(counterfactual_index)
        payload = json.loads(self.counterfactual_index_path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != COUNTERFACTUAL_INDEX_VERSION:
            raise ValueError("counterfactual index is not the coherent-pair v3 format")
        if int(payload["seed"]) != seed or float(payload["train_ratio"]) != train_ratio:
            raise ValueError("counterfactual index split does not match dataset split")
        expected_files = [path.name for path in self.series_files]
        if payload.get("train_series") != expected_files:
            raise ValueError("counterfactual index train-series manifest does not match dataset")
        self.counterfactual_records = payload["records"]
        self.index_metadata = payload
        self.supervision_cache_path = Path(supervision_cache) if supervision_cache else None
        self.supervision_metadata: dict[str, Any] | None = None
        self.supervision_records: dict[str, dict[str, Any]] | None = None
        if self.supervision_cache_path is not None:
            supervision = json.loads(self.supervision_cache_path.read_text(encoding="utf-8"))
            if int(supervision.get("version", 0)) != SUPERVISION_CACHE_VERSION:
                raise ValueError("question-type supervision cache version mismatch")
            if int(supervision.get("seed", -1)) != seed:
                raise ValueError("question-type supervision seed mismatch")
            if float(supervision.get("train_ratio", -1.0)) != train_ratio:
                raise ValueError("question-type supervision split mismatch")
            if supervision.get("train_series") != expected_files:
                raise ValueError("question-type supervision series manifest mismatch")
            digest = hashlib.sha256(self.counterfactual_index_path.read_bytes()).hexdigest()
            if supervision.get("counterfactual_index_sha256") != digest:
                raise ValueError("question-type supervision was built for another counterfactual index")
            self.supervision_metadata = supervision
            self.supervision_records = supervision["records"]

    def _materialize_counterfactual(self, reference: dict, anchor_data: dict, anchor_window: dict) -> dict:
        start = int(anchor_window["window_range"]["start"])
        end = int(anchor_window["window_range"]["end"])
        anchor = torch.tensor(anchor_data["original_data"]["time_series"], dtype=torch.float32)
        anchor_values = anchor[start:end]
        state = int(bool(anchor_window["has_anomaly"]))
        valid = bool(reference.get("valid", False))
        kind = reference.get("kind", "invalid")
        counterfactual = anchor.clone()

        if not valid:
            values = anchor_values.clone()
            kind = "invalid"
        elif state == 1 and kind == "self_normal_patch":
            normal = torch.tensor(anchor_data["original_data"]["normal_series"], dtype=torch.float32)
            if normal.shape != anchor.shape:
                raise ValueError("paired normal series has a different length")
            values = normal[start:end].clone()
        elif state == 0 and kind == "residual_transplant":
            donor_data = self._load_series(self.series_dir / reference["donor_series_file"])
            donor_window = donor_data["windows"][int(reference["donor_window_index"])]
            if not bool(donor_window["has_anomaly"]):
                raise ValueError("residual donor is not anomalous")
            donor_start = int(donor_window["window_range"]["start"])
            donor_end = int(donor_window["window_range"]["end"])
            donor_current = torch.tensor(donor_data["original_data"]["time_series"], dtype=torch.float32)
            donor_normal = torch.tensor(donor_data["original_data"]["normal_series"], dtype=torch.float32)
            residual = donor_current[donor_start:donor_end] - donor_normal[donor_start:donor_end]
            if residual.numel() != anchor_values.numel():
                raise ValueError("residual donor length does not match anchor window")
            values = anchor_values + residual
        else:
            raise ValueError(f"counterfactual kind/state mismatch: {kind}/{state}")

        if valid:
            if values.numel() != end - start or not bool(torch.isfinite(values).all()):
                raise ValueError("counterfactual window is invalid")
            counterfactual[start:end] = values
        return {
            "valid": valid,
            "kind": kind,
            "series": counterfactual,
            "values": values,
            "target_state": 1 - state,
        }

    def __getitem__(self, idx: int) -> dict:
        file_path = self.series_files[idx]
        data = self._load_series(file_path)
        counterfactuals = []
        supervisions = []
        for window_index, window in enumerate(data["windows"]):
            key = f"{file_path.name}:{window_index}"
            reference = self.counterfactual_records.get(key, {"valid": False, "kind": "invalid"})
            if bool(reference.get("has_anomaly", window["has_anomaly"])) != bool(window["has_anomaly"]):
                raise ValueError(f"counterfactual label mismatch for {key}")
            counterfactuals.append(self._materialize_counterfactual(reference, data, window))
            if self.supervision_records is not None:
                if key not in self.supervision_records:
                    raise ValueError(f"question-type supervision is missing {key}")
                supervisions.append(self.supervision_records[key])
            else:
                supervisions.append({
                    "question_type": canonical_question_type(window.get("question_type", "unknown")),
                    "mc_pair_valid": False,
                    "tf_pair_valid": False,
                    "tf_target": None,
                    "oe_pair_valid": False,
                })
        return {
            "time_series": torch.tensor(data["original_data"]["time_series"], dtype=torch.float32),
            "analysis_data": data["windows"],
            "series_file": file_path.name,
            "counterfactuals": counterfactuals,
            "supervisions": supervisions,
        }


class CounterfactualQAView(Dataset):
    """QA-row view over a counterfactual series dataset.

    The original dataset samples whole series, so an auxiliary component is
    active only when a randomly selected series happens to contain an eligible
    row. This view indexes the high-precision supervision cache directly and
    lets the post-training runner construct deterministic MC/TF/OE queues while
    keeping the factual answer stream unchanged.
    """

    COMPONENT_FIELDS = {
        "mc": "mc_pair_valid",
        "tf": "tf_pair_valid",
        "oe": "oe_pair_valid",
    }

    def __init__(self, base: CounterfactualAXISDataset) -> None:
        if base.supervision_records is None or base.supervision_metadata is None:
            raise ValueError("CounterfactualQAView requires a supervision cache")
        self.base = base
        self.records: list[tuple[int, int, str]] = []
        self.positions_by_component: dict[str, list[int]] = {
            component: [] for component in self.COMPONENT_FIELDS
        }
        series_indices = {path.name: index for index, path in enumerate(base.series_files)}
        routed = []
        for record_id, supervision in base.supervision_records.items():
            try:
                series_name, window_text = record_id.rsplit(":", 1)
                series_index = series_indices[series_name]
                window_index = int(window_text)
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid supervision record id: {record_id}") from error
            active = [
                component
                for component, field in self.COMPONENT_FIELDS.items()
                if bool(supervision.get(field, False))
            ]
            if len(active) > 1:
                raise ValueError(f"auxiliary row is routed to multiple components: {record_id}")
            if active:
                routed.append((series_index, window_index, record_id, active[0]))

        for series_index, window_index, record_id, component in sorted(routed):
            position = len(self.records)
            self.records.append((series_index, window_index, record_id))
            self.positions_by_component[component].append(position)

        stats = base.supervision_metadata.get("stats", {})
        expected = {
            "mc": int(stats.get("valid/mc_pair_valid", -1)),
            "tf": int(stats.get("valid/tf_pair_valid", -1)),
            "oe": int(stats.get("valid/oe_pair_valid", -1)),
        }
        actual = {
            component: len(positions)
            for component, positions in self.positions_by_component.items()
        }
        if actual != expected:
            raise ValueError(f"QA view count mismatch: actual={actual}, expected={expected}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        series_index, window_index, record_id = self.records[index]
        item = self.base[series_index]
        if f'{item["series_file"]}:{window_index}' != record_id:
            raise RuntimeError("counterfactual QA index is not stable")
        return {
            "time_series": item["time_series"],
            "analysis_data": [item["analysis_data"][window_index]],
            "series_file": item["series_file"],
            "counterfactuals": [item["counterfactuals"][window_index]],
            "supervisions": [item["supervisions"][window_index]],
        }


def _pad_with_mask(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    padded = pad_sequence([sequence.to(torch.bfloat16) for sequence in sequences], batch_first=True, padding_value=0.0)
    mask = torch.zeros(padded.shape[:2], dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        mask[index, : sequence.numel()] = True
    return padded, mask


def counterfactual_collate_fn(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("empty counterfactual batch")
    positive_sequences: list[torch.Tensor] = []
    counterfactual_sequences: list[torch.Tensor] = []
    positive_window_values: list[torch.Tensor] = []
    counterfactual_window_values: list[torch.Tensor] = []
    questions, answers, starts, ends, question_types = [], [], [], [], []
    states, valid, record_ids, kinds = [], [], [], []
    mc_valid, tf_valid, tf_targets, oe_valid = [], [], [], []
    for item in batch:
        rows = zip(item["analysis_data"], item["counterfactuals"], item["supervisions"])
        for window_index, (window, counterfactual, supervision) in enumerate(rows):
            start = int(window["window_range"]["start"])
            end = int(window["window_range"]["end"])
            positive_sequences.append(item["time_series"])
            counterfactual_sequences.append(counterfactual["series"])
            positive_window_values.append(item["time_series"][start:end].float())
            counterfactual_window_values.append(counterfactual["values"].float())
            questions.append(window["question"])
            answers.append(window["answer"])
            starts.append(start)
            ends.append(end)
            question_type = canonical_question_type(window.get("question_type", "unknown"))
            if supervision.get("question_type") != question_type:
                raise ValueError("question-type supervision disagrees with source data")
            question_types.append(question_type)
            states.append(int(bool(window["has_anomaly"])))
            valid.append(bool(counterfactual["valid"]))
            record_ids.append(f'{item["series_file"]}:{window_index}')
            kinds.append(counterfactual["kind"])
            mc_valid.append(bool(supervision.get("mc_pair_valid", False)))
            tf_valid.append(bool(supervision.get("tf_pair_valid", False)))
            tf_target = supervision.get("tf_target")
            tf_targets.append(-1 if tf_target is None else int(tf_target))
            oe_valid.append(bool(supervision.get("oe_pair_valid", False)))
    positive_padded, positive_masks = _pad_with_mask(positive_sequences)
    counterfactual_padded, counterfactual_masks = _pad_with_mask(counterfactual_sequences)
    if positive_padded.shape != counterfactual_padded.shape:
        raise ValueError("counterfactual patch changed full-series shape")
    return {
        "padded_sequences": positive_padded,
        "attention_masks": positive_masks,
        "counterfactual_sequences": counterfactual_padded,
        "counterfactual_masks": counterfactual_masks,
        "positive_window_values": positive_window_values,
        "counterfactual_window_values": counterfactual_window_values,
        "questions": questions,
        "answers": answers,
        "start_indices": starts,
        "end_indices": ends,
        "question_types": question_types,
        "state_targets": torch.tensor(states, dtype=torch.long),
        "counterfactual_valid": torch.tensor(valid, dtype=torch.bool),
        "record_ids": record_ids,
        "counterfactual_kinds": kinds,
        "mc_pair_valid": torch.tensor(mc_valid, dtype=torch.bool),
        "tf_pair_valid": torch.tensor(tf_valid, dtype=torch.bool),
        "tf_targets": torch.tensor(tf_targets, dtype=torch.long),
        "oe_pair_valid": torch.tensor(oe_valid, dtype=torch.bool),
    }



def answer_collate_fn(batch: list[dict]) -> dict:
    """Collate factual Phase-II rows without materializing counterfactual data."""
    if not batch:
        raise ValueError("empty answer-only batch")
    sequences: list[torch.Tensor] = []
    questions, answers, starts, ends, question_types = [], [], [], [], []
    for item in batch:
        sequence = torch.as_tensor(item["time_series"], dtype=torch.float32)
        for window in item["analysis_data"]:
            sequences.append(sequence)
            questions.append(window["question"])
            answers.append(window["answer"])
            starts.append(int(window["window_range"]["start"]))
            ends.append(int(window["window_range"]["end"]))
            question_types.append(canonical_question_type(window.get("question_type", "unknown")))
    padded, masks = _pad_with_mask(sequences)
    return {
        "padded_sequences": padded,
        "attention_masks": masks,
        "questions": questions,
        "answers": answers,
        "start_indices": starts,
        "end_indices": ends,
        "question_types": question_types,
    }
