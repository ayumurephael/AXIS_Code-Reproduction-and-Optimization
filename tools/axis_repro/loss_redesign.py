from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from src.models.AXIS.dataset import AXISAnomalyQADataset


SLR_MODES = ("local_only", "window_only", "full_local", "full_window")


def _normalize_question_type(question_type: str) -> str:
    normalized = re.sub(r"[^a-z]+", "_", question_type.lower()).strip("_")
    aliases = {
        "multiplechoice": "multiple_choice",
        "openended": "open_ended",
        "truefalse": "true_false",
    }
    return aliases.get(normalized.replace("_", ""), normalized)


def _first_sentence_without_decimal_split(text: str) -> str:
    text = text.strip()
    for index, char in enumerate(text):
        if char != ".":
            continue
        previous_is_digit = index > 0 and text[index - 1].isdigit()
        next_is_digit = index + 1 < len(text) and text[index + 1].isdigit()
        if previous_is_digit and next_is_digit:
            continue
        return text[: index + 1].strip()
    return text


def extract_answer_head(answer: str, question_type: str) -> Optional[str]:
    """Extract the short answer head used by the source-likelihood ratio."""
    question_type = _normalize_question_type(question_type)
    answer = answer.strip()
    if question_type == "true_false":
        match = re.match(r"(?i)^(true|false)\b", answer)
        return match.group(1) if match else None

    head = _first_sentence_without_decimal_split(answer)
    if not head:
        return None
    if question_type == "open_ended":
        decision = re.search(r"(?i)\b(?:anomal(?:y|ies|ous)|abnormal|normal)\b", head)
        if decision is None:
            return None
    if question_type == "multiple_choice":
        return head
    if question_type == "open_ended":
        return head
    return None


def evidence_mode_spec(mode: str) -> tuple[bool, bool, str]:
    specs = {
        "local_only": (True, False, "local"),
        "window_only": (False, True, "window"),
        "full_local": (True, True, "local"),
        "full_window": (True, True, "window"),
    }
    try:
        return specs[mode]
    except KeyError as error:
        raise ValueError(f"unknown SLR mode: {mode}") from error


def build_answer_head_mask_from_offsets(
    answers: list[str],
    question_types: list[str],
    offset_mapping: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    answer_prefix: str = "Answer: ",
) -> tuple[torch.Tensor, list[Optional[str]]]:
    if len(answers) != len(question_types) or offset_mapping.shape[:2] != attention_mask.shape:
        raise ValueError("answer-head mask inputs have inconsistent batch shapes")
    result = torch.zeros(offset_mapping.shape[:2], dtype=torch.bool)
    heads: list[Optional[str]] = []
    for row, (answer, question_type) in enumerate(zip(answers, question_types)):
        head = extract_answer_head(answer, question_type)
        heads.append(head)
        if head is None:
            continue
        head_start = len(answer_prefix)
        head_end = head_start + len(head)
        starts = offset_mapping[row, :, 0]
        ends = offset_mapping[row, :, 1]
        overlaps = (ends > head_start) & (starts < head_end) & (ends > starts)
        result[row] = overlaps & attention_mask[row].bool().cpu()
    return result, heads


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


def compute_source_likelihood_ratio_loss(
    answer_loss: torch.Tensor,
    positive_head_nll: torch.Tensor,
    negative_head_nll: torch.Tensor,
    head_token_counts: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    beta: float,
    margin: float = math.log(2.0),
) -> tuple[torch.Tensor, dict[str, float | int]]:
    valid_rows = valid_rows.bool() & (head_token_counts > 0)
    safe_counts = head_token_counts.clamp_min(1).to(positive_head_nll.dtype)
    hinge = F.relu(margin + positive_head_nll - negative_head_nll)
    row_loss = (positive_head_nll + hinge) / safe_counts
    if bool(valid_rows.any()):
        slr_loss = row_loss[valid_rows].mean()
    else:
        slr_loss = answer_loss.new_zeros(())
    total = answer_loss + beta * slr_loss
    stats: dict[str, float | int] = {
        "answer_loss": float(answer_loss.detach()),
        "slr_loss": float(slr_loss.detach()),
        "valid_rows": int(valid_rows.sum().detach()),
        "active_hinges": int(((hinge > 0) & valid_rows).sum().detach()),
    }
    return total, stats


def memory_safe_token_nll_sums(
    lm_head,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    head_mask: torch.Tensor,
    *,
    chunk_size: int = 64,
    checkpoint_chunks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = hidden.shape[0]
    full_sums = hidden.new_zeros(batch_size, dtype=torch.float32)
    head_sums = hidden.new_zeros(batch_size, dtype=torch.float32)
    full_counts = torch.zeros(batch_size, dtype=torch.long, device=hidden.device)
    head_counts = torch.zeros(batch_size, dtype=torch.long, device=hidden.device)
    for start in range(0, hidden.shape[1], chunk_size):
        chunk_hidden = hidden[:, start:start + chunk_size]
        chunk_targets = targets[:, start:start + chunk_size]
        chunk_head_mask = head_mask[:, start:start + chunk_size].bool()

        def token_losses(current_hidden):
            logits = lm_head(current_hidden).float()
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                chunk_targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(chunk_targets.shape)

        if checkpoint_chunks and torch.is_grad_enabled():
            nll = checkpoint(token_losses, chunk_hidden, use_reentrant=False)
        else:
            nll = token_losses(chunk_hidden)
        valid = chunk_targets.ne(-100)
        selected_head = valid & chunk_head_mask
        full_sums = full_sums + (nll * valid).sum(dim=1)
        head_sums = head_sums + (nll * selected_head).sum(dim=1)
        full_counts = full_counts + valid.sum(dim=1)
        head_counts = head_counts + selected_head.sum(dim=1)
    return full_sums, full_counts, head_sums, head_counts


@torch.no_grad()
def retrieve_exact(
    anchor: torch.Tensor,
    donor_normal: torch.Tensor,
    donor_abnormal: torch.Tensor,
    control_gap: torch.Tensor,
    *,
    gamma: float,
    tau: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor = anchor.float()
    donor_normal = donor_normal.float()
    donor_abnormal = donor_abnormal.float()
    control_gap = control_gap.float()
    dimensions = anchor.shape[1]
    d2 = (
        anchor.square().sum(dim=1, keepdim=True)
        + donor_normal.square().sum(dim=1).unsqueeze(0)
        - 2.0 * anchor @ donor_normal.T
    ).clamp_min_(0.0) / dimensions
    a2 = (donor_abnormal - donor_normal).square().mean(dim=1)
    donor_valid = (a2 > gamma**2) & (control_gap.square() <= tau**2 * a2)
    pair_valid = donor_valid.unsqueeze(0) & (d2 <= tau**2 * a2.unsqueeze(0))
    cost = d2.masked_fill(~pair_valid, float("inf"))
    best_cost, best_index = cost.min(dim=1)
    return best_index, torch.isfinite(best_cost)



class CounterfactualAXISDataset(AXISAnomalyQADataset):
    """Phase-II dataset that materializes source-specific counterfactuals."""

    def __init__(self, dataset_dir: str, counterfactual_index: str, *, split: str = "train", train_ratio: float = 0.95, seed: int = 72, cache_size: int = 1000) -> None:
        super().__init__(dataset_dir, split=split, train_ratio=train_ratio, seed=seed, cache_size=cache_size)
        self.counterfactual_index_path = Path(counterfactual_index)
        payload = json.loads(self.counterfactual_index_path.read_text(encoding="utf-8"))
        if int(payload["seed"]) != seed or float(payload["train_ratio"]) != train_ratio:
            raise ValueError("counterfactual index split does not match dataset split")
        self.counterfactual_records = payload["records"]

    def _materialize_reference(self, reference: dict, anchor_data: dict, anchor_window: dict) -> dict:
        anchor_start = int(anchor_window["window_range"]["start"])
        anchor_end = int(anchor_window["window_range"]["end"])
        anchor_time = torch.tensor(anchor_data["original_data"]["time_series"], dtype=torch.float32)
        valid = bool(reference.get("valid", False))
        kind = reference.get("kind", "invalid")
        if valid and kind == "self_normal":
            source = torch.tensor(anchor_data["original_data"]["normal_series"], dtype=torch.float32)
            source_start, source_end = anchor_start, anchor_end
        elif valid and kind == "donor":
            donor_data = self._load_series(self.series_dir / reference["series_file"])
            donor_window = donor_data["windows"][int(reference["window_index"])]
            source_start = int(donor_window["window_range"]["start"])
            source_end = int(donor_window["window_range"]["end"])
            source = torch.tensor(donor_data["original_data"]["time_series"], dtype=torch.float32)
        else:
            source = anchor_time
            source_start, source_end = anchor_start, anchor_end
            valid = False
        return {"valid": valid, "series": source, "start": source_start, "end": source_end, "values": source[source_start:source_end].clone()}

    def __getitem__(self, idx: int) -> dict:
        file_path = self.series_files[idx]
        data = self._load_series(file_path)
        counterfactuals = []
        for window_index, window in enumerate(data["windows"]):
            record = self.counterfactual_records.get(f"{file_path.name}:{window_index}", {})
            counterfactuals.append({
                "window": self._materialize_reference(record.get("window", {}), data, window),
                "local": self._materialize_reference(record.get("local", {}), data, window),
            })
        return {
            "time_series": torch.tensor(data["original_data"]["time_series"], dtype=torch.float32),
            "normal_series": torch.tensor(data["original_data"]["normal_series"], dtype=torch.float32),
            "analysis_data": data["windows"],
            "series_file": file_path.name,
            "counterfactuals": counterfactuals,
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
    positive_sequences, negative_local_sequences, negative_window_values = [], [], []
    questions, answers, starts, ends, question_types = [], [], [], [], []
    negative_starts, negative_ends, local_valid, window_valid, record_ids = [], [], [], [], []
    for item in batch:
        for window_index, (window, counterfactual) in enumerate(zip(item["analysis_data"], item["counterfactuals"])):
            positive_sequences.append(item["time_series"])
            questions.append(window["question"])
            answers.append(window["answer"])
            starts.append(int(window["window_range"]["start"]))
            ends.append(int(window["window_range"]["end"]))
            question_types.append(window.get("question_type", "unknown"))
            local = counterfactual["local"]
            window_source = counterfactual["window"]
            negative_local_sequences.append(local["series"])
            negative_starts.append(int(local["start"]))
            negative_ends.append(int(local["end"]))
            negative_window_values.append(window_source["values"].float())
            local_valid.append(bool(local["valid"]))
            window_valid.append(bool(window_source["valid"]))
            record_ids.append(f'{item["series_file"]}:{window_index}')
    padded, attention_masks = _pad_with_mask(positive_sequences)
    negative_padded, negative_masks = _pad_with_mask(negative_local_sequences)
    return {
        "padded_sequences": padded, "attention_masks": attention_masks,
        "questions": questions, "answers": answers,
        "start_indices": starts, "end_indices": ends, "question_types": question_types,
        "negative_local_sequences": negative_padded, "negative_local_masks": negative_masks,
        "negative_local_start_indices": negative_starts, "negative_local_end_indices": negative_ends,
        "negative_window_values": negative_window_values,
        "local_source_valid": torch.tensor(local_valid, dtype=torch.bool),
        "window_source_valid": torch.tensor(window_valid, dtype=torch.bool),
        "record_ids": record_ids,
    }
