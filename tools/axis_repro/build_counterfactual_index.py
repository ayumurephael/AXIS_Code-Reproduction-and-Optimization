"""Build coherent full-sequence counterfactual pairs for explicit state supervision."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.configs.axis_config import default_config
from src.models.AXIS.Pretrain_ts_encoder import TimeSeriesPretrainModel
from .loss_redesign import COUNTERFACTUAL_INDEX_VERSION, retrieve_consistent_donors
from .model_utils import sha256_file


def _rms(values: torch.Tensor) -> float:
    return float(values.float().square().mean().sqrt())


def _formatted(values: np.ndarray) -> torch.Tensor:
    return torch.round(torch.from_numpy(values.astype(np.float32, copy=False)) * 100.0).float()


def _percentile_nonzero(values: list[float], percentile: float, name: str) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value) and value > 0.0], dtype=np.float64)
    if finite.size == 0:
        raise RuntimeError(f"no nonzero finite {name} effects for gamma selection")
    return float(np.percentile(finite, percentile, method="linear"))


def _load_encoder(phase1: str, device: torch.device) -> TimeSeriesPretrainModel:
    model = TimeSeriesPretrainModel(default_config)
    payload = torch.load(phase1, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state.get("ts_pretrain_model", state), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _encode_sequences(
    encoder: TimeSeriesPretrainModel,
    sequences: list[np.ndarray],
    device: torch.device,
) -> list[torch.Tensor]:
    if not sequences:
        return []
    max_length = max(len(sequence) for sequence in sequences)
    values = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    masks = torch.zeros((len(sequences), max_length), dtype=torch.bool, device=device)
    lengths = []
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        lengths.append(length)
        values[index, :length] = torch.from_numpy(sequence).to(device)
        masks[index, :length] = True
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = encoder(values, mask=masks)
    embeddings = embeddings.to(torch.float16).cpu()
    return [embeddings[index, :length].clone() for index, length in enumerate(lengths)]


def _patched(sequence: np.ndarray, start: int, end: int, values: np.ndarray) -> np.ndarray:
    if end - start != len(values):
        raise ValueError("patch length mismatch")
    result = sequence.copy()
    result[start:end] = values
    return result


def _invalid_reference(has_anomaly: bool, reason: str) -> dict:
    return {
        "has_anomaly": bool(has_anomaly),
        "valid": False,
        "kind": "invalid",
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/anomaly_llava_training_dataset")
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--gamma-percentile", type=float, default=5.0)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--anchor-chunk-size", type=int, default=128)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--max-series", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("counterfactual Local indexing requires CUDA")
    if not 0.0 <= args.gamma_percentile <= 100.0:
        raise ValueError("gamma percentile must be in [0, 100]")
    if args.tau <= 0.0:
        raise ValueError("tau must be positive")

    started = time.perf_counter()
    data_root = Path(args.data)
    series_dir = data_root / "series"
    files = sorted(series_dir.glob("series_*.json"))
    random.Random(args.seed).shuffle(files)
    files = files[: int(len(files) * args.train_ratio)]
    if args.max_series is not None:
        files = files[:args.max_series]

    payloads: list[dict] = []
    records: list[dict] = []
    records_by_series: dict[int, list[int]] = defaultdict(list)
    for file_index, path in enumerate(files):
        payload = json.loads(path.read_text(encoding="utf-8"))
        current = np.asarray(payload["original_data"]["time_series"], dtype=np.float32)
        normal = np.asarray(payload["original_data"]["normal_series"], dtype=np.float32)
        if current.shape != normal.shape or current.ndim != 1:
            raise ValueError(f"paired univariate series mismatch: {path.name}")
        item = {
            "series_file": path.name,
            "current": current,
            "normal": normal,
            "windows": payload["windows"],
        }
        payloads.append(item)
        for window_index, window in enumerate(payload["windows"]):
            start = int(window["window_range"]["start"])
            end = int(window["window_range"]["end"])
            if not 0 <= start < end <= len(current):
                raise ValueError(f"invalid window in {path.name}:{window_index}")
            record = {
                "key": f"{path.name}:{window_index}",
                "series_file": path.name,
                "series_index": file_index,
                "window_index": window_index,
                "start": start,
                "end": end,
                "length": end - start,
                "has_anomaly": bool(window["has_anomaly"]),
                "window_current": _formatted(current[start:end]),
                "window_normal": _formatted(normal[start:end]),
                "local_current": None,
                "local_normal_full": None,
            }
            records.append(record)
            records_by_series[file_index].append(len(records) - 1)
        if (file_index + 1) % 2000 == 0:
            print(json.dumps({"stage": "load", "series": file_index + 1}), flush=True)

    device = torch.device("cuda", args.device_index)
    encoder = _load_encoder(args.phase1, device)

    for batch_start in range(0, len(payloads), args.encode_batch_size):
        batch = payloads[batch_start:batch_start + args.encode_batch_size]
        embeddings = _encode_sequences(
            encoder,
            [item["current"] for item in batch] + [item["normal"] for item in batch],
            device,
        )
        for local_index, _item in enumerate(batch):
            series_index = batch_start + local_index
            current_embedding = embeddings[local_index]
            normal_embedding = embeddings[local_index + len(batch)]
            for record_index in records_by_series[series_index]:
                record = records[record_index]
                start, end = record["start"], record["end"]
                record["local_current"] = current_embedding[start:end].clone()
                record["local_normal_full"] = normal_embedding[start:end].clone()
        if (batch_start // args.encode_batch_size + 1) % 50 == 0:
            print(json.dumps({
                "stage": "encode_original_and_normal",
                "series": min(len(payloads), batch_start + len(batch)),
                "elapsed_minutes": (time.perf_counter() - started) / 60,
            }), flush=True)

    controls: dict[int, dict[str, float]] = {}
    for series_index, record_indices in records_by_series.items():
        normal_indices = [index for index in record_indices if not records[index]["has_anomaly"]]
        if not normal_indices:
            controls[series_index] = {"window": math.inf, "local": math.inf, "count": 0}
            continue
        controls[series_index] = {
            "window": max(_rms(records[index]["window_current"] - records[index]["window_normal"]) for index in normal_indices),
            "local": max(_rms(records[index]["local_current"] - records[index]["local_normal_full"]) for index in normal_indices),
            "count": len(normal_indices),
        }

    abnormal_indices = [index for index, record in enumerate(records) if record["has_anomaly"]]
    for batch_start in range(0, len(abnormal_indices), args.encode_batch_size):
        chunk = abnormal_indices[batch_start:batch_start + args.encode_batch_size]
        patched_sequences = []
        for index in chunk:
            record = records[index]
            item = payloads[record["series_index"]]
            patched_sequences.append(_patched(
                item["current"],
                record["start"],
                record["end"],
                item["normal"][record["start"]:record["end"]],
            ))
        embeddings = _encode_sequences(encoder, patched_sequences, device)
        for index, embedding in zip(chunk, embeddings):
            record = records[index]
            start, end = record["start"], record["end"]
            record["window_effect"] = _rms(record["window_current"] - record["window_normal"])
            record["local_effect"] = _rms(record["local_current"] - embedding[start:end])
        if (batch_start // args.encode_batch_size + 1) % 50 == 0:
            print(json.dumps({
                "stage": "encode_anomaly_deletions",
                "windows": min(len(abnormal_indices), batch_start + len(chunk)),
                "elapsed_minutes": (time.perf_counter() - started) / 60,
            }), flush=True)

    gamma_window = _percentile_nonzero(
        [records[index]["window_effect"] for index in abnormal_indices],
        args.gamma_percentile,
        "Window",
    )
    gamma_local = _percentile_nonzero(
        [records[index]["local_effect"] for index in abnormal_indices],
        args.gamma_percentile,
        "Local",
    )

    donor_valid: dict[int, bool] = {}
    donor_reason: dict[int, str] = {}
    for index in abnormal_indices:
        record = records[index]
        control = controls[record["series_index"]]
        checks = {
            "normal_control_present": control["count"] > 0,
            "window_effect": record["window_effect"] >= gamma_window,
            "local_effect": record["local_effect"] >= gamma_local,
            "window_control_gap": control["window"] <= args.tau * record["window_effect"],
            "local_control_gap": control["local"] <= args.tau * record["local_effect"],
        }
        donor_valid[index] = all(checks.values())
        donor_reason[index] = "ok" if donor_valid[index] else ",".join(name for name, passed in checks.items() if not passed)
        record["control_window"] = control["window"]
        record["control_local"] = control["local"]

    output_records = {
        record["key"]: _invalid_reference(record["has_anomaly"], "not_processed")
        for record in records
    }
    for index in abnormal_indices:
        record = records[index]
        if donor_valid[index]:
            output_records[record["key"]] = {
                "has_anomaly": True,
                "valid": True,
                "kind": "self_normal_patch",
                "target_state": 0,
                "window_effect": record["window_effect"],
                "local_effect": record["local_effect"],
                "control_window": record["control_window"],
                "control_local": record["control_local"],
            }
        else:
            output_records[record["key"]] = _invalid_reference(True, donor_reason[index])

    normal_indices = [index for index, record in enumerate(records) if not record["has_anomaly"]]
    donors_by_length: dict[int, list[int]] = defaultdict(list)
    anchors_by_length: dict[int, list[int]] = defaultdict(list)
    for index in abnormal_indices:
        if donor_valid[index]:
            donors_by_length[records[index]["length"]].append(index)
    for index in normal_indices:
        anchors_by_length[records[index]["length"]].append(index)

    matches: dict[int, int] = {}
    match_ratios: dict[int, float] = {}
    for length, anchors in anchors_by_length.items():
        donors = donors_by_length[length]
        if not donors:
            continue
        donor_normal = torch.stack([records[index]["window_normal"] for index in donors]).to(device)
        donor_abnormal = torch.stack([records[index]["window_current"] for index in donors]).to(device)
        valid_tensor = torch.ones(len(donors), dtype=torch.bool, device=device)
        for start in range(0, len(anchors), args.anchor_chunk_size):
            chunk = anchors[start:start + args.anchor_chunk_size]
            anchor_values = torch.stack([records[index]["window_current"] for index in chunk]).to(device)
            best, found, ratios = retrieve_consistent_donors(
                anchor_values,
                donor_normal,
                donor_abnormal,
                valid_tensor,
                tau=args.tau,
            )
            for row, anchor_index in enumerate(chunk):
                if bool(found[row]):
                    matches[anchor_index] = donors[int(best[row])]
                    match_ratios[anchor_index] = float(ratios[row])

    matched_normal_indices = sorted(matches)
    accepted_normal = 0
    rejected_post_patch = 0
    for batch_start in range(0, len(matched_normal_indices), args.encode_batch_size):
        chunk = matched_normal_indices[batch_start:batch_start + args.encode_batch_size]
        patched_sequences = []
        patched_windows = []
        for anchor_index in chunk:
            donor_index = matches[anchor_index]
            anchor = records[anchor_index]
            donor = records[donor_index]
            anchor_item = payloads[anchor["series_index"]]
            donor_item = payloads[donor["series_index"]]
            residual = (
                donor_item["current"][donor["start"]:donor["end"]]
                - donor_item["normal"][donor["start"]:donor["end"]]
            )
            window = anchor_item["current"][anchor["start"]:anchor["end"]] + residual
            patched_windows.append(window)
            patched_sequences.append(_patched(
                anchor_item["current"],
                anchor["start"],
                anchor["end"],
                window,
            ))
        embeddings = _encode_sequences(encoder, patched_sequences, device)
        for anchor_index, window, embedding in zip(chunk, patched_windows, embeddings):
            donor_index = matches[anchor_index]
            anchor = records[anchor_index]
            start, end = anchor["start"], anchor["end"]
            window_effect = _rms(_formatted(window) - anchor["window_current"])
            local_effect = _rms(embedding[start:end] - anchor["local_current"])
            if window_effect >= gamma_window and local_effect >= gamma_local:
                donor = records[donor_index]
                output_records[anchor["key"]] = {
                    "has_anomaly": False,
                    "valid": True,
                    "kind": "residual_transplant",
                    "target_state": 1,
                    "donor_series_file": donor["series_file"],
                    "donor_window_index": donor["window_index"],
                    "donor_key": donor["key"],
                    "match_ratio": match_ratios[anchor_index],
                    "window_effect": window_effect,
                    "local_effect": local_effect,
                }
                accepted_normal += 1
            else:
                output_records[anchor["key"]] = _invalid_reference(False, "post_patch_effect_below_gamma")
                rejected_post_patch += 1
        if (batch_start // args.encode_batch_size + 1) % 50 == 0:
            print(json.dumps({
                "stage": "validate_residual_transplants",
                "windows": min(len(matched_normal_indices), batch_start + len(chunk)),
                "elapsed_minutes": (time.perf_counter() - started) / 60,
            }), flush=True)

    del encoder
    torch.cuda.empty_cache()
    valid_abnormal = sum(donor_valid.values())
    valid_total = valid_abnormal + accepted_normal
    stats = {
        "series": len(payloads),
        "qa": len(records),
        "normal_anchors": len(normal_indices),
        "anomalous_anchors": len(abnormal_indices),
        "eligible_anomalous_donors": valid_abnormal,
        "valid_anomaly_deletions": valid_abnormal,
        "normal_matches_before_post_patch": len(matches),
        "valid_residual_transplants": accepted_normal,
        "rejected_after_post_patch": rejected_post_patch,
        "valid_pairs": valid_total,
        "valid_pair_rate": valid_total / max(1, len(records)),
        "normal_pair_rate": accepted_normal / max(1, len(normal_indices)),
        "anomalous_pair_rate": valid_abnormal / max(1, len(abnormal_indices)),
        "gamma_window": gamma_window,
        "gamma_local": gamma_local,
        "wall_seconds": time.perf_counter() - started,
    }
    output = {
        "version": COUNTERFACTUAL_INDEX_VERSION,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "phase1_sha256": sha256_file(args.phase1),
        "data_root": str(data_root.resolve()),
        "train_series": [item["series_file"] for item in payloads],
        "policy": {
            "name": "coherent_full_sequence_patch_with_explicit_state_targets",
            "tau": args.tau,
            "gamma_percentile": args.gamma_percentile,
            "gamma_window": gamma_window,
            "gamma_local": gamma_local,
            "window_distance_space": "round(value*100)",
            "local_effect_space": "frozen_phase1_encoder",
            "normal_anchor_method": "paired_anomaly_residual_transplant",
            "anomalous_anchor_method": "anchor_coordinate_self_normal_patch",
            "control_gap": "max_paired_gap_over_normal_windows",
            "same_length_required": True,
            "phase_fallback": False,
            "post_patch_dual_source_validation": True,
        },
        "records": output_records,
        "stats": stats,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"wrote": str(destination), "stats": stats}), flush=True)


if __name__ == "__main__":
    main()
