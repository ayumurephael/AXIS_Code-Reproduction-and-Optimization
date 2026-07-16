"""Build exact Window/Local counterfactual donor indices for SLR training."""
from __future__ import annotations

import argparse
import hashlib
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
from .loss_redesign import retrieve_exact
from .model_utils import sha256_file


def _rms(values: torch.Tensor) -> float:
    return float(values.float().square().mean().sqrt())


def _load_encoder(phase1: str, device: torch.device) -> TimeSeriesPretrainModel:
    model = TimeSeriesPretrainModel(default_config)
    payload = torch.load(phase1, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state.get("ts_pretrain_model", state), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _invalid_reference() -> dict:
    return {"valid": False, "kind": "invalid"}


def _self_normal_reference(valid: bool) -> dict:
    return {"valid": bool(valid), "kind": "self_normal" if valid else "invalid"}


def _donor_reference(record: dict) -> dict:
    return {
        "valid": True,
        "kind": "donor",
        "series_file": record["series_file"],
        "window_index": record["window_index"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/anomaly_llava_training_dataset")
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--gamma-window", type=float, default=0.0)
    parser.add_argument("--gamma-local", type=float, default=0.0)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--anchor-chunk-size", type=int, default=128)
    parser.add_argument("--max-series", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("counterfactual Local indexing requires CUDA")

    started = time.perf_counter()
    data_root = Path(args.data)
    series_dir = data_root / "series"
    files = sorted(series_dir.glob("series_*.json"))
    random.Random(args.seed).shuffle(files)
    files = files[: int(len(files) * args.train_ratio)]
    if args.max_series is not None:
        files = files[: args.max_series]

    payloads = []
    records = []
    records_by_series = defaultdict(list)
    for file_index, path in enumerate(files):
        payload = json.loads(path.read_text(encoding="utf-8"))
        current = np.asarray(payload["original_data"]["time_series"], dtype=np.float32)
        normal = np.asarray(payload["original_data"]["normal_series"], dtype=np.float32)
        if current.shape != normal.shape:
            raise ValueError(f"paired series length mismatch: {path.name}")
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
            if end <= start:
                raise ValueError(f"invalid window in {path.name}:{window_index}")
            current_window = torch.from_numpy(current[start:end].copy())
            normal_window = torch.from_numpy(normal[start:end].copy())
            record = {
                "key": f"{path.name}:{window_index}",
                "series_file": path.name,
                "series_index": file_index,
                "window_index": window_index,
                "start": start,
                "end": end,
                "length": end - start,
                "phase": start % default_config.ts_config.patch_size,
                "has_anomaly": bool(window["has_anomaly"]),
                "raw_changed": bool(torch.any(current_window.ne(normal_window))),
                "window_abnormal": torch.round(current_window * 100),
                "window_normal": torch.round(normal_window * 100),
                "local_abnormal": None,
                "local_normal": None,
            }
            records.append(record)
            records_by_series[file_index].append(len(records) - 1)
        if (file_index + 1) % 2000 == 0:
            print(json.dumps({"stage": "load", "series": file_index + 1}), flush=True)

    device = torch.device("cuda", 0)
    encoder = _load_encoder(args.phase1, device)
    for batch_start in range(0, len(payloads), args.encode_batch_size):
        batch = payloads[batch_start: batch_start + args.encode_batch_size]
        max_length = max(len(item["current"]) for item in batch)
        combined = torch.zeros((2 * len(batch), max_length), dtype=torch.float32, device=device)
        mask = torch.zeros((2 * len(batch), max_length), dtype=torch.bool, device=device)
        for index, item in enumerate(batch):
            length = len(item["current"])
            combined[index, :length] = torch.from_numpy(item["current"]).to(device)
            combined[index + len(batch), :length] = torch.from_numpy(item["normal"]).to(device)
            mask[index, :length] = True
            mask[index + len(batch), :length] = True
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = encoder(combined, mask=mask)
        embeddings = embeddings.to(torch.float16).cpu()
        for local_index, item in enumerate(batch):
            series_index = batch_start + local_index
            for record_index in records_by_series[series_index]:
                record = records[record_index]
                start, end = record["start"], record["end"]
                record["local_abnormal"] = embeddings[local_index, start:end].reshape(-1).clone()
                record["local_normal"] = embeddings[local_index + len(batch), start:end].reshape(-1).clone()
        if (batch_start // args.encode_batch_size + 1) % 50 == 0:
            print(json.dumps({
                "stage": "encode",
                "series": min(len(payloads), batch_start + len(batch)),
                "elapsed_minutes": (time.perf_counter() - started) / 60,
            }), flush=True)
    del encoder
    torch.cuda.empty_cache()

    control_record = {}
    for series_index, record_indices in records_by_series.items():
        controls = [index for index in record_indices if not records[index]["has_anomaly"]]
        for index in record_indices:
            if records[index]["has_anomaly"]:
                control_record[index] = controls[0] if controls else None

    effects = {"window": {}, "local": {}}
    controls = {"window": {}, "local": {}}
    for index, record in enumerate(records):
        if not record["has_anomaly"]:
            continue
        effects["window"][index] = _rms(record["window_abnormal"] - record["window_normal"])
        effects["local"][index] = _rms(record["local_abnormal"] - record["local_normal"])
        control_index = control_record[index]
        if control_index is None:
            controls["window"][index] = math.inf
            controls["local"][index] = math.inf
        else:
            control = records[control_index]
            controls["window"][index] = _rms(control["window_abnormal"] - control["window_normal"])
            controls["local"][index] = _rms(control["local_abnormal"] - control["local_normal"])

    output_records = {
        record["key"]: {
            "has_anomaly": record["has_anomaly"],
            "window": _invalid_reference(),
            "local": _invalid_reference(),
        }
        for record in records
    }
    for index, record in enumerate(records):
        if not record["has_anomaly"]:
            continue
        window_valid = effects["window"][index] > args.gamma_window
        local_valid = record["raw_changed"] and effects["local"][index] > args.gamma_local
        output_records[record["key"]]["window"] = _self_normal_reference(window_valid)
        output_records[record["key"]]["local"] = _self_normal_reference(local_valid)

    normal_indices = [index for index, record in enumerate(records) if not record["has_anomaly"]]
    abnormal_indices = [index for index, record in enumerate(records) if record["has_anomaly"]]
    match_stats = {}

    def representation(record: dict, source: str, abnormal: bool) -> torch.Tensor:
        name = f"{source}_{'abnormal' if abnormal else 'normal'}"
        return record[name].float()

    def eligible_donors(source: str) -> list[int]:
        gamma = args.gamma_window if source == "window" else args.gamma_local
        result = []
        for index in abnormal_indices:
            effect = effects[source][index]
            if effect <= gamma or controls[source][index] > args.tau * effect:
                continue
            if source == "local" and not records[index]["raw_changed"]:
                continue
            result.append(index)
        return result

    def match_group(anchor_indices: list[int], donor_indices: list[int], source: str) -> dict[int, int]:
        if not anchor_indices or not donor_indices:
            return {}
        donor_normal = torch.stack([representation(records[index], source, False) for index in donor_indices]).to(device)
        donor_abnormal = torch.stack([representation(records[index], source, True) for index in donor_indices]).to(device)
        control_gap = torch.tensor([controls[source][index] for index in donor_indices], device=device)
        gamma = args.gamma_window if source == "window" else args.gamma_local
        matched = {}
        for start in range(0, len(anchor_indices), args.anchor_chunk_size):
            chunk_indices = anchor_indices[start:start + args.anchor_chunk_size]
            anchor = torch.stack([
                representation(records[index], source, True) for index in chunk_indices
            ]).to(device)
            best, found = retrieve_exact(
                anchor,
                donor_normal,
                donor_abnormal,
                control_gap,
                gamma=gamma,
                tau=args.tau,
            )
            for row, anchor_index in enumerate(chunk_indices):
                if bool(found[row]):
                    matched[anchor_index] = donor_indices[int(best[row])]
        return matched

    for source in ("window", "local"):
        donors = eligible_donors(source)
        donors_by_length = defaultdict(list)
        donors_by_phase = defaultdict(list)
        anchors_by_length = defaultdict(list)
        anchors_by_phase = defaultdict(list)
        for index in donors:
            record = records[index]
            donors_by_length[record["length"]].append(index)
            donors_by_phase[(record["length"], record["phase"])].append(index)
        for index in normal_indices:
            record = records[index]
            anchors_by_length[record["length"]].append(index)
            anchors_by_phase[(record["length"], record["phase"])].append(index)

        matches = {}
        phase_fallback = 0
        if source == "local":
            for key, anchors in anchors_by_phase.items():
                matches.update(match_group(anchors, donors_by_phase[key], source))
            missing_by_length = defaultdict(list)
            for index in normal_indices:
                if index not in matches:
                    missing_by_length[records[index]["length"]].append(index)
            for length, anchors in missing_by_length.items():
                fallback_matches = match_group(anchors, donors_by_length[length], source)
                phase_fallback += len(fallback_matches)
                matches.update(fallback_matches)
        else:
            for length, anchors in anchors_by_length.items():
                matches.update(match_group(anchors, donors_by_length[length], source))

        for anchor_index, donor_index in matches.items():
            output_records[records[anchor_index]["key"]][source] = _donor_reference(records[donor_index])
        anomaly_valid = sum(
            output_records[records[index]["key"]][source]["valid"] for index in abnormal_indices
        )
        match_stats[source] = {
            "eligible_donors": len(donors),
            "normal_anchors": len(normal_indices),
            "normal_matches": len(matches),
            "normal_match_rate": len(matches) / max(1, len(normal_indices)),
            "abnormal_anchors": len(abnormal_indices),
            "abnormal_self_normal_valid": anomaly_valid,
            "abnormal_self_normal_valid_rate": anomaly_valid / max(1, len(abnormal_indices)),
            "phase_fallback_matches": phase_fallback,
        }
        print(json.dumps({"stage": "retrieve", "source": source, **match_stats[source]}), flush=True)

    output = {
        "version": 1,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "tau": args.tau,
        "gamma_window": args.gamma_window,
        "gamma_local": args.gamma_local,
        "phase1_sha256": sha256_file(args.phase1),
        "data_root": str(data_root.resolve()),
        "train_series": [item["series_file"] for item in payloads],
        "records": output_records,
        "stats": {
            "series": len(payloads),
            "qa": len(records),
            "window": match_stats["window"],
            "local": match_stats["local"],
            "wall_seconds": time.perf_counter() - started,
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"wrote": str(destination), "stats": output["stats"]}), flush=True)


if __name__ == "__main__":
    main()
