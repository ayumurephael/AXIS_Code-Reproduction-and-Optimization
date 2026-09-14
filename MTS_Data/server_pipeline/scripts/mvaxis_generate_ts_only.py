from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.legacy_adapter import convert_legacy_sample
from src.mvaxis.legacy_tsad_generator import load_legacy_tsad_generate_dataset
from src.mvaxis.progress import ProgressPrinter
from src.mvaxis.utils import save_json, write_jsonl


def _ts_only_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only raw multivariate time-series fields and anomaly metadata."""

    series = row.get("series") or {}
    values = series.get("values") or []
    return {
        "schema_version": row.get("schema_version"),
        "sample_id": row.get("sample_id"),
        "base_sample_id": row.get("base_sample_id"),
        "source": row.get("source"),
        "original_data": row.get("original_data"),
        "normal_series": row.get("normal_series"),
        "series": row.get("series"),
        "channels": row.get("channels"),
        "causal_graph": row.get("causal_graph"),
        "root_cause": row.get("root_cause"),
        "synthetic_label": row.get("synthetic_label"),
        "sequence_length": int(len(values)),
    }


def _with_order_metadata(row: Dict[str, Any], *, global_index: int, shard_index: int | None, index_in_shard: int | None) -> Dict[str, Any]:
    """Attach stable global/shard ordering metadata without changing sample identity."""

    out = dict(row)
    out["global_sample_index"] = int(global_index)
    if shard_index is not None:
        out["shard_index"] = int(shard_index)
    if index_in_shard is not None:
        out["index_in_shard"] = int(index_in_shard)
    return out


def _iter_with_order(rows: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for idx, row in enumerate(rows):
        yield _with_order_metadata(row, global_index=idx, shard_index=None, index_in_shard=None)


def _write_sharded_jsonl(
    rows: List[Dict[str, Any]],
    *,
    shards_dir: Path,
    shard_size: int,
    prefix: str = "raw_series",
) -> List[Dict[str, Any]]:
    """Write one dataset into fixed-size JSONL shards and return a manifest list."""

    shards_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    if shard_size <= 0:
        return manifest

    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        end = min(start + shard_size, len(rows))
        shard_rows = [
            _with_order_metadata(
                rows[global_index],
                global_index=global_index,
                shard_index=shard_index,
                index_in_shard=global_index - start,
            )
            for global_index in range(start, end)
        ]
        shard_path = shards_dir / f"{prefix}_{shard_index:04d}.jsonl"
        write_jsonl(shard_rows, shard_path)
        manifest.append(
            {
                "shard_index": int(shard_index),
                "path": str(shard_path),
                "start_global_index": int(start),
                "end_global_index_exclusive": int(end),
                "num_rows": int(end - start),
            }
        )
    return manifest


def _summary(
    *,
    output_dir: Path,
    raw_path: Path,
    num_samples: int,
    seq_len: int | None,
    num_features: int | None,
    seed: int,
    anomaly_ratio: float,
    anomalous_series_count: int,
    normal_series_count: int,
    example_keys: Sequence[str] | None = None,
    example_sample_id: str | None = None,
    sampled_seq_lens: Sequence[int] | None = None,
    sampled_feature_counts: Sequence[int] | None = None,
    shard_size: int = 0,
    shard_manifest: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    seq_lens = list(sampled_seq_lens or [])
    if not seq_lens and seq_len is not None:
        seq_lens = [int(seq_len)] * int(num_samples)
    seq_len_summary: Dict[str, Any] = {
        "mode": "fixed" if seq_len is not None else "sampled",
    }
    if seq_lens:
        counts = Counter(int(v) for v in seq_lens)
        seq_len_summary.update(
            {
                "min": int(min(seq_lens)),
                "max": int(max(seq_lens)),
                "mean": round(float(sum(seq_lens) / len(seq_lens)), 2),
                "num_unique": int(len(counts)),
                "top_counts": [
                    {"seq_len": int(length), "count": int(count)}
                    for length, count in counts.most_common(12)
                ],
            }
        )
    feature_counts = list(sampled_feature_counts or [])
    if not feature_counts and num_features is not None:
        feature_counts = [int(num_features)] * int(num_samples)
    feature_summary: Dict[str, Any] = {
        "mode": "fixed" if num_features is not None else "sampled",
    }
    if feature_counts:
        counts = Counter(int(v) for v in feature_counts)
        feature_summary.update(
            {
                "min": int(min(feature_counts)),
                "max": int(max(feature_counts)),
                "mean": round(float(sum(feature_counts) / len(feature_counts)), 2),
                "num_unique": int(len(counts)),
                "top_counts": [
                    {"num_features": int(feature_count), "count": int(count)}
                    for feature_count, count in counts.most_common(12)
                ],
            }
        )
    return {
        "output_dir": str(output_dir),
        "raw_series_path": str(raw_path),
        "num_samples": int(num_samples),
        "seq_len": int(seq_len) if seq_len is not None else None,
        "sequence_length_summary": seq_len_summary,
        "num_features": int(num_features) if num_features is not None else None,
        "feature_count_summary": feature_summary,
        "seed": int(seed),
        "legacy_anomaly_sample_ratio": float(anomaly_ratio),
        "anomalous_series_count": int(anomalous_series_count),
        "normal_series_count": int(normal_series_count),
        "example_keys": sorted(example_keys or []),
        "example_sample_id": example_sample_id,
        "shard_size": int(shard_size),
        "num_shards": len(shard_manifest or []),
        "shards": shard_manifest or [],
    }


def _sample_sequence_lengths(
    *,
    num_samples: int,
    fixed_seq_len: int | None,
    seq_len_min: int | None,
    seq_len_max: int | None,
    rng: random.Random,
) -> List[int]:
    """Return one sequence length per sample, either fixed or uniformly sampled."""

    if seq_len_min is None and seq_len_max is None:
        if fixed_seq_len is None:
            raise ValueError("Either fixed_seq_len or seq_len_min/seq_len_max must be provided.")
        return [int(fixed_seq_len)] * int(num_samples)
    if seq_len_min is None or seq_len_max is None:
        raise ValueError("seq_len_min and seq_len_max must be provided together.")
    lo = int(min(seq_len_min, seq_len_max))
    hi = int(max(seq_len_min, seq_len_max))
    return [rng.randint(lo, hi) for _ in range(int(num_samples))]


def _sample_feature_counts(
    *,
    num_samples: int,
    fixed_num_features: int | None,
    num_features_min: int | None,
    num_features_max: int | None,
    rng: random.Random,
) -> List[int]:
    """Return one channel-count per sample, either fixed or uniformly sampled."""

    if num_features_min is None and num_features_max is None:
        if fixed_num_features is None:
            raise ValueError("Either fixed_num_features or num_features_min/num_features_max must be provided.")
        return [int(fixed_num_features)] * int(num_samples)
    if num_features_min is None or num_features_max is None:
        raise ValueError("num_features_min and num_features_max must be provided together.")
    lo = int(min(num_features_min, num_features_max))
    hi = int(max(num_features_min, num_features_max))
    return [rng.randint(lo, hi) for _ in range(int(num_samples))]


def _generate_variable_length_dataset(
    *,
    generate_dataset: Any,
    seq_lens: Sequence[int],
    feature_counts: Sequence[int],
    anomaly_ratio: float,
    activate_function: bool,
    use_attribute_set: bool,
) -> List[Dict[str, Any]]:
    """Generate legacy rows while preserving per-sample length/channel schedules."""

    if len(seq_lens) != len(feature_counts):
        raise ValueError("seq_lens and feature_counts must have the same length.")

    buckets: Dict[tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    counts = Counter((int(seq_len), int(feature_count)) for seq_len, feature_count in zip(seq_lens, feature_counts))
    for (seq_len, feature_count), count in counts.items():
        bucket_rows = generate_dataset(
            num_samples=int(count),
            seq_len=int(seq_len),
            anomaly_sample_ratio=float(anomaly_ratio),
            is_multivariate=True,
            num_features=int(feature_count),
            activate_function=bool(activate_function),
            use_attribute_set=bool(use_attribute_set),
        )
        buckets[(int(seq_len), int(feature_count))].extend(bucket_rows)

    ordered_rows: List[Dict[str, Any]] = []
    for seq_len, feature_count in zip(seq_lens, feature_counts):
        bucket = buckets[(int(seq_len), int(feature_count))]
        if not bucket:
            raise RuntimeError(
                f"Legacy generator bucket for seq_len={seq_len}, num_features={feature_count} was exhausted unexpectedly."
            )
        ordered_rows.append(bucket.pop())
    return ordered_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw multivariate time-series data only, without QA fields.")
    parser.add_argument(
        "--legacy-src",
        default=str(ROOT / "legacy" / "TSAD_dataset_gen-axis"),
        help="Path to the TSAD_dataset_gen-axis directory.",
    )
    parser.add_argument("--output-dir", default="outputs/ts_data/ts_531")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seq-len", type=int, default=180)
    parser.add_argument("--seq-len-min", type=int, default=None, help="If set with --seq-len-max, sample one sequence length per sample from this inclusive range.")
    parser.add_argument("--seq-len-max", type=int, default=None, help="If set with --seq-len-min, sample one sequence length per sample from this inclusive range.")
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--num-features-min", type=int, default=None, help="If set with --num-features-max, sample one channel count per sample from this inclusive range.")
    parser.add_argument("--num-features-max", type=int, default=None, help="If set with --num-features-min, sample one channel count per sample from this inclusive range.")
    parser.add_argument("--anomaly-ratio", type=float, default=1.0)
    parser.add_argument("--activate-function", action="store_true")
    parser.add_argument("--use-attribute-set", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=531)
    parser.add_argument("--sample-id-prefix", default=None, help="Prefix for sample_id/base_sample_id. Defaults to the output directory name.")
    parser.add_argument("--shard-size", type=int, default=0, help="If > 0, also write JSONL shards with this many rows each.")
    parser.add_argument("--progress-step-percent", type=float, default=2.5)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id_prefix = str(args.sample_id_prefix or output_dir.name)
    raw_path = output_dir / "raw_series.jsonl"
    summary_path = output_dir / "dataset_summary.json"
    shards_dir = output_dir / "shards"

    print(
        json.dumps(
            {
                "status": "starting",
                "output_dir": str(output_dir),
                "num_samples": int(args.num_samples),
                "seq_len": int(args.seq_len),
                "seq_len_min": None if args.seq_len_min is None else int(args.seq_len_min),
                "seq_len_max": None if args.seq_len_max is None else int(args.seq_len_max),
                "num_features": int(args.num_features),
                "num_features_min": None if args.num_features_min is None else int(args.num_features_min),
                "num_features_max": None if args.num_features_max is None else int(args.num_features_max),
                "anomaly_ratio": float(args.anomaly_ratio),
                "seed": int(args.seed),
                "sample_id_prefix": sample_id_prefix,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    generate_dataset = load_legacy_tsad_generate_dataset(Path(args.legacy_src))
    seq_len_rng = random.Random(int(args.seed))
    sampled_seq_lens = _sample_sequence_lengths(
        num_samples=int(args.num_samples),
        fixed_seq_len=int(args.seq_len),
        seq_len_min=args.seq_len_min,
        seq_len_max=args.seq_len_max,
        rng=seq_len_rng,
    )
    feature_rng = random.Random(int(args.seed) + 1009)
    sampled_feature_counts = _sample_feature_counts(
        num_samples=int(args.num_samples),
        fixed_num_features=int(args.num_features),
        num_features_min=args.num_features_min,
        num_features_max=args.num_features_max,
        rng=feature_rng,
    )
    legacy_rows = _generate_variable_length_dataset(
        generate_dataset=generate_dataset,
        seq_lens=sampled_seq_lens,
        feature_counts=sampled_feature_counts,
        anomaly_ratio=float(args.anomaly_ratio),
        activate_function=bool(args.activate_function),
        use_attribute_set=bool(args.use_attribute_set),
    )

    progress = ProgressPrinter(
        len(legacy_rows),
        label="ts-only-convert",
        step_percent=float(args.progress_step_percent),
    )
    progress.start(extra="converted=0 written=0")

    if shards_dir.exists():
        for stale_shard in shards_dir.glob("raw_series_*.jsonl"):
            stale_shard.unlink()
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_manifest: List[Dict[str, Any]] = []
    current_shard_handle = None
    current_shard_path: Path | None = None
    current_shard_index = -1
    current_shard_start = 0
    current_shard_rows = 0
    num_samples = len(legacy_rows)
    anomalous_series_count = 0
    normal_series_count = 0
    example_keys: List[str] = []
    example_sample_id: str | None = None

    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for idx, legacy_row in enumerate(legacy_rows):
            converted = convert_legacy_sample(
                legacy_row,
                sample_id=f"{sample_id_prefix}_{idx:05d}",
                base_sample_id=f"{sample_id_prefix}_{idx:05d}",
            )
            raw_row = _with_order_metadata(
                _ts_only_row(converted),
                global_index=idx,
                shard_index=None,
                index_in_shard=None,
            )
            raw_handle.write(json.dumps(raw_row, ensure_ascii=False) + "\n")

            if int(args.shard_size) > 0:
                shard_index = idx // int(args.shard_size)
                if shard_index != current_shard_index:
                    if current_shard_handle is not None and current_shard_path is not None:
                        current_shard_handle.close()
                        shard_manifest.append(
                            {
                                "shard_index": int(current_shard_index),
                                "path": str(current_shard_path),
                                "start_global_index": int(current_shard_start),
                                "end_global_index_exclusive": int(current_shard_start + current_shard_rows),
                                "num_rows": int(current_shard_rows),
                            }
                        )
                    current_shard_index = shard_index
                    current_shard_start = idx
                    current_shard_rows = 0
                    current_shard_path = shards_dir / f"raw_series_{shard_index:04d}.jsonl"
                    current_shard_handle = current_shard_path.open("w", encoding="utf-8")

                shard_row = _with_order_metadata(
                    _ts_only_row(converted),
                    global_index=idx,
                    shard_index=shard_index,
                    index_in_shard=idx - current_shard_start,
                )
                current_shard_handle.write(json.dumps(shard_row, ensure_ascii=False) + "\n")
                current_shard_rows += 1

            if bool(((raw_row.get("synthetic_label") or {}).get("affected_channels") or [])):
                anomalous_series_count += 1
            else:
                normal_series_count += 1
            if not example_keys:
                example_keys = list(raw_row.keys())
                example_sample_id = raw_row.get("sample_id")

            legacy_rows[idx] = None
            progress.update(idx + 1, extra=f"converted={idx + 1} written={idx + 1}")

    if current_shard_handle is not None and current_shard_path is not None:
        current_shard_handle.close()
        shard_manifest.append(
            {
                "shard_index": int(current_shard_index),
                "path": str(current_shard_path),
                "start_global_index": int(current_shard_start),
                "end_global_index_exclusive": int(current_shard_start + current_shard_rows),
                "num_rows": int(current_shard_rows),
            }
        )

    summary = _summary(
        output_dir=output_dir,
        raw_path=raw_path,
        num_samples=num_samples,
        seq_len=None if args.seq_len_min is not None and args.seq_len_max is not None else int(args.seq_len),
        num_features=None if args.num_features_min is not None and args.num_features_max is not None else int(args.num_features),
        seed=int(args.seed),
        anomaly_ratio=float(args.anomaly_ratio),
        anomalous_series_count=anomalous_series_count,
        normal_series_count=normal_series_count,
        example_keys=example_keys,
        example_sample_id=example_sample_id,
        sampled_seq_lens=sampled_seq_lens,
        sampled_feature_counts=sampled_feature_counts,
        shard_size=int(args.shard_size),
        shard_manifest=shard_manifest,
    )
    save_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
