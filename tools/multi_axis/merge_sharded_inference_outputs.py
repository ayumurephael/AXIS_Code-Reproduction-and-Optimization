from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.multi_axis.datasets import SUPPORTED_EVALUATION_DATASETS, validate_datasets
from tools.multi_axis.merge_inference_outputs import COMMON_FIELDS


def read_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path: Path, rows) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and merge disjoint group-sharded Multi-AXIS inference runs"
    )
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_EVALUATION_DATASETS,
        required=True,
    )
    args = parser.parse_args()
    datasets = validate_datasets(args.datasets)
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common = None
    shard_count = None
    assigned_shards = set()
    runs = []
    per_dataset_rows = {dataset: [] for dataset in datasets}
    for source in args.source:
        for required in (source / "inference_manifest.json", source / "INFERENCE_COMPLETE"):
            if not required.is_file():
                raise FileNotFoundError(required)
        manifest = json.loads(
            (source / "inference_manifest.json").read_text(encoding="utf-8")
        )
        if tuple(manifest.get("datasets", ())) != datasets:
            raise RuntimeError(
                f"{source}: datasets {manifest.get('datasets')} do not match {datasets}"
            )
        selected = {field: manifest.get(field) for field in COMMON_FIELDS}
        if common is None:
            common = selected
        elif selected != common:
            differing = [field for field in COMMON_FIELDS if selected[field] != common[field]]
            raise RuntimeError(f"{source}: common inference fields differ: {differing}")
        count = int(manifest.get("inference_shard_count", 0))
        indices = tuple(int(value) for value in manifest.get("inference_shard_indices", ()))
        if count <= 1 or not indices:
            raise RuntimeError(f"{source}: not an explicit multi-shard inference run")
        if shard_count is None:
            shard_count = count
        elif count != shard_count:
            raise RuntimeError(f"{source}: shard count {count} != {shard_count}")
        overlap = assigned_shards & set(indices)
        if overlap:
            raise RuntimeError(f"Duplicate inference shard assignments: {sorted(overlap)}")
        assigned_shards.update(indices)
        for dataset in datasets:
            per_dataset_rows[dataset].extend(
                read_jsonl(source / f"{dataset}.predictions.jsonl")
            )
        runs.append(
            {
                "source_output_dir": str(source.resolve()),
                "inference_shard_indices": list(indices),
                "world_size": manifest.get("world_size"),
                "distributed_topology": manifest.get("distributed_topology"),
            }
        )

    assert common is not None and shard_count is not None
    if assigned_shards != set(range(shard_count)):
        raise RuntimeError(
            f"Shard coverage is {sorted(assigned_shards)}, expected 0..{shard_count - 1}"
        )
    for dataset in datasets:
        references = read_jsonl(manifest_dir / f"eval_{dataset}.jsonl")
        rows = per_dataset_rows[dataset]
        by_index = {}
        for row in rows:
            index = int(row["index"])
            if index in by_index:
                raise RuntimeError(f"{dataset}: duplicate prediction index {index}")
            by_index[index] = row
        if set(by_index) != set(range(len(references))):
            missing = sorted(set(range(len(references))) - set(by_index))
            raise RuntimeError(f"{dataset}: merged shard output is missing {missing[:20]}")
        ordered = [by_index[index] for index in range(len(references))]
        for index, (reference, prediction) in enumerate(zip(references, ordered)):
            if prediction.get("sample_id") != reference.get("sample_id"):
                raise RuntimeError(f"{dataset}: sample identity mismatch at {index}")
        atomic_write_jsonl(output_dir / f"{dataset}.predictions.jsonl", ordered)
        (output_dir / f"{dataset}.complete.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "examples": len(ordered),
                    "merged_shards": shard_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    merged = {
        **common,
        "datasets": list(datasets),
        "inference_execution": "independent-group-sharded-runs",
        "inference_shard_count": shard_count,
        "inference_shard_indices": list(range(shard_count)),
        "shard_runs": runs,
        "world_size": None,
        "expected_inference_world_size": None,
    }
    (output_dir / "inference_manifest.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "INFERENCE_COMPLETE").write_text(
        json.dumps({"datasets": list(datasets), "merged_shards": shard_count}),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"datasets": list(datasets), "merged_shards": shard_count, "complete": True}
        )
    )


if __name__ == "__main__":
    main()
