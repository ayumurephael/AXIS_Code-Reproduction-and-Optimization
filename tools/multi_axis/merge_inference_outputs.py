from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tools.multi_axis.datasets import SUPPORTED_EVALUATION_DATASETS, validate_datasets


COMMON_FIELDS = (
    "model",
    "hint_checkpoint_sha256",
    "hint_checkpoint_epoch",
    "timercd_sha256",
    "training_world_size",
    "generation",
    "llm",
    "vision",
    "attention_backend_audit",
    "ablation_variant",
    "ablation_spec",
    "inference_source_commit",
)


def parse_source(value: str) -> tuple[str, Path]:
    dataset, separator, raw_path = value.partition("=")
    if not separator or dataset not in SUPPORTED_EVALUATION_DATASETS or not raw_path:
        raise argparse.ArgumentTypeError(
            "--source must be a supported DATASET=/path/to/inference-run pair"
        )
    return dataset, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge independently distributed per-dataset Multi-AXIS inference runs"
    )
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    datasets = validate_datasets([dataset for dataset, _path in args.source])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    common = None
    dataset_runs = {}
    for dataset, source in args.source:
        manifest_path = source / "inference_manifest.json"
        complete_path = source / "INFERENCE_COMPLETE"
        predictions_path = source / f"{dataset}.predictions.jsonl"
        dataset_complete_path = source / f"{dataset}.complete.json"
        for path in (
            manifest_path,
            complete_path,
            predictions_path,
            dataset_complete_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("datasets") != [dataset]:
            raise RuntimeError(
                f"{dataset}: source manifest datasets are {manifest.get('datasets')}"
            )
        selected = {field: manifest.get(field) for field in COMMON_FIELDS}
        if common is None:
            common = selected
        elif selected != common:
            differing = [field for field in COMMON_FIELDS if selected[field] != common[field]]
            raise RuntimeError(
                f"{dataset}: inference source differs on common fields {differing}"
            )
        manifests[dataset] = manifest
        dataset_runs[dataset] = {
            "world_size": manifest.get("world_size"),
            "expected_inference_world_size": manifest.get(
                "expected_inference_world_size"
            ),
            "distributed_topology": manifest.get("distributed_topology"),
            "source_output_dir": str(source.resolve()),
        }
        shutil.copy2(predictions_path, output_dir / predictions_path.name)
        shutil.copy2(dataset_complete_path, output_dir / dataset_complete_path.name)

    assert common is not None
    merged = {
        **common,
        "datasets": list(datasets),
        "inference_execution": "independent-per-dataset-distributed-runs",
        "dataset_runs": dataset_runs,
        "world_size": None,
        "expected_inference_world_size": None,
    }
    (output_dir / "inference_manifest.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "INFERENCE_COMPLETE").write_text(
        json.dumps({"datasets": list(datasets), "merged": True}), encoding="utf-8"
    )
    print(json.dumps({"datasets": list(datasets), "merged": True}))


if __name__ == "__main__":
    main()
