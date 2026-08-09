from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.multi_axis.datasets import (
    OFFICIAL_BIAS_NEUTRALIZED_DATASETS,
    SUPPORTED_EVALUATION_DATASETS,
    validate_datasets,
)

TYPE_NAMES = {"MC": "multiple_choice", "OE": "open_ended", "TF": "true_false"}
DATASETS = OFFICIAL_BIAS_NEUTRALIZED_DATASETS


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_EVALUATION_DATASETS,
        default=list(DATASETS),
    )
    args = parser.parse_args()
    manifest_dir, predictions_dir, output_dir = map(Path, (args.manifest_dir, args.predictions_dir, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = validate_datasets(args.datasets)
    summary = {}
    for dataset in datasets:
        references = read_jsonl(manifest_dir / f"eval_{dataset}.jsonl")
        predictions = read_jsonl(predictions_dir / f"{dataset}.predictions.jsonl")
        if len(references) != len(predictions):
            raise RuntimeError(f"{dataset}: count mismatch")
        output_path = output_dir / f"{dataset}.judge_input.jsonl"
        counts = {name: 0 for name in TYPE_NAMES.values()}
        with output_path.open("w", encoding="utf-8") as handle:
            for index, (reference, prediction) in enumerate(zip(references, predictions)):
                if prediction["index"] != index or prediction["sample_id"] != reference["sample_id"]:
                    raise RuntimeError(f"{dataset}: identity mismatch at {index}")
                question_type = TYPE_NAMES[reference["question_group"]]
                counts[question_type] += 1
                item = {"record_id": f"{dataset}:{index}", "index": index, "sample_id": reference["sample_id"], "dataset": dataset, "mode": "multi_axis", "question_type": question_type, "question": reference["question"], "answer": reference["teacher_answer"], "response": prediction["raw_response"]}
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary[dataset] = {"examples": len(references), "counts": counts, "path": output_path.name}
    (output_dir / "judge_input_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
