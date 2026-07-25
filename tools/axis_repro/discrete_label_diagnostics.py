"""Deterministic discrete-label diagnostics for AXIS TF and MC answers.

The parser intentionally operates on the raw generated response, including any
``<think>`` block.  It extracts the first valid label occurrence.  Open-ended
records are retained in the input coverage audit but are not assigned a label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .common import read_jsonl


PARSER_VERSION = "axis-discrete-label-v1"
TF_LABELS = ("True", "False")
MC_LABELS = ("A", "B", "C", "D")
DISCRETE_QUESTION_TYPES = ("true_false", "multiple_choice")

_TF_PATTERN = re.compile(r"(?<![A-Za-z])(?P<label>true|false)(?![A-Za-z])", re.IGNORECASE)
_MC_PATTERNS = (
    (
        "answer_marker",
        re.compile(
            r"(?i:\b(?:final\s+)?answer\b)\s*(?::|=|-|\bis\b)?\s*"
            r"(?P<label>[A-D])(?=$|[\s\)\].,:;!?])"
        ),
    ),
    (
        "parenthesized_label",
        re.compile(r"(?<![A-Za-z0-9])\((?P<label>[A-D])\)"),
    ),
    (
        "option_label",
        re.compile(
            r"(?<![A-Za-z0-9])(?P<label>[A-D])(?=[\)\].:](?:\s|$))"
        ),
    ),
    (
        "standalone_uppercase",
        re.compile(r"(?<![A-Za-z0-9])(?P<label>[A-D])(?![A-Za-z0-9])"),
    ),
)


@dataclass(frozen=True)
class LabelParse:
    label: str | None
    strategy: str
    start: int | None
    end: int | None
    matched_text: str | None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_discrete_label(text: object, question_type: str) -> LabelParse:
    """Extract the first valid TF or MC label from raw text.

    TF is case-insensitive for the standalone words ``True`` and ``False``.
    MC labels must be uppercase A--D and appear in an answer marker, ``(A)``,
    ``A)``/``A.``/``A:``, or as a standalone uppercase token.  Lowercase
    articles and letters embedded in words are deliberately rejected.
    """

    raw = "" if text is None else str(text)
    if question_type == "true_false":
        match = _TF_PATTERN.search(raw)
        if match is None:
            return LabelParse(None, "unparseable", None, None, None)
        label = match.group("label").lower()
        normalized = "True" if label == "true" else "False"
        return LabelParse(
            normalized,
            "standalone_true_false",
            match.start(),
            match.end(),
            match.group(0),
        )
    if question_type == "multiple_choice":
        candidates: list[tuple[int, int, int, str, re.Match[str]]] = []
        for priority, (strategy, pattern) in enumerate(_MC_PATTERNS):
            match = pattern.search(raw)
            if match is not None:
                candidates.append(
                    (match.start(), priority, match.end(), strategy, match)
                )
        if not candidates:
            return LabelParse(None, "unparseable", None, None, None)
        _, _, _, strategy, match = min(candidates)
        return LabelParse(
            match.group("label"),
            strategy,
            match.start(),
            match.end(),
            match.group(0),
        )
    raise ValueError(f"unsupported discrete question type: {question_type}")


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _question_type_metrics(
    samples: Sequence[Mapping[str, object]],
    labels: Sequence[str],
) -> dict[str, object]:
    total = len(samples)
    correct = sum(row["classification"] == "label_correct" for row in samples)
    unparseable = sum(row["classification"] == "U" for row in samples)
    parseable = total - unparseable
    confusion = {
        true_label: {
            pred_label: sum(
                row["reference_label"] == true_label
                and row["predicted_label"] == pred_label
                for row in samples
            )
            for pred_label in labels
        }
        for true_label in labels
    }
    support = {
        label: sum(row["reference_label"] == label for row in samples)
        for label in labels
    }
    unparseable_by_true = {
        label: sum(
            row["reference_label"] == label
            and row["classification"] == "U"
            for row in samples
        )
        for label in labels
    }
    recall = {
        label: _safe_ratio(confusion[label][label], support[label])
        for label in labels
    }
    balanced_accuracy = (
        sum(value for value in recall.values() if value is not None)
        / sum(value is not None for value in recall.values())
        if any(value is not None for value in recall.values())
        else None
    )
    row_percent = {
        true_label: {
            pred_label: _safe_ratio(confusion[true_label][pred_label], support[true_label])
            for pred_label in labels
        }
        for true_label in labels
    }
    return {
        "records": total,
        "correct": correct,
        "incorrect_parseable": parseable - correct,
        "unparseable": unparseable,
        "accuracy_all_records": _safe_ratio(correct, total),
        "accuracy_parseable_only": _safe_ratio(correct, parseable),
        "unparseable_rate": _safe_ratio(unparseable, total),
        "labels": list(labels),
        "support_by_reference_label": support,
        "recall_by_reference_label": recall,
        "balanced_accuracy": _finite_or_none(balanced_accuracy),
        "confusion_counts_parseable": confusion,
        "confusion_row_fraction_full_support": row_percent,
        "unparseable_by_reference_label": unparseable_by_true,
        "parse_strategy_counts": dict(
            Counter(str(row["prediction_parse_strategy"]) for row in samples)
        ),
    }


def evaluate_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    model_name: str,
    expected_records: int | None = None,
    expected_record_ids: set[str] | None = None,
    mode: str = "base",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate one prediction file and return aggregate and per-sample rows."""

    filtered = [row for row in rows if str(row.get("mode", "")) == mode]
    record_ids = [str(row["record_id"]) for row in filtered]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{model_name}: duplicate record_id values for mode={mode}")
    if expected_records is not None and len(filtered) != expected_records:
        raise ValueError(
            f"{model_name}: got {len(filtered)} records, expected {expected_records}"
        )
    if expected_record_ids is not None and set(record_ids) != expected_record_ids:
        raise ValueError(
            f"{model_name}: record-id mismatch "
            f"got={len(set(record_ids))} expected={len(expected_record_ids)}"
        )

    qtype_counts = Counter(str(row.get("question_type", "")) for row in filtered)
    unexpected_qtypes = set(qtype_counts) - {
        "true_false", "multiple_choice", "open_ended"
    }
    if unexpected_qtypes:
        raise ValueError(
            f"{model_name}: unexpected question types {sorted(unexpected_qtypes)}"
        )

    samples: list[dict[str, object]] = []
    for row in filtered:
        qtype = str(row.get("question_type", ""))
        if qtype not in DISCRETE_QUESTION_TYPES:
            continue
        reference = parse_discrete_label(row.get("answer"), qtype)
        if reference.label is None:
            raise ValueError(
                f"{model_name}: reference answer is unparseable at "
                f"{row.get('record_id')}"
            )
        prediction = parse_discrete_label(row.get("response"), qtype)
        if prediction.label is None:
            classification = "U"
        elif prediction.label == reference.label:
            classification = "label_correct"
        else:
            classification = "label_incorrect"
        samples.append(
            {
                "model": model_name,
                "record_id": str(row["record_id"]),
                "series_file": str(row.get("series_file", "")),
                "series_index": row.get("series_index"),
                "window_index": row.get("window_index"),
                "sample_id": row.get("sample_id"),
                "mode": mode,
                "question_type": qtype,
                "reference_label": reference.label,
                "predicted_label": prediction.label,
                "classification": classification,
                "reference_parse_strategy": reference.strategy,
                "prediction_parse_strategy": prediction.strategy,
                "reference_match_start": reference.start,
                "prediction_match_start": prediction.start,
                "reference_matched_text": reference.matched_text,
                "prediction_matched_text": prediction.matched_text,
                "question": str(row.get("question", "")),
                "reference_answer": str(row.get("answer", "")),
                "generated_response": str(row.get("response", "")),
            }
        )

    by_type: dict[str, object] = {}
    for qtype, labels in (
        ("true_false", TF_LABELS),
        ("multiple_choice", MC_LABELS),
    ):
        subset = [row for row in samples if row["question_type"] == qtype]
        by_type[qtype] = _question_type_metrics(subset, labels)

    discrete_total = len(samples)
    discrete_correct = sum(
        row["classification"] == "label_correct" for row in samples
    )
    discrete_unparseable = sum(row["classification"] == "U" for row in samples)
    discrete_parseable = discrete_total - discrete_unparseable
    summary = {
        "model": model_name,
        "coverage": {
            "prediction_records": len(filtered),
            "question_type_counts": dict(qtype_counts),
            "discrete_records": discrete_total,
            "open_ended_records_excluded_from_label_metrics": qtype_counts.get(
                "open_ended", 0
            ),
        },
        "combined_discrete": {
            "records": discrete_total,
            "correct": discrete_correct,
            "incorrect_parseable": discrete_parseable - discrete_correct,
            "unparseable": discrete_unparseable,
            "accuracy_all_records": _safe_ratio(
                discrete_correct, discrete_total
            ),
            "accuracy_parseable_only": _safe_ratio(
                discrete_correct, discrete_parseable
            ),
            "unparseable_rate": _safe_ratio(
                discrete_unparseable, discrete_total
            ),
        },
        "by_question_type": by_type,
        "classification_counts": dict(
            Counter(str(row["classification"]) for row in samples)
        ),
    }
    return summary, samples


def _comparison_value(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    path: Sequence[str],
) -> dict[str, float | None]:
    def resolve(source: Mapping[str, object]) -> float | None:
        current: object = source
        for part in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(part)
        if current is None:
            return None
        return float(current)

    candidate_value = resolve(candidate)
    baseline_value = resolve(baseline)
    absolute = (
        None
        if candidate_value is None or baseline_value is None
        else candidate_value - baseline_value
    )
    relative = (
        None
        if absolute is None or baseline_value == 0.0
        else absolute / abs(baseline_value)
    )
    return {
        "candidate": candidate_value,
        "baseline": baseline_value,
        "absolute_difference": absolute,
        "percentage_point_difference": (
            None if absolute is None else 100.0 * absolute
        ),
        "relative_change": relative,
        "relative_change_percent": (
            None if relative is None else 100.0 * relative
        ),
    }


def build_comparisons(
    models: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    pairs = (
        ("control_vs_released", "control", "released"),
        ("treatment_vs_released", "treatment", "released"),
        ("treatment_vs_control", "treatment", "control"),
    )
    paths = {
        "combined_accuracy": ("combined_discrete", "accuracy_all_records"),
        "combined_parseable_accuracy": (
            "combined_discrete",
            "accuracy_parseable_only",
        ),
        "combined_unparseable_rate": (
            "combined_discrete",
            "unparseable_rate",
        ),
        "tf_accuracy": (
            "by_question_type",
            "true_false",
            "accuracy_all_records",
        ),
        "tf_true_recall": (
            "by_question_type",
            "true_false",
            "recall_by_reference_label",
            "True",
        ),
        "tf_false_recall": (
            "by_question_type",
            "true_false",
            "recall_by_reference_label",
            "False",
        ),
        "tf_balanced_accuracy": (
            "by_question_type",
            "true_false",
            "balanced_accuracy",
        ),
        "tf_unparseable_rate": (
            "by_question_type",
            "true_false",
            "unparseable_rate",
        ),
        "mc_accuracy": (
            "by_question_type",
            "multiple_choice",
            "accuracy_all_records",
        ),
        "mc_balanced_accuracy": (
            "by_question_type",
            "multiple_choice",
            "balanced_accuracy",
        ),
        **{
            f"mc_{label.lower()}_recall": (
                "by_question_type",
                "multiple_choice",
                "recall_by_reference_label",
                label,
            )
            for label in MC_LABELS
        },
        "mc_unparseable_rate": (
            "by_question_type",
            "multiple_choice",
            "unparseable_rate",
        ),
    }
    result: dict[str, object] = {}
    for title, candidate_name, baseline_name in pairs:
        if candidate_name not in models or baseline_name not in models:
            continue
        result[title] = {
            metric: _comparison_value(
                models[candidate_name],
                models[baseline_name],
                path,
            )
            for metric, path in paths.items()
        }
    return result


def _parse_named_paths(values: Iterable[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"--predictions must be NAME=PATH, got {value!r}"
            )
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in names:
            raise ValueError(f"invalid or duplicate model name: {name!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        names.add(name)
        result.append((name, path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-records", type=int, default=284)
    parser.add_argument("--mode", default="base")
    args = parser.parse_args()

    named_paths = _parse_named_paths(args.predictions)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = set(str(value) for value in manifest["record_ids"])
    if manifest.get("subset") != "full":
        raise ValueError("discrete full-set diagnostic requires subset=full")
    if len(expected_ids) != args.expected_records:
        raise ValueError(
            f"manifest has {len(expected_ids)} ids, "
            f"expected {args.expected_records}"
        )

    models: dict[str, dict[str, object]] = {}
    all_samples: list[dict[str, object]] = []
    inputs: dict[str, object] = {}
    for name, path in named_paths:
        rows = read_jsonl(path)
        summary, samples = evaluate_prediction_rows(
            rows,
            model_name=name,
            expected_records=args.expected_records,
            expected_record_ids=expected_ids,
            mode=args.mode,
        )
        run_manifest_path = path.parent / "run_manifest.json"
        input_metadata: dict[str, object] = {
            "predictions_path": str(path),
            "predictions_sha256": sha256_file(path),
        }
        if run_manifest_path.is_file():
            run_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
            input_metadata.update(
                {
                    "run_manifest_path": str(run_manifest_path),
                    "run_manifest_sha256": sha256_file(run_manifest_path),
                    "run_manifest": run_manifest,
                }
            )
        summary["input"] = input_metadata
        models[name] = summary
        inputs[name] = input_metadata
        all_samples.extend(samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "schema_version": 1,
        "parser_version": PARSER_VERSION,
        "protocol": {
            "scope": "full284",
            "manifest_subset": manifest.get("subset"),
            "manifest_record_count": manifest.get("record_count"),
            "manifest_sha256": sha256_file(manifest_path),
            "mode": args.mode,
            "raw_response_including_think": True,
            "reference_and_prediction_use_same_parser": True,
            "unparseable_counts_as_incorrect": True,
            "recall_denominator_includes_unparseable": True,
            "open_ended_excluded_from_label_metrics": True,
            "semantic_explanation_categories_A_to_E_evaluated": False,
            "classification_labels": [
                "label_correct",
                "label_incorrect",
                "U",
            ],
        },
        "models": models,
        "comparisons": build_comparisons(models),
    }
    summary_path = output_dir / "discrete_label_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sample_path = output_dir / "discrete_label_samples.jsonl"
    sample_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in sorted(
                all_samples,
                key=lambda row: (
                    str(row["model"]),
                    str(row["record_id"]),
                ),
            )
        ),
        encoding="utf-8",
    )
    audit = {
        "ok": True,
        "summary_sha256": sha256_file(summary_path),
        "samples_sha256": sha256_file(sample_path),
        "model_count": len(models),
        "models": list(models),
        "prediction_records_per_model": {
            name: model["coverage"]["prediction_records"]
            for name, model in models.items()
        },
        "discrete_records_per_model": {
            name: model["coverage"]["discrete_records"]
            for name, model in models.items()
        },
        "sample_rows": len(all_samples),
    }
    audit_path = output_dir / "discrete_label_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
