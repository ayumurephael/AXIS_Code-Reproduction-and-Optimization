import argparse
import json
import tempfile
import unittest
from pathlib import Path

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round3_pipeline import (
    SOURCE_MODE,
    assemble,
    route_changed,
)


BALANCED = (
    "How would you assess whether this boundary event is anomalous, and what "
    "evidence would support or challenge the interpretation?"
)
ONE_SIDED = "What evidence supports the conclusion that this is normal?"


def prediction(record_id, question, mode):
    return {
        "record_id": record_id,
        "series_file": "series_000001.json",
        "question": question,
        "question_type": "open_ended",
        "start_index": 1,
        "end_index": 2,
        "answer": "gold",
        "response": f"{mode} response",
        "mode": mode,
    }


def score(record_id, mode, dimension, value):
    return {
        "record_id": record_id,
        "mode": mode,
        "question_type": "open_ended",
        "dimension": dimension,
        "weight": {"accuracy": 0.35, "completeness": 0.35, "relevance": 0.30}[dimension],
        "score": value,
    }


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class Round3PipelineTests(unittest.TestCase):
    def test_route_predicates(self):
        matched = prediction("m", BALANCED, "base")
        unmatched = prediction("u", ONE_SIDED, "base")
        self.assertTrue(route_changed(matched, "v2_r3_01_oe_balanced_support"))
        self.assertTrue(route_changed(matched, "v2_r3_02_oe_boundary_balanced"))
        self.assertTrue(route_changed(matched, "v2_r3_03_oe_assessment_balanced"))
        for mode in (
            "v2_r3_01_oe_balanced_support",
            "v2_r3_02_oe_boundary_balanced",
            "v2_r3_03_oe_assessment_balanced",
        ):
            self.assertFalse(route_changed(unmatched, mode))

    def test_assemble_reuses_source_only_for_matched_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline_predictions = root / "base_predictions.jsonl"
            baseline_scores = root / "base_scores.jsonl"
            component_predictions = root / "components.jsonl"
            component_scores = root / "component_scores.jsonl"
            split = root / "split.json"
            output_predictions = root / "out_predictions.jsonl"
            output_scores = root / "out_scores.jsonl"
            provenance = root / "provenance.json"
            base = [
                prediction("m", BALANCED, "base"),
                prediction("u", ONE_SIDED, "base"),
            ]
            write_jsonl(baseline_predictions, base)
            write_jsonl(
                baseline_scores,
                [
                    score(record_id, "base", dimension, 3)
                    for record_id in ("m", "u")
                    for dimension in ("accuracy", "completeness", "relevance")
                ],
            )
            write_jsonl(
                component_predictions,
                [prediction("m", BALANCED, SOURCE_MODE)],
            )
            write_jsonl(
                component_scores,
                [
                    score("m", SOURCE_MODE, dimension, 4)
                    for dimension in ("accuracy", "completeness", "relevance")
                ],
            )
            split.write_text(
                json.dumps({"test": ["series_000001.json"]}),
                encoding="utf-8",
            )
            assemble(
                argparse.Namespace(
                    baseline_predictions=str(baseline_predictions),
                    baseline_scores=str(baseline_scores),
                    component_predictions=str(component_predictions),
                    component_scores=str(component_scores),
                    split_manifest=str(split),
                    split_key="test",
                    modes=["v2_r3_01_oe_balanced_support"],
                    output_predictions=str(output_predictions),
                    output_scores=str(output_scores),
                    provenance=str(provenance),
                )
            )
            rows = [
                json.loads(line)
                for line in output_predictions.open(encoding="utf-8")
            ]
            routed = {
                row["record_id"]: row
                for row in rows
                if row["mode"] == "v2_r3_01_oe_balanced_support"
            }
            self.assertEqual(routed["m"]["component_source_mode"], SOURCE_MODE)
            self.assertEqual(routed["u"]["component_source_mode"], "base")


if __name__ == "__main__":
    unittest.main()
