import argparse
import json
import tempfile
import unittest
from pathlib import Path

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src import (
    round1_pipeline as pipeline,
)
from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round1_selection import (
    ACTIVE_FAMILY,
    prompt_changed,
)
from tools.axis_repro.build_tables import DIMS


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def base_row(record_id, series_file, question_type, question):
    return {
        "record_id": record_id,
        "series_file": series_file,
        "mode": "base",
        "question_type": question_type,
        "question": question,
        "answer": "gold",
        "response": "baseline",
        "start_index": 10,
        "end_index": 13,
    }


class Round1PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "splits.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "screening_series": ["mc.json", "oe.json", "tf.json"],
                    "validation_series": [],
                    "holdout_series": [],
                }
            ),
            encoding="utf-8",
        )
        self.baseline = [
            base_row(
                "mc:0",
                "mc.json",
                "multiple_choice",
                "Which option? A) Normal. B) Anomalous.",
            ),
            base_row("oe:0", "oe.json", "open_ended", "Describe this window."),
            base_row(
                "tf:0",
                "tf.json",
                "true_false",
                "True or False: No anomaly is present.",
            ),
        ]
        self.baseline_path = self.root / "baseline.jsonl"
        write_jsonl(self.baseline_path, self.baseline)

    def tearDown(self):
        self.temp.cleanup()

    def selected_components(self):
        rows = []
        for base in self.baseline:
            record = pipeline.selection_record(base)
            for mode, family in ACTIVE_FAMILY.items():
                if (
                    base["question_type"] == family
                    and prompt_changed(record, mode)
                ):
                    row = dict(base)
                    row.update(
                        {
                            "mode": mode,
                            "response": "candidate",
                            "selected_component": True,
                        }
                    )
                    rows.append(row)
        return rows

    def test_prepare_and_assemble_exact_components(self):
        raw_path = self.root / "raw.jsonl"
        component_path = self.root / "components.jsonl"
        prepare_provenance = self.root / "prepare.json"
        write_jsonl(raw_path, self.selected_components())
        pipeline.prepare(
            argparse.Namespace(
                baseline_predictions=str(self.baseline_path),
                raw_predictions=str(raw_path),
                split_manifest=str(self.manifest),
                split_key="search_pool_series",
                output=str(component_path),
                provenance=str(prepare_provenance),
            )
        )
        components = pipeline.read_jsonl(component_path)
        self.assertEqual(len(components), 8)
        self.assertNotIn(
            "v2_r1_03_oe_evidence_router",
            {row["mode"] for row in components},
        )

        base_scores = []
        for row in self.baseline:
            for dimension in DIMS[row["question_type"]]:
                base_scores.append(
                    {
                        "record_id": row["record_id"],
                        "mode": "base",
                        "question_type": row["question_type"],
                        "dimension": dimension,
                        "weight": 1 / len(DIMS[row["question_type"]]),
                        "score": 3.0,
                    }
                )
        component_scores = []
        for row in components:
            for dimension in DIMS[row["question_type"]]:
                component_scores.append(
                    {
                        "record_id": row["record_id"],
                        "mode": row["mode"],
                        "question_type": row["question_type"],
                        "dimension": dimension,
                        "weight": 1 / len(DIMS[row["question_type"]]),
                        "score": 4.0,
                    }
                )
        base_scores_path = self.root / "base_scores.jsonl"
        component_scores_path = self.root / "component_scores.jsonl"
        write_jsonl(base_scores_path, base_scores)
        write_jsonl(component_scores_path, component_scores)
        output_predictions = self.root / "assembled_predictions.jsonl"
        output_scores = self.root / "assembled_scores.jsonl"
        pipeline.assemble(
            argparse.Namespace(
                baseline_predictions=str(self.baseline_path),
                baseline_scores=str(base_scores_path),
                component_predictions=str(component_path),
                component_scores=str(component_scores_path),
                split_manifest=str(self.manifest),
                split_key="search_pool_series",
                modes=list(ACTIVE_FAMILY),
                output_predictions=str(output_predictions),
                output_scores=str(output_scores),
                provenance=str(self.root / "assemble.json"),
            )
        )
        self.assertEqual(len(pipeline.read_jsonl(output_predictions)), 30)
        self.assertEqual(len(pipeline.read_jsonl(output_scores)), 70)


if __name__ == "__main__":
    unittest.main()
