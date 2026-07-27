import unittest

from experiments.reproduction.prompt_final_full284_deepseek_v4_pro_20260727.src.extract_final_round9_cases import (
    select_best_cases,
)
from experiments.reproduction.prompt_final_full284_deepseek_v4_pro_20260727.src.final_round9_variants import (
    JOINT_MODE as FINAL_MODE,
)
from tools.axis_repro.build_tables import DIMS


class ExtractFinalRound9CasesTest(unittest.TestCase):
    def test_selects_largest_weighted_delta_per_family(self):
        predictions = []
        scores = []
        families = ["multiple_choice", "open_ended", "true_false"]
        for family_index, family in enumerate(families):
            for case_index in (0, 1):
                record_id = f"r{family_index}_{case_index}"
                base = {
                    "record_id": record_id,
                    "mode": "base",
                    "question_type": family,
                    "question": "Q",
                    "answer": "G",
                    "response": "base",
                }
                candidate = dict(
                    base,
                    mode=FINAL_MODE,
                    response=f"candidate {case_index}",
                    component_source_mode="source",
                )
                predictions.extend([base, candidate])
                for dimension in DIMS[family]:
                    scores.append({
                        "record_id": record_id,
                        "mode": "base",
                        "dimension": dimension,
                        "score": 2.0,
                        "weight": 1 / len(DIMS[family]),
                    })
                    scores.append({
                        "record_id": record_id,
                        "mode": FINAL_MODE,
                        "dimension": dimension,
                        "score": 3.0 + case_index,
                        "weight": 1 / len(DIMS[family]),
                    })
        result = select_best_cases(predictions, scores)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(row["record_id"].endswith("_1") for row in result))


if __name__ == "__main__":
    unittest.main()