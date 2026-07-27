import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).with_name("round1_selection.py")
SPEC = importlib.util.spec_from_file_location("round1_selection", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def record(question_type, question):
    return SimpleNamespace(
        question=question,
        question_type=question_type,
        start_index=10,
        end_index=13,
    )


class Round1InferenceSelectionTests(unittest.TestCase):
    def test_regular_single_family_modes_select_only_active_family(self):
        examples = {
            "multiple_choice": "Which option? A) Normal. B) Anomalous.",
            "open_ended": "Describe this window.",
            "true_false": "True or False: No anomaly is present.",
        }
        for mode, active in MODULE.ACTIVE_FAMILY.items():
            if mode == "v2_r1_03_oe_evidence_router":
                continue
            for question_type, question in examples.items():
                with self.subTest(mode=mode, question_type=question_type):
                    changed = MODULE.prompt_changed(
                        record(question_type, question),
                        mode,
                    )
                    self.assertEqual(changed, question_type == active)

    def test_evidence_router_selects_only_matching_oe_questions(self):
        mode = "v2_r1_03_oe_evidence_router"
        self.assertTrue(
            MODULE.prompt_changed(
                record(
                    "open_ended",
                    "What evidence would you examine near the boundary?",
                ),
                mode,
            )
        )
        self.assertFalse(
            MODULE.prompt_changed(
                record("open_ended", "Describe this window."),
                mode,
            )
        )
        self.assertFalse(
            MODULE.prompt_changed(
                record(
                    "true_false",
                    "What evidence would make this statement true?",
                ),
                mode,
            )
        )


if __name__ == "__main__":
    unittest.main()
