import unittest

from tools.axis_repro.analyze_prompt_outputs import analyze, diagnose


class PromptOutputDiagnosticsTest(unittest.TestCase):
    def test_mc_strict_format_and_exact_option_copy(self):
        row = {
            "record_id": "r1",
            "mode": "task_protocol",
            "question_type": "multiple_choice",
            "question": "Pick one.\nA) Flat\nB) Sharp spike",
            "answer": "B) Sharp spike",
            "response": "B) Sharp spike\nExplanation: The spike is localized.",
            "start_index": 0,
            "end_index": 2,
            "time_series": [0.0, 1.0],
        }
        result = diagnose(row)
        self.assertTrue(result["raw_answer_first"])
        self.assertTrue(result["strict_parse"])
        self.assertTrue(result["selected_option_text_exact"])
        self.assertTrue(result["heuristic_correct"])
        self.assertEqual(result["think_state"], "none")

    def test_thinking_prefix_is_not_answer_first(self):
        row = {
            "record_id": "r2",
            "mode": "base",
            "question_type": "true_false",
            "question": "True or False: flat.",
            "answer": "True.",
            "response": "<think>reasoning",
            "start_index": 0,
            "end_index": 1,
            "time_series": [0.0],
        }
        result = diagnose(row)
        self.assertFalse(result["raw_answer_first"])
        self.assertFalse(result["strict_parse"])
        self.assertEqual(result["think_state"], "open_only")

    def test_summary_is_split_by_mode_and_type(self):
        rows = [
            {
                "record_id": "a",
                "mode": "base",
                "question_type": "true_false",
                "question": "True or False: flat.",
                "answer": "True.",
                "response": "True. It is flat.",
                "start_index": 0,
                "end_index": 1,
                "time_series": [0.0],
            },
            {
                "record_id": "b",
                "mode": "base",
                "question_type": "open_ended",
                "question": "Describe the pattern.",
                "answer": "It is flat.",
                "response": "The supplied window is flat.",
                "start_index": 0,
                "end_index": 1,
                "time_series": [0.0],
            },
        ]
        summary, details = analyze(rows)
        self.assertEqual(len(details), 2)
        self.assertEqual(summary["modes"]["base"]["overall"]["count"], 2)
        self.assertEqual(
            summary["modes"]["base"]["by_question_type"]["true_false"][
                "strict_parse_rate"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
