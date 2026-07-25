from __future__ import annotations

import unittest

from tools.axis_repro.discrete_label_diagnostics import (
    evaluate_prediction_rows,
    parse_discrete_label,
)


class ParseDiscreteLabelTests(unittest.TestCase):
    def test_tf_is_case_insensitive_and_standalone(self):
        self.assertEqual(
            parse_discrete_label("The claim is TRUE.", "true_false").label,
            "True",
        )
        self.assertEqual(
            parse_discrete_label("falsehood, then False", "true_false").label,
            "False",
        )

    def test_tf_uses_first_occurrence_in_raw_think_text(self):
        parsed = parse_discrete_label(
            "<think>Could be False, but revise.</think> True.",
            "true_false",
        )
        self.assertEqual(parsed.label, "False")
        self.assertEqual(parsed.matched_text, "False")

    def test_mc_supported_forms(self):
        cases = {
            "Answer: C": "C",
            "The final answer is D.": "D",
            "(B) localized anomaly": "B",
            "A) normal": "A",
            "I choose C": "C",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(
                    parse_discrete_label(text, "multiple_choice").label,
                    expected,
                )

    def test_mc_rejects_lowercase_articles_and_embedded_letters(self):
        for text in (
            "a normal sequence",
            "Option b is tempting",
            "DATA contains capital letters but no label token",
        ):
            with self.subTest(text=text):
                self.assertIsNone(
                    parse_discrete_label(text, "multiple_choice").label
                )

    def test_mc_uses_earliest_valid_occurrence(self):
        parsed = parse_discrete_label(
            "<think>(B) first thought, then Answer: C</think>",
            "multiple_choice",
        )
        self.assertEqual(parsed.label, "B")
        self.assertEqual(parsed.strategy, "parenthesized_label")


class EvaluatePredictionRowsTests(unittest.TestCase):
    def _row(
        self,
        record_id: str,
        question_type: str,
        answer: str,
        response: str,
    ):
        return {
            "record_id": record_id,
            "mode": "base",
            "question_type": question_type,
            "answer": answer,
            "response": response,
            "question": "q",
        }

    def test_metrics_count_unparseable_as_incorrect_and_in_recall_denominator(self):
        rows = [
            self._row("tf1", "true_false", "True.", "True."),
            self._row("tf2", "true_false", "False.", "No label."),
            self._row("mc1", "multiple_choice", "A) ref", "B) pred"),
            self._row("oe1", "open_ended", "free", "free"),
        ]
        summary, samples = evaluate_prediction_rows(
            rows,
            model_name="test",
            expected_records=4,
            expected_record_ids={"tf1", "tf2", "mc1", "oe1"},
        )
        self.assertEqual(len(samples), 3)
        self.assertEqual(summary["combined_discrete"]["records"], 3)
        self.assertEqual(summary["combined_discrete"]["correct"], 1)
        self.assertEqual(summary["combined_discrete"]["unparseable"], 1)
        self.assertAlmostEqual(
            summary["combined_discrete"]["accuracy_all_records"], 1 / 3
        )
        tf = summary["by_question_type"]["true_false"]
        self.assertEqual(tf["recall_by_reference_label"]["True"], 1.0)
        self.assertEqual(tf["recall_by_reference_label"]["False"], 0.0)
        self.assertEqual(tf["balanced_accuracy"], 0.5)

    def test_unparseable_reference_fails_closed(self):
        rows = [
            self._row("tf1", "true_false", "No label.", "True."),
        ]
        with self.assertRaisesRegex(ValueError, "reference answer"):
            evaluate_prediction_rows(rows, model_name="test")


if __name__ == "__main__":
    unittest.main()
