import unittest

from experiments.reproduction.prompt_literature_autoresearch_deepseek_v4_pro_20260727.src.assemble_literature_round5 import (
    BASE,
    MC_STATUS,
    MC_STRUCTURED,
    TF_SOURCE,
    source_mode,
)


def row(question_type, question="Question"):
    return {"question_type": question_type, "question": question}


class LiteratureRound5AssemblyTest(unittest.TestCase):
    def test_tf_only_fallback_changes_only_explicit_negative_tf(self):
        self.assertEqual(source_mode(TF_SOURCE, row("multiple_choice")), BASE)
        self.assertEqual(source_mode(TF_SOURCE, row("open_ended")), BASE)
        self.assertEqual(
            source_mode(
                TF_SOURCE,
                row("true_false", "There is no evidence of an anomaly."),
            ),
            TF_SOURCE,
        )
        self.assertEqual(
            source_mode(
                TF_SOURCE,
                row("true_false", "An upward spike is anomalous."),
            ),
            BASE,
        )

    def test_structured_joint_routes_exact_components(self):
        route = "lit_r5_03_joint_structured_tf_neg_re2"
        self.assertEqual(source_mode(route, row("multiple_choice")), MC_STRUCTURED)
        self.assertEqual(source_mode(route, row("open_ended")), BASE)
        self.assertEqual(
            source_mode(route, row("true_false", "This is not normal.")),
            TF_SOURCE,
        )
        self.assertEqual(
            source_mode(route, row("true_false", "A spike is anomalous.")),
            BASE,
        )

    def test_status_joint_routes_exact_components(self):
        route = "lit_r5_04_joint_status_tf_neg_re2"
        self.assertEqual(source_mode(route, row("multiple_choice")), MC_STATUS)
        self.assertEqual(source_mode(route, row("open_ended")), BASE)

    def test_rejects_unknown_route_and_question_type(self):
        with self.assertRaises(ValueError):
            source_mode("unknown", row("multiple_choice"))
        with self.assertRaises(ValueError):
            source_mode(MC_STRUCTURED, row("unknown"))


if __name__ == "__main__":
    unittest.main()
