import unittest

from experiments.reproduction.prompt_final_full284_deepseek_v4_pro_20260727.src.final_round8_pipeline import (
    JOINT_MODE,
    ROUND5_SOURCE_MODE,
    component_source,
    displayed,
)


def row(question_type, question="Question", response="Base"):
    return {
        "question_type": question_type,
        "question": question,
        "response": response,
    }


class FinalRound8PipelineTest(unittest.TestCase):
    def test_mc_uses_joint_component(self):
        base = row("multiple_choice")
        joint = row("multiple_choice", response="Joint")
        source, selected = component_source(base, joint, base)
        self.assertEqual(source, JOINT_MODE)
        self.assertIs(selected, joint)

    def test_tf_routes_only_explicit_negative_cues(self):
        negative = row("true_false", "True or False: No anomaly is present.")
        positive = row("true_false", "True or False: An anomaly is present.")
        joint = row("true_false", response="Joint")
        self.assertEqual(component_source(negative, joint, negative)[0], JOINT_MODE)
        self.assertEqual(component_source(positive, joint, positive)[0], "base")

    def test_oe_uses_only_round8_eligible_revision(self):
        base = row(
            "open_ended",
            response="<think>There is no evidence of anomalies.",
        )
        revision = row(
            "open_ended",
            response="Answer: There are no anomalies.",
        )
        source, selected = component_source(base, base, revision)
        self.assertEqual(source, ROUND5_SOURCE_MODE)
        self.assertIs(selected, revision)
        unsafe = row("open_ended", response="Answer: A spike is anomalous.")
        self.assertEqual(component_source(base, base, unsafe)[0], "base")

    def test_table_gate_uses_four_decimal_display(self):
        self.assertEqual(displayed(3.674999943), 3.675)
        self.assertEqual(displayed(3.674999780), 3.675)


if __name__ == "__main__":
    unittest.main()