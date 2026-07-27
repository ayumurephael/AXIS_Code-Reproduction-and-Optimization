import unittest

from experiments.reproduction.prompt_final_full284_deepseek_v4_pro_20260727.src.final_round9_variants import (
    JOINT_MODE,
    MC_OE_MODE,
    TF_OE_MODE,
    route_source,
)


def row(question_type, label):
    return {"question_type": question_type, "label": label}


class FinalRound9VariantsTest(unittest.TestCase):
    def test_joint_uses_every_joint_component(self):
        for family in ("multiple_choice", "open_ended", "true_false"):
            base = row(family, "base"); joint = row(family, "joint")
            self.assertIs(route_source(JOINT_MODE, base, joint), joint)

    def test_mc_oe_protects_tf(self):
        base = row("true_false", "base"); joint = row("true_false", "joint")
        self.assertIs(route_source(MC_OE_MODE, base, joint), base)
        self.assertIs(route_source(MC_OE_MODE, row("multiple_choice", "base"), joint), joint)

    def test_tf_oe_protects_mc(self):
        base = row("multiple_choice", "base"); joint = row("multiple_choice", "joint")
        self.assertIs(route_source(TF_OE_MODE, base, joint), base)
        self.assertIs(route_source(TF_OE_MODE, row("true_false", "base"), joint), joint)

    def test_all_routes_keep_round9_oe(self):
        base = row("open_ended", "base"); joint = row("open_ended", "joint")
        for mode in (MC_OE_MODE, TF_OE_MODE, JOINT_MODE):
            self.assertIs(route_source(mode, base, joint), joint)


if __name__ == "__main__":
    unittest.main()