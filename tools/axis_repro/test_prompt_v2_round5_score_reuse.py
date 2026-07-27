import unittest

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round5_score_reuse import (
    normalized_response,
    unique_index,
)


class Round5ScoreReuseTest(unittest.TestCase):
    def test_only_outer_whitespace_is_ignored(self):
        self.assertEqual(
            normalized_response("\n Answer: normal. \r\n"),
            "Answer: normal.",
        )
        self.assertNotEqual(
            normalized_response("Answer:  normal."),
            normalized_response("Answer: normal."),
        )

    def test_single_field_index_uses_a_tuple_key(self):
        rows = [{"record_id": "x"}]
        index = unique_index(rows, ("record_id",))
        self.assertEqual(index[("x",)], rows[0])

    def test_unique_index_rejects_duplicate_keys(self):
        rows = [{"record_id": "x"}, {"record_id": "x"}]
        with self.assertRaises(RuntimeError):
            unique_index(rows, ("record_id",))


if __name__ == "__main__":
    unittest.main()
