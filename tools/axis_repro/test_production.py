import unittest
from .runtime_fixes import score_from_text
from .table_runner import aggregate

class TestProduction(unittest.TestCase):
    def test_markdown_score(self): self.assertEqual(score_from_text("**Score:** 4"),4)
    def test_partial_aggregate(self):
        rows=[{"record_id":"r","mode":"base","question_type":"multiple_choice","dimension":d,"weight":w,"score":s}
              for d,w,s in [("correctness",.7,5),("reasoning_quality",.3,3)]]
        self.assertAlmostEqual(aggregate(rows)[0]["base"]["macro_final"],4.4)

if __name__=="__main__": unittest.main()
