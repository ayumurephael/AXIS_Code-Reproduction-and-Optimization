import unittest
from . import common
from .io_utils import append_jsonl
common.append_jsonl=append_jsonl
from .geval_correct import final_score_distribution

class TestFinalScorePosition(unittest.TestCase):
    def test_ignores_reasoning_digits(self):
        entries=[]
        for token in ["Reasoning 5 and 4. ","**Score:** ","1"]:
            alts=[{"token":str(i),"logprob":0 if i==1 else -10} for i in range(1,6)] if token=="1" else [{"token":"5","logprob":0}]
            entries.append({"token":token,"top_logprobs":alts})
        response={"choices":[{"message":{"content":"Reasoning 5 and 4. **Score:** 1"},"logprobs":{"content":entries}}]}
        score,_=final_score_distribution(response);self.assertLess(score,1.01)

if __name__=="__main__":unittest.main()
