import unittest
from unittest import mock

from .build_tables import aggregate
from . import common
from .io_utils import append_jsonl
common.append_jsonl = append_jsonl
from .geval_deepseek import logprob_distribution, normalize_type, score_from_text


class TestPipeline(unittest.TestCase):
    def test_score_parser(self):
        self.assertEqual(score_from_text("**Score:** 4"), 4)
        self.assertEqual(normalize_type("Multiple Choice"), "multiple_choice")

    def test_logprob_expectation(self):
        alts = [{"token": str(i), "logprob": 0.0 if i == 5 else -10.0} for i in range(1, 6)]
        result = logprob_distribution({"choices": [{"logprobs": {"content": [{"top_logprobs": alts}]}}]})
        self.assertIsNotNone(result)
        self.assertGreater(result[0], 4.99)

    def test_configurable_deepseek_endpoint(self):
        seen = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"choices": []}'

        def fake_urlopen(request, timeout):
            seen["url"] = request.full_url
            seen["timeout"] = timeout
            return FakeResponse()

        from . import geval_deepseek
        endpoint = "https://judge.example/v1/chat/completions"
        with mock.patch.object(geval_deepseek, "API_ENDPOINT", endpoint), \
             mock.patch.object(geval_deepseek.urllib.request, "urlopen", fake_urlopen):
            response = geval_deepseek.api_call("secret", "deepseek-v4-pro", "prompt", 0, True)

        self.assertEqual(response, {"choices": []})
        self.assertEqual(seen, {"url": endpoint, "timeout": 180})
    def test_weighted_table(self):
        rows = [{"record_id":"r", "mode":"base", "question_type":"multiple_choice",
                 "dimension":d, "weight":w, "score":s}
                for d,w,s in [("correctness",.7,5),("reasoning_quality",.3,3)]]
        result,_ = aggregate(rows)
        self.assertAlmostEqual(result["base"]["multiple_choice/final"], 4.4)


if __name__ == "__main__": unittest.main()
