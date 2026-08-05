import json
import random
import unittest

from framework.profiling.topics import _label_texts, _parse_label_lines, profile_topics


class FakeCallApi:
    """Canned LLM: returns queued responses in order; an Exception raises."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ParseLabelLinesTests(unittest.TestCase):
    def test_accepts_number_separator_variants_and_skips_garbage(self):
        raw = "1: greetings\n2. spam offer\nnot a label line\n3) daily chat"
        self.assertEqual(
            _parse_label_lines(raw, expected=3),
            {0: "greetings", 1: "spam offer", 2: "daily chat"},
        )

    def test_ignores_out_of_range_indices_and_normalizes_case(self):
        self.assertEqual(_parse_label_lines("1: Sports\n9: nope", expected=2), {0: "sports"})

    def test_empty_response(self):
        self.assertEqual(_parse_label_lines("", expected=3), {})


class LabelTextsTests(unittest.TestCase):
    def test_batches_and_failsoft(self):
        texts = ["t1", "t2", "t3", "t4", "t5"]
        call = FakeCallApi(["1: a\n2: b", "garbage", "1: e"])
        labels = _label_texts(texts, call, batch_size=2)
        self.assertEqual(len(call.prompts), 3)  # ceil(5/2)
        self.assertEqual(labels, ["a", "b", None, None, "e"])

    def test_api_exception_skips_batch(self):
        call = FakeCallApi([RuntimeError("boom"), "1: ok"])
        labels = _label_texts(["t1", "t2"], call, batch_size=1)
        self.assertEqual(labels, [None, "ok"])


LABELS_RESPONSE = "1: prize scam\n2: prize scam\n3: banking\n4: daily chat"
CONSOLIDATION = json.dumps({
    "scams": {
        "description": "Messages about scams and fraud.",
        "members": ["prize scam", "banking"],
    }
})


class ProfileTopicsTests(unittest.TestCase):
    def _texts(self):
        return ["win a prize", "free prize now", "your bank account", "see you at 5"]

    def test_happy_path_fractions_computed_in_code(self):
        call = FakeCallApi([LABELS_RESPONSE, CONSOLIDATION])
        result = profile_topics(self._texts(), call, sample_size=10, batch_size=10)
        self.assertEqual(result["n_sampled"], 4)
        self.assertEqual(result["n_labeled"], 4)
        self.assertIsNone(result["note"])
        self.assertAlmostEqual(result["topics"]["scams"]["fraction"], 0.75)
        # unclaimed raw label survives as its own topic (mass is preserved):
        self.assertAlmostEqual(result["topics"]["daily chat"]["fraction"], 0.25)
        self.assertEqual(len(result["topics"]["scams"]["examples"]), 3)
        self.assertEqual(result["raw_labels"]["prize scam"], 2)
        self.assertEqual(len(call.prompts), 2)  # 1 label batch + 1 consolidation

    def test_consolidation_failure_falls_back_to_raw_labels(self):
        call = FakeCallApi([LABELS_RESPONSE, "not json", "still not json"])
        result = profile_topics(self._texts(), call, sample_size=10, batch_size=10)
        self.assertIn("consolidation failed", result["note"])
        self.assertAlmostEqual(result["topics"]["prize scam"]["fraction"], 0.5)
        self.assertEqual(len(call.prompts), 3)  # 1 label + 2 consolidation attempts

    def test_code_fenced_json_is_accepted(self):
        fenced = "```json\n" + CONSOLIDATION + "\n```"
        call = FakeCallApi([LABELS_RESPONSE, fenced])
        result = profile_topics(self._texts(), call, sample_size=10, batch_size=10)
        self.assertIn("scams", result["topics"])

    def test_no_labels_at_all(self):
        call = FakeCallApi(["garbage"])
        result = profile_topics(self._texts(), call, sample_size=10, batch_size=10)
        self.assertEqual(result["n_labeled"], 0)
        self.assertEqual(result["topics"], {})
        self.assertEqual(result["note"], "no batches labeled successfully")
        self.assertEqual(len(call.prompts), 1)  # no consolidation attempted

    def test_sampling_is_deterministic(self):
        texts = [f"text {i}" for i in range(30)]
        prompts = []
        for _ in range(2):
            call = FakeCallApi(["", ""])
            profile_topics(texts, call, sample_size=10, batch_size=10,
                           rng=random.Random(42))
            prompts.append(call.prompts[0])
        self.assertEqual(prompts[0], prompts[1])


if __name__ == "__main__":
    unittest.main()
