import unittest

from framework.profiling.topics import _label_texts, _parse_label_lines


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


if __name__ == "__main__":
    unittest.main()
