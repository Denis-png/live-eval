import unittest

from framework.generators.base_generator import BaseGenerator


class FakeGenerator(BaseGenerator):
    """Test double: returns a queued response per call_api invocation."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def call_api(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def _cot(error_type, generated, ground_truth):
    return (
        f"Error type: {error_type}\n"
        f"Generated: {generated}\n"
        f"Ground truth: {ground_truth}\n"
    )


class GenerateFilteringTests(unittest.TestCase):
    def _run(self, responses, sample_size=10):
        gen = FakeGenerator(responses)
        samples = [{"incorrect": f"in {i}", "correct": f"out {i}"} for i in range(len(responses))]
        out = gen.generate(
            real_samples=samples,
            error_types=["article"],
            prompt_instruction="Fix: {sentence}",
            sample_size=sample_size,
        )
        return out

    def test_keeps_valid_sample(self):
        out = self._run([_cot("verb_tense", "He go to school yesterday.", "He went to school yesterday.")])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["corrupted"], "He go to school yesterday.")
        self.assertEqual(out[0]["original"], "He went to school yesterday.")
        self.assertEqual(out[0]["error_type"], "verb_tense")

    def test_skips_corrupted_under_three_words(self):
        out = self._run([_cot("article", "He go.", "He goes.")])
        self.assertEqual(out, [])

    def test_skips_identical_corrupted_and_gold(self):
        out = self._run([_cot("article", "This is fine here.", "This is fine here.")])
        self.assertEqual(out, [])

    def test_skips_parse_failure(self):
        out = self._run(["no structured fields at all"])
        self.assertEqual(out, [])

    def test_call_api_is_the_public_contract(self):
        # The judge/generator call sites depend on a public call_api method.
        self.assertTrue(hasattr(BaseGenerator, "call_api"))
        self.assertFalse(hasattr(BaseGenerator, "_call_api"))


if __name__ == "__main__":
    unittest.main()
