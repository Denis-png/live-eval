import random
import unittest

from framework.generators.base_generator import BaseGenerator


class FakeGenerator(BaseGenerator):
    """Returns a queued response per call_api invocation; records prompts."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def call_api(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


_PROMPT = "Errors: {error_spec}\nCorrect: {sentence}"
_DESCRIPTIONS = {"R:VERB:TENSE": "use a wrong verb tense"}


class GenerateInverseTests(unittest.TestCase):
    def _run(self, responses, judge_prompt=None, judge_call=None):
        gen = FakeGenerator(responses)
        samples = [{"incorrect": f"bad {i}", "correct": f"This sentence is correct {i}."}
                   for i in range(len(responses))]
        out = gen.generate_inverse(
            real_samples=samples,
            inverse_prompt=_PROMPT,
            error_descriptions=_DESCRIPTIONS,
            type_dist={"R:VERB:TENSE": 1.0},
            count_dist={1: 1.0},
            sample_size=10,
            judge_prompt=judge_prompt,
            judge_call=judge_call,
            rng=random.Random(0),
        )
        return out, gen

    def test_keeps_valid_sample_with_clean_gold_passthrough(self):
        (out, gen) = self._run(["Corrupted: This sentence are correct 0."])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["corrupted"], "This sentence are correct 0.")
        self.assertEqual(out[0]["original"], "This sentence is correct 0.")  # from source_field
        self.assertEqual(out[0]["error_type"], "R:VERB:TENSE")
        # error_spec rendered from the description map into the prompt:
        self.assertIn("use a wrong verb tense", gen.calls[0])

    def test_skips_parse_failure(self):
        # Multiline output without a Corrupted: field (e.g. a reasoning dump)
        # is unparseable. A bare SINGLE line is now accepted as the corrupted
        # text — models often obey "one line" but drop the prefix.
        (out, _) = self._run(["<think>\nreasoning dump\nwithout the field\n</think>"])
        self.assertEqual(out, [])

    def test_skips_identical_corrupted_and_gold(self):
        (out, _) = self._run(["Corrupted: This sentence is correct 0."])
        self.assertEqual(out, [])

    def test_skips_corrupted_under_three_words(self):
        (out, _) = self._run(["Corrupted: Too short."])
        self.assertEqual(out, [])

    def test_judge_drops_sample(self):
        judged = []

        def judge(prompt):
            judged.append(prompt)
            return "Redundancy: trivial\nCorrection: correct"

        (out, _) = self._run(
            ["Corrupted: This sentence are correct 0."],
            judge_prompt="J {sentence} / {correction}",
            judge_call=judge,
        )
        self.assertEqual(out, [])
        self.assertEqual(len(judged), 1)  # judge was consulted

    def test_judge_keeps_valid_sample(self):
        (out, _) = self._run(
            ["Corrupted: This sentence are correct 0."],
            judge_prompt="J {sentence} / {correction}",
            judge_call=lambda p: "Redundancy: valid\nCorrection: correct",
        )
        self.assertEqual(len(out), 1)

    def test_skips_missing_source_field(self):
        gen = FakeGenerator([])  # call_api never reached
        out = gen.generate_inverse(
            real_samples=[{"incorrect": "x"}],  # no "correct" field
            inverse_prompt=_PROMPT,
            error_descriptions=_DESCRIPTIONS,
            type_dist={"R:VERB:TENSE": 1.0},
            count_dist={1: 1.0},
            sample_size=10,
            rng=random.Random(0),
        )
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
