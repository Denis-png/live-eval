import random
import unittest

from framework.generators.base_generator import BaseGenerator
from framework.pipeline import _run_generation


class FakeGenerator(BaseGenerator):
    def __init__(self, responses):
        self._responses = list(responses)

    def call_api(self, prompt: str) -> str:
        return self._responses.pop(0)


class FakeTask:
    # forward-mode hooks
    def get_error_types(self):       return ["article"]
    def get_prompt_instruction(self): return "Fix: {sentence}"
    def get_judge_prompt(self):      return None
    # inverse-mode hooks
    def get_inverse_prompt(self):       return "Errors: {error_spec}\nCorrect: {sentence}"
    def get_inverse_judge_prompt(self): return None
    def get_error_descriptions(self):   return {"R:VERB:TENSE": "use a wrong verb tense"}


_REAL = [{"incorrect": "He go to school.", "correct": "He goes to school daily."}]
_DIST = {"type_dist": {"R:VERB:TENSE": 1.0}, "count_dist": {1: 1.0}}


class RunGenerationTests(unittest.TestCase):
    def test_forward_mode_uses_generate(self):
        gen = FakeGenerator([
            "Error type: verb_tense\nGenerated: She go there often now.\nGround truth: She goes there often now."
        ])
        config = {"generation": {"mode": "forward", "sample_size": 5}}
        out = _run_generation(gen, FakeTask(), config, _REAL, None, judge_call=None)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["corrupted"], "She go there often now.")

    def test_inverse_mode_uses_generate_inverse(self):
        gen = FakeGenerator(["Corrupted: He go to school daily."])
        config = {"generation": {"mode": "inverse", "sample_size": 5,
                                 "inverse": {"source_field": "correct"}}}
        out = _run_generation(gen, FakeTask(), config, _REAL, _DIST, judge_call=None)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["original"], "He goes to school daily.")  # clean source
        self.assertEqual(out[0]["corrupted"], "He go to school daily.")
        self.assertEqual(out[0]["error_type"], "R:VERB:TENSE")

    def test_defaults_to_forward_when_mode_absent(self):
        gen = FakeGenerator([
            "Error type: verb_tense\nGenerated: She go there often now.\nGround truth: She goes there often now."
        ])
        config = {"generation": {"sample_size": 5}}
        out = _run_generation(gen, FakeTask(), config, _REAL, None, judge_call=None)
        self.assertEqual(len(out), 1)

    def test_inverse_raises_when_source_field_missing_from_all_samples(self):
        """_run_generation must raise ValueError if the configured source_field is
        absent from every real sample — prevents silent zero-output runs (e.g.
        mode=inverse, task=spam, but source_field left at the 'correct' default)."""
        gen = FakeGenerator([])  # generate_inverse should never be reached
        config = {
            "generation": {
                "mode": "inverse",
                "sample_size": 5,
                "inverse": {"source_field": "correct"},
            }
        }
        # Only 'incorrect' key present — no 'correct' key on any sample
        data_without_correct = [{"incorrect": "x y z"}]
        with self.assertRaises(ValueError) as ctx:
            _run_generation(gen, FakeTask(), config, data_without_correct, _DIST, judge_call=None)
        self.assertIn("source_field", str(ctx.exception))
        self.assertIn("correct", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
