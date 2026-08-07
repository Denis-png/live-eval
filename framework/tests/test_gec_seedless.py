import random
import unittest
from unittest import mock

from framework.generators.base_generator import BaseGenerator


class FakeGenerator(BaseGenerator):
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def call_api(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


PROMPT = "spec: {spec}\nerrors: {error_spec}\nError type / Generated / Ground truth"
GOOD = ("Error type: R:VERB:TENSE\n"
        "Generated: Yesterday she go to the market.\n"
        "Ground truth: Yesterday she went to the market.")


class GenerateSeedlessPairsTests(unittest.TestCase):
    def test_produces_records_and_injects_spec_and_error(self):
        gen = FakeGenerator([GOOD])
        out = gen.generate_seedless_pairs(
            ["topic: travel; roughly 8 words"], PROMPT,
            {"R:VERB:TENSE": "use a wrong verb tense"},
            {"R:VERB:TENSE": 1.0}, {1: 1.0}, rng=random.Random(0),
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["corrupted"], "Yesterday she go to the market.")
        self.assertEqual(out[0]["original"], "Yesterday she went to the market.")
        self.assertEqual(out[0]["error_type"], "R:VERB:TENSE")
        self.assertIn("topic: travel", gen.prompts[0])
        self.assertIn("use a wrong verb tense", gen.prompts[0])

    def test_no_real_text_needed_and_identical_pair_skipped(self):
        gen = FakeGenerator([
            "Error type: R:SPELL\nGenerated: same text here\nGround truth: same text here",
            GOOD,
        ])
        out = gen.generate_seedless_pairs(
            ["a", "b"], PROMPT, {"R:VERB:TENSE": "x"},
            {"R:VERB:TENSE": 1.0}, {1: 1.0}, rng=random.Random(0),
        )
        self.assertEqual(len(out), 1)

    def test_judge_can_drop_a_sample(self):
        gen = FakeGenerator([GOOD])
        out = gen.generate_seedless_pairs(
            ["a"], PROMPT, {"R:VERB:TENSE": "x"}, {"R:VERB:TENSE": 1.0}, {1: 1.0},
            judge_prompt="{sentence}|{correction}",
            judge_call=lambda p: "Redundancy: trivial\nCorrection: correct",
            rng=random.Random(0),
        )
        self.assertEqual(out, [])


class GecSeedlessDispatchTests(unittest.TestCase):
    """The corruption branch must route each (mode, seedless) cell correctly."""

    def _config(self, mode, seedless):
        return {"generation": {"mode": mode, "seedless": seedless, "sample_size": 2}}

    def test_forward_seedless_calls_generate_seedless_pairs(self):
        from framework.pipeline import _run_generation
        from framework.tasks.gec.task import GECTask

        task = GECTask()
        generator = mock.Mock()
        generator.generate_seedless_pairs.return_value = [{"original": "a", "corrupted": "b"}]
        profile = {"profile_version": 2, "topics": {"t": {"fraction": 1.0}},
                   "length_distributions": {"incorrect": {"words": {
                       "bins": {"6-10": 1.0}, "quantiles": {"p90": 9}}}},
                   "style": {"incorrect": {}}}
        out = _run_generation(generator, task, self._config("forward", True), [],
                              {"type_dist": {"R:SPELL": 1.0}, "count_dist": {1: 1.0}},
                              None, 0.5, profile=profile)
        self.assertTrue(generator.generate_seedless_pairs.called)
        self.assertFalse(generator.generate.called)
        self.assertEqual(len(out), 1)

    def test_inverse_seedless_feeds_carriers_into_generate_inverse(self):
        from framework.pipeline import _run_generation
        from framework.tasks.gec.task import GECTask

        task = GECTask()
        generator = mock.Mock()
        generator.generate_carriers.return_value = ["A clean sentence here."]
        generator.generate_inverse.return_value = [{"original": "A clean sentence here.",
                                                    "corrupted": "A clean sentence her."}]
        profile = {"profile_version": 2, "topics": {"t": {"fraction": 1.0}},
                   "length_distributions": {"correct": {"words": {
                       "bins": {"6-10": 1.0}, "quantiles": {"p90": 9}}}},
                   "style": {"correct": {}}}
        _run_generation(generator, task, self._config("inverse", True), [],
                        {"type_dist": {"R:SPELL": 1.0}, "count_dist": {1: 1.0}},
                        None, 0.5, profile=profile)
        self.assertTrue(generator.generate_carriers.called)
        samples = generator.generate_inverse.call_args.kwargs["real_samples"]
        self.assertEqual(samples, [{"correct": "A clean sentence here."}])

    def test_inverse_seedless_raises_without_carrier_prompt(self):
        """A task that lacks a carrier_prompt has no way to synthesize a seedless
        carrier for inverse mode — must fail fast, not silently skip to seeded
        behavior or crash deeper inside generate_carriers."""
        from framework.pipeline import _run_generation
        from framework.tasks.gec.task import GECTask

        task = GECTask()
        generator = mock.Mock()
        profile = {"profile_version": 2, "topics": {"t": {"fraction": 1.0}},
                   "length_distributions": {"correct": {"words": {
                       "bins": {"6-10": 1.0}, "quantiles": {"p90": 9}}}},
                   "style": {"correct": {}}}
        with mock.patch.object(GECTask, "get_carrier_prompt", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                _run_generation(generator, task, self._config("inverse", True), [],
                                {"type_dist": {"R:SPELL": 1.0}, "count_dist": {1: 1.0}},
                                None, 0.5, profile=profile)
        message = str(ctx.exception)
        self.assertIn("gec", message)
        self.assertIn("mode=inverse", message)
        self.assertIn("carrier_prompt", message)
        self.assertFalse(generator.generate_carriers.called)

    def test_forward_seedless_raises_without_seedless_forward_prompt(self):
        """A task that lacks a seedless_forward_prompt has no template to drive
        generate_seedless_pairs — must fail fast rather than call it with None."""
        from framework.pipeline import _run_generation
        from framework.tasks.gec.task import GECTask

        task = GECTask()
        generator = mock.Mock()
        profile = {"profile_version": 2, "topics": {"t": {"fraction": 1.0}},
                   "length_distributions": {"incorrect": {"words": {
                       "bins": {"6-10": 1.0}, "quantiles": {"p90": 9}}}},
                   "style": {"incorrect": {}}}
        with mock.patch.object(GECTask, "get_seedless_forward_prompt", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                _run_generation(generator, task, self._config("forward", True), [],
                                {"type_dist": {"R:SPELL": 1.0}, "count_dist": {1: 1.0}},
                                None, 0.5, profile=profile)
        message = str(ctx.exception)
        self.assertIn("gec", message)
        self.assertIn("mode=forward", message)
        self.assertIn("seedless_forward_prompt", message)
        self.assertFalse(generator.generate_seedless_pairs.called)


if __name__ == "__main__":
    unittest.main()
