import random
import unittest

from framework.generators.base_generator import BaseGenerator


class FakeGenerator(BaseGenerator):
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def call_api(self, prompt):
        self.prompts.append(prompt)
        return self.response


COMMON = dict(
    class_prob=1.0, positive_label="SPAM", negative_label="HAM",
    type_dist={"phishing_link": 1.0}, count_dist={1: 1.0},
    error_descriptions={"phishing_link": "insert a link"},
    inject_prompt="inject {error_spec} into {sentence}",
    negative_prompt="rewrite {sentence}",
)


class SeedPolicyTests(unittest.TestCase):
    def test_cross_class_seeds_from_the_negative_class_only(self):
        gen = FakeGenerator("Corrupted: Win a FREE prize http://x.com now")
        out = gen.generate_class_conditional(
            real_seeds=[{"incorrect": "see you at lunch"}], seed_field="incorrect",
            sample_size=1, seed_policy="cross_class", rng=random.Random(0), **COMMON,
        )
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertIn("see you at lunch", gen.prompts[0])

    def test_same_class_picks_a_seed_of_the_drawn_class(self):
        gen = FakeGenerator("Rewritten: CLAIM your FREE reward today")
        seeds = [{"text": "see you at lunch", "label": "HAM"},
                 {"text": "WIN cash now", "label": "SPAM"}]
        out = gen.generate_class_conditional(
            real_seeds=seeds, seed_field="text", sample_size=1,
            seed_policy="same_class", forward_prompt="new {class_name} like: {sentence}",
            rng=random.Random(0), **COMMON,
        )
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertIn("WIN cash now", gen.prompts[0])
        self.assertNotIn("see you at lunch", gen.prompts[0])

    def test_same_class_missing_class_in_pool_raises(self):
        gen = FakeGenerator("Rewritten: x")
        with self.assertRaises(RuntimeError) as ctx:
            gen.generate_class_conditional(
                real_seeds=[{"text": "hi", "label": "HAM"}], seed_field="text",
                sample_size=1, seed_policy="same_class",
                forward_prompt="{class_name} {sentence}", rng=random.Random(0), **COMMON,
            )
        self.assertIn("SPAM", str(ctx.exception))

    def test_none_policy_uses_specs_and_never_touches_seeds(self):
        gen = FakeGenerator("Message: FREE prize, click http://x.com")
        out = gen.generate_class_conditional(
            real_seeds=None, sample_size=1, seed_policy="none",
            seedless_prompts={"SPAM": "spam {spec} using {error_spec}",
                              "HAM": "ham {spec}"},
            specs_by_label={"SPAM": ["topic: prizes; roughly 12 words"],
                            "HAM": ["topic: chat; roughly 8 words"]},
            rng=random.Random(0), **COMMON,
        )
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertIn("topic: prizes", gen.prompts[0])
        self.assertIn("insert a link", gen.prompts[0])

    def test_none_policy_negative_class_uses_its_own_prompt(self):
        gen = FakeGenerator("Message: are we still on for lunch tomorrow")
        common = {**COMMON, "class_prob": 0.0}
        out = gen.generate_class_conditional(
            real_seeds=None, sample_size=1, seed_policy="none",
            seedless_prompts={"SPAM": "spam {spec} {error_spec}", "HAM": "ham {spec}"},
            specs_by_label={"SPAM": ["x"], "HAM": ["topic: chat; roughly 8 words"]},
            rng=random.Random(0), **common,
        )
        self.assertEqual(out[0]["label"], "HAM")
        self.assertIn("topic: chat", gen.prompts[0])


if __name__ == "__main__":
    unittest.main()
