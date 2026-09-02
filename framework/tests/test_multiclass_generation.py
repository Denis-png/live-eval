"""class_conditional must hold N labels, not two.

SentimentTask already defines {0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE} and
macro-averaged evaluators, and declares the `corruption` strategy only because
this strategy could not hold it.
"""
import unittest
from collections import Counter
from random import Random

from framework.generators.base_generator import BaseGenerator

_LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
_PROMPTS = {
    "NEGATIVE": "Write a negative review of: {sentence}",
    "NEUTRAL": "Write a neutral review of: {sentence}",
    "POSITIVE": "Write a positive review of: {sentence}",
}


class _Echo(BaseGenerator):
    def __init__(self):
        self.prompts = []

    def call_api(self, prompt):
        self.prompts.append(prompt)
        return "Message: a perfectly ordinary sentence about the thing"


def _run(generator, balance, prompts=_PROMPTS, seeds=None, rng_seed=0):
    return generator.generate_class_conditional(
        class_balance=balance,
        labels=_LABELS,
        inverse_prompts=prompts,
        type_dist={"tone": 1.0},
        count_dist={1: 1.0},
        error_descriptions={"tone": "an emphatic tone"},
        sample_size=60,
        seed_policy="impose",
        real_seeds=seeds or [{"incorrect": f"product {i}"} for i in range(60)],
        seed_field="incorrect",
        rng=Random(rng_seed),
    )


class ThreeLabelTests(unittest.TestCase):
    def test_all_three_labels_are_produced(self):
        out = _run(_Echo(), {"NEGATIVE": 1 / 3, "NEUTRAL": 1 / 3, "POSITIVE": 1 / 3})
        self.assertEqual(set(r["label"] for r in out), set(_LABELS))

    def test_the_balance_vector_shapes_the_output(self):
        out = _run(_Echo(), {"NEGATIVE": 0.8, "NEUTRAL": 0.1, "POSITIVE": 0.1})
        counts = Counter(r["label"] for r in out)
        self.assertGreater(counts["NEGATIVE"], counts["NEUTRAL"] + counts["POSITIVE"])

    def test_attrition_is_tracked_for_every_label(self):
        gen = _Echo()
        _run(gen, {"NEGATIVE": 1 / 3, "NEUTRAL": 1 / 3, "POSITIVE": 1 / 3})
        self.assertEqual(set(gen.last_class_attrition), set(_LABELS))
        for label, counts in gen.last_class_attrition.items():
            self.assertGreaterEqual(counts["attempted"], counts["survived"])

    def test_each_label_renders_its_own_prompt(self):
        gen = _Echo()
        _run(gen, {"NEGATIVE": 1 / 3, "NEUTRAL": 1 / 3, "POSITIVE": 1 / 3})
        joined = "\n".join(gen.prompts)
        for word in ("negative", "neutral", "positive"):
            self.assertIn(f"Write a {word} review", joined)


class TemplateDrivenSignalTests(unittest.TestCase):
    def test_a_label_gets_signals_only_when_its_template_asks(self):
        prompts = {
            "NEGATIVE": "Write a negative review using {error_spec} of: {sentence}",
            "NEUTRAL": "Write a neutral review of: {sentence}",
            "POSITIVE": "Write a positive review of: {sentence}",
        }
        gen = _Echo()
        out = _run(gen, {"NEGATIVE": 1 / 3, "NEUTRAL": 1 / 3, "POSITIVE": 1 / 3},
                   prompts=prompts)
        by_label = {r["label"]: r for r in out}
        self.assertIn("tone", by_label["NEGATIVE"]["technique"])
        self.assertNotIn("tone", by_label["NEUTRAL"]["technique"])
        self.assertNotIn("tone", by_label["POSITIVE"]["technique"])

    def test_no_template_asking_means_no_signals_anywhere(self):
        gen = _Echo()
        out = _run(gen, {"NEGATIVE": 0.5, "POSITIVE": 0.5},
                   prompts={k: v for k, v in _PROMPTS.items() if k != "NEUTRAL"})
        for record in out:
            self.assertNotIn("tone", record["technique"])
