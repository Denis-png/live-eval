"""Inverse draws the target label independently, so it must work in every
direction — including HAM produced FROM a spam seed, which is the hard negative
the benchmark currently lacks: spam-like topic carrying no spam signals."""
import unittest
from random import Random

from framework.generators.base_generator import BaseGenerator

_LABELS = ("SPAM", "HAM")
_PROMPTS = {
    "SPAM": "Make it spam using {error_spec} from: {sentence}",
    "HAM": "Remove any spam characteristics from: {sentence}",
}


class _Echo(BaseGenerator):
    def call_api(self, prompt):
        return "Message: a rewritten line of text here"


def _run(balance, seeds, rng_seed=0):
    return _Echo().generate_class_conditional(
        class_balance=balance, labels=_LABELS, inverse_prompts=_PROMPTS,
        type_dist={"link": 1.0}, count_dist={1: 1.0},
        error_descriptions={"link": "a suspicious link"},
        sample_size=40, seed_policy="impose",
        real_seeds=seeds, seed_field="incorrect", rng=Random(rng_seed),
    )


_MIXED = [{"incorrect": f"text {i}", "label": "SPAM" if i % 2 else "HAM"}
          for i in range(40)]


class SourceLabelTests(unittest.TestCase):
    def test_records_carry_the_seed_s_label(self):
        out = _run({"SPAM": 0.5, "HAM": 0.5}, _MIXED)
        self.assertTrue(out)
        for record in out:
            self.assertIn(record["source_label"], ("SPAM", "HAM"))

    def test_ham_is_produced_from_spam_seeds(self):
        # The behaviour this task exists for. Without the source label this
        # assertion could not be written at all.
        out = _run({"SPAM": 0.0, "HAM": 1.0}, _MIXED)
        self.assertTrue(out)
        self.assertEqual({r["label"] for r in out}, {"HAM"})
        self.assertIn("SPAM", {r["source_label"] for r in out},
                      "no HAM sample derived from a spam seed")

    def test_spam_is_produced_from_spam_seeds_too(self):
        # Imposition does not require the classes to differ; a drawn label may
        # coincide with the seed's, which is why "cross_class" was a lie.
        out = _run({"SPAM": 1.0, "HAM": 0.0}, _MIXED)
        self.assertIn("SPAM", {r["source_label"] for r in out})

    def test_missing_seed_label_records_none_rather_than_guessing(self):
        out = _run({"HAM": 1.0, "SPAM": 0.0},
                   [{"incorrect": f"t {i}"} for i in range(40)])
        self.assertTrue(out)
        self.assertEqual({r["source_label"] for r in out}, {None})
