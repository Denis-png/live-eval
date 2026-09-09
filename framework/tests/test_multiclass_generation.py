"""class_conditional must hold N labels, not two.

SentimentTask already defines {0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE} and
macro-averaged evaluators, and declares the `corruption` strategy only because
this strategy could not hold it.
"""
import unittest
from collections import Counter
from random import Random

from framework import pipeline
from framework.generators.base_generator import BaseGenerator
from framework.tasks.base_task import BaseTask

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


class _ThreeClassTask(BaseTask):
    """A synthetic three-class class_conditional task.

    Only NEGATIVE's template carries {error_spec}, so the signal-bearing label
    is neither the first nor the last of the set — a positional rule would have
    to pick one of those and would be caught here.
    """

    def get_class_labels(self):
        return _LABELS

    def get_generation_strategy(self):
        return "class_conditional"

    def get_inverse_class_prompts(self):
        return {
            "NEGATIVE": "Write a negative review using {error_spec} of: {sentence}",
            "NEUTRAL": "Write a neutral review of: {sentence}",
            "POSITIVE": "Write a positive review of: {sentence}",
        }

    def get_error_descriptions(self):
        return {"tone": "an emphatic tone"}

    def get_task_name(self):
        return "three-class"

    # unused-but-abstract
    def get_error_types(self): return []
    def get_prompt_instruction(self): return ""
    def get_evaluators(self): return []
    def get_evaluator_fns(self): return {}
    def get_model(self, model_config): return None
    def parse_row(self, row): return row


class ThreeLabelDispatchTests(unittest.TestCase):
    """The three-class task driven through the REAL dispatch.

    The tests above call generate_class_conditional directly, so a two-label
    assumption reintroduced in pipeline._run_generation would not be caught by
    them — and the only other dispatch-level test uses a TWO-label task. This
    is the spec's "synthetic three-class task driven through the real dispatch".
    """

    # Large enough that a label at 0.15 going unseen, or the 0.7 label failing
    # to outweigh the other two combined, is a ~1e-5 event.
    SAMPLE_SIZE = 90

    def _dispatch(self, class_balance):
        gen = _Echo()
        cfg = {"generation": {"sample_size": self.SAMPLE_SIZE,
                              "mode": "inverse", "seedless": False},
               "task": {"name": "three-class"}}
        out = pipeline._run_generation(
            gen, _ThreeClassTask(), cfg,
            [{"incorrect": f"product number {i}"} for i in range(self.SAMPLE_SIZE)],
            {"type_dist": {"tone": 1.0}, "count_dist": {1: 1.0}},
            None, class_balance,
        )
        return out, gen

    def test_dispatch_honours_the_balance_vector_across_all_three_labels(self):
        out, _ = self._dispatch({"NEGATIVE": 0.7, "NEUTRAL": 0.15, "POSITIVE": 0.15})
        counts = Counter(r["label"] for r in out)
        self.assertEqual(set(counts), set(_LABELS),
                         "every declared label must be reachable through the dispatch")
        self.assertGreater(counts["NEGATIVE"],
                           counts["NEUTRAL"] + counts["POSITIVE"])

    def test_dispatch_can_suppress_a_label_entirely(self):
        # A zero weight is honoured rather than rounded into existence, and the
        # remaining two still both appear — so the vector is being consumed as a
        # vector, not collapsed to a single P(first label).
        out, _ = self._dispatch({"NEGATIVE": 0.0, "NEUTRAL": 0.5, "POSITIVE": 0.5})
        counts = Counter(r["label"] for r in out)
        self.assertEqual(set(counts), {"NEUTRAL", "POSITIVE"})

    def test_dispatch_puts_signals_only_on_the_error_spec_bearing_label(self):
        out, gen = self._dispatch(
            {"NEGATIVE": 1 / 3, "NEUTRAL": 1 / 3, "POSITIVE": 1 / 3})
        by_label = {}
        for record in out:
            by_label.setdefault(record["label"], []).append(record)
        self.assertEqual(set(by_label), set(_LABELS))
        for record in by_label["NEGATIVE"]:
            self.assertEqual(record["technique"], "tone")
        for label in ("NEUTRAL", "POSITIVE"):
            for record in by_label[label]:
                self.assertEqual(record["technique"], "rewrite")
        # ... and the rendered prompts agree: only the negative template ever
        # carries the description text.
        emphatic = [p for p in gen.prompts if "an emphatic tone" in p]
        self.assertTrue(emphatic)
        for prompt in emphatic:
            self.assertIn("Write a negative review", prompt)

    def test_dispatch_fails_before_any_call_when_a_label_has_no_prompt(self):
        class _Incomplete(_ThreeClassTask):
            def get_inverse_class_prompts(self):
                prompts = super().get_inverse_class_prompts()
                del prompts["NEUTRAL"]
                return prompts

        gen = _Echo()
        cfg = {"generation": {"sample_size": 4, "mode": "inverse", "seedless": False},
               "task": {"name": "three-class"}}
        with self.assertRaises(RuntimeError) as ctx:
            pipeline._run_generation(
                gen, _Incomplete(), cfg, [{"incorrect": "product one"}],
                {"type_dist": {"tone": 1.0}, "count_dist": {1: 1.0}},
                None, {"NEGATIVE": 1 / 3, "NEUTRAL": 1 / 3, "POSITIVE": 1 / 3},
            )
        message = str(ctx.exception)
        self.assertIn("NEUTRAL", message)
        self.assertIn("get_inverse_class_prompts", message)
        self.assertEqual(gen.prompts, [])
