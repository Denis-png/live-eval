"""Forward taxonomy: generate freely, then measure what structure emerged.

This is the baseline that answers whether targeting plus the feedback loop
earns its cost — there is no way to ask that question with inverse alone.
"""
import unittest

from framework import pipeline
from framework.generators.base_generator import BaseGenerator
from framework.tasks.taxonomy.task import TaxonomyTask

_PROFILE = {"taxonomies": [{
    "domain": "Pizza",
    "n_classes": 42,
    "n_subclass_axioms": 55,
    "n_roots": 3,
    "n_leaves": 20,
    "max_depth": 5,
    "mean_depth": 2.4,
    "depth_distribution": {"1": 3, "2": 20},
    "parent_count_distribution": {"1": 40},
    "child_count_distribution": {"0": 20},
    "multiple_parent_fraction": 0.1,
}]}

_VALID = ('{"domain": "Pizza", "classes": ["Margherita", "Pizza"], '
          '"subclass_axioms": [["Margherita", "Pizza"]]}')


class _Recorder(BaseGenerator):
    def __init__(self, reply=_VALID):
        self.prompts = []
        self._reply = reply

    def call_api(self, prompt):
        self.prompts.append(prompt)
        return self._reply


class ForwardPromptTests(unittest.TestCase):
    def test_forward_prompt_carries_the_domain(self):
        prompt = TaxonomyTask().build_structured_generation_prompt(
            _PROFILE, mode="forward")
        self.assertIn("Pizza", prompt)

    def test_forward_prompt_withholds_every_structural_target(self):
        # THE assertion that pins "all structure emerges". Without it the two
        # modes could silently converge while every other test stays green.
        prompt = TaxonomyTask().build_structured_generation_prompt(
            _PROFILE, mode="forward")
        for leaked in ("n_classes", "n_subclass_axioms", "n_roots", "n_leaves",
                       "max_depth", "mean_depth", "depth_distribution",
                       "parent_count_distribution", "child_count_distribution",
                       "multiple_parent_fraction"):
            self.assertNotIn(leaked, prompt, f"{leaked} leaked into forward")
        for value in ("42", "55", "2.4"):
            self.assertNotIn(value, prompt, f"target value {value} leaked")

    def test_inverse_prompt_still_carries_the_targets(self):
        prompt = TaxonomyTask().build_structured_generation_prompt(
            _PROFILE, mode="inverse")
        self.assertIn("n_classes", prompt)
        self.assertIn("42", prompt)

    def test_mode_defaults_to_inverse(self):
        task = TaxonomyTask()
        self.assertEqual(task.build_structured_generation_prompt(_PROFILE),
                         task.build_structured_generation_prompt(_PROFILE,
                                                                 mode="inverse"))


class ForwardDispatchTests(unittest.TestCase):
    def _run(self, gen_cfg, generator=None):
        gen = generator or _Recorder()
        cfg = {"generation": {"sample_size": 1, **gen_cfg},
               "task": {"name": "taxonomy"}}
        out = pipeline._run_generation(gen, TaxonomyTask(), cfg, [], None, None,
                                       0.5, profile=_PROFILE)
        return out, gen

    def test_forward_runs_no_feedback_loop(self):
        out, _ = self._run({"mode": "forward"})
        self.assertEqual(len(out), 1)
        meta = out[0]["generation_feedback"]
        self.assertFalse(meta["feedback_enabled"])
        self.assertEqual(meta["rounds"], [])

    def test_forward_asks_the_model_once_per_sample(self):
        # No feedback rounds means one call per sample when parsing succeeds.
        _, gen = self._run({"mode": "forward"})
        self.assertEqual(len(gen.prompts), 1)

    def test_forward_with_feedback_enabled_fails_before_any_api_call(self):
        gen = _Recorder()
        cfg = {"generation": {"sample_size": 1, "mode": "forward",
                              "feedback": {"enabled": True, "max_rounds": 1}},
               "task": {"name": "taxonomy"}}
        with self.assertRaises(RuntimeError) as ctx:
            pipeline._run_generation(gen, TaxonomyTask(), cfg, [], None, None,
                                     0.5, profile=_PROFILE)
        self.assertIn("forward", str(ctx.exception))
        self.assertEqual(gen.prompts, [], "must fail before spending a call")
