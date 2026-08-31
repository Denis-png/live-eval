"""The `structured` strategy must declare its hooks like every other strategy.

Three methods the structured branch needs were absent from BaseTask:
get_feedback_config and parse_structured_generation_with_diagnostics were
probed with hasattr() — so a task that omitted them silently lost its feedback
loop — and build_structural_feedback was called unconditionally, so omitting it
crashed with AttributeError mid-loop, after paying for an API call.

That contradicts the framework's own rule: an unsupported capability raises
before any API call, naming what is missing, and never silently degrades.
"""
import unittest

from framework import pipeline
from framework.generators.base_generator import BaseGenerator
from framework.tasks.base_task import BaseTask
from framework.tasks.taxonomy.task import TaxonomyTask


class _Structured(BaseTask):
    """Minimal structured task that implements only the declared essentials."""

    def get_generation_strategy(self): return "structured"
    def get_task_name(self): return "structured_stub"
    def build_structured_generation_prompt(self, profile, rng=None, feedback=None):
        return "make one"
    def parse_structured_generation(self, text):
        return {"classes": ["A"]} if "good" in text else None

    def get_error_types(self): return []
    def get_prompt_instruction(self): return ""
    def get_evaluators(self): return []
    def get_evaluator_fns(self): return {}
    def get_model(self, model_config): return None
    def parse_row(self, row): return row


class _Gen(BaseGenerator):
    def __init__(self, replies): self._r = list(replies)
    def call_api(self, prompt): return self._r.pop(0)


class ContractTests(unittest.TestCase):
    def test_all_three_structured_hooks_are_declared(self):
        for hook in ("get_feedback_config",
                     "parse_structured_generation_with_diagnostics",
                     "build_structural_feedback"):
            self.assertTrue(hasattr(BaseTask, hook), f"BaseTask lacks {hook}")

    def test_feedback_is_off_by_default(self):
        cfg = _Structured().get_feedback_config({})
        self.assertFalse(cfg.get("enabled"))

    def test_diagnostics_default_wraps_the_plain_parser(self):
        """A task implementing only parse_structured_generation still gets
        diagnostics, so the dispatcher needs no branch."""
        t = _Structured()
        ok = t.parse_structured_generation_with_diagnostics("good")
        self.assertEqual(ok["artifact"], {"classes": ["A"]})
        self.assertTrue(ok["diagnostic"]["valid"])

        bad = t.parse_structured_generation_with_diagnostics("bad")
        self.assertIsNone(bad["artifact"])
        self.assertFalse(bad["diagnostic"]["valid"])
        self.assertIn("rejection_reason", bad["diagnostic"])

    def test_taxonomy_still_overrides_with_its_richer_diagnostics(self):
        t = TaxonomyTask()
        out = t.parse_structured_generation_with_diagnostics("not json at all")
        self.assertIsNone(out["artifact"])
        self.assertIn("rejection_reason", out["diagnostic"])


class FailFastTests(unittest.TestCase):
    def _run(self, task, replies=("good",)):
        cfg = {"generation": {"sample_size": 1}, "task": {"name": task.get_task_name()}}
        return pipeline._run_generation(
            _Gen(replies), task, cfg, [], None, None, 0.5,
            profile={"domain": "x"},
        )

    def test_feedback_enabled_without_a_builder_fails_before_any_api_call(self):
        class _WantsFeedback(_Structured):
            def get_feedback_config(self, generation_config=None):
                return {"enabled": True, "max_rounds": 1}

        gen = _Gen(["good"])
        cfg = {"generation": {"sample_size": 1}, "task": {"name": "structured_stub"}}
        with self.assertRaises(RuntimeError) as ctx:
            pipeline._run_generation(gen, _WantsFeedback(), cfg, [], None, None, 0.5,
                                     profile={"domain": "x"})
        self.assertIn("build_structural_feedback", str(ctx.exception))
        self.assertEqual(len(gen._r), 1, "must fail before spending an API call")

    def test_structured_task_without_feedback_still_generates(self):
        out = self._run(_Structured())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["classes"], ["A"])
