"""The class_conditional strategy must not be hardcoded to spam's vocabulary.

Before this, `_run_generation` passed positive_label="SPAM"/negative_label="HAM"
literally and called `task.get_ham_generation_prompt()` — a method that was not
on BaseTask at all. Any second classification task would have had to implement a
spam-named method and would still have been generated with spam's labels.
"""
import unittest
from random import Random
from unittest import mock

from framework import pipeline
from framework.generators.base_generator import BaseGenerator
from framework.tasks.base_task import BaseTask
from framework.tasks.spam.task import SpamTask


class _Toxicity(BaseTask):
    """A second class_conditional task that is not spam."""

    def get_class_labels(self):
        return ("TOXIC", "CIVIL")

    def get_generation_strategy(self):
        return "class_conditional"

    def get_negative_generation_prompt(self):
        return "Rewrite civilly: {sentence}"

    def get_inverse_prompt(self):
        return "Make TOXIC using {error_spec} from: {sentence}"

    def get_error_descriptions(self):
        return {"slur": "include a slur"}

    def get_task_name(self):
        return "toxicity"

    # unused-but-abstract
    def get_error_types(self): return []
    def get_prompt_instruction(self): return ""
    def get_evaluators(self): return []
    def get_evaluator_fns(self): return {}
    def get_model(self, model_config): return None
    def parse_row(self, row): return row


class _Fake(BaseGenerator):
    def __init__(self, n):
        self._n = n
        self.calls = []

    def call_api(self, prompt):
        self.calls.append(prompt)
        return "Corrupted: this is a generated line"


class ContractTests(unittest.TestCase):
    def test_base_task_declares_both_hooks(self):
        # The old code called get_ham_generation_prompt(), which BaseTask never
        # declared — the contract lied about what a class_conditional task needs.
        self.assertTrue(hasattr(BaseTask, "get_class_labels"))
        self.assertTrue(hasattr(BaseTask, "get_negative_generation_prompt"))

    def test_non_classification_tasks_declare_no_labels(self):
        from framework.tasks.gec.task import GECTask
        self.assertIsNone(GECTask().get_class_labels())

    def test_spam_declares_its_labels(self):
        self.assertEqual(SpamTask().get_class_labels(), ("SPAM", "HAM"))

    def test_spam_negative_prompt_survives_the_rename(self):
        self.assertTrue(SpamTask().get_negative_generation_prompt())


class DispatchTests(unittest.TestCase):
    def _run(self, task, class_prob):
        gen = _Fake(4)
        cfg = {"generation": {"sample_size": 4, "mode": "inverse", "seedless": False},
               "task": {"name": task.get_task_name()}}
        return pipeline._run_generation(
            gen, task, cfg,
            [{"incorrect": f"a civil sentence number {i}"} for i in range(4)],
            {"type_dist": {"slur": 1.0}, "count_dist": {1: 1.0}},
            None, class_prob,
        )

    def test_generation_uses_the_tasks_own_labels(self):
        out = self._run(_Toxicity(), class_prob=1.0)
        self.assertTrue(out)
        self.assertEqual({r["label"] for r in out}, {"TOXIC"})

    def test_negative_class_uses_the_tasks_own_labels(self):
        out = self._run(_Toxicity(), class_prob=0.0)
        self.assertTrue(out)
        self.assertEqual({r["label"] for r in out}, {"CIVIL"})

    def test_class_conditional_task_without_labels_fails_fast(self):
        class _Unlabelled(_Toxicity):
            def get_class_labels(self): return None

        with self.assertRaises(RuntimeError) as ctx:
            self._run(_Unlabelled(), class_prob=1.0)
        self.assertIn("get_class_labels", str(ctx.exception))


class BalanceTests(unittest.TestCase):
    def test_empirical_balance_counts_the_tasks_positive_label(self):
        rows = [{"label": "TOXIC"}, {"label": "CIVIL"}, {"label": "TOXIC"},
                {"label": "CIVIL"}]
        self.assertAlmostEqual(
            pipeline._resolve_class_prob({}, rows, _Toxicity()), 0.5)

    def test_spam_balance_is_unchanged(self):
        rows = [{"label": "SPAM"}] + [{"label": "HAM"}] * 3
        self.assertAlmostEqual(
            pipeline._resolve_class_prob({}, rows, SpamTask()), 0.25)


class InformativeCountTests(unittest.TestCase):
    def test_counts_the_tasks_positive_label(self):
        from framework import calibrate
        rows = [{"label": "TOXIC"}, {"label": "CIVIL"}, {"label": "TOXIC"}]
        self.assertEqual(calibrate.informative_count(_Toxicity(), rows), 2)
