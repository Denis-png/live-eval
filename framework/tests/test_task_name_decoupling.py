"""main.py must branch on the task's generation STRATEGY, not on its name.

Two checks were keyed on the literal string "taxonomy". A second structured
task would silently skip the seeded-generation guard and be mislabelled at
startup — and the framework has spent this whole line of work removing exactly
this coupling from the dispatch (get_class_labels, get_generation_strategy,
get_inverse_class_prompts all exist so that no shared code names one task).
"""
import unittest
from unittest import mock

from framework import main as fmain
from framework.tasks.base_task import BaseTask


class _SecondStructuredTask(BaseTask):
    """A structured task that is NOT called 'taxonomy'."""

    def get_generation_strategy(self):
        return "structured"

    def get_task_name(self):
        return "ontology"

    def get_error_types(self): return []
    def get_prompt_instruction(self): return ""
    def get_evaluators(self): return []
    def get_evaluator_fns(self): return {}
    def get_model(self, model_config): return None
    def parse_row(self, row): return row


def _config(task_name, **gen):
    return {
        "task": {"name": task_name},
        "dataset": {"source": "local", "local": {"path": "x.jsonl", "format": "jsonl"}},
        "generation": {"provider": "openrouter", "model": "m", "num_runs": 1,
                       "sample_size": 5, **gen},
        "task_models": [{"name": "m", "type": "llm"}],
    }


class SeededStructuredGuardTests(unittest.TestCase):
    def test_a_second_structured_task_also_rejects_seeded_generation(self):
        cfg = _config("ontology", seedless=False)
        with mock.patch.object(fmain, "load_task", return_value=_SecondStructuredTask()):
            with self.assertRaises(ValueError) as ctx:
                fmain.validate_config(cfg)
        self.assertIn("not implemented", str(ctx.exception))
        self.assertIn("ontology", str(ctx.exception))

    def test_a_corruption_task_may_be_seeded(self):
        # The guard must not leak onto tasks whose strategy supports seeds.
        cfg = _config("gec", seedless=False, mode="forward")
        self.assertIsNone(fmain.validate_config(cfg))

    def test_taxonomy_itself_still_rejects_seeded_generation(self):
        cfg = _config("taxonomy", seedless=False)
        with self.assertRaises(ValueError) as ctx:
            fmain.validate_config(cfg)
        self.assertIn("not implemented", str(ctx.exception))


class DisplayModeTests(unittest.TestCase):
    def test_a_second_structured_task_reports_its_resolved_mode(self):
        cfg = _config("ontology", seedless=True)
        with mock.patch.object(fmain, "load_task", return_value=_SecondStructuredTask()):
            self.assertEqual(fmain._display_generation_mode(cfg), "inverse")

    def test_a_class_conditional_task_is_not_labelled_n_a_any_more(self):
        # Spam dispatches four (mode, seedless) cells; calling its mode "n/a"
        # predates that and contradicts what the run actually does.
        cfg = _config("spam", mode="inverse", seedless=False)
        self.assertEqual(fmain._display_generation_mode(cfg), "inverse")

    def test_a_corruption_task_reports_its_mode(self):
        cfg = _config("gec", mode="forward", seedless=False)
        self.assertEqual(fmain._display_generation_mode(cfg), "forward")


if __name__ == "__main__":
    unittest.main()
