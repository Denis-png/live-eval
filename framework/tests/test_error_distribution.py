import unittest

from framework.pipeline import load_error_distribution
from framework.tasks.base_task import BaseTask


class NoProfileTask:
    """Task whose empirical profiler cannot derive a distribution."""
    def get_task_name(self):
        return "gec"

    def profile_error_distribution(self, real_data, count_max=5, config=None):
        return None


class EmpiricalTask(NoProfileTask):
    """Returns a canned empirical distribution."""
    def profile_error_distribution(self, real_data, count_max=5, config=None):
        return {"type_dist": {"R:VERB:TENSE": 1.0}, "count_dist": {1: 1.0}}


class MinimalTask(BaseTask):
    """Concrete BaseTask with trivial abstractmethod bodies, to test the
    default profile_error_distribution."""
    def get_error_types(self): return []
    def get_prompt_instruction(self): return ""
    def get_evaluators(self): return []
    def get_evaluator_fns(self): return {}
    def get_model(self, model_config): return None
    def get_task_name(self): return "min"
    def parse_row(self, row): return None


class LoadErrorDistributionTests(unittest.TestCase):
    def test_returns_task_empirical_distribution_when_present(self):
        dist = load_error_distribution(
            {}, [{"incorrect": "a", "correct": "b"}], EmpiricalTask()
        )
        self.assertEqual(
            dist, {"type_dist": {"R:VERB:TENSE": 1.0}, "count_dist": {1: 1.0}}
        )

    def test_raises_actionable_error_when_profiler_returns_none(self):
        with self.assertRaises(RuntimeError) as ctx:
            load_error_distribution({}, [], NoProfileTask())
        message = str(ctx.exception)
        self.assertIn("empirical error distribution", message)
        self.assertIn("'gec'", message)
        self.assertIn("generation.sample_size", message)
        self.assertIn("dataset.reference_size", message)

    def test_base_default_profile_returns_none(self):
        self.assertIsNone(MinimalTask().profile_error_distribution([]))

    def test_forwards_config_to_task_profiler(self):
        seen = {}

        class RecordingTask(EmpiricalTask):
            def profile_error_distribution(self, real_data, count_max=5, config=None):
                seen["config"] = config
                return super().profile_error_distribution(real_data, count_max, config)

        cfg = {"generation": {}}
        load_error_distribution(cfg, [], RecordingTask())
        self.assertIs(seen["config"], cfg)


if __name__ == "__main__":
    unittest.main()
