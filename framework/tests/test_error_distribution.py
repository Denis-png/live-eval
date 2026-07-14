import unittest

from framework.pipeline import _poisson_pmf, load_error_distribution
from framework.tasks.base_task import BaseTask


class FakeTask:
    """Placeholder-path task: has categories but no empirical profiler."""
    def get_error_descriptions(self):
        return {"R:VERB:TENSE": "use a wrong verb tense", "M:DET": "omit a required article"}

    def profile_error_distribution(self, real_data, count_max=5, config=None):
        return None


class EmpiricalTask(FakeTask):
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


class PoissonPmfTests(unittest.TestCase):
    def test_keys_span_range_and_sum_to_one(self):
        pmf = _poisson_pmf(1.5, n_min=1, n_max=5)
        self.assertEqual(sorted(pmf.keys()), [1, 2, 3, 4, 5])
        self.assertAlmostEqual(sum(pmf.values()), 1.0)
        self.assertTrue(all(p >= 0 for p in pmf.values()))


class LoadErrorDistributionTests(unittest.TestCase):
    def _config(self, **inverse):
        return {"generation": {"inverse": {"placeholder_distribution": inverse}}}

    def test_returns_task_empirical_distribution_when_present(self):
        dist = load_error_distribution(self._config(), [{"incorrect": "a", "correct": "b"}],
                                       EmpiricalTask())
        self.assertEqual(dist, {"type_dist": {"R:VERB:TENSE": 1.0}, "count_dist": {1: 1.0}})

    def test_falls_back_to_placeholder_when_task_returns_none(self):
        dist = load_error_distribution(self._config(), [], FakeTask())
        self.assertEqual(set(dist["type_dist"].keys()), {"R:VERB:TENSE", "M:DET"})
        self.assertAlmostEqual(dist["type_dist"]["R:VERB:TENSE"], 0.5)
        self.assertAlmostEqual(sum(dist["type_dist"].values()), 1.0)

    def test_placeholder_count_dist_respects_count_max_and_sums_to_one(self):
        dist = load_error_distribution(self._config(count_mean=2.0, count_max=3), [], FakeTask())
        self.assertEqual(sorted(dist["count_dist"].keys()), [1, 2, 3])
        self.assertAlmostEqual(sum(dist["count_dist"].values()), 1.0)

    def test_raises_when_task_has_no_categories(self):
        class Empty(FakeTask):
            def get_error_descriptions(self):
                return {}
        with self.assertRaises(ValueError):
            load_error_distribution(self._config(), [], Empty())

    def test_base_default_profile_returns_none(self):
        self.assertIsNone(MinimalTask().profile_error_distribution([]))

    def test_forwards_config_to_task_profiler(self):
        seen = {}

        class RecordingTask(FakeTask):
            def profile_error_distribution(self, real_data, count_max=5, config=None):
                seen["config"] = config
                return {"type_dist": {"R:VERB:TENSE": 1.0}, "count_dist": {1: 1.0}}

        cfg = self._config()
        load_error_distribution(cfg, [], RecordingTask())
        self.assertIs(seen["config"], cfg)


if __name__ == "__main__":
    unittest.main()
