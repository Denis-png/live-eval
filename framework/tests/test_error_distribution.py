import unittest

from framework.pipeline import _poisson_pmf, load_error_distribution


class FakeTask:
    def get_error_descriptions(self):
        return {"R:VERB:TENSE": "use a wrong verb tense", "M:DET": "omit a required article"}


class PoissonPmfTests(unittest.TestCase):
    def test_keys_span_range_and_sum_to_one(self):
        pmf = _poisson_pmf(1.5, n_min=1, n_max=5)
        self.assertEqual(sorted(pmf.keys()), [1, 2, 3, 4, 5])
        self.assertAlmostEqual(sum(pmf.values()), 1.0)
        self.assertTrue(all(p >= 0 for p in pmf.values()))


class LoadErrorDistributionTests(unittest.TestCase):
    def _config(self, **inverse):
        return {"generation": {"inverse": {"placeholder_distribution": inverse}}}

    def test_type_dist_uniform_over_task_categories(self):
        dist = load_error_distribution(self._config(), [], FakeTask())
        self.assertEqual(set(dist["type_dist"].keys()), {"R:VERB:TENSE", "M:DET"})
        self.assertAlmostEqual(dist["type_dist"]["R:VERB:TENSE"], 0.5)
        self.assertAlmostEqual(sum(dist["type_dist"].values()), 1.0)

    def test_count_dist_respects_count_max_and_sums_to_one(self):
        dist = load_error_distribution(self._config(count_mean=2.0, count_max=3), [], FakeTask())
        self.assertEqual(sorted(dist["count_dist"].keys()), [1, 2, 3])
        self.assertAlmostEqual(sum(dist["count_dist"].values()), 1.0)

    def test_raises_when_task_has_no_categories(self):
        class Empty:
            def get_error_descriptions(self):
                return {}
        with self.assertRaises(ValueError):
            load_error_distribution(self._config(), [], Empty())


if __name__ == "__main__":
    unittest.main()
