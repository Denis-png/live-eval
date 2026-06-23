import argparse
import math
import unittest

from framework.pipeline import _get_field, aggregate, _mean_std


class GetFieldTests(unittest.TestCase):
    def test_returns_first_matching_candidate(self):
        row = {"input": "bad sentence", "output": "good sentence"}
        self.assertEqual(_get_field(row, ["input", "text"]), "bad sentence")

    def test_skips_empty_candidate(self):
        row = {"input": "", "text": "fallback text"}
        self.assertEqual(_get_field(row, ["input", "text"]), "fallback text")

    def test_returns_none_when_no_candidate_matches(self):
        # Must NOT silently grab an arbitrary unrelated column.
        row = {"some_unrelated_column": "noise", "another": "more noise"}
        self.assertIsNone(_get_field(row, ["input", "output"]))


class AggregateTests(unittest.TestCase):
    def test_scalar_mean_and_sample_std(self):
        runs = [
            {"m": {"gleu": 0.2}},
            {"m": {"gleu": 0.4}},
            {"m": {"gleu": 0.6}},
        ]
        out = aggregate(runs)
        self.assertAlmostEqual(out["m"]["gleu"]["mean"], 0.4)
        # sample std (ddof=1) of [.2,.4,.6] = 0.2, not population std 0.1633
        self.assertAlmostEqual(out["m"]["gleu"]["std"], 0.2)

    def test_single_run_has_zero_std(self):
        out = aggregate([{"m": {"gleu": 0.5}}])
        self.assertEqual(out["m"]["gleu"]["mean"], 0.5)
        self.assertEqual(out["m"]["gleu"]["std"], 0.0)
        self.assertFalse(math.isnan(out["m"]["gleu"]["std"]))

    def test_nested_dict_metric(self):
        runs = [
            {"m": {"errant": {"precision": 0.5}}},
            {"m": {"errant": {"precision": 0.7}}},
        ]
        out = aggregate(runs)
        self.assertAlmostEqual(out["m"]["errant"]["precision"]["mean"], 0.6)

    def test_run_missing_a_model_does_not_crash(self):
        # A later run that failed to score a model should be skipped for that
        # model, not raise KeyError mid-aggregation.
        runs = [
            {"a": {"gleu": 0.2}, "b": {"gleu": 0.8}},
            {"a": {"gleu": 0.4}},  # model "b" missing this run
        ]
        out = aggregate(runs)
        self.assertAlmostEqual(out["a"]["gleu"]["mean"], 0.3)
        self.assertAlmostEqual(out["b"]["gleu"]["mean"], 0.8)


if __name__ == "__main__":
    unittest.main()
