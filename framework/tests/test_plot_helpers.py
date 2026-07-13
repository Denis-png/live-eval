import unittest

from framework.plotting.plots import flatten_mean_std, flatten_point


class FlattenPointTests(unittest.TestCase):
    def test_flat_metrics_pass_through(self):
        self.assertEqual(flatten_point({"f1": 0.9, "fpr": 0.0}), {"f1": 0.9, "fpr": 0.0})

    def test_nested_metrics_become_dotted(self):
        out = flatten_point({"errant": {"precision": 0.8, "recall": 0.6}, "gleu": 0.5})
        self.assertEqual(out, {"errant.precision": 0.8, "errant.recall": 0.6, "gleu": 0.5})

    def test_empty_block(self):
        self.assertEqual(flatten_point({}), {})
        self.assertEqual(flatten_point(None), {})


class FlattenMeanStdTests(unittest.TestCase):
    def test_flat_mean_std(self):
        self.assertEqual(flatten_mean_std({"f1": {"mean": 0.8, "std": 0.1}}), {"f1": (0.8, 0.1)})

    def test_nested_mean_std_becomes_dotted(self):
        out = flatten_mean_std({"errant": {"precision": {"mean": 0.7, "std": 0.05}}})
        self.assertEqual(out, {"errant.precision": (0.7, 0.05)})

    def test_missing_std_defaults_to_zero(self):
        self.assertEqual(flatten_mean_std({"f1": {"mean": 0.5}}), {"f1": (0.5, 0.0)})

    def test_empty_block(self):
        self.assertEqual(flatten_mean_std({}), {})


if __name__ == "__main__":
    unittest.main()
