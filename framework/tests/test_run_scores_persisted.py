import unittest

from framework.main import format_results_lines
from framework.pipeline import _nest_results


class NestResultsRunsTests(unittest.TestCase):
    def test_runs_are_persisted_per_model(self):
        generated = {"m": {"f1": {"mean": 0.8, "std": 0.05}}}
        real = {"m": {"f1": 0.9}}
        all_runs = [{"m": {"f1": 0.75}}, {"m": {"f1": 0.85}}]
        out = _nest_results(generated, real, all_runs)
        self.assertEqual(out["m"]["runs"], [{"f1": 0.75}, {"f1": 0.85}])
        self.assertEqual(out["m"]["generated"]["f1"]["mean"], 0.8)
        self.assertEqual(out["m"]["real"]["f1"], 0.9)

    def test_runs_skip_runs_where_model_absent(self):
        generated = {"m": {"f1": {"mean": 0.8, "std": 0.0}}}
        all_runs = [{"m": {"f1": 0.8}}, {"other": {"f1": 0.1}}]
        out = _nest_results(generated, {}, all_runs)
        self.assertEqual(out["m"]["runs"], [{"f1": 0.8}])

    def test_printer_unaffected_by_extra_runs_key(self):
        results = {"m": {"generated": {"f1": {"mean": 0.8, "std": 0.1}},
                         "real": {"f1": 0.9}, "runs": [{"f1": 0.8}]}}
        text = "\n".join(format_results_lines(results))
        self.assertIn("generated.f1", text)
        self.assertIn("real.f1", text)
        self.assertNotIn("runs", text)


if __name__ == "__main__":
    unittest.main()
