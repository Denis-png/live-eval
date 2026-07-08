import unittest

from framework.main import format_results_lines


class ResultsPrinterTests(unittest.TestCase):
    def test_prints_generated_and_real(self):
        results = {
            "m": {
                "generated": {"f1": {"mean": 0.8, "std": 0.1}},
                "real": {"f1": 0.9},
            }
        }
        text = "\n".join(format_results_lines(results))
        self.assertIn("m", text)
        self.assertIn("generated", text)
        self.assertIn("0.8", text)
        self.assertIn("real", text)
        self.assertIn("0.9", text)

    def test_handles_missing_real(self):
        results = {"m": {"generated": {"f1": {"mean": 0.5, "std": 0.0}}}}
        text = "\n".join(format_results_lines(results))
        self.assertIn("0.5", text)

    def test_prints_nested_metric_real_values(self):
        results = {
            "g": {
                "generated": {"errant": {"precision": {"mean": 0.7, "std": 0.05}}},
                "real": {"errant": {"precision": 0.8}},
            }
        }
        text = "\n".join(format_results_lines(results))
        self.assertIn("generated.errant.precision", text)
        self.assertIn("0.7", text)
        self.assertIn("real.errant.precision", text)
        self.assertIn("0.8", text)


if __name__ == "__main__":
    unittest.main()
