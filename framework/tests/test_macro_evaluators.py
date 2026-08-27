import unittest

from framework.evaluators.classification.macro_f1 import compute_macro_f1
from framework.evaluators.classification.macro_precision import compute_macro_precision
from framework.evaluators.classification.macro_recall import compute_macro_recall

CLASSES = ("NEGATIVE", "NEUTRAL", "POSITIVE")


def _results(pairs):
    """pairs of (prediction, gold)."""
    return [{"prediction": p, "label": g} for p, g in pairs]


class MacroPrecisionRecallTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        results = _results([("NEGATIVE", "NEGATIVE"), ("NEUTRAL", "NEUTRAL"),
                            ("POSITIVE", "POSITIVE")])
        self.assertAlmostEqual(compute_macro_precision(results, CLASSES), 1.0)
        self.assertAlmostEqual(compute_macro_recall(results, CLASSES), 1.0)
        self.assertAlmostEqual(compute_macro_f1(results, CLASSES), 1.0)

    def test_all_wrong_scores_zero(self):
        results = _results([("POSITIVE", "NEGATIVE"), ("NEGATIVE", "POSITIVE")])
        self.assertAlmostEqual(compute_macro_precision(results, CLASSES), 0.0)
        self.assertAlmostEqual(compute_macro_recall(results, CLASSES), 0.0)
        self.assertAlmostEqual(compute_macro_f1(results, CLASSES), 0.0)

    def test_macro_averages_over_classes_not_samples(self):
        # NEGATIVE: 2 gold, both found. POSITIVE: 1 gold, missed (predicted NEGATIVE).
        # Per-class recall = 1.0, 0.0 (NEUTRAL absent), 0.0 -> macro 1/3.
        results = _results([("NEGATIVE", "NEGATIVE"), ("NEGATIVE", "NEGATIVE"),
                            ("NEGATIVE", "POSITIVE")])
        self.assertAlmostEqual(compute_macro_recall(results, CLASSES), 1 / 3)
        # NEGATIVE precision = 2/3; NEUTRAL and POSITIVE never predicted -> 0.
        self.assertAlmostEqual(compute_macro_precision(results, CLASSES), (2 / 3) / 3)

    def test_absent_class_contributes_zero_rather_than_dividing_by_zero(self):
        results = _results([("NEGATIVE", "NEGATIVE")])
        self.assertAlmostEqual(compute_macro_precision(results, CLASSES), 1 / 3)
        self.assertAlmostEqual(compute_macro_recall(results, CLASSES), 1 / 3)

    def test_empty_results_are_zero_not_an_error(self):
        self.assertEqual(compute_macro_precision([], CLASSES), 0.0)
        self.assertEqual(compute_macro_recall([], CLASSES), 0.0)
        self.assertEqual(compute_macro_f1([], CLASSES), 0.0)

    def test_empty_label_set_is_zero_not_an_error(self):
        results = _results([("NEGATIVE", "NEGATIVE")])
        self.assertEqual(compute_macro_precision(results, ()), 0.0)
        self.assertEqual(compute_macro_recall(results, ()), 0.0)


class MacroF1Tests(unittest.TestCase):
    def test_is_the_harmonic_mean_of_macro_precision_and_recall(self):
        results = _results([("NEGATIVE", "NEGATIVE"), ("NEGATIVE", "POSITIVE"),
                            ("NEUTRAL", "NEUTRAL")])
        p = compute_macro_precision(results, CLASSES)
        r = compute_macro_recall(results, CLASSES)
        self.assertAlmostEqual(compute_macro_f1(results, CLASSES), 2 * p * r / (p + r))

    def test_zero_precision_and_recall_does_not_divide_by_zero(self):
        self.assertEqual(
            compute_macro_f1(_results([("POSITIVE", "NEGATIVE")]), CLASSES), 0.0)


if __name__ == "__main__":
    unittest.main()
