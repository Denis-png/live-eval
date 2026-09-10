"""The one precision/recall/F-beta formula every evaluator family shares.

It was reimplemented 16 times across 8 files, in all three evaluator families.
Each copy has to agree bit-for-bit with the others, or two tasks report the
"same" metric computed two ways. These tests pin the formula against the exact
expressions the inline copies used, so the refactor onto it is provably a no-op.
"""
import unittest

from framework.evaluators.prf import f_beta, precision, precision_recall_f, recall


class PrecisionRecallTests(unittest.TestCase):
    def test_precision_and_recall(self):
        self.assertEqual(precision(3, 1), 0.75)
        self.assertEqual(recall(3, 1), 0.75)

    def test_undefined_ratios_are_zero_not_an_error(self):
        # Nothing predicted / nothing to find. Every copy in the framework used
        # 0.0 here, so the shared helper must too.
        self.assertEqual(precision(0, 0), 0.0)
        self.assertEqual(recall(0, 0), 0.0)

    def test_soft_float_counts_are_accepted(self):
        # errant_dist sums proportions, so its tp/fp/fn are floats.
        self.assertAlmostEqual(precision(0.3, 0.1), 0.75)


class FBetaTests(unittest.TestCase):
    def test_f1_is_bit_identical_to_the_classification_expression(self):
        for p, r in [(0.75, 0.6), (1 / 3, 2 / 7), (0.9, 0.1), (1.0, 1.0)]:
            with self.subTest(p=p, r=r):
                self.assertEqual(f_beta(p, r), 2 * p * r / (p + r))

    def test_f05_is_bit_identical_to_the_errant_expression(self):
        # GEC weights precision over recall: over-correcting fluent text is
        # worse than missing an error.
        for p, r in [(0.75, 0.6), (1 / 3, 2 / 7), (0.9, 0.1)]:
            with self.subTest(p=p, r=r):
                self.assertEqual(f_beta(p, r, beta=0.5),
                                 (1.25 * p * r) / (0.25 * p + r))

    def test_zero_precision_and_recall_give_zero(self):
        self.assertEqual(f_beta(0.0, 0.0), 0.0)
        self.assertEqual(f_beta(0.0, 0.0, beta=0.5), 0.0)

    def test_one_side_zero_gives_zero(self):
        self.assertEqual(f_beta(0.8, 0.0), 0.0)
        self.assertEqual(f_beta(0.0, 0.8, beta=0.5), 0.0)

    def test_beta_above_one_favours_recall(self):
        self.assertGreater(f_beta(0.2, 0.9, beta=2.0), f_beta(0.2, 0.9, beta=1.0))


class CombinedTests(unittest.TestCase):
    def test_precision_recall_f_from_counts(self):
        p, r, f = precision_recall_f(3, 1, 2)
        self.assertEqual((p, r), (0.75, 0.6))
        self.assertEqual(f, 2 * 0.75 * 0.6 / (0.75 + 0.6))


if __name__ == "__main__":
    unittest.main()
