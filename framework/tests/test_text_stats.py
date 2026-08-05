import unittest

from framework.profiling.text_stats import (
    CHAR_BINS,
    WORD_BINS,
    bin_label,
    length_distribution,
    style_profile,
)


class LengthDistributionTests(unittest.TestCase):
    def test_bin_labels(self):
        self.assertEqual(bin_label(1, 5), "1-5")
        self.assertEqual(bin_label(51, None), "51+")

    def test_fractions_sum_to_one_and_use_canonical_labels(self):
        dist = length_distribution([3, 8, 8, 60], WORD_BINS)
        self.assertEqual(
            set(dist["bins"]),
            {"1-5", "6-10", "11-15", "16-20", "21-30", "31-50", "51+"},
        )
        self.assertAlmostEqual(sum(dist["bins"].values()), 1.0, places=3)
        self.assertAlmostEqual(dist["bins"]["6-10"], 0.5)
        self.assertAlmostEqual(dist["bins"]["51+"], 0.25)
        self.assertEqual(dist["count"], 4)

    def test_out_of_range_clamps_into_boundary_bins(self):
        dist = length_distribution([0, 9999], WORD_BINS)
        self.assertAlmostEqual(dist["bins"]["1-5"], 0.5)
        self.assertAlmostEqual(dist["bins"]["51+"], 0.5)

    def test_quantiles_inclusive_method(self):
        dist = length_distribution(list(range(1, 101)), WORD_BINS)
        self.assertAlmostEqual(dist["quantiles"]["p50"], 50.5)
        self.assertAlmostEqual(dist["quantiles"]["p90"], 90.1)

    def test_empty_input_gives_zero_stats(self):
        dist = length_distribution([], WORD_BINS)
        self.assertEqual(dist["count"], 0)
        self.assertEqual(sum(dist["bins"].values()), 0.0)
        self.assertEqual(dist["quantiles"]["p50"], 0)

    def test_single_value_quantiles_are_that_value(self):
        dist = length_distribution([7], WORD_BINS)
        for key in ("p10", "p25", "p50", "p75", "p90"):
            self.assertEqual(dist["quantiles"][key], 7)

    def test_char_bins_exist(self):
        dist = length_distribution([30], CHAR_BINS)
        self.assertAlmostEqual(dist["bins"]["26-50"], 1.0)


class StyleProfileTests(unittest.TestCase):
    def test_rates_on_crafted_texts(self):
        texts = ["Are you FREE now?", "i'm at home", "call her 2day plz!"]
        style = style_profile(texts)
        self.assertAlmostEqual(style["question_rate"], 1 / 3, places=3)
        self.assertAlmostEqual(style["exclaim_rate"], 1 / 3, places=3)
        self.assertAlmostEqual(style["first_person_rate"], 1 / 3, places=3)   # i'm
        self.assertAlmostEqual(style["second_person_rate"], 1 / 3, places=3)  # you
        self.assertAlmostEqual(style["contraction_rate"], 1 / 3, places=3)    # i'm
        self.assertAlmostEqual(style["digit_rate"], 1 / 3, places=3)          # 2day
        self.assertAlmostEqual(style["uppercase_word_rate"], 1 / 3, places=3) # FREE
        self.assertAlmostEqual(style["texting_slang_rate"], 1 / 3, places=3)  # 2day, plz
        self.assertGreater(style["punctuation_density"], 0.0)

    def test_empty_input_all_zero(self):
        style = style_profile([])
        for value in style.values():
            self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
