import unittest

from framework.profiling.spam_distribution import profile_spam_distribution

_CATS = {"phishing_link", "money_promise", "excessive_caps", "urgency", "spam_keywords"}


class ProfileSpamDistributionTests(unittest.TestCase):
    def test_weights_by_frequency_and_smooths_full_vocab(self):
        # 6 rows, each contains a URL (phishing_link) only.
        rows = [{"text": "visit http://x.com please here"}] * 6
        dist = profile_spam_distribution(rows, _CATS)
        td = dist["type_dist"]
        self.assertEqual(set(td.keys()), _CATS)                 # full vocab
        self.assertEqual(max(td, key=td.get), "phishing_link")  # most frequent
        self.assertTrue(all(p > 0 for p in td.values()))        # Laplace smoothing
        self.assertAlmostEqual(sum(td.values()), 1.0)
        self.assertEqual(dist["count_dist"], {1: 1.0})          # one signal per row

    def test_returns_none_below_min_rows(self):
        rows = [{"text": "win $100 now!"}] * 2
        self.assertIsNone(profile_spam_distribution(rows, _CATS, min_rows=5))

    def test_skips_rows_with_no_signal_and_clamps_count(self):
        rows = [{"text": "just a normal message here"}] * 10  # no signals
        self.assertIsNone(profile_spam_distribution(rows, _CATS))

    def test_clamps_signals_per_row_to_count_max(self):
        # Text fires multiple signals (5 total): phishing_link, money_promise,
        # excessive_caps, urgency, spam_keywords. With count_max=2, clamping occurs.
        text = "WIN $500 FREE money click http://x.com NOW!"
        rows = [{"text": text}] * 6  # 6 rows to exceed min_rows=5
        dist = profile_spam_distribution(rows, _CATS, count_max=2)
        # Verify clamping: max key in count_dist must be <= count_max
        self.assertLessEqual(max(dist["count_dist"].keys()), 2)
        # Verify count_dist sums to ~1.0 (probability distribution)
        self.assertAlmostEqual(sum(dist["count_dist"].values()), 1.0)
