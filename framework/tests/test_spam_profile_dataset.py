import unittest

from framework.tasks.spam.task import SpamTask


class SpamProfileDatasetTests(unittest.TestCase):
    def setUp(self):
        self.task = SpamTask()

    def test_profile_class_balance_and_signals(self):
        rows = [
            {"text": "win $500 http://x.com NOW!", "label": "SPAM"},
            {"text": "claim your free prize money now", "label": "SPAM"},
            {"text": "see you at lunch tomorrow", "label": "HAM"},
        ]
        prof = self.task.profile_dataset(rows)
        self.assertEqual(prof["n"], 3)
        self.assertEqual(prof["class_balance"]["SPAM"], 2)
        self.assertEqual(prof["class_balance"]["HAM"], 1)
        self.assertAlmostEqual(prof["class_balance"]["spam_fraction"], 2 / 3)
        # phishing_link fired on 1 of 2 spam messages:
        self.assertAlmostEqual(prof["signal_rate"]["phishing_link"], 0.5)
        self.assertAlmostEqual(sum(prof["signal_type_dist"].values()), 1.0)

    def test_compare_profiles_identical_zero_divergence(self):
        rows = [
            {"text": "win $500 http://x.com NOW!", "label": "SPAM"},
            {"text": "cheap deals click http://y.com free", "label": "SPAM"},
            {"text": "lunch tomorrow?", "label": "HAM"},
        ]
        prof = self.task.profile_dataset(rows)
        cmp = self.task.compare_profiles(prof, prof)
        self.assertAlmostEqual(cmp["type_dist_jsd"], 0.0)
        self.assertAlmostEqual(cmp["count_dist_jsd"], 0.0)
        self.assertAlmostEqual(cmp["class_balance_delta"], 0.0)
        self.assertTrue(all(abs(d) < 1e-9 for d in cmp["signal_deltas"].values()))


if __name__ == "__main__":
    unittest.main()
