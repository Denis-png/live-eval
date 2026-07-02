import unittest

from framework.tasks.spam.task import SpamTask

_CATS = {"phishing_link", "money_promise", "excessive_caps", "urgency", "spam_keywords"}


class SpamInverseConfigTests(unittest.TestCase):
    def setUp(self):
        self.task = SpamTask()

    def test_inverse_prompt_has_placeholders(self):
        p = self.task.get_inverse_prompt()
        self.assertIsNotNone(p)
        self.assertIn("{sentence}", p)
        self.assertIn("{error_spec}", p)
        self.assertIn("Corrupted:", p)

    def test_inverse_judge_prompt_has_placeholders(self):
        p = self.task.get_inverse_judge_prompt()
        self.assertIsNotNone(p)
        self.assertIn("{sentence}", p)
        self.assertIn("{correction}", p)

    def test_error_descriptions_are_the_five_signals(self):
        self.assertEqual(set(self.task.get_error_descriptions().keys()), _CATS)


class SpamProfileTests(unittest.TestCase):
    def test_profile_loads_spam_subset_and_delegates(self, ):
        task = SpamTask()
        captured = {}

        def fake_load(**kwargs):
            captured.update(kwargs)
            return [{"text": "win $500 http://x.com NOW!", "label": "SPAM"}] * 6 + \
                   [{"text": "lunch tomorrow", "label": "HAM"}] * 3

        import framework.profiling.spam_profiler as sp
        original = sp.load_spam_rows
        sp.load_spam_rows = fake_load
        try:
            config = {
                "dataset": {"name": "deysi/spam-detection-dataset", "split": "train"},
                "generation": {"inverse": {"profile_size": 42}},
            }
            dist = task.profile_error_distribution([], count_max=5, config=config)
        finally:
            sp.load_spam_rows = original

        self.assertEqual(captured["sample_size"], 42)          # profile_size forwarded
        self.assertEqual(set(dist["type_dist"].keys()), _CATS)  # HAM rows excluded, vocab full
        self.assertAlmostEqual(sum(dist["type_dist"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()
