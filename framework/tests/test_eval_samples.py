import unittest

from framework.tasks.gec.task import GECTask
from framework.tasks.spam.task import SpamTask


class BaseDefaultTests(unittest.TestCase):
    def test_gec_default_one_row_per_item_no_label(self):
        synthetic = [{"original": "a good sentence", "corrupted": "a bad sentence",
                      "error_type": "R:VERB:TENSE"}]
        out = GECTask().get_eval_samples(synthetic)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "a bad sentence")
        self.assertNotIn("label", out[0])


class SpamBalancedTests(unittest.TestCase):
    def test_spam_item_adds_ham_negative(self):
        synthetic = [{"original": "let us meet tomorrow", "corrupted": "WIN $500 now!",
                      "error_type": "money_promise, urgency"}]
        out = SpamTask().get_eval_samples(synthetic)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "WIN $500 now!")
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertEqual(out[1]["text"], "let us meet tomorrow")
        self.assertEqual(out[1]["label"], "HAM")

    def test_paraphrase_item_has_no_negative(self):
        synthetic = [{"original": "are we still meeting?", "corrupted": "hey, still on for later?",
                      "error_type": "paraphrase"}]
        out = SpamTask().get_eval_samples(synthetic)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], "HAM")


if __name__ == "__main__":
    unittest.main()
