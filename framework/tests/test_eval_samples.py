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


class SpamPassThroughTests(unittest.TestCase):
    def test_labeled_records_pass_through_unchanged(self):
        synthetic = [
            {"text": "WIN $500 now http://x.com", "label": "SPAM", "technique": "money_promise"},
            {"text": "are we still meeting tomorrow", "label": "HAM", "technique": "paraphrase"},
        ]
        out = SpamTask().get_eval_samples(synthetic)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "WIN $500 now http://x.com")
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertEqual(out[1]["label"], "HAM")


if __name__ == "__main__":
    unittest.main()
