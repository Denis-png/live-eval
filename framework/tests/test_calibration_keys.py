import unittest

from framework.tasks.gec.task import GECTask
from framework.tasks.sentiment.task import SentimentTask
from framework.tasks.spam.task import SpamTask
from framework.tasks.taxonomy.task import TaxonomyTask


class CalibrationKeysTests(unittest.TestCase):
    def test_gec_maps_control_inputs_to_errant_profile_keys(self):
        self.assertEqual(
            GECTask().get_calibration_keys(),
            {"type_dist": "error_type_dist", "count_dist": "error_count_dist"},
        )

    def test_spam_maps_control_inputs_to_signal_profile_keys(self):
        self.assertEqual(
            SpamTask().get_calibration_keys(),
            {"type_dist": "signal_type_dist", "count_dist": "signal_count_dist"},
        )

    def test_uncalibratable_tasks_opt_out(self):
        self.assertIsNone(TaxonomyTask().get_calibration_keys())
        self.assertIsNone(SentimentTask().get_calibration_keys())

    def test_named_keys_exist_in_the_task_profile(self):
        # The mapping is only useful if profile_dataset actually emits them.
        rows = [{"text": "WIN a FREE $500 prize now http://x.example !",
                 "label": "SPAM"},
                {"text": "are we still meeting tomorrow afternoon", "label": "HAM"}]
        profile = SpamTask().profile_dataset(rows)
        for key in SpamTask().get_calibration_keys().values():
            self.assertIn(key, profile)
