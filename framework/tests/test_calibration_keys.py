import unittest

from framework.tasks.gec.task import GECTask
from framework.tasks.sentiment.task import SentimentTask
from framework.tasks.spam.task import SpamTask
from framework.tasks.taxonomy.task import TaxonomyTask


class FakeEdit:
    """ERRANT stand-in: minimal object with a .type attribute."""
    def __init__(self, type_):
        self.type = type_


class FakeAnnotator:
    """Canned ERRANT stand-in: maps a source sentence to a list of edit types."""
    def __init__(self, edits_by_src):
        self.edits_by_src = edits_by_src

    def parse(self, text):
        return text

    def annotate(self, src, ref):
        return [FakeEdit(t) for t in self.edits_by_src.get(src, [])]


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

    def test_spam_profile_keys_exist_and_are_shape_correct(self):
        # The mapping is only useful if profile_dataset actually emits them with
        # the correct shapes: type_dist keyed by string categories, count_dist by int counts.
        # This catches swapped mappings (e.g., accidentally assigning signal_count_dist
        # to "type_dist").
        rows = [{"text": "WIN a FREE $500 prize now http://x.example !",
                 "label": "SPAM"},
                {"text": "are we still meeting tomorrow afternoon", "label": "HAM"}]
        profile = SpamTask().profile_dataset(rows)
        keys = SpamTask().get_calibration_keys()

        # Verify keys exist
        type_key = keys["type_dist"]
        count_key = keys["count_dist"]
        self.assertIn(type_key, profile)
        self.assertIn(count_key, profile)

        # Verify type_dist is non-empty and has string keys
        type_dist = profile[type_key]
        self.assertGreater(len(type_dist), 0, "type_dist must be non-empty")
        for k in type_dist.keys():
            self.assertIsInstance(k, str, f"type_dist keys must be strings, got {type(k)}")

        # Verify count_dist is non-empty and has int keys
        count_dist = profile[count_key]
        self.assertGreater(len(count_dist), 0, "count_dist must be non-empty")
        for k in count_dist.keys():
            self.assertIsInstance(k, int, f"count_dist keys must be ints, got {type(k)}")

    def test_gec_profile_keys_exist_and_are_shape_correct(self):
        # Like Spam test, verify GEC's declared keys exist in the profile with
        # correct shapes. Uses an injected annotator to avoid real ERRANT dependency.
        task = GECTask()
        annotator = FakeAnnotator({
            "a bad": ["R:SPELL", "M:DET"],
            "b bad": ["R:SPELL"],
            "c good": [],
        })
        rows = [
            {"corrupted": "a bad", "original": "a good"},
            {"corrupted": "b bad", "original": "b good"},
            {"corrupted": "c good", "original": "c good"},
        ]
        profile = task.profile_dataset(rows, annotator=annotator)
        keys = task.get_calibration_keys()

        # Verify keys exist
        type_key = keys["type_dist"]
        count_key = keys["count_dist"]
        self.assertIn(type_key, profile)
        self.assertIn(count_key, profile)

        # Verify type_dist is non-empty and has string keys
        type_dist = profile[type_key]
        self.assertGreater(len(type_dist), 0, "type_dist must be non-empty")
        for k in type_dist.keys():
            self.assertIsInstance(k, str, f"type_dist keys must be strings, got {type(k)}")

        # Verify count_dist is non-empty and has int keys
        count_dist = profile[count_key]
        self.assertGreater(len(count_dist), 0, "count_dist must be non-empty")
        for k in count_dist.keys():
            self.assertIsInstance(k, int, f"count_dist keys must be ints, got {type(k)}")
