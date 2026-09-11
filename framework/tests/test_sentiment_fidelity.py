"""Sentiment's fidelity profile: the hooks the pipeline calls, fed what the
pipeline actually passes, measured the same way on both sides.

The task defined profile_dataset / compare_profiles, which nothing calls; the
pipeline calls build_fidelity_profile / compare_fidelity_profiles, whose BaseTask
defaults opt a task out. So no sentiment session ever wrote a profile.json.

Renaming alone would not have been enough. The pipeline passes real eval samples
({"text", "label"}) and raw generated records ({"original", "corrupted",
"error_type"}): real tweets carry a label and no error type, generated records the
reverse. Each JSD over a field only one side has compared a filled distribution
with an empty one -- a constant 0.5, however alike the two datasets were.
"""
import json
import os
import tempfile
import unittest

from framework import pipeline
from framework.plotting import session as S
from framework.tasks.sentiment.task import SentimentTask

# What get_real_eval_samples returns.
_REAL = [
    {"text": "i hate mondays so much", "label": "NEGATIVE"},
    {"text": "the meeting is at noon today", "label": "NEUTRAL"},
    {"text": "what a lovely sunny day", "label": "POSITIVE"},
    {"text": "worst service i have ever had", "label": "NEGATIVE"},
]
# The same tweets as generated records, each error type imposing its row's label.
_GENERATED = [
    {"original": "x", "corrupted": "i hate mondays so much",
     "error_type": "sentiment_flip_negative"},
    {"original": "x", "corrupted": "the meeting is at noon today",
     "error_type": "intensity_reduction"},
    {"original": "x", "corrupted": "what a lovely sunny day",
     "error_type": "sentiment_flip_positive"},
    {"original": "x", "corrupted": "worst service i have ever had",
     "error_type": "negation_insertion"},
]


class SentimentFidelityTests(unittest.TestCase):
    def setUp(self):
        self.task = SentimentTask()

    def _compare(self, real, generated):
        return self.task.compare_fidelity_profiles(
            self.task.build_fidelity_profile(real),
            self.task.build_fidelity_profile(generated))

    def test_the_pipeline_hooks_are_implemented(self):
        # BaseTask's defaults return None, which opts a task out of fidelity.
        self.assertIsNotNone(self.task.build_fidelity_profile(_REAL))
        self.assertIsNotNone(self._compare(_REAL, _GENERATED))

    def test_a_generated_record_is_labelled_by_its_error_type(self):
        # The label it is scored against in evaluation (get_label).
        profile = self.task.build_fidelity_profile(_GENERATED)
        self.assertEqual(profile["label_dist"],
                         {"NEGATIVE": 0.5, "NEUTRAL": 0.25, "POSITIVE": 0.25})

    def test_identical_data_scores_as_identical(self):
        # The invariant: both sides measured the same way.
        fid = self._compare(_REAL, _GENERATED)
        self.assertEqual(fid["label_dist_jsd"], 0.0)
        self.assertEqual(fid["length_jsd"], 0.0)
        self.assertEqual(set(fid["label_deltas"].values()), {0.0})

    def test_a_shifted_label_mix_is_detected(self):
        positive = [dict(r, error_type="sentiment_flip_positive") for r in _GENERATED]
        fid = self._compare(_REAL, positive)
        self.assertGreater(fid["label_dist_jsd"], 0.0)
        self.assertEqual(fid["label_deltas"]["POSITIVE"], 0.75)

    def test_a_record_with_no_deterministic_label_is_left_out(self):
        # paraphrase keeps a sentiment that is unknown; evaluation skips it too.
        paraphrase = {"original": "x", "corrupted": "a b c d", "error_type": "paraphrase"}
        profile = self.task.build_fidelity_profile(_GENERATED + [paraphrase])
        self.assertEqual(profile["num_samples"], 4)

    def test_the_error_type_mix_is_not_compared(self):
        # Real tweets have no error type, so there is nothing to compare it with.
        self.assertNotIn("type_dist_jsd", self._compare(_REAL, _GENERATED))

    def test_an_empty_generated_side_still_compares(self):
        fid = self._compare(_REAL, [])
        self.assertEqual(fid["profile_type"], "sentiment_fidelity")


class SentimentSessionTests(unittest.TestCase):
    def test_a_session_writes_its_profile(self):
        with tempfile.TemporaryDirectory() as d:
            paths = {"real_sample": os.path.join(d, "real_sample.json"),
                     "profile": os.path.join(d, "profile.json")}
            pipeline._write_fidelity_artifacts(SentimentTask(), _REAL, _GENERATED, paths)
            with open(paths["profile"], encoding="utf-8") as f:
                profile = json.load(f)
        self.assertEqual(profile["fidelity"]["profile_type"], "sentiment_fidelity")

    def test_a_session_renders_the_sentiment_figure_not_the_spam_one(self):
        with tempfile.TemporaryDirectory() as d:
            paths = {"real_sample": os.path.join(d, "real_sample.json"),
                     "profile": os.path.join(d, "profile.json")}
            pipeline._write_fidelity_artifacts(SentimentTask(), _REAL, _GENERATED, paths)
            with open(os.path.join(d, "results.json"), "w", encoding="utf-8") as f:
                json.dump({"meta": {"task": "sentiment", "mode": "forward",
                                    "model": "minimax-m3"}, "results": {}}, f)
            names = sorted(os.path.basename(p) for p in S.render_session(d))
        self.assertIn("sentiment_fidelity.png", names)
        self.assertNotIn("fidelity.png", names)


if __name__ == "__main__":
    unittest.main()
