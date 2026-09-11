"""A sentiment benchmark profile can be built, and seedless generation can use it.

Seedless sentiment needs framework/data/profiles/sentiment/*_sentiment_profile.json,
and nothing could write one: profile_dataset.py's --task refused "sentiment", and
its sentiment branch called two names that were never defined. The profiling logic
in sentiment_profiler only ran on rows it downloaded itself, whatever dataset the
config named.
"""
import csv
import io
import json
import os
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework import pipeline, profile_dataset
from framework.profiling.spec_sampler import load_profile, sample_content_spec
from framework.tasks.sentiment.task import SentimentTask

_TOPICS = {"topics": {"daily life": {"description": "everyday events", "fraction": 1.0}}}
_ROWS = [("i hate mondays so much", "0"), ("the meeting is at noon today", "1"),
         ("what a lovely sunny day", "2"), ("worst service i have ever had", "0")]


class SentimentProfileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        path = os.path.join(self.dir.name, "tweets.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerows(_ROWS)
        self.config = {
            "task": {"name": "sentiment"},
            "dataset": {"source": "local", "local": {"path": path, "format": "csv"}},
            "generation": {"mode": "forward", "seedless": True, "sample_size": 4},
        }

    def _profile(self, output=None):
        with mock.patch("framework.profiling.topics.profile_topics", return_value=_TOPICS), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return profile_dataset._profile_sentiment(
                self.config, output, topic_call=lambda prompt: "", topic_sample_size=10)

    def test_the_cli_accepts_the_sentiment_task(self):
        argv = ["profile_dataset", "--task", "sentiment", "--config", "c.yaml"]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(profile_dataset.parse_args().task, "sentiment")

    def test_it_profiles_the_rows_a_run_evaluates(self):
        # The run's own loader and parse_row, so CSV digit labels become classes.
        with open(self._profile(os.path.join(self.dir.name, "p.json")),
                  encoding="utf-8") as f:
            profile = json.load(f)
        self.assertEqual(profile["num_samples"], 4)
        self.assertEqual(profile["label_distribution"],
                         {"NEGATIVE": 2, "NEUTRAL": 1, "POSITIVE": 1})

    def test_seedless_generation_can_load_and_sample_it(self):
        profile = load_profile(self._profile(os.path.join(self.dir.name, "p.json")),
                               topics_key="topics")
        side = SentimentTask().get_profile_side("forward")
        spec = sample_content_spec(profile, random.Random(0), side=side)
        self.assertEqual(spec["topic"], "daily life")

    def test_the_default_output_is_where_the_pipeline_looks(self):
        with mock.patch.object(pipeline, "DEFAULT_PROFILE_DIR", self.dir.name):
            written = self._profile()
            found = pipeline._resolve_benchmark_profile_path(self.config, SentimentTask())
        self.assertEqual(found, written)
        self.assertEqual(os.path.basename(written), "tweets_4_sentiment_profile.json")


if __name__ == "__main__":
    unittest.main()
