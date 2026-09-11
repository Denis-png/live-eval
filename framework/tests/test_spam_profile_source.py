"""The spam profile describes the benchmark the run evaluates on.

_profile_spam read dataset.name for the HuggingFace split and fell back to the
profiler's default dataset whenever it was missing -- which it always is for a
local benchmark. So the shipped config (a local SMS CSV) was profiled from a
different HuggingFace dataset, and the seedless cells would have generated
from another benchmark's statistics while being scored against this one.
"""
import csv
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework import profile_dataset
from framework.profiling import spam_profiler
from framework.profiling.spec_sampler import load_profile

_ROWS = [("SPAM", "WIN a FREE prize now, call 0800 123 456!"),
         ("HAM", "see you at lunch tomorrow"),
         ("HAM", "running late, start without me"),
         ("SPAM", "URGENT: your account is locked, reply YES")]
_TOPICS = {"topics": {"everyday": {"description": "daily life", "fraction": 1.0}}}


class SpamProfileSourceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        path = os.path.join(self.dir.name, "sms.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "text"])
            writer.writerows(_ROWS)
        self.config = {"task": {"name": "spam"},
                       "dataset": {"source": "local", "local": {"path": path, "format": "csv"}},
                       "generation": {"sample_size": 4}}

    def _profile(self):
        out = os.path.join(self.dir.name, "p.json")
        refuse = mock.patch.object(spam_profiler, "load_spam_rows",
                                   side_effect=AssertionError("downloaded another dataset"))
        with refuse, mock.patch("framework.profiling.topics.profile_topics",
                                return_value=_TOPICS), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            profile_dataset._profile_spam(self.config, out, topic_call=lambda p: "",
                                          topic_sample_size=10)
        return out

    def test_it_profiles_the_configured_local_benchmark(self):
        with open(self._profile(), encoding="utf-8") as f:
            profile = json.load(f)
        self.assertEqual(profile["num_samples"], 4)
        self.assertEqual(profile["label_distribution"], {"HAM": 2, "SPAM": 2})

    def test_seedless_generation_can_load_it(self):
        profile = load_profile(self._profile(), topics_key="topics_per_label")
        self.assertEqual(set(profile["topics_per_label"]), {"HAM", "SPAM"})


if __name__ == "__main__":
    unittest.main()
