"""Every sentiment cell runs, and the cells that impose a mix can be calibrated.

Only forward+seeded ran. The other three cells need an empirical error
distribution (pipeline._should_load_error_distribution) and SentimentTask had no
profile_error_distribution: BaseTask's default returns None, which
load_error_distribution turns into a RuntimeError on every such run.

Real tweets carry a sentiment label, not an error type, so the distribution is
the real LABEL balance spread over the error types that produce each label.
"""
import csv
import io
import json
import os
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import yaml

from framework import calibrate, pipeline
from framework.generators.base_generator import BaseGenerator
from framework.main import _expand_env_vars
from framework.profiling.sentiment_profiler import profile_sentiment_rows
from framework.tasks.sentiment.task import SentimentTask

_LABELS = ["0", "0", "1", "2", "0", "1", "2", "0"]          # NEG 4, NEU 2, POS 2
_TWEETS = [f"tweet number {i} about the weather in town today" for i in range(8)]
_TOPICS = {"topics": {"weather": {"description": "the weather", "fraction": 1.0}}}


def _real():
    task = SentimentTask()
    return [task.parse_row({"text": t, "label": lab}) for t, lab in zip(_TWEETS, _LABELS)]


class _Stub(BaseGenerator):
    """Answers each sentiment prompt family in the format its parser expects."""

    def call_api(self, prompt):
        if "Rewritten tweet:" in prompt or "Transformed tweet:" in prompt:
            return "Redundancy: ok\nFaithfulness: ok"
        if "completely neutral" in prompt:
            return "Sentence: the weather is mild in town today"
        if "Requested sentiment:" in prompt:
            return "Corrupted: honestly i really hate how mild the weather is today"
        return ("Error type: sentiment_flip_negative\n"
                "Generated: honestly i really hate the weather in town today\n"
                "Ground truth: the weather is mild in town today")


class _Model:
    def predict(self, texts):
        return ["NEGATIVE"] * len(texts)


class EmpiricalDistributionTests(unittest.TestCase):
    def setUp(self):
        self.task = SentimentTask()
        self.dist = self.task.profile_error_distribution(_real())

    def test_the_imposed_mix_reproduces_the_real_label_balance(self):
        by_label = Counter()
        for etype, share in self.dist["type_dist"].items():
            by_label[self.task.get_label({"error_type": etype})] += share
        self.assertAlmostEqual(by_label["NEGATIVE"], 0.5)
        self.assertAlmostEqual(by_label["NEUTRAL"], 0.25)
        self.assertAlmostEqual(by_label["POSITIVE"], 0.25)

    def test_every_imposed_type_can_be_rendered_and_labelled(self):
        descriptions = self.task.get_error_descriptions()
        for etype in self.dist["type_dist"]:
            self.assertIn(etype, descriptions)
            self.assertIsNotNone(self.task.get_label({"error_type": etype}))

    def test_a_sample_carries_exactly_one_transformation(self):
        # Two would contradict each other and leave the label ambiguous.
        self.assertEqual(self.dist["count_dist"], {1: 1.0})

    def test_too_few_rows_give_none(self):
        self.assertIsNone(self.task.profile_error_distribution(_real()[:4]))


class _Session(unittest.TestCase):
    """The shipped config, pointed at a temporary CSV and profile."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        data = os.path.join(self.dir.name, "tweets.csv")
        with open(data, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerows(zip(_TWEETS, _LABELS))
        profile = profile_sentiment_rows(
            [{"text": r["incorrect"], "label": r["sentiment_label"]} for r in _real()])
        profile["topics"] = _TOPICS
        self.profile_path = os.path.join(self.dir.name, "p_sentiment_profile.json")
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f)
        with open("framework/configs/sentiment/config.yaml", encoding="utf-8") as f:
            self.base = _expand_env_vars(yaml.safe_load(f))
        self.base["dataset"] = {"source": "local", "local": {"path": data, "format": "csv"}}

    def config(self, mode, seedless, **generation):
        cfg = json.loads(json.dumps(self.base))
        cfg["generation"].update(mode=mode, seedless=seedless, sample_size=8, num_runs=1,
                                 request_delay=0, profile_path=self.profile_path,
                                 calibration_path=None, **generation)
        cfg["output"] = {"base_dir": self.dir.name, "plots": False,
                         "session_id": f"{mode}_{seedless}"}
        return cfg


class EveryCellRunsTests(_Session):
    def test_every_cell_runs_end_to_end_from_the_shipped_config(self):
        for mode in ("forward", "inverse"):
            for seedless in (False, True):
                with self.subTest(mode=mode, seedless=seedless):
                    with mock.patch.object(pipeline, "load_generator", return_value=_Stub()), \
                            mock.patch.object(SentimentTask, "get_model",
                                              return_value=_Model()), \
                            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        pipeline.run_pipeline(self.config(mode, seedless))
                    session = os.path.join(self.dir.name, "sentiment", f"{mode}_{seedless}")
                    with open(os.path.join(session, "results.json"), encoding="utf-8") as f:
                        results = json.load(f)["results"]
                    for scores in results.values():
                        self.assertIn("macro_f1", scores["generated"])
                        self.assertIn("macro_f1", scores["real"])
                    with open(os.path.join(session, "profile.json"), encoding="utf-8") as f:
                        fidelity = json.load(f)["fidelity"]
                    self.assertEqual(fidelity["profile_type"], "sentiment_fidelity")


class CalibrationTests(_Session):
    def _calibrate(self, mode, seedless):
        with mock.patch.object(pipeline, "load_generator", return_value=_Stub()), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return calibrate.run_calibration(
                self.config(mode, seedless), rounds=0, sample_size=8,
                output_path=os.path.join(self.dir.name, "calibration.json"))

    def test_an_imposed_cell_calibrates_toward_the_real_label_balance(self):
        payload = self._calibrate("inverse", False)
        self.assertEqual(payload["target"]["type_dist"],
                         SentimentTask().profile_error_distribution(_real())["type_dist"])
        self.assertTrue(payload["rounds"][0]["measured"]["type_dist"])

    def test_every_typed_record_counts_as_informative(self):
        # calibrate.informative_count reads n_annotated for corruption tasks;
        # without it every round warned that it measured 0 samples.
        payload = self._calibrate("inverse", False)
        self.assertEqual(payload["rounds"][0]["informative_samples"], 8)

    def test_forward_seeded_refuses_with_a_reason(self):
        # The model picks the transformation there, and real tweets carry no
        # error type to aim a seed mix at.
        with self.assertRaisesRegex(RuntimeError, "nothing to calibrate"):
            self._calibrate("forward", False)


if __name__ == "__main__":
    unittest.main()
