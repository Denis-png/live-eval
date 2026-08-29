import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from framework import pipeline
from framework.calibration.artifact import write_calibration
from framework.tasks.spam.task import SpamTask

_ROWS = "label,text\n" + "".join(
    f"ham,could we move the meeting to three tomorrow number {i}\n" for i in range(20)
) + "".join(
    f"spam,claim your FREE prize now $50 http://y{i}.example !\n" for i in range(12)
)


class _Bench(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.bench = os.path.join(self.dir.name, "bench.csv")
        with open(self.bench, "w", encoding="utf-8") as f:
            f.write(_ROWS)
        self.artifact = os.path.join(self.dir.name, "cal.json")
        self.task = SpamTask()

    def _config(self, calibration_path=...):
        cfg = {
            "dataset": {"source": "local",
                        "local": {"path": self.bench, "format": "csv"}},
            "generation": {"sample_size": 5, "mode": "inverse", "seedless": False},
            "task": {"name": "spam"},
        }
        if calibration_path is not ...:
            cfg["generation"]["calibration_path"] = calibration_path
        return cfg

    def _real_data(self, cfg):
        return pipeline.load_real_data(cfg, self.task)


class ConsumptionTests(_Bench):
    def test_no_artifact_returns_the_empirical_distribution_unchanged(self):
        cfg = self._config()
        empirical = self.task.profile_error_distribution(
            self._real_data(cfg), config=cfg)
        out = pipeline.load_error_distribution(cfg, self._real_data(cfg), self.task)
        self.assertEqual(out, empirical)

    def test_matching_artifact_is_used(self):
        cfg = self._config(self.artifact)
        empirical = self.task.profile_error_distribution(
            self._real_data(cfg), config=cfg)
        calibrated = {"type_dist": {k: v for k, v in empirical["type_dist"].items()},
                      "count_dist": dict(empirical["count_dist"])}
        first = sorted(calibrated["type_dist"])[0]
        calibrated["type_dist"][first] = calibrated["type_dist"][first] + 0.05
        write_calibration(self.artifact, {
            "meta": {}, "target": empirical, "calibrated": calibrated,
            "selected_round": 1, "rounds": [],
        })
        out = pipeline.load_error_distribution(cfg, self._real_data(cfg), self.task)
        self.assertEqual(out["type_dist"], calibrated["type_dist"])

    def test_count_dist_keys_are_ints_after_load(self):
        cfg = self._config(self.artifact)
        empirical = self.task.profile_error_distribution(
            self._real_data(cfg), config=cfg)
        write_calibration(self.artifact, {
            "meta": {}, "target": empirical, "calibrated": empirical,
            "selected_round": 0, "rounds": [],
        })
        out = pipeline.load_error_distribution(cfg, self._real_data(cfg), self.task)
        self.assertTrue(all(isinstance(k, int) for k in out["count_dist"]))

    def test_drifted_target_warns_and_falls_back(self):
        cfg = self._config(self.artifact)
        empirical = self.task.profile_error_distribution(
            self._real_data(cfg), config=cfg)
        stale = {"type_dist": {"phishing_link": 1.0}, "count_dist": {1: 1.0}}
        write_calibration(self.artifact, {
            "meta": {}, "target": stale, "calibrated": stale,
            "selected_round": 0, "rounds": [],
        })
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            out = pipeline.load_error_distribution(cfg, self._real_data(cfg), self.task)
        self.assertEqual(out, empirical)
        self.assertIn("WARN", err.getvalue())

    def test_explicit_null_opts_out(self):
        cfg = self._config(None)
        empirical = self.task.profile_error_distribution(
            self._real_data(cfg), config=cfg)
        write_calibration(self.artifact, {
            "meta": {}, "target": empirical,
            "calibrated": {"type_dist": {"urgency": 1.0}, "count_dist": {1: 1.0}},
            "selected_round": 0, "rounds": [],
        })
        out = pipeline.load_error_distribution(cfg, self._real_data(cfg), self.task)
        self.assertEqual(out, empirical)


class MetaTests(_Bench):
    def test_meta_records_calibration_provenance(self):
        cfg = self._config(self.artifact)
        cfg["generation"].update({"provider": "openai", "model": "gpt-x",
                                  "num_runs": 1})
        empirical = self.task.profile_error_distribution(
            self._real_data(cfg), config=cfg)
        write_calibration(self.artifact, {
            "meta": {}, "target": empirical, "calibrated": empirical,
            "selected_round": 2, "rounds": [],
        })
        with redirect_stdout(io.StringIO()):
            pipeline.load_error_distribution(cfg, self._real_data(cfg), self.task)
        meta = pipeline._build_meta(cfg, self.task, runs_completed=1,
                                    effective_samples_per_run=[5],
                                    real_baseline=False)
        self.assertEqual(meta["calibration"]["path"], self.artifact)
        self.assertEqual(meta["calibration"]["selected_round"], 2)

    def test_meta_calibration_is_none_without_an_artifact(self):
        cfg = self._config()
        cfg["generation"].update({"provider": "openai", "model": "gpt-x",
                                  "num_runs": 1})
        meta = pipeline._build_meta(cfg, self.task, runs_completed=1,
                                    effective_samples_per_run=[5],
                                    real_baseline=False)
        self.assertIsNone(meta["calibration"])
