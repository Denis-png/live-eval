"""A calibration artifact measured under asymmetric inverse must not steer
symmetric generation.

targets_match cannot catch this: the artifact's stored target is derived from
the REAL benchmark, which does not change when generation semantics do.
"""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from framework import pipeline
from framework.calibration.artifact import write_calibration
from framework.generators.base_generator import CLASS_CONDITIONAL_SEMANTICS
from framework.tasks.spam.task import SpamTask

_ROWS = "label,text\n" + "".join(
    f"ham,could we move the meeting to three tomorrow number {i}\n" for i in range(20)
) + "".join(
    f"spam,claim your FREE prize now $50 http://y{i}.example !\n" for i in range(12)
)


class SemanticsMarkerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.bench = os.path.join(self.dir.name, "bench.csv")
        with open(self.bench, "w", encoding="utf-8") as f:
            f.write(_ROWS)
        self.artifact = os.path.join(self.dir.name, "cal.json")
        self.task = SpamTask()

    def _config(self):
        return {"dataset": {"source": "local",
                            "local": {"path": self.bench, "format": "csv"}},
                "generation": {"sample_size": 5, "mode": "inverse", "seedless": False,
                               "calibration_path": self.artifact,
                               "provider": "openai", "model": "m", "num_runs": 1},
                "task": {"name": "spam"}}

    def _write(self, semantics):
        cfg = self._config()
        real = pipeline.load_real_data(cfg, self.task)
        empirical = self.task.profile_error_distribution(real, config=cfg)
        write_calibration(self.artifact, {
            "meta": {"class_conditional_semantics": semantics} if semantics else {},
            "target": empirical, "calibrated": empirical,
            "selected_round": 0, "rounds": [],
        })
        return cfg, real, empirical

    def test_current_semantics_is_accepted(self):
        cfg, real, empirical = self._write(CLASS_CONDITIONAL_SEMANTICS)
        with redirect_stdout(io.StringIO()):
            out = pipeline.load_error_distribution(cfg, real, self.task)
        self.assertEqual(out["type_dist"], empirical["type_dist"])

    def test_asymmetric_artifact_is_refused_and_names_recalibration(self):
        cfg, real, empirical = self._write("asymmetric")
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            out = pipeline.load_error_distribution(cfg, real, self.task)
        self.assertEqual(out, empirical)          # fell back
        message = err.getvalue()
        self.assertIn("WARN", message)
        self.assertIn("recalibrat", message.lower())

    def test_absent_marker_reads_as_legacy_and_is_refused(self):
        cfg, real, empirical = self._write(None)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            out = pipeline.load_error_distribution(cfg, real, self.task)
        self.assertEqual(out, empirical)
        self.assertIn("WARN", err.getvalue())

    def test_meta_records_the_semantics_for_a_classification_run(self):
        cfg = self._config()
        meta = pipeline._build_meta(cfg, self.task, runs_completed=1,
                                    effective_samples_per_run=[5], real_baseline=False)
        self.assertEqual(meta["class_conditional_semantics"],
                         CLASS_CONDITIONAL_SEMANTICS)

    def test_meta_omits_the_marker_for_a_non_classification_task(self):
        from framework.tasks.gec.task import GECTask
        cfg = self._config()
        cfg["task"]["name"] = "gec"
        meta = pipeline._build_meta(cfg, GECTask(), runs_completed=1,
                                    effective_samples_per_run=[5], real_baseline=False)
        self.assertIsNone(meta["class_conditional_semantics"])
