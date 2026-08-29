import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework import pipeline
from framework.calibration.artifact import write_calibration
from framework.tasks.gec.task import GECTask
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


class _StubGenerator:
    def call_api(self, prompt):
        return "Corrupted: stub"


class ProvenanceIsolationTests(unittest.TestCase):
    """Regression: _LAST_CALIBRATION must not leak from one config to the next
    in the same process, even when the second config's strategy/mode never
    reaches _apply_calibration at all.

    _apply_calibration only runs inside load_error_distribution, which
    build_generation_context only calls when _should_load_error_distribution(...)
    is True. GEC forward+seeded (corruption strategy, mode="forward",
    seedless=False) is False on all three of its conditions, so that call is
    skipped entirely -- the ONLY thing that can still clear a calibration set by
    a prior config is the unconditional reset at the top of
    build_generation_context itself. scripts/compare_models.py loops
    run_pipeline() over multiple configs in one process, so this is a real
    contamination path, not a hypothetical one.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_non_calibratable_config_does_not_inherit_a_prior_configs_calibration(self):
        # 1) Populate _LAST_CALIBRATION via a calibratable spam config with a
        # matching artifact -- this is the "prior config" sharing the process.
        spam_bench = os.path.join(self.dir.name, "spam.csv")
        with open(spam_bench, "w", encoding="utf-8") as f:
            f.write(_ROWS)
        spam_task = SpamTask()
        spam_cfg = {
            "dataset": {"source": "local",
                        "local": {"path": spam_bench, "format": "csv"}},
            "generation": {"sample_size": 5, "mode": "inverse", "seedless": False},
            "task": {"name": "spam"},
        }
        spam_real = pipeline.load_real_data(spam_cfg, spam_task)
        empirical = spam_task.profile_error_distribution(spam_real, config=spam_cfg)
        artifact = os.path.join(self.dir.name, "spam_cal.json")
        write_calibration(artifact, {
            "meta": {}, "target": empirical, "calibrated": empirical,
            "selected_round": 3, "rounds": [],
        })
        spam_cfg["generation"]["calibration_path"] = artifact
        with redirect_stdout(io.StringIO()):
            pipeline.load_error_distribution(spam_cfg, spam_real, spam_task)
        # Sanity: the contamination source is real, not a no-op.
        self.assertEqual(pipeline._LAST_CALIBRATION,
                         {"path": artifact, "selected_round": 3})

        # 2) A GEC forward+seeded config never reaches _apply_calibration --
        # build_generation_context must still start clean for it.
        gec_bench = os.path.join(self.dir.name, "gec.csv")
        with open(gec_bench, "w", encoding="utf-8") as f:
            f.write("incorrect,correct\n" + "".join(
                f"He go to school {i} yesterday.,He went to school {i} yesterday.\n"
                for i in range(5)
            ))
        gec_task = GECTask()
        gec_cfg = {
            "dataset": {"source": "local",
                        "local": {"path": gec_bench, "format": "csv"}},
            "generation": {"provider": "openai", "model": "gpt-x", "num_runs": 1,
                           "sample_size": 5, "mode": "forward", "seedless": False},
            "task": {"name": "gec"},
        }
        with mock.patch.object(pipeline, "load_generator",
                               return_value=_StubGenerator()):
            with redirect_stdout(io.StringIO()):
                ctx = pipeline.build_generation_context(gec_cfg)
        # Sanity: this cell really does skip the calibration lookup entirely.
        self.assertIsNone(ctx["error_dist"])

        meta = pipeline._build_meta(gec_cfg, gec_task, runs_completed=1,
                                    effective_samples_per_run=[5],
                                    real_baseline=False)
        self.assertIsNone(meta["calibration"])


class MalformedArtifactTests(_Bench):
    """A JSON-valid but structurally corrupt artifact must warn and fall back
    to the empirical distribution, never crash the run. Complements
    test_drifted_target_warns_and_falls_back, which covers a well-formed but
    stale artifact; these cover artifacts that are simply broken."""

    def test_non_dict_target_warns_and_falls_back_instead_of_crashing(self):
        cfg = self._config(self.artifact)
        empirical = self.task.profile_error_distribution(
            self._real_data(cfg), config=cfg)
        write_calibration(self.artifact, {
            "meta": {}, "target": "oops", "calibrated": empirical,
            "selected_round": 0, "rounds": [],
        })
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            out = pipeline.load_error_distribution(cfg, self._real_data(cfg), self.task)
        self.assertEqual(out, empirical)
        self.assertIn("WARN", err.getvalue())

    def test_non_object_top_level_json_warns_and_falls_back_instead_of_crashing(self):
        cfg = self._config(self.artifact)
        empirical = self.task.profile_error_distribution(
            self._real_data(cfg), config=cfg)
        write_calibration(self.artifact, ["not", "an", "object"])
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            out = pipeline.load_error_distribution(cfg, self._real_data(cfg), self.task)
        self.assertEqual(out, empirical)
        self.assertIn("WARN", err.getvalue())
