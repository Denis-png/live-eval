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
        # Task 7: _LAST_CALIBRATION also carries class_prob (None here — this
        # artifact's `calibrated` has no class_prob key, i.e. no Stage B ran).
        self.assertEqual(pipeline._LAST_CALIBRATION,
                         {"path": artifact, "selected_round": 3, "class_prob": None})

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


class ClassProbPrecedenceTests(unittest.TestCase):
    """I2/I4: Component 5's consumption half. A calibrated balance must
    correct the EMPIRICAL balance for differential attrition, but an explicit
    MAPPING in generation.class_balance is a user instruction and always wins —
    calibration included. Nothing previously exercised _resolve_class_prob
    pulling a calibrated value out of _LAST_CALIBRATION at all."""

    def tearDown(self):
        pipeline._LAST_CALIBRATION = None

    def test_empirical_class_balance_uses_the_calibrated_value(self):
        pipeline._LAST_CALIBRATION = {"path": "x", "selected_round": 1,
                                      "class_prob": {"SPAM": 0.77, "HAM": 0.23}}
        cfg = {"generation": {"class_balance": "empirical"}}
        self.assertEqual(
            pipeline._resolve_class_prob(cfg, [], SpamTask()),
            {"SPAM": 0.77, "HAM": 0.23})

    def test_explicit_class_balance_mapping_wins_over_a_calibrated_value(self):
        pipeline._LAST_CALIBRATION = {"path": "x", "selected_round": 1,
                                      "class_prob": {"SPAM": 0.77, "HAM": 0.23}}
        cfg = {"generation": {"class_balance": {"SPAM": 0.25, "HAM": 0.75}}}
        self.assertEqual(
            pipeline._resolve_class_prob(cfg, [], SpamTask()),
            {"SPAM": 0.25, "HAM": 0.75})

    def test_no_calibration_falls_back_to_the_real_reference_fraction(self):
        cfg = {"generation": {"class_balance": "empirical"}}
        real_reference = [{"label": "SPAM"}, {"label": "HAM"},
                          {"label": "HAM"}, {"label": "HAM"}]
        # The task is what says which labels exist; this used to fall back
        # to a hardcoded "SPAM", which is the coupling that was removed.
        self.assertEqual(
            pipeline._resolve_class_prob(cfg, real_reference, SpamTask()),
            {"SPAM": 0.25, "HAM": 0.75})


class _ThreeLabelTask:
    """Duck-typed stand-in for a task with more than two labels — only
    get_class_labels() is needed by _resolve_class_prob's numeric branch."""
    def get_class_labels(self):
        return ("NEGATIVE", "NEUTRAL", "POSITIVE")

    def get_task_name(self):
        return "sentiment-fixture"


class ClassBalanceFloatTests(unittest.TestCase):
    """A bare `generation.class_balance` float is the binary legacy spelling of
    the mapping: P(labels[0]) is what it has always meant for a two-label task
    (P(SPAM) with ("SPAM", "HAM")), so every existing two-label config must
    keep working unchanged rather than silently falling through to the
    empirical/calibrated fallback (I1: a documented config value must never
    silently do nothing)."""

    def tearDown(self):
        pipeline._LAST_CALIBRATION = None

    def test_two_labels_float_is_honoured_as_the_first_labels_share(self):
        pipeline._LAST_CALIBRATION = {"path": "x", "selected_round": 1,
                                      "class_prob": {"SPAM": 0.05, "HAM": 0.95}}
        cfg = {"generation": {"class_balance": 0.9}}
        result = pipeline._resolve_class_prob(cfg, [], SpamTask())
        self.assertEqual(set(result), {"SPAM", "HAM"})
        self.assertAlmostEqual(result["SPAM"], 0.9)
        self.assertAlmostEqual(result["HAM"], 0.1)
        # Must differ from what empirical/calibrated would give here, or this
        # test cannot tell "the float was honoured" from "it was ignored".
        self.assertNotAlmostEqual(result["SPAM"], 0.05)

    def test_more_than_two_labels_float_raises_naming_the_mapping_form(self):
        cfg = {"generation": {"class_balance": 0.9}}
        with self.assertRaises(RuntimeError) as ctx:
            pipeline._resolve_class_prob(cfg, [], _ThreeLabelTask())
        message = str(ctx.exception)
        self.assertIn("ambiguous", message)
        # Names the mapping form to use instead, with the task's own labels.
        self.assertIn("NEGATIVE", message)
        self.assertIn("NEUTRAL", message)
        self.assertIn("POSITIVE", message)

    def test_float_outside_zero_to_one_raises(self):
        cfg = {"generation": {"class_balance": 1.5}}
        with self.assertRaises(RuntimeError) as ctx:
            pipeline._resolve_class_prob(cfg, [], SpamTask())
        self.assertIn("0..1", str(ctx.exception))

        cfg_negative = {"generation": {"class_balance": -0.1}}
        with self.assertRaises(RuntimeError):
            pipeline._resolve_class_prob(cfg_negative, [], SpamTask())

    def test_bool_is_rejected_not_silently_treated_as_a_probability(self):
        # bool is a subclass of int in Python, so `class_balance: true` would
        # otherwise silently pass the numeric branch as 1.0 — reject it
        # explicitly instead.
        cfg = {"generation": {"class_balance": True}}
        with self.assertRaises(RuntimeError) as ctx:
            pipeline._resolve_class_prob(cfg, [], SpamTask())
        self.assertIn("boolean", str(ctx.exception))
