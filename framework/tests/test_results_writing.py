import json
import os
import tempfile
import unittest

from framework.pipeline import _build_meta, _write_results


class _CorruptionTask:
    """Fake task with a real forward/inverse mode (mirrors GEC)."""
    def get_generation_strategy(self):
        return "corruption"

    def get_task_name(self):
        return "gec"


class _ClassConditionalTask:
    """Fake task where mode never applies (mirrors spam)."""
    def get_generation_strategy(self):
        return "class_conditional"

    def get_task_name(self):
        return "spam"


def _config(results_path):
    return {
        "dataset": {"name": "d/ds", "split": "train", "sample_size": 50},
        "generation": {"mode": "inverse", "provider": "openrouter",
                       "model": "minimax-m2.7", "num_runs": 3, "sample_size": 20},
        "task": {"name": "spam"},
        "output": {"results_path": results_path},
    }


class BuildMetaTests(unittest.TestCase):
    def test_meta_records_provenance(self):
        meta = _build_meta(_config("r.json"), _CorruptionTask(), runs_completed=3,
                           effective_samples_per_run=[18, 20, 19], real_baseline=True)
        self.assertEqual(meta["task"], "spam")
        self.assertEqual(meta["strategy"], "corruption")
        self.assertEqual(meta["mode"], "inverse")
        self.assertEqual(meta["provider"], "openrouter")
        self.assertEqual(meta["model"], "minimax-m2.7")
        self.assertEqual(meta["num_runs"], 3)
        self.assertEqual(meta["runs_completed"], 3)
        self.assertFalse(meta["partial"])
        self.assertEqual(meta["dataset"]["name"], "d/ds")
        # sample_size comes from generation.sample_size, NOT the dataset pool size —
        # this is the provenance fix: dataset.sample_size (50) must NOT leak in.
        self.assertEqual(meta["dataset"]["sample_size"], 20)
        self.assertEqual(meta["effective_samples_per_run"], [18, 20, 19])
        self.assertTrue(meta["real_baseline"])
        self.assertEqual(meta["class_balance"], "empirical")
        self.assertNotIn("generation_sample_size", meta)
        self.assertIn("created", meta)

    def test_meta_marks_partial_when_runs_incomplete(self):
        meta = _build_meta(_config("r.json"), _CorruptionTask(), runs_completed=1,
                           effective_samples_per_run=[18], real_baseline=False)
        self.assertTrue(meta["partial"])
        self.assertEqual(meta["runs_completed"], 1)
        self.assertFalse(meta["real_baseline"])

    def test_meta_records_local_source_path(self):
        cfg = _config("r.json")
        cfg["dataset"] = {"source": "local", "sample_size": 300,
                          "local": {"path": "framework/data/spam/sms.csv", "format": "csv"}}
        meta = _build_meta(cfg, _CorruptionTask(), runs_completed=1, effective_samples_per_run=[3],
                           real_baseline=True)
        self.assertEqual(meta["dataset"]["source"], "local")
        self.assertEqual(meta["dataset"]["path"], "framework/data/spam/sms.csv")
        # generation.sample_size (20) is authoritative, not dataset.sample_size (300).
        self.assertEqual(meta["dataset"]["sample_size"], 20)
        self.assertNotIn("name", meta["dataset"])

    def test_meta_records_hf_source(self):
        meta = _build_meta(_config("r.json"), _CorruptionTask(), runs_completed=1,
                           effective_samples_per_run=[3], real_baseline=True)
        self.assertEqual(meta["dataset"]["source"], "huggingface")
        self.assertEqual(meta["dataset"]["name"], "d/ds")

    def test_meta_judge_none_when_judging_off(self):
        cfg = _config("r.json")
        cfg["judge"] = {"enabled": False, "provider": "groq", "model": "m"}
        meta = _build_meta(cfg, _CorruptionTask(), runs_completed=1, effective_samples_per_run=[1],
                           real_baseline=True)
        self.assertIsNone(meta["judge"])

    def test_meta_judge_recorded_when_enabled(self):
        cfg = _config("r.json")
        cfg["judge"] = {"enabled": True, "provider": "groq", "model": "llama"}
        meta = _build_meta(cfg, _CorruptionTask(), runs_completed=1, effective_samples_per_run=[1],
                           real_baseline=True)
        self.assertEqual(meta["judge"], {"provider": "groq", "model": "llama"})

    def test_meta_records_explicit_class_balance(self):
        cfg = _config("r.json")
        cfg["generation"]["class_balance"] = 0.3
        meta = _build_meta(cfg, _CorruptionTask(), runs_completed=1, effective_samples_per_run=[1],
                           real_baseline=True)
        self.assertEqual(meta["class_balance"], 0.3)

    def test_class_conditional_task_echoes_configured_mode(self):
        # spam now has real forward/inverse modes — meta must echo whatever
        # generation.mode the config actually set, same as a corruption task.
        cfg = _config("r.json")  # generation.mode: "inverse"
        meta = _build_meta(cfg, _ClassConditionalTask(), runs_completed=1,
                           effective_samples_per_run=[1], real_baseline=True)
        self.assertEqual(meta["strategy"], "class_conditional")
        self.assertEqual(meta["mode"], "inverse")

    def test_class_conditional_task_defaults_to_inverse_without_config_mode_key(self):
        # Omitting generation.mode must keep reproducing today's production
        # behavior (cross_class over real seeds), so the recorded default is
        # "inverse", not the corruption strategy's "forward" default.
        cfg = _config("r.json")
        del cfg["generation"]["mode"]
        meta = _build_meta(cfg, _ClassConditionalTask(), runs_completed=1,
                           effective_samples_per_run=[1], real_baseline=True)
        self.assertEqual(meta["mode"], "inverse")


class SeedlessMetaTests(unittest.TestCase):
    """Task 9: meta["seedless"] / meta["profile_path"] are new, additive keys.

    meta["strategy"] already means the task's generation SHAPE ("corruption"
    vs "class_conditional") and must NOT be repurposed to encode seedless-ness
    — that was the brief's mistake; these tests pin the corrected contract."""

    def _meta(self, task, mode, seedless, profile_path=None):
        cfg = _config("r.json")
        cfg["generation"]["mode"] = mode
        cfg["generation"]["seedless"] = seedless
        if profile_path is not None:
            cfg["generation"]["profile_path"] = profile_path
        return _build_meta(cfg, task, runs_completed=1,
                           effective_samples_per_run=[1], real_baseline=True)

    def test_seedless_and_profile_path_for_each_mode_seedless_combination(self):
        for mode in ("forward", "inverse"):
            for seedless in (False, True):
                meta = self._meta(_CorruptionTask(), mode, seedless,
                                  profile_path="prof.json" if seedless else None)
                self.assertEqual(meta["seedless"], seedless)
                self.assertEqual(meta["profile_path"], "prof.json" if seedless else None)
                # strategy stays the task shape, untouched by mode/seedless.
                self.assertEqual(meta["strategy"], "corruption")
                self.assertEqual(meta["mode"], mode)

    def test_seedless_defaults_false_and_profile_path_none_when_absent(self):
        # Configs predating this feature have no "seedless" key at all.
        cfg = _config("r.json")
        meta = _build_meta(cfg, _CorruptionTask(), runs_completed=1,
                           effective_samples_per_run=[1], real_baseline=True)
        self.assertFalse(meta["seedless"])
        self.assertIsNone(meta["profile_path"])

    def test_profile_path_ignored_when_not_seedless(self):
        meta = self._meta(_CorruptionTask(), "inverse", False, profile_path="prof.json")
        self.assertIsNone(meta["profile_path"])

    def test_class_conditional_strategy_not_repurposed_by_seedless(self):
        meta = self._meta(_ClassConditionalTask(), "inverse", True, profile_path="prof.json")
        self.assertEqual(meta["strategy"], "class_conditional")
        self.assertTrue(meta["seedless"])
        self.assertEqual(meta["profile_path"], "prof.json")


class WriteResultsTests(unittest.TestCase):
    def test_writes_meta_and_results(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "results.json")
            cfg = _config(path)
            scores = {"m": {"generated": {"f1": {"mean": 0.5, "std": 0.1}}}}
            meta = _build_meta(cfg, _CorruptionTask(), runs_completed=3, effective_samples_per_run=[20, 20, 20],
                               real_baseline=False)
            written = _write_results(scores, path, meta)
            self.assertEqual(written, path)
            with open(path) as f:
                payload = json.load(f)
            self.assertEqual(payload["results"], scores)
            self.assertEqual(payload["meta"]["task"], "spam")


if __name__ == "__main__":
    unittest.main()
