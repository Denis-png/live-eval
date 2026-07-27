import json
import os
import tempfile
import unittest

from framework.pipeline import _build_meta, _write_results, class_conditional_mode_notice


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

    def test_class_conditional_task_gets_null_mode_regardless_of_config(self):
        # cfg carries generation.mode: "inverse" (a leftover/stray value) but a
        # class_conditional task (spam) never reads it — meta must not echo it.
        cfg = _config("r.json")
        meta = _build_meta(cfg, _ClassConditionalTask(), runs_completed=1,
                           effective_samples_per_run=[1], real_baseline=True)
        self.assertEqual(meta["strategy"], "class_conditional")
        self.assertIsNone(meta["mode"])

    def test_class_conditional_task_null_mode_even_without_config_mode_key(self):
        cfg = _config("r.json")
        del cfg["generation"]["mode"]
        meta = _build_meta(cfg, _ClassConditionalTask(), runs_completed=1,
                           effective_samples_per_run=[1], real_baseline=True)
        self.assertIsNone(meta["mode"])


class ClassConditionalModeNoticeTests(unittest.TestCase):
    def test_warns_when_class_conditional_task_config_sets_mode(self):
        cfg = _config("r.json")  # generation.mode: "inverse"
        notice = class_conditional_mode_notice(cfg, _ClassConditionalTask(), "class_conditional")
        self.assertIsNotNone(notice)
        self.assertIn("generation.mode", notice)
        self.assertIn("inverse", notice)

    def test_silent_when_class_conditional_config_omits_mode(self):
        cfg = _config("r.json")
        del cfg["generation"]["mode"]
        notice = class_conditional_mode_notice(cfg, _ClassConditionalTask(), "class_conditional")
        self.assertIsNone(notice)

    def test_silent_for_corruption_strategy_even_with_mode_set(self):
        cfg = _config("r.json")
        notice = class_conditional_mode_notice(cfg, _CorruptionTask(), "corruption")
        self.assertIsNone(notice)


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
