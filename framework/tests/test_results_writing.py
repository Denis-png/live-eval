import json
import os
import tempfile
import unittest

from framework.pipeline import _build_meta, _write_results


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
        meta = _build_meta(_config("r.json"), runs_completed=3,
                           effective_samples_per_run=[18, 20, 19])
        self.assertEqual(meta["task"], "spam")
        self.assertEqual(meta["mode"], "inverse")
        self.assertEqual(meta["provider"], "openrouter")
        self.assertEqual(meta["model"], "minimax-m2.7")
        self.assertEqual(meta["num_runs"], 3)
        self.assertEqual(meta["runs_completed"], 3)
        self.assertFalse(meta["partial"])
        self.assertEqual(meta["dataset"]["name"], "d/ds")
        self.assertEqual(meta["generation_sample_size"], 20)
        self.assertEqual(meta["effective_samples_per_run"], [18, 20, 19])
        self.assertIn("created", meta)

    def test_meta_marks_partial_when_runs_incomplete(self):
        meta = _build_meta(_config("r.json"), runs_completed=1,
                           effective_samples_per_run=[18])
        self.assertTrue(meta["partial"])
        self.assertEqual(meta["runs_completed"], 1)

    def test_meta_records_local_source_path(self):
        cfg = _config("r.json")
        cfg["dataset"] = {"source": "local", "sample_size": 300,
                          "local": {"path": "framework/data/spam/sms.csv", "format": "csv"}}
        meta = _build_meta(cfg, runs_completed=1, effective_samples_per_run=[3])
        self.assertEqual(meta["dataset"]["source"], "local")
        self.assertEqual(meta["dataset"]["path"], "framework/data/spam/sms.csv")
        self.assertEqual(meta["dataset"]["sample_size"], 300)
        self.assertNotIn("name", meta["dataset"])

    def test_meta_records_hf_source(self):
        meta = _build_meta(_config("r.json"), runs_completed=1,
                           effective_samples_per_run=[3])
        self.assertEqual(meta["dataset"]["source"], "huggingface")
        self.assertEqual(meta["dataset"]["name"], "d/ds")

    def test_meta_judge_none_when_judging_off(self):
        cfg = _config("r.json")
        cfg["judge"] = {"enabled": False, "provider": "groq", "model": "m"}
        meta = _build_meta(cfg, runs_completed=1, effective_samples_per_run=[1])
        self.assertIsNone(meta["judge"])

    def test_meta_judge_recorded_when_enabled(self):
        cfg = _config("r.json")
        cfg["judge"] = {"enabled": True, "provider": "groq", "model": "llama"}
        meta = _build_meta(cfg, runs_completed=1, effective_samples_per_run=[1])
        self.assertEqual(meta["judge"], {"provider": "groq", "model": "llama"})


class WriteResultsTests(unittest.TestCase):
    def test_writes_meta_and_results(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "results.json")
            cfg = _config(path)
            scores = {"m": {"f1": {"mean": 0.5, "std": 0.1}}}
            meta = _build_meta(cfg, runs_completed=3, effective_samples_per_run=[20, 20, 20])
            written = _write_results(scores, cfg, meta)
            self.assertEqual(written, path)
            with open(path) as f:
                payload = json.load(f)
            self.assertEqual(payload["results"], scores)
            self.assertEqual(payload["meta"]["task"], "spam")


if __name__ == "__main__":
    unittest.main()
