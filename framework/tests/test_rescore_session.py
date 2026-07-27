import json
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.rescore_session import rescore_session


class FakeModel:
    """Predicts SPAM for everything."""
    def predict(self, texts):
        return ["SPAM"] * len(texts)


def _write_session(d):
    os.makedirs(os.path.join(d, "generated"))
    meta = {
        "created": "2026-07-08T00:00:00", "task": "spam", "mode": "inverse",
        "provider": "openrouter", "model": "m3", "num_runs": 2, "runs_completed": 2,
        "partial": False, "dataset": {"source": "local", "path": "x.csv",
                                      "format": "csv", "sample_size": 2},
        "effective_samples_per_run": [2, 2], "judge": None,
        "real_baseline": True, "class_balance": "empirical",
    }
    # Old results: only old_model, no per-run scores.
    results = {"old_model": {"generated": {"accuracy": {"mean": 0.5, "std": 0.0}},
                             "real": {"accuracy": 1.0}}}
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    for i in (1, 2):
        with open(os.path.join(d, "generated", f"run_{i}.json"), "w") as f:
            json.dump([{"text": "WIN a FREE prize now!!", "label": "SPAM",
                        "technique": "spam_keywords", "seed": "hi"},
                       {"text": "see you at lunch", "label": "HAM",
                        "technique": "paraphrase", "seed": "lunch?"}], f)
    with open(os.path.join(d, "real_sample.json"), "w") as f:
        json.dump([{"text": "free money!!!", "label": "SPAM"},
                   {"text": "meeting at 3", "label": "HAM"}], f)


def _config():
    return {
        "task": {"name": "spam"},
        "task_models": [{"name": "old_model", "type": "roberta"},
                        {"name": "new_model", "type": "bert"}],
        "evaluation": {"real_baseline": True},
    }


class RescoreSessionTests(unittest.TestCase):
    def _run(self, d, **kwargs):
        with patch("framework.tasks.spam.task.SpamTask.get_model",
                   lambda self, cfg: FakeModel()):
            return rescore_session(d, _config(), **kwargs)

    def test_restores_per_run_scores_and_adds_new_model(self):
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            self._run(d)
            out = json.load(open(os.path.join(d, "results.json")))
            for name in ("old_model", "new_model"):
                self.assertEqual(len(out["results"][name]["runs"]), 2)
                # all-SPAM predictions on a 50/50 sample -> accuracy 0.5, recall 1.0
                self.assertEqual(out["results"][name]["generated"]["accuracy"]["mean"], 0.5)
                self.assertEqual(out["results"][name]["runs"][0]["recall"], 1.0)
                self.assertEqual(out["results"][name]["real"]["accuracy"], 0.5)
            self.assertIn("rescored", out["meta"])
            self.assertEqual(out["meta"]["created"], "2026-07-08T00:00:00")
            self.assertFalse(out["meta"]["partial"])

    def test_writes_profile_json(self):
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            self._run(d)
            prof = json.load(open(os.path.join(d, "profile.json")))
            self.assertEqual(set(prof), {"real", "generated", "fidelity"})
            self.assertEqual(prof["generated"]["n"], 4)  # 2 runs x 2 rows

    def test_skip_eval_leaves_results_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            before = json.load(open(os.path.join(d, "results.json")))
            self._run(d, skip_eval=True)
            after = json.load(open(os.path.join(d, "results.json")))
            self.assertEqual(before, after)
            self.assertTrue(os.path.exists(os.path.join(d, "profile.json")))

    def test_num_runs_recomputed_from_actual_run_files(self):
        # Stale meta claims 5 runs even though only 2 generated/run_*.json exist
        # (e.g. after a manual merge) — rescoring must trust the files, not meta.
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            stale = json.load(open(os.path.join(d, "results.json")))
            stale["meta"]["num_runs"] = 5
            stale["meta"]["runs_completed"] = 3
            stale["meta"]["partial"] = True
            json.dump(stale, open(os.path.join(d, "results.json"), "w"))
            self._run(d)
            out = json.load(open(os.path.join(d, "results.json")))
            self.assertEqual(out["meta"]["num_runs"], 2)
            self.assertEqual(out["meta"]["runs_completed"], 2)
            self.assertFalse(out["meta"]["partial"])

    def test_mode_and_strategy_recomputed_from_task_not_stale_meta(self):
        # _write_session seeds a stale "mode": "inverse" (pre-dates the
        # provenance fix) — spam is class_conditional, so mode must become
        # None and strategy must be filled in, regardless of the seed value.
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            self._run(d)
            out = json.load(open(os.path.join(d, "results.json")))
            self.assertEqual(out["meta"]["strategy"], "class_conditional")
            self.assertIsNone(out["meta"]["mode"])

    def test_task_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            cfg = _config()
            cfg["task"]["name"] = "gec"
            with self.assertRaises(ValueError):
                rescore_session(d, cfg)


if __name__ == "__main__":
    unittest.main()
