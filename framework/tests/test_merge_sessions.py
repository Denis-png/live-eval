import json
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.merge_sessions import merge_sessions


class FakeModel:
    def predict(self, texts):
        return ["SPAM"] * len(texts)


def _write_session(d, *, created, n_runs, model="m3"):
    os.makedirs(os.path.join(d, "generated"))
    meta = {
        "created": created, "task": "spam", "mode": None, "strategy": "class_conditional",
        "provider": "openrouter", "model": model, "num_runs": n_runs,
        "runs_completed": n_runs, "partial": False,
        "dataset": {"source": "local", "path": "x.csv", "format": "csv", "sample_size": 2},
        "effective_samples_per_run": [2] * n_runs, "judge": None,
        "real_baseline": True, "class_balance": "empirical",
    }
    results = {"old_model": {"generated": {"accuracy": {"mean": 0.5, "std": 0.0}},
                             "real": {"accuracy": 1.0}}}
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    for i in range(1, n_runs + 1):
        with open(os.path.join(d, "generated", f"run_{i}.json"), "w") as f:
            json.dump([{"text": f"WIN a FREE prize {created}-{i}!!", "label": "SPAM",
                        "technique": "spam_keywords", "seed": "hi"},
                       {"text": "see you at lunch", "label": "HAM",
                        "technique": "paraphrase", "seed": "lunch?"}], f)
    with open(os.path.join(d, "real_sample.json"), "w") as f:
        json.dump([{"text": "free money!!!", "label": "SPAM"},
                   {"text": "meeting at 3", "label": "HAM"}], f)


def _config():
    return {
        "task": {"name": "spam"},
        "task_models": [{"name": "old_model", "type": "roberta"}],
        "evaluation": {"real_baseline": True},
    }


class MergeSessionsTests(unittest.TestCase):
    def test_merges_run_files_and_recomputes_from_combined_count(self):
        with tempfile.TemporaryDirectory() as root:
            a, b, out = (os.path.join(root, x) for x in ("a", "b", "out"))
            _write_session(a, created="2026-07-08T00:00:00", n_runs=3)
            _write_session(b, created="2026-07-14T00:00:00", n_runs=3)
            with patch("framework.tasks.spam.task.SpamTask.get_model",
                       lambda self, cfg: FakeModel()):
                merge_sessions([a, b], out, _config())

            run_files = sorted(os.listdir(os.path.join(out, "generated")))
            self.assertEqual(run_files, [f"run_{i}.json" for i in range(1, 7)])

            out_meta = json.load(open(os.path.join(out, "results.json")))["meta"]
            self.assertEqual(out_meta["num_runs"], 6)
            self.assertEqual(out_meta["runs_completed"], 6)
            self.assertFalse(out_meta["partial"])
            self.assertEqual(
                [m["session"] for m in out_meta["merged_from"]], [a, b]
            )
            self.assertEqual(len(json.load(open(os.path.join(out, "results.json")))
                                  ["results"]["old_model"]["runs"]), 6)

    def test_mismatched_model_raises(self):
        with tempfile.TemporaryDirectory() as root:
            a, b, out = (os.path.join(root, x) for x in ("a", "b", "out"))
            _write_session(a, created="2026-07-08T00:00:00", n_runs=2, model="m3")
            _write_session(b, created="2026-07-14T00:00:00", n_runs=2, model="other-model")
            with self.assertRaises(ValueError):
                merge_sessions([a, b], out, _config())

    def test_requires_at_least_two_sessions(self):
        with tempfile.TemporaryDirectory() as root:
            a, out = os.path.join(root, "a"), os.path.join(root, "out")
            _write_session(a, created="2026-07-08T00:00:00", n_runs=2)
            with self.assertRaises(ValueError):
                merge_sessions([a], out, _config())


if __name__ == "__main__":
    unittest.main()
