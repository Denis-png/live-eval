import json
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.merge_sessions import merge_sessions


class FakeModel:
    def predict(self, texts):
        return ["SPAM"] * len(texts)


def _write_session(d, *, created, n_runs, model="m3", mode=None, seedless=False):
    os.makedirs(os.path.join(d, "generated"))
    meta = {
        "created": created, "task": "spam", "mode": mode, "seedless": seedless,
        "strategy": "class_conditional",
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

    def test_forward_and_inverse_sessions_refuse_to_merge(self):
        # IMPORTANT 4: forward and inverse are genuinely different generation
        # cells now (not just a label on an identical process), so merging
        # across them must be refused.
        with tempfile.TemporaryDirectory() as root:
            a, b, out = (os.path.join(root, x) for x in ("a", "b", "out"))
            _write_session(a, created="2026-07-08T00:00:00", n_runs=2, mode="forward")
            _write_session(b, created="2026-07-14T00:00:00", n_runs=2, mode="inverse")
            with self.assertRaises(ValueError):
                merge_sessions([a, b], out, _config())

    def test_seedless_and_seeded_sessions_of_the_same_mode_refuse_to_merge(self):
        with tempfile.TemporaryDirectory() as root:
            a, b, out = (os.path.join(root, x) for x in ("a", "b", "out"))
            _write_session(a, created="2026-07-08T00:00:00", n_runs=2,
                           mode="inverse", seedless=False)
            _write_session(b, created="2026-07-14T00:00:00", n_runs=2,
                           mode="inverse", seedless=True)
            with self.assertRaises(ValueError):
                merge_sessions([a, b], out, _config())

    def test_legacy_null_mode_still_merges_with_new_style_inverse(self):
        # A legacy session with "mode": null (the old _build_meta forced it)
        # and a new-style session recording "mode": "inverse" explicitly are
        # the SAME cell (class_conditional's per-strategy default is
        # "inverse") — merging them must still succeed.
        with tempfile.TemporaryDirectory() as root:
            a, b, out = (os.path.join(root, x) for x in ("a", "b", "out"))
            _write_session(a, created="2026-07-08T00:00:00", n_runs=2, mode=None)
            _write_session(b, created="2026-07-14T00:00:00", n_runs=2, mode="inverse")
            with patch("framework.tasks.spam.task.SpamTask.get_model",
                       lambda self, cfg: FakeModel()):
                merge_sessions([a, b], out, _config())
            run_files = sorted(os.listdir(os.path.join(out, "generated")))
            self.assertEqual(run_files, [f"run_{i}.json" for i in range(1, 5)])


if __name__ == "__main__":
    unittest.main()
