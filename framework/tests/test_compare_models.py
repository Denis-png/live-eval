import json
import os
import tempfile
import unittest

import scripts.compare_models as cm


class SlugTests(unittest.TestCase):
    def test_slug_sanitizes(self):
        self.assertEqual(cm._slug("claude-haiku-4-5"), "claude_haiku_4_5")
        self.assertEqual(cm._slug("minimax/m2.7"), "minimax_m2_7")


class PerModelConfigTests(unittest.TestCase):
    def _base(self, results_dir):
        return {
            "api_keys": {"anthropic": "k1", "openrouter": "k2"},
            "dataset": {"name": "d", "split": "train", "sample_size": 5},
            "generation": {"mode": "inverse", "provider": "x", "model": "y"},
            "task": {"name": "spam"},
            "task_models": [{"name": "m", "type": "roberta"}],
            "output": {"results_dir": results_dir},
        }

    def test_merges_entry_and_names_output(self):
        with tempfile.TemporaryDirectory() as d:
            base = self._base(d)
            cfg = cm._per_model_config(base, {"provider": "anthropic", "model": "claude-haiku-4-5"})
            self.assertEqual(cfg["generation"]["provider"], "anthropic")
            self.assertEqual(cfg["generation"]["api_key"], "k1")  # re-resolved
            self.assertEqual(
                cfg["output"]["results_path"],
                os.path.join(d, "spam_inverse_anthropic_claude_haiku_4_5.json"),
            )


class RunComparisonTests(unittest.TestCase):
    def test_runs_each_model_and_writes_combined(self):
        with tempfile.TemporaryDirectory() as d:
            base = {
                "api_keys": {"openrouter": "k"},
                "dataset": {"name": "d", "split": "train", "sample_size": 5},
                "generation": {"mode": "forward", "provider": "x", "model": "y"},
                "task": {"name": "spam"},
                "task_models": [{"name": "m", "type": "roberta"}],
                "output": {"results_dir": d},
                "generation_models": [
                    {"provider": "openrouter", "model": "a"},
                    {"provider": "openrouter", "model": "b"},
                ],
            }
            seen_paths = []

            def fake_run(cfg):
                seen_paths.append(cfg["output"]["results_path"])
                return {"m": {"f1": {"mean": 0.5, "std": 0.1}}}

            original = cm.run_pipeline
            cm.run_pipeline = fake_run
            try:
                results = cm.run_comparison(base)
            finally:
                cm.run_pipeline = original

            self.assertEqual(len(seen_paths), 2)
            self.assertEqual(len(set(seen_paths)), 2)  # distinct per-model files
            self.assertIn("openrouter/a", results)
            self.assertTrue(os.path.exists(os.path.join(d, "comparison_spam_forward.json")))

    def test_empty_generation_models_raises(self):
        with self.assertRaises(ValueError):
            cm.run_comparison({"generation": {}, "task": {"name": "spam"}, "output": {}})


if __name__ == "__main__":
    unittest.main()
