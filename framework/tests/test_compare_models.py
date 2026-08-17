import json
import os
import tempfile
import unittest
from unittest import mock

import scripts.compare_models as cm


class SlugTests(unittest.TestCase):
    def test_slug_sanitizes(self):
        self.assertEqual(cm._slug("claude-haiku-4-5"), "claude_haiku_4_5")
        self.assertEqual(cm._slug("minimax/m2.7"), "minimax_m2_7")


class PerModelConfigTests(unittest.TestCase):
    def _base(self, base_dir):
        return {
            "api_keys": {"anthropic": "k1", "openrouter": "k2"},
            "dataset": {"name": "d", "split": "train", "sample_size": 5},
            "generation": {"mode": "inverse", "provider": "x", "model": "y"},
            "task": {"name": "spam"},
            "task_models": [{"name": "m", "type": "roberta"}],
            "output": {"base_dir": base_dir},
        }

    def test_merges_entry_and_names_output(self):
        with tempfile.TemporaryDirectory() as d:
            base = self._base(d)
            cfg = cm._per_model_config(base, {"provider": "anthropic", "model": "claude-haiku-4-5"})
            self.assertEqual(cfg["generation"]["provider"], "anthropic")
            self.assertEqual(cfg["generation"]["api_key"], "k1")  # re-resolved
            self.assertEqual(cfg["output"]["base_dir"], d)
            # The session id carries the generation cell (mode + seedless) so a
            # seedless comparison cannot overwrite the seeded one.
            self.assertEqual(cfg["output"]["session_id"],
                             "anthropic_claude_haiku_4_5_inverse")


class ParseCompareArgsTests(unittest.TestCase):
    """compare_models must own its CLI: per-model provider/model come from the
    generation_models list, so accepting --provider/--model (as main.py's parser
    does) silently leaked a CLI provider into entries that omitted one."""

    def test_rejects_provider_flag(self):
        with self.assertRaises(SystemExit):
            cm.parse_compare_args(["--provider", "openai"])

    def test_rejects_model_flag(self):
        with self.assertRaises(SystemExit):
            cm.parse_compare_args(["--model", "gpt"])

    def test_accepts_sample_shaping_flags(self):
        args = cm.parse_compare_args(
            ["--config", "c.yaml", "--task", "spam", "--runs", "2", "--sample-size", "9"]
        )
        self.assertEqual(args.config, "c.yaml")
        self.assertEqual(args.task, "spam")
        self.assertEqual(args.runs, 2)
        self.assertEqual(args.sample_size, 9)


class RunComparisonTests(unittest.TestCase):
    def test_runs_each_model_and_writes_combined(self):
        with tempfile.TemporaryDirectory() as d:
            base = {
                "api_keys": {"openrouter": "k"},
                "dataset": {"name": "d", "split": "train", "sample_size": 5},
                "generation": {"mode": "forward", "provider": "x", "model": "y"},
                "task": {"name": "spam"},
                "task_models": [{"name": "m", "type": "roberta"}],
                "output": {"base_dir": d},
                "generation_models": [
                    {"provider": "openrouter", "model": "a"},
                    {"provider": "openrouter", "model": "b"},
                ],
            }
            seen_sessions = []

            def fake_run(cfg):
                seen_sessions.append(
                    os.path.join(cfg["output"]["base_dir"], cfg["output"]["session_id"])
                )
                return {
                    "m": {
                        "generated": {"f1": {"mean": 0.5, "std": 0.1}},
                        "real": {"f1": 0.6},
                    }
                }

            original = cm.run_pipeline
            cm.run_pipeline = fake_run
            try:
                results = cm.run_comparison(base)
            finally:
                cm.run_pipeline = original

            self.assertEqual(len(seen_sessions), 2)
            self.assertEqual(len(set(seen_sessions)), 2)  # distinct per-model session dirs
            self.assertIn("openrouter/a", results)
            self.assertTrue(
                os.path.exists(os.path.join(d, "spam", "comparison", "comparison.json"))
            )

    def test_empty_generation_models_raises(self):
        with self.assertRaises(ValueError):
            cm.run_comparison({"generation": {}, "task": {"name": "spam"}, "output": {}})


class RunComparisonFailureTests(unittest.TestCase):
    def _config(self, base_dir):
        return {
            "api_keys": {"openrouter": "k"},
            "dataset": {"name": "d", "split": "train"},
            "generation": {"provider": "openrouter", "model": "base"},
            "task": {"name": "spam"},
            "task_models": [{"name": "m", "type": "roberta"}],
            "output": {"base_dir": base_dir},
            "generation_models": [
                {"provider": "openrouter", "model": "good"},
                {"provider": "openrouter", "model": "bad"},
            ],
        }

    def test_failed_entry_recorded_and_others_complete(self):
        ok_result = {"m": {"generated": {"accuracy": {"mean": 1.0, "std": 0.0}}}}

        def fake_run(cfg):
            if cfg["generation"]["model"] == "bad":
                raise RuntimeError("generation produced 0 samples")
            return ok_result

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(cm, "run_pipeline", side_effect=fake_run):
                results = cm.run_comparison(self._config(d))
            self.assertEqual(results["openrouter/good"], ok_result)
            self.assertEqual(
                results["openrouter/bad"],
                {"error": "generation produced 0 samples"},
            )
            combined = os.path.join(d, "spam", "comparison", "comparison.json")
            with open(combined, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["openrouter/bad"]["error"],
                             "generation produced 0 samples")
            self.assertIn("openrouter/good", data)

    def test_all_entries_failed_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(
                cm, "run_pipeline", side_effect=RuntimeError("boom")
            ):
                with self.assertRaises(SystemExit) as ctx:
                    cm.run_comparison(self._config(d))
            self.assertNotEqual(ctx.exception.code, 0)

    def test_print_table_tolerates_error_entries(self):
        cm._print_table({
            "openrouter/good": {"m": {"generated": {"acc": {"mean": 1.0, "std": 0.0}}}},
            "openrouter/bad": {"error": "boom"},
        })  # must not raise


if __name__ == "__main__":
    unittest.main()


class SessionIdIncludesCellTests(unittest.TestCase):
    """The generation cell is part of session identity: without it a seedless
    comparison writes into the same directory as the seeded one for the same
    provider/model and silently overwrites it."""

    def _base(self, base_dir, **generation):
        gen = {"provider": "x", "model": "y"}
        gen.update(generation)
        return {
            "api_keys": {"openrouter": "k"},
            "dataset": {"name": "d", "split": "train"},
            "generation": gen,
            "task": {"name": "gec"},
            "task_models": [{"name": "m", "type": "t5"}],
            "output": {"base_dir": base_dir},
        }

    def _session(self, **generation):
        with tempfile.TemporaryDirectory() as d:
            cfg = cm._per_model_config(self._base(d, **generation),
                                       {"provider": "openrouter", "model": "minimax-m3"})
            return cfg["output"]["session_id"]

    def test_seeded_and_seedless_get_distinct_sessions(self):
        seeded = self._session(mode="inverse", seedless=False)
        seedless = self._session(mode="inverse", seedless=True)
        self.assertNotEqual(seeded, seedless)
        self.assertIn("seedless", seedless)
        self.assertNotIn("seedless", seeded)

    def test_modes_get_distinct_sessions(self):
        self.assertNotEqual(self._session(mode="forward"), self._session(mode="inverse"))

    def test_provider_and_model_still_present(self):
        session = self._session(mode="inverse")
        self.assertTrue(session.startswith("openrouter_minimax_m3"))
