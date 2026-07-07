import argparse
import unittest

from framework.main import _resolve_api_keys, apply_overrides, validate_config


def _args(**kw):
    base = dict(task=None, provider=None, model=None, runs=None, sample_size=None,
                mode=None, output=None, judge=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _config():
    return {
        "task": {"name": "gec"},
        "generation": {"provider": "openai", "model": "gpt", "num_runs": 3, "sample_size": 20},
    }


def _full_config():
    return {
        "api_keys": {"openai": "sk-x"},
        "dataset": {"name": "d/ds", "split": "train", "sample_size": 50},
        "generation": {"provider": "openai", "model": "gpt", "num_runs": 3, "sample_size": 20},
        "task": {"name": "gec"},
        "task_models": [{"name": "m", "type": "roberta"}],
    }


class ApplyOverridesTests(unittest.TestCase):
    def test_none_args_leave_config_untouched(self):
        cfg = apply_overrides(_config(), _args())
        self.assertEqual(cfg["generation"]["num_runs"], 3)
        self.assertEqual(cfg["generation"]["sample_size"], 20)

    def test_explicit_values_override(self):
        cfg = apply_overrides(_config(), _args(runs=5, sample_size=50, provider="groq"))
        self.assertEqual(cfg["generation"]["num_runs"], 5)
        self.assertEqual(cfg["generation"]["sample_size"], 50)
        self.assertEqual(cfg["generation"]["provider"], "groq")

    def test_zero_is_a_real_override_not_ignored(self):
        cfg = apply_overrides(_config(), _args(runs=0, sample_size=0))
        self.assertEqual(cfg["generation"]["num_runs"], 0)
        self.assertEqual(cfg["generation"]["sample_size"], 0)

    def test_mode_override(self):
        cfg = apply_overrides(_config(), _args(mode="inverse"))
        self.assertEqual(cfg["generation"]["mode"], "inverse")

    def test_output_override_creates_output_block(self):
        cfg = apply_overrides(_config(), _args(output="out/runs"))
        self.assertEqual(cfg["output"]["base_dir"], "out/runs")

    def test_no_judge_disables_judge_block(self):
        cfg = _config()
        cfg["judge"] = {"enabled": True, "provider": "groq", "model": "m"}
        cfg = apply_overrides(cfg, _args(judge=False))
        self.assertFalse(cfg["judge"]["enabled"])

    def test_judge_enables_existing_judge_block(self):
        cfg = _config()
        cfg["judge"] = {"enabled": False, "provider": "groq", "model": "m"}
        cfg = apply_overrides(cfg, _args(judge=True))
        self.assertTrue(cfg["judge"]["enabled"])


class MainErrorHandlingTests(unittest.TestCase):
    def test_pipeline_runtime_error_exits_cleanly_without_traceback(self):
        """User-facing pipeline failures (e.g. 0 usable samples) must exit with
        a clean [ERROR] message, not a raw traceback."""
        import tempfile

        import yaml

        import framework.main as fm

        cfg = _full_config()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
            cfg_path = f.name

        def boom(config):
            raise RuntimeError("Generation produced 0 usable samples")

        orig_run, orig_argv = fm.run_pipeline, argparse._sys.argv
        fm.run_pipeline = boom
        argparse._sys.argv = ["main", "--config", cfg_path]
        try:
            with self.assertRaises(SystemExit) as ctx:
                fm.main()
        finally:
            fm.run_pipeline = orig_run
            argparse._sys.argv = orig_argv
        self.assertIn("0 usable samples", str(ctx.exception))


class ResolveApiKeysTests(unittest.TestCase):
    def test_injects_provider_key_into_generation(self):
        cfg = _full_config()
        _resolve_api_keys(cfg)
        self.assertEqual(cfg["generation"]["api_key"], "sk-x")

    def test_strict_raises_on_missing_generator_key(self):
        """A missing key must stop the run up front — warned-then-continued runs
        burn minutes before failing (or worse, score 0 samples as all-zero)."""
        cfg = _full_config()
        cfg["api_keys"] = {}
        with self.assertRaises(ValueError) as ctx:
            _resolve_api_keys(cfg, strict=True)
        self.assertIn("openai", str(ctx.exception).lower())

    def test_strict_raises_on_missing_key_for_enabled_judge(self):
        cfg = _full_config()
        cfg["judge"] = {"enabled": True, "provider": "groq", "model": "m"}
        with self.assertRaises(ValueError) as ctx:
            _resolve_api_keys(cfg, strict=True)
        self.assertIn("groq", str(ctx.exception).lower())

    def test_strict_ignores_disabled_judge(self):
        cfg = _full_config()
        cfg["judge"] = {"enabled": False, "provider": "groq", "model": "m"}
        _resolve_api_keys(cfg, strict=True)  # must not raise

    def test_non_strict_warns_but_continues(self):
        cfg = _full_config()
        cfg["api_keys"] = {}
        _resolve_api_keys(cfg)  # must not raise
        self.assertEqual(cfg["generation"]["api_key"], "")


class ValidateConfigTests(unittest.TestCase):
    def test_valid_config_passes(self):
        validate_config(_full_config())  # must not raise

    def test_missing_required_key_names_the_key(self):
        cfg = _full_config()
        del cfg["generation"]["num_runs"]
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("generation.num_runs", str(ctx.exception))

    def test_missing_section_names_the_section(self):
        cfg = _full_config()
        del cfg["task"]
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("task", str(ctx.exception))

    def test_zero_runs_rejected(self):
        cfg = _full_config()
        cfg["generation"]["num_runs"] = 0
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("num_runs", str(ctx.exception))

    def test_generation_sample_size_may_exceed_dataset_pool(self):
        # dataset.sample_size no longer exists; generation.sample_size is the
        # single source of truth and is not cross-checked against the dataset.
        cfg = _full_config()
        cfg["generation"]["sample_size"] = 99  # dataset pool (unused) is 50
        validate_config(cfg)  # must NOT raise

    def test_unknown_mode_rejected(self):
        cfg = _full_config()
        cfg["generation"]["mode"] = "sideways"
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("mode", str(ctx.exception))

    def test_empty_task_models_rejected(self):
        cfg = _full_config()
        cfg["task_models"] = []
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("task_models", str(ctx.exception))

    def test_local_source_requires_only_path_not_name_split(self):
        cfg = _full_config()
        cfg["dataset"] = {"source": "local", "sample_size": 50,
                          "local": {"path": "framework/data/spam/x.csv"}}
        validate_config(cfg)  # must not raise: name/split are HF-only

    def test_local_source_missing_path_names_the_key(self):
        cfg = _full_config()
        cfg["dataset"] = {"source": "local", "sample_size": 10}
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("dataset.local.path", str(ctx.exception))

    def test_local_source_undeterminable_format_rejected(self):
        cfg = _full_config()
        cfg["dataset"] = {"source": "local", "sample_size": 50,
                          "local": {"path": "data/things.jsonl"}}
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("format", str(ctx.exception).lower())

    def test_hf_source_missing_name_names_the_key(self):
        cfg = _full_config()
        cfg["dataset"] = {"source": "huggingface", "sample_size": 50,
                          "huggingface": {"split": "train"}}
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("dataset.huggingface.name", str(ctx.exception))

    def test_nested_hf_block_accepted_without_flat_keys(self):
        cfg = _full_config()
        cfg["dataset"] = {"source": "huggingface", "sample_size": 50,
                          "huggingface": {"name": "d/ds", "split": "train"}}
        validate_config(cfg)  # must not raise

    def test_unknown_source_rejected(self):
        cfg = _full_config()
        cfg["dataset"] = {"source": "ftp", "sample_size": 50}
        with self.assertRaises(ValueError) as ctx:
            validate_config(cfg)
        self.assertIn("source", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
