import argparse
import unittest

from framework.main import apply_overrides


def _args(**kw):
    base = dict(task=None, provider=None, model=None, runs=None, sample_size=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _config():
    return {
        "task": {"name": "gec"},
        "generation": {"provider": "openai", "model": "gpt", "num_runs": 3, "sample_size": 20},
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


if __name__ == "__main__":
    unittest.main()
