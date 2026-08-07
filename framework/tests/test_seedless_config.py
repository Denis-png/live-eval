import json
import os
import tempfile
import unittest

from framework.main import validate_config
from framework.pipeline import _load_generation_profile


class FakeTask:
    def __init__(self, strategy="corruption", name="gec"):
        self._strategy = strategy
        self._name = name

    def get_generation_strategy(self):
        return self._strategy

    def get_task_name(self):
        return self._name


def _base_config(**generation):
    gen = {"provider": "openrouter", "model": "m", "num_runs": 1,
           "sample_size": 2, "mode": "inverse"}
    gen.update(generation)
    return {
        "dataset": {"source": "local", "local": {"path": "p", "format": "csv"}},
        "generation": gen,
        "task": {"name": "gec"},
        "task_models": [{"name": "n", "type": "t5"}],
    }


class ValidateSeedlessTests(unittest.TestCase):
    def test_non_boolean_seedless_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_config(_base_config(seedless="yes"))
        self.assertIn("seedless", str(ctx.exception))

    def test_boolean_seedless_accepted(self):
        validate_config(_base_config(seedless=True))
        validate_config(_base_config(seedless=False))

    def test_absent_seedless_accepted(self):
        validate_config(_base_config())


class LoadGenerationProfileTests(unittest.TestCase):
    def test_returns_none_when_not_seedless(self):
        self.assertIsNone(
            _load_generation_profile(_base_config(seedless=False), FakeTask())
        )

    def test_loads_configured_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"profile_version": 2, "topics": {"a": {"fraction": 1.0}}}, f)
            profile = _load_generation_profile(
                _base_config(seedless=True, profile_path=path), FakeTask()
            )
            self.assertEqual(profile["profile_version"], 2)

    def test_default_path_is_derived_from_task_name(self):
        with self.assertRaises(RuntimeError) as ctx:
            _load_generation_profile(_base_config(seedless=True), FakeTask())
        self.assertIn("framework/data/profiles/gec_profile.json", str(ctx.exception))

    def test_classification_profile_validated_against_per_label_topics(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"profile_version": 2, "topics": {"a": {"fraction": 1.0}}}, f)
            with self.assertRaises(RuntimeError) as ctx:
                _load_generation_profile(
                    _base_config(seedless=True, profile_path=path),
                    FakeTask(strategy="class_conditional", name="spam"),
                )
            self.assertIn("topics_per_label", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
