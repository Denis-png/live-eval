import json
import os
import tempfile
import unittest
from unittest import mock

import framework.pipeline as pipeline
from framework.main import validate_config
from framework.pipeline import _load_generation_profile
from framework.tasks.base_task import BaseTask


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
        # Pin DEFAULT_PROFILE_DIR to a directory that cannot exist rather than
        # relying on the real default (framework/data/profiles/) being empty —
        # the seedless prerequisite (profile_dataset --topics) legitimately
        # writes gec_profile.json there for local dev, and this test must not
        # depend on that ambient file being absent.
        with mock.patch.object(pipeline, "DEFAULT_PROFILE_DIR", "/nonexistent/profiles/dir"):
            with self.assertRaises(RuntimeError) as ctx:
                _load_generation_profile(_base_config(seedless=True), FakeTask())
        self.assertIn("/nonexistent/profiles/dir/gec_profile.json", str(ctx.exception))

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


class _FakeModel:
    def predict(self, texts):
        return ["x"] * len(texts)


class _FakeGenerator:
    """No API calls: forward's two cells (seeded/seedless) return canned records."""
    def call_api(self, prompt):
        return ""

    def generate(self, **kw):
        return [{"original": "a", "corrupted": "b b b", "error_type": "article"}
                for _ in range(kw["sample_size"])]

    def generate_seedless_pairs(self, *a, **kw):
        return [{"original": "a", "corrupted": "b b b", "error_type": "R:VERB:TENSE"}]


class _FakeCorruptionTask(BaseTask):
    """Minimal corruption-strategy task double. Subclassing BaseTask (rather than
    duck-typing every method) means get_real_eval_samples/profile_dataset default to
    None, so the real baseline and fidelity profiling stay skipped without needing
    to fake ERRANT/COLA/GLEU machinery — this test only cares whether run_pipeline
    loads the empirical error distribution for the forward+seedless cell."""
    def get_error_types(self): return ["article"]
    def get_prompt_instruction(self): return "Fix: {sentence}"
    def get_evaluators(self): return ["score"]
    def get_evaluator_fns(self): return {"score": lambda results: 1.0}
    def get_model(self, model_config): return _FakeModel()
    def get_task_name(self): return "gec"
    def parse_row(self, row): return row
    def get_seedless_forward_prompt(self): return "spec: {spec}\nerrors: {error_spec}"
    def get_error_descriptions(self): return {"R:VERB:TENSE": "use a wrong verb tense"}
    def get_profile_side(self, mode): return "incorrect"


_FAKE_PROFILE = {
    "profile_version": 2,
    "topics": {"t": {"fraction": 1.0}},
    "length_distributions": {"incorrect": {"words": {
        "bins": {"6-10": 1.0}, "quantiles": {"p90": 9}}}},
    "style": {"incorrect": {}},
}


def _pipeline_config(seedless, base_dir):
    return {
        "task": {"name": "gec"},
        "dataset": {"source": "local", "local": {"path": "unused.csv", "format": "csv"}},
        "generation": {"provider": "openrouter", "model": "m", "num_runs": 1,
                       "sample_size": 2, "mode": "forward", "seedless": seedless},
        "evaluation": {"real_baseline": True},
        "task_models": [{"name": "fake", "type": "t5"}],
        "output": {"base_dir": base_dir, "plots": False},
    }


class ErrorDistLoadedForSeedlessForwardTests(unittest.TestCase):
    """Regression: seedless forward samples its error type from the same empirical
    ERRANT distribution inverse mode uses (see spec Component 1's GEC table), so
    run_pipeline must load it for that cell too — not only for mode == 'inverse'.
    Before the fix, forward+seedless left error_dist as None and _run_generation's
    error_dist["type_dist"] lookup crashed with TypeError instead of running."""

    def test_forward_seedless_loads_error_dist(self):
        with tempfile.TemporaryDirectory() as d:
            config = _pipeline_config(seedless=True, base_dir=os.path.join(d, "runs"))
            loader = mock.Mock(return_value={"type_dist": {"R:VERB:TENSE": 1.0},
                                             "count_dist": {1: 1.0}})
            with mock.patch.object(pipeline, "load_task", lambda name: _FakeCorruptionTask()), \
                 mock.patch.object(pipeline, "load_generator", lambda c: _FakeGenerator()), \
                 mock.patch.object(pipeline, "load_real_data", lambda cfg, task: []), \
                 mock.patch.object(pipeline, "_load_generation_profile", lambda cfg, task: _FAKE_PROFILE), \
                 mock.patch.object(pipeline, "load_error_distribution", loader):
                pipeline.run_pipeline(config)
            self.assertTrue(loader.called)

    def test_forward_seeded_does_not_load_error_dist(self):
        with tempfile.TemporaryDirectory() as d:
            config = _pipeline_config(seedless=False, base_dir=os.path.join(d, "runs"))
            loader = mock.Mock(return_value={"type_dist": {}, "count_dist": {}})
            with mock.patch.object(pipeline, "load_task", lambda name: _FakeCorruptionTask()), \
                 mock.patch.object(pipeline, "load_generator", lambda c: _FakeGenerator()), \
                 mock.patch.object(pipeline, "load_real_data", lambda cfg, task: []), \
                 mock.patch.object(pipeline, "load_error_distribution", loader):
                pipeline.run_pipeline(config)
            self.assertFalse(loader.called)


if __name__ == "__main__":
    unittest.main()
