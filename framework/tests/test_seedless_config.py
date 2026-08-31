import json
import os
import tempfile
import unittest
from unittest import mock

import framework.pipeline as pipeline
from framework.main import validate_config
from framework.pipeline import _load_benchmark_profile
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
            _load_benchmark_profile(_base_config(seedless=False), FakeTask())
        )

    def test_loads_configured_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"profile_version": 2,
                           "topics": {"a": {"fraction": 1.0}},
                           "length_distributions": {"correct": {"words": {"bins": {"6-10": 1.0}}}},
                           "style": {"correct": {}}}, f)
            profile = _load_benchmark_profile(
                _base_config(seedless=True, profile_path=path), FakeTask()
            )
            self.assertEqual(profile["profile_version"], 2)

    def test_missing_profile_names_the_expected_per_task_pattern(self):
        # Pin DEFAULT_PROFILE_DIR to a directory that cannot exist rather than
        # relying on the real default being empty — the seedless prerequisite
        # (profile_dataset --topics) legitimately writes a profile there for
        # local dev, and this test must not depend on that being absent.
        with mock.patch.object(pipeline, "DEFAULT_PROFILE_DIR", "/nonexistent/profiles/dir"):
            with self.assertRaises(RuntimeError) as ctx:
                _load_benchmark_profile(_base_config(seedless=True), FakeTask())
        message = str(ctx.exception)
        # Profiles live per task and are named for the benchmark and sample size,
        # so the error shows the pattern searched, not one invented filename.
        self.assertIn("/nonexistent/profiles/dir/gec/*_gec_profile.json", message)
        self.assertIn("profile_dataset", message)

    def test_several_profiles_for_a_task_is_ambiguous_and_raises(self):
        with tempfile.TemporaryDirectory() as d:
            task_dir = os.path.join(d, "gec")
            os.makedirs(task_dir)
            for name in ("fce_150_gec_profile.json", "conll_500_gec_profile.json"):
                with open(os.path.join(task_dir, name), "w", encoding="utf-8") as f:
                    json.dump({"profile_version": 2}, f)
            with mock.patch.object(pipeline, "DEFAULT_PROFILE_DIR", d):
                with self.assertRaises(RuntimeError) as ctx:
                    _load_benchmark_profile(_base_config(seedless=True), FakeTask())
            message = str(ctx.exception)
            self.assertIn("fce_150_gec_profile.json", message)
            self.assertIn("conll_500_gec_profile.json", message)
            self.assertIn("generation.profile_path", message)

    def test_single_profile_in_the_task_dir_resolves_silently(self):
        with tempfile.TemporaryDirectory() as d:
            task_dir = os.path.join(d, "gec")
            os.makedirs(task_dir)
            path = os.path.join(task_dir, "fce_150_gec_profile.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"profile_version": 2, "topics": {"a": {"fraction": 1.0}},
                           "length_distributions": {"correct": {"words": {"bins": {"6-10": 1.0}}}},
                           "style": {"correct": {}}}, f)
            with mock.patch.object(pipeline, "DEFAULT_PROFILE_DIR", d):
                profile = _load_benchmark_profile(_base_config(seedless=True), FakeTask())
            self.assertEqual(profile["profile_version"], 2)

    def test_classification_profile_validated_against_per_label_topics(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"profile_version": 2, "topics": {"a": {"fraction": 1.0}}}, f)
            with self.assertRaises(RuntimeError) as ctx:
                _load_benchmark_profile(
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

    def generate_forward(self, **kw):
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
                 mock.patch.object(pipeline, "_load_benchmark_profile", lambda cfg, task: _FAKE_PROFILE), \
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


class GenerationCellSlugTests(unittest.TestCase):
    """Session directories are named after the setup that produced them, so a
    listing shows what was run without opening results.json."""

    def _slug(self, strategy, **generation):
        return pipeline.generation_cell_slug({"generation": generation}, strategy)

    def test_corruption_cells(self):
        self.assertEqual(self._slug("corruption", mode="inverse"), "inverse_seeded")
        self.assertEqual(self._slug("corruption", mode="forward", seedless=True),
                         "forward_seedless")

    def test_corruption_defaults_to_forward(self):
        self.assertEqual(self._slug("corruption"), "forward_seeded")

    def test_class_conditional_defaults_to_inverse(self):
        # Matches _build_meta's per-strategy default, so the directory name and
        # the recorded meta.mode can never disagree.
        self.assertEqual(self._slug("class_conditional"), "inverse_seeded")
        self.assertEqual(self._slug("class_conditional", seedless=True),
                         "inverse_seedless")

    def test_structured_has_no_mode_or_seed_axis(self):
        self.assertEqual(self._slug("structured"), "structured")
        self.assertEqual(self._slug("structured", mode="forward"), "structured")

    def test_slug_matches_the_mode_meta_records(self):
        for strategy in ("corruption", "class_conditional"):
            config = {"generation": {}}
            self.assertTrue(
                self._slug(strategy).startswith(pipeline.resolve_mode(config, strategy)),
                strategy,
            )


class ProfileNamingTests(unittest.TestCase):
    """Profiles are grouped per task and named for the benchmark they describe
    plus the number of rows profiled, so a directory listing distinguishes two
    profiles of the same task built from different benchmarks or sample sizes."""

    def _cfg(self, dataset):
        return {"dataset": dataset, "generation": {}}

    def test_local_benchmark_uses_the_file_stem(self):
        cfg = self._cfg({"source": "local",
                         "local": {"path": "framework/data/benchmarks/gec/fce.m2",
                                   "format": "m2"}})
        self.assertEqual(pipeline.benchmark_slug(cfg), "fce")
        self.assertEqual(pipeline.benchmark_profile_filename(cfg, "gec", 150),
                         "fce_150_gec_profile.json")

    def test_huggingface_benchmark_uses_the_last_name_component(self):
        cfg = self._cfg({"source": "huggingface",
                         "huggingface": {"name": "cardiffnlp/tweet_eval",
                                         "split": "test"}})
        self.assertEqual(pipeline.benchmark_slug(cfg), "tweet_eval")
        self.assertEqual(pipeline.benchmark_profile_filename(cfg, "sentiment", 3000),
                         "tweet_eval_3000_sentiment_profile.json")

    def test_slug_is_filesystem_safe(self):
        cfg = self._cfg({"source": "huggingface",
                         "huggingface": {"name": "deysi/spam-detection.v2"}})
        self.assertEqual(pipeline.benchmark_slug(cfg), "spam_detection_v2")

    def test_profiles_are_grouped_per_task(self):
        self.assertEqual(pipeline.benchmark_profile_dir("spam"),
                         os.path.join(pipeline.DEFAULT_PROFILE_DIR, "spam"))

    def test_two_sample_sizes_do_not_collide(self):
        cfg = self._cfg({"source": "local", "local": {"path": "b/fce.m2"}})
        self.assertNotEqual(pipeline.benchmark_profile_filename(cfg, "gec", 150),
                            pipeline.benchmark_profile_filename(cfg, "gec", 500))
