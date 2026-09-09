"""Seeded structured cells reach the generator, and are named distinctly."""
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework import pipeline
from framework.generators.base_generator import BaseGenerator
from framework.tasks.taxonomy.task import TaxonomyTask

_REAL = [{
    "domain": "pizza",
    "classes": ["Food", "Pizza", "Margherita", "Napoletana", "Dessert", "Gelato"],
    "subclass_axioms": [["Pizza", "Food"], ["Margherita", "Pizza"],
                        ["Napoletana", "Pizza"], ["Dessert", "Food"],
                        ["Gelato", "Dessert"]],
}]
_PROFILE = {"taxonomies": [{"domain": "pizza", "n_classes": 6, "max_depth": 2,
                            "depth_distribution": {"0": 0.2, "1": 0.4, "2": 0.4}}]}


def _config(mode, seedless, **gen):
    return {"task": {"name": "taxonomy"},
            "generation": {"mode": mode, "seedless": seedless, "sample_size": 2,
                           "seed_pool": {"max_depth": 3, "min_classes": 3},
                           **gen}}


class _Gen(BaseGenerator):
    def __init__(self):
        self.calls = 0

    def call_api(self, prompt):
        self.calls += 1
        return ('{"domain": "marine biology", "classes": ["X", "Y", "Z"], '
                '"subclass_axioms": [["Y", "X"], ["Z", "X"]]}')


def _run(cfg, profile=_PROFILE):
    gen = _Gen()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        out = pipeline._run_generation(gen, TaxonomyTask(), cfg, _REAL,
                                       None, None, None, profile=profile)
    return gen, out


class CellSlugTests(unittest.TestCase):
    def test_a_seeded_structured_run_is_named_seeded(self):
        # generation_cell_slug hardcoded "seedless" for every structured task.
        # Left alone, a seeded session would be written into the seedless
        # session's directory name and the two cells would be indistinguishable
        # in the archive and in analyze_results.
        self.assertEqual(
            pipeline.generation_cell_slug(_config("inverse", False), "structured"),
            "inverse_seeded")
        self.assertEqual(
            pipeline.generation_cell_slug(_config("forward", False), "structured"),
            "forward_seeded")

    def test_a_seedless_structured_run_is_still_named_seedless(self):
        self.assertEqual(
            pipeline.generation_cell_slug(_config("inverse", True), "structured"),
            "inverse_seedless")


class DispatchTests(unittest.TestCase):
    def test_forward_seeded_generates_without_a_profile(self):
        gen, out = _run(_config("forward", False), profile=None)
        self.assertTrue(out)
        self.assertGreater(gen.calls, 0)

    def test_inverse_seeded_generates(self):
        gen, out = _run(_config("inverse", False))
        self.assertTrue(out)

    def test_inverse_seeded_without_a_profile_fails_before_any_api_call(self):
        gen = _Gen()
        with self.assertRaises(RuntimeError) as ctx:
            with redirect_stdout(io.StringIO()):
                pipeline._run_generation(gen, TaxonomyTask(),
                                         _config("inverse", False), _REAL,
                                         None, None, None, profile=None)
        self.assertEqual(gen.calls, 0)
        self.assertIn("profile", str(ctx.exception).lower())

    def test_feedback_on_a_seeded_cell_fails_before_any_api_call(self):
        gen = _Gen()
        cfg = _config("inverse", False, feedback={"enabled": True})
        with self.assertRaises(RuntimeError) as ctx:
            with redirect_stdout(io.StringIO()):
                pipeline._run_generation(gen, TaxonomyTask(), cfg, _REAL,
                                         None, None, None, profile=_PROFILE)
        self.assertEqual(gen.calls, 0)
        self.assertIn("feedback", str(ctx.exception).lower())

    def test_an_empty_seed_pool_fails_before_any_api_call(self):
        gen = _Gen()
        cfg = _config("forward", False)
        cfg["generation"]["seed_pool"] = {"min_classes": 999}
        with self.assertRaises(RuntimeError) as ctx:
            with redirect_stdout(io.StringIO()):
                pipeline._run_generation(gen, TaxonomyTask(), cfg, _REAL,
                                         None, None, None, profile=_PROFILE)
        self.assertEqual(gen.calls, 0)
        self.assertIn("get_seed_pool", str(ctx.exception))


class CliGuardTests(unittest.TestCase):
    def test_validate_config_no_longer_rejects_seeded_structured(self):
        from framework.main import validate_config
        cfg = {"task": {"name": "taxonomy"},
               "dataset": {"source": "local",
                           "local": {"path": "x.jsonl", "format": "jsonl"}},
               "generation": {"provider": "p", "model": "m", "num_runs": 1,
                              "sample_size": 2, "mode": "forward", "seedless": False},
               "task_models": [{"name": "m", "type": "llm"}]}
        self.assertIsNone(validate_config(cfg))


if __name__ == "__main__":
    unittest.main()
