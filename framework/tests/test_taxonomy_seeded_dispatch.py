"""Seeded structured cells reach the generator, and are named distinctly."""
import io
import json
import os
import re
import shutil
import tempfile
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
    """A compliant model: one new name per anonymised class, in the order given,
    structure untouched.

    A fake returning a FIXED response cannot answer a prompt whose class count
    varies with the seed drawn, so it would only ever prove the code path is
    reachable. Answering properly makes these tests exercise the whole wiring --
    seed pool, gold construction, prompt rendering, parse and verification.
    """

    def __init__(self):
        self.calls = 0

    def call_api(self, prompt):
        self.calls += 1
        block = re.search(r'\{\s*"classes":\s*\[.*?\]\s*\}', prompt, re.S).group(0)
        struct = json.loads(block)
        names = {c: f"Species{i}" for i, c in enumerate(struct["classes"])}
        return json.dumps({
            "domain": "marine biology",
            "classes": [names[c] for c in struct["classes"]],
            "subclass_axioms": [[names[c], names[p]]
                                for c, p in struct["subclass_axioms"]],
        })


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
        # The artifacts carry the MODEL's names, not the source ontology's --
        # re-verbalisation is what keeps the seed ontology out of the benchmark.
        for art in out:
            self.assertTrue(all(c.startswith("Species") for c in art["classes"]))
            for child, parent in art["subclass_axioms"]:
                self.assertIn(child, art["classes"])
                self.assertIn(parent, art["classes"])

    def test_inverse_seeded_generates(self):
        gen, out = _run(_config("inverse", False))
        self.assertTrue(out)
        for art in out:
            self.assertIn("source_max_depth", art)

    def test_a_model_that_changes_the_structure_produces_nothing(self):
        # The gate, end to end: gold is computed, so a drifting model loses its
        # sample instead of redefining the reference.
        class _Drifting(_Gen):
            def call_api(self, prompt):
                payload = json.loads(super().call_api(prompt))
                payload["subclass_axioms"] = payload["subclass_axioms"][:-1]
                return json.dumps(payload)

        gen = _Drifting()
        # Not an empty list handed back to run_pipeline: the seeded branch used
        # to return before the shared "0 usable samples" guard, so a model that
        # failed EVERY verification produced a results.json full of 0.0 scores
        # for a run that generated nothing. The spec's error table puts this
        # cell under the same guard as every other.
        with self.assertRaises(RuntimeError) as ctx:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                pipeline._run_generation(gen, TaxonomyTask(),
                                         _config("forward", False), _REAL,
                                         None, None, None, profile=None)
        self.assertIn("0 usable samples", str(ctx.exception))
        self.assertGreater(gen.calls, 0)

    def test_a_partly_drifting_model_still_yields_its_verified_samples(self):
        # The guard fires on an EMPTY result, not on any skip: one bad sample
        # out of two must not take the good one down with it.
        class _HalfDrifting(_Gen):
            def call_api(self, prompt):
                text = super().call_api(prompt)
                if self.calls > 1:
                    return text
                payload = json.loads(text)
                payload["subclass_axioms"] = payload["subclass_axioms"][:-1]
                return json.dumps(payload)

        gen = _HalfDrifting()
        cfg = _config("forward", False)
        cfg["generation"]["max_parse_attempts"] = 1
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = pipeline._run_generation(gen, TaxonomyTask(), cfg, _REAL,
                                           None, None, None, profile=None)
        self.assertEqual(len(out), 1)

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


class GenerationContextProfileTests(unittest.TestCase):
    """The layer that SUPPLIES `profile` to the dispatch, which no seeded test
    exercised: every other test in this file passes `profile=` by hand.

    `_load_benchmark_profile` short-circuited to None for every structured cell
    with `seedless: false`, before `generation.profile_path` was consulted. That
    made `inverse+seeded` — what the shipped config produces by flipping
    `seedless` — unreachable: it always died asking for a profile no config
    could give it."""

    def _context(self, cfg):
        with mock.patch.object(pipeline, "load_generator", return_value=_Gen()), \
             mock.patch.object(pipeline, "load_real_data", return_value=_REAL):
            return pipeline.build_generation_context(cfg)

    def _cfg_with_profile_on_disk(self, mode):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        path = os.path.join(d, "onto_1_taxonomy_profile.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_PROFILE, f)
        cfg = _config(mode, False, profile_path=path)
        cfg["generation"].update(provider="stub", model="stub", num_runs=1)
        cfg["dataset"] = {"source": "local",
                          "local": {"path": "x.jsonl", "format": "jsonl"}}
        return cfg, path

    def test_inverse_seeded_gets_the_configured_profile(self):
        cfg, path = self._cfg_with_profile_on_disk("inverse")
        ctx = self._context(cfg)
        self.assertEqual(ctx["profile"], _PROFILE)
        self.assertFalse(ctx["seedless"])
        self.assertEqual(ctx["mode"], "inverse")
        self.assertTrue(os.path.exists(path))

    def test_inverse_seeded_generates_from_the_context_it_is_given(self):
        # End to end through the supplying layer: the cell the branch made
        # unreachable now produces artifacts.
        cfg, _ = self._cfg_with_profile_on_disk("inverse")
        ctx = self._context(cfg)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = pipeline._run_generation(ctx["generator"], ctx["task"], cfg,
                                           ctx["real_data"], ctx["error_dist"],
                                           ctx["judge_call"], ctx["class_prob"],
                                           profile=ctx["profile"])
        self.assertTrue(out)
        for art in out:
            self.assertIn("source_max_depth", art)

    def test_forward_seeded_still_consults_no_profile(self):
        # Forward inherits the seed's structure, so it must run with no profile
        # on disk at all -- the spec's "Profile missing, forward+seeded" row.
        cfg = _config("forward", False)
        cfg["generation"].update(provider="stub", model="stub", num_runs=1)
        cfg["dataset"] = {"source": "local",
                          "local": {"path": "x.jsonl", "format": "jsonl"}}
        ctx = self._context(cfg)
        self.assertIsNone(ctx["profile"])
        self.assertFalse(ctx["seedless"])

    def test_seedless_structured_still_loads_its_profile(self):
        cfg, _ = self._cfg_with_profile_on_disk("inverse")
        cfg["generation"]["seedless"] = True
        ctx = self._context(cfg)
        self.assertEqual(ctx["profile"], _PROFILE)
        self.assertTrue(ctx["seedless"])


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
