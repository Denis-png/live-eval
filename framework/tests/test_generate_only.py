"""framework.main --generate-only, then scripts.rescore_session: generation and
evaluation as two separate commands on one session format.

A generate-only run must archive everything the rescore needs (generated runs,
real sample, meta) and load no task model. Rescoring it must yield the same
results shape a normal run writes, so the analysis cannot tell them apart.
"""
import argparse
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework import pipeline
from framework.main import apply_overrides, generate_only, validate_config
from framework.tasks.taxonomy.task import TaxonomyTask
from framework.tests.test_taxonomy_real_reference import _E2E_REAL, _POOL_OPTS
from framework.tests.test_taxonomy_seeded_dispatch import _Gen
from scripts import analyze_results as ar
from scripts.rescore_session import rescore_session

TASK_MODELS = [{"name": "lexical", "type": "lexical"}, {"name": "star", "type": "star"}]


def _config(base, name, *, only):
    data = os.path.join(base, "onto.jsonl")
    with open(data, "w", encoding="utf-8") as f:
        f.write(json.dumps(_E2E_REAL) + "\n")
    return {
        "task": {"name": "taxonomy"},
        "dataset": {"source": "local", "local": {"path": data, "format": "jsonl"}},
        "generation": {"provider": "stub", "model": "stub", "mode": "forward",
                       "seedless": False, "num_runs": 2, "sample_size": 3,
                       "max_parse_attempts": 1,
                       "seed_pool": {**_POOL_OPTS, "domains": ["marine biology"]}},
        "evaluation": {"generate_only": only},
        "task_models": TASK_MODELS,
        "output": {"base_dir": base, "plots": False, "session_id": name},
    }


def _run(cfg):
    with mock.patch.object(pipeline, "load_generator", return_value=_Gen()), \
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        pipeline.run_pipeline(cfg)
    session = os.path.join(cfg["output"]["base_dir"], "taxonomy", cfg["output"]["session_id"])
    with open(os.path.join(session, "results.json"), encoding="utf-8") as f:
        return session, json.load(f)


class GenerateOnlyRunTests(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

    def test_it_archives_the_session_and_loads_no_task_model(self):
        cfg = _config(self.base, "g", only=True)
        with mock.patch.object(TaxonomyTask, "get_model",
                               side_effect=AssertionError("a task model was loaded")):
            session, written = _run(cfg)
        self.assertEqual(written["results"], {})
        meta = written["meta"]
        self.assertIs(meta["generate_only"], True)
        self.assertIs(meta["real_baseline"], False)
        self.assertEqual(meta["runs_completed"], 2)
        self.assertEqual(meta["effective_samples_per_run"], [3, 3])
        for name in ("generated/run_1.json", "generated/run_2.json", "real_sample.json"):
            self.assertTrue(os.path.exists(os.path.join(session, name)), name)

    def test_rescoring_it_writes_what_a_normal_run_writes(self):
        session, _ = _run(_config(self.base, "g", only=True))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rescore_session(session, _config(self.base, "g", only=False))
        with open(os.path.join(session, "results.json"), encoding="utf-8") as f:
            rescored = json.load(f)
        _, normal = _run(_config(self.base, "n", only=False))

        self.assertEqual(set(rescored["results"]), {"lexical", "star"})
        for model, blocks in normal["results"].items():
            self.assertEqual(set(rescored["results"][model]), set(blocks), model)
            self.assertEqual(len(rescored["results"][model]["runs"]), 2)
        self.assertNotIn("generate_only", rescored["meta"])
        self.assertIs(rescored["meta"]["real_baseline"], True)
        self.assertIs(rescored["meta"]["paired_real"], normal["meta"]["paired_real"])

    def test_the_analysis_skips_it_until_it_is_scored(self):
        scored, _ = _run(_config(self.base, "a", only=False))
        _run(_config(self.base, "b", only=True))
        with redirect_stderr(io.StringIO()) as err:
            found = ar.discover_sessions([self.base])
        self.assertEqual([s["dir"] for s in found], [scored])
        self.assertIn("unscored session", err.getvalue())


class GenerateOnlyConfigTests(unittest.TestCase):
    def _config(self, **evaluation):
        return {"task": {"name": "spam"}, "dataset": {"source": "local",
                "local": {"path": "x.csv"}},
                "generation": {"provider": "p", "model": "m", "num_runs": 1,
                               "sample_size": 1},
                "evaluation": evaluation}

    def test_the_flag_sets_the_config_key(self):
        args = argparse.Namespace(task=None, provider=None, model=None, runs=None,
                                  sample_size=None, mode=None, output=None, judge=None,
                                  real_baseline=None, plots=None, seedless=None,
                                  generate_only=True)
        self.assertTrue(generate_only(apply_overrides(self._config(), args)))

    def test_task_models_are_required_only_when_scoring(self):
        with self.assertRaisesRegex(ValueError, "task_models"):
            validate_config(self._config())
        validate_config(self._config(generate_only=True))


if __name__ == "__main__":
    unittest.main()
