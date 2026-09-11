"""Per-run paired real scoring.

A seeded taxonomy session's real side was the WHOLE seed pool, scored once, while
each run's synthetic side is what that run drew and what passed verification --
which drops the largest subtrees most often. Micro scores are dominated by those
items, so the generated-vs-real gap mixed generation fidelity with attrition.
Each run is now also scored against the real items it actually delivered.
"""
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework import pipeline
from framework.tasks.base_task import BaseTask
from framework.tasks.taxonomy.task import TaxonomyTask
from framework.tests.test_taxonomy_real_reference import _E2E_REAL, _POOL_OPTS
from framework.tests.test_taxonomy_seeded_dispatch import _Gen

_REAL = [{"pool_index": 0, "classes": ["A"]}, {"pool_index": 1, "classes": ["B"]},
         {"pool_index": 2, "classes": ["C"]}]


class PairingHookTests(unittest.TestCase):
    def test_the_base_task_does_not_pair(self):
        self.assertIsNone(BaseTask.paired_real_indices(object(), _REAL, [{}]))

    def test_each_record_maps_to_the_real_item_with_its_index(self):
        synthetic = [{"source_pool_index": 2}, {"source_pool_index": 0}]
        self.assertEqual(TaxonomyTask().paired_real_indices(_REAL, synthetic), [2, 0])

    def test_a_repeated_draw_is_paired_once_per_repeat(self):
        synthetic = [{"source_pool_index": 1}, {"source_pool_index": 1}]
        self.assertEqual(TaxonomyTask().paired_real_indices(_REAL, synthetic), [1, 1])

    def test_positions_not_indices_are_returned(self):
        # real_sample.json need not be in pool order.
        real = [_REAL[2], _REAL[0], _REAL[1]]
        synthetic = [{"source_pool_index": 2}]
        self.assertEqual(TaxonomyTask().paired_real_indices(real, synthetic), [0])

    def test_seedless_records_are_not_paired(self):
        # A seedless artifact comes from no particular real item.
        self.assertIsNone(TaxonomyTask().paired_real_indices(_REAL, [{"classes": ["X"]}]))

    def test_an_index_matching_no_real_item_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "7"):
            TaxonomyTask().paired_real_indices(_REAL, [{"source_pool_index": 7}])


class _RejectLargest(_Gen):
    """Compliant, except on the largest subtree, whose answer drops a relation:
    verification rejects it on every attempt, as it rejects Pizza's 91-class
    subtree in real runs."""

    def call_api(self, prompt):
        answer = json.loads(super().call_api(prompt))
        if len(answer["classes"]) >= 10:
            answer["subclass_axioms"] = answer["subclass_axioms"][1:]
        return json.dumps(answer)


def _session(base, name, generator):
    data = os.path.join(base, "onto.jsonl")
    with open(data, "w", encoding="utf-8") as f:
        f.write(json.dumps(_E2E_REAL) + "\n")
    cfg = {
        "task": {"name": "taxonomy"},
        "dataset": {"source": "local", "local": {"path": data, "format": "jsonl"}},
        "generation": {"provider": "stub", "model": "stub", "mode": "forward",
                       "seedless": False, "num_runs": 2, "sample_size": 3,
                       "max_parse_attempts": 1,
                       "seed_pool": {**_POOL_OPTS, "domains": ["marine biology"]}},
        "task_models": [{"name": "lexical", "type": "lexical"},
                        {"name": "star", "type": "star"}],
        "output": {"base_dir": base, "plots": False, "session_id": name},
    }
    with mock.patch.object(pipeline, "load_generator", return_value=generator), \
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        pipeline.run_pipeline(cfg)
    with open(os.path.join(base, "taxonomy", name, "results.json"), encoding="utf-8") as f:
        return cfg, json.load(f)


def _gold_edges(scores):
    """Gold relations the scored items hold: every one is a tp or an fn."""
    return scores["diagnostics"]["tp"] + scores["diagnostics"]["fn"]


class PairedSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp)
        cls.cfg, cls.all_kept = _session(cls.tmp, "kept", _Gen())
        _, cls.one_dropped = _session(cls.tmp, "dropped", _RejectLargest())

    def test_a_paired_session_says_so(self):
        self.assertIs(self.one_dropped["meta"]["paired_real"], True)

    def test_every_run_has_a_paired_score_aligned_with_it(self):
        for scores in self.one_dropped["results"].values():
            self.assertEqual(len(scores["real_paired_runs"]), len(scores["runs"]))
            self.assertIn("f1", scores["real_paired"])
            self.assertEqual(set(scores["real_paired"]["f1"]), {"mean", "std"})

    def test_the_unpaired_real_is_unchanged(self):
        for model, scores in self.one_dropped["results"].items():
            self.assertEqual(scores["real"], self.all_kept["results"][model]["real"])

    def test_a_rejected_item_leaves_the_paired_real_side(self):
        # _RejectLargest drops the 10-class subtree in every run, so each run's
        # paired real holds fewer gold relations than the whole reference.
        for scores in self.one_dropped["results"].values():
            whole = _gold_edges(scores["real"])
            for paired in scores["real_paired_runs"]:
                self.assertLess(_gold_edges(paired), whole)

    def test_with_nothing_rejected_paired_equals_real(self):
        # The same items in a shuffled order: counts match exactly, and the
        # scores match up to float summation order.
        for scores in self.all_kept["results"].values():
            for paired in scores["real_paired_runs"]:
                self.assertEqual(_gold_edges(paired), _gold_edges(scores["real"]))
                for metric in ("precision", "recall", "f1"):
                    self.assertAlmostEqual(paired[metric], scores["real"][metric],
                                           places=12)


class UnpairedSessionTests(unittest.TestCase):
    def test_nest_results_without_paired_runs_is_unchanged(self):
        final = pipeline._nest_results({"m": {"f1": {"mean": 1.0, "std": 0.0}}},
                                       {"m": {"f1": 0.9}}, [{"m": {"f1": 1.0}}])
        self.assertEqual(set(final["m"]), {"generated", "real", "runs"})


class PrinterTests(unittest.TestCase):
    def test_the_paired_real_is_printed_beside_the_real(self):
        from framework.main import format_results_lines
        text = "\n".join(format_results_lines({"m": {
            "generated": {"f1": {"mean": 0.9, "std": 0.01}},
            "real": {"f1": 0.8},
            "real_paired": {"f1": {"mean": 0.85, "std": 0.02}}}}))
        self.assertIn("paired real.f1: 0.85 ± 0.02", text)


if __name__ == "__main__":
    unittest.main()
