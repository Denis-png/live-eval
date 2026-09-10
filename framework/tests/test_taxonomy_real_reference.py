"""The real side is drawn the way each cell's synthetic side is drawn.

The real reference used to be the whole ontology, as one item, for every cell.
Seeded cells evaluate on subtrees of it, and F1 on a small graph is far easier
than on a large one -- so a seeded benchmark looked "easier" than the real one
purely because of size, and the real side was n=1. The same reference also
feeds the real side of the structural fidelity profile, so the size confound
hit both.

Seeded sessions now compare against the real subtrees of the SAME seed pool;
seedless sessions against the whole ontology, which is what they target.
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
from framework.tasks.taxonomy.task import TaxonomyTask
from framework.tests.test_taxonomy_seeded_dispatch import _Gen

# A 7-class ontology deep enough to yield several subtrees.
_REAL = [{
    "ontology_id": "t",
    "domain": "pizza",
    "classes": ["Food", "Pizza", "Margherita", "Napoletana", "Dessert",
                "Gelato", "Sorbet"],
    "subclass_axioms": [["Pizza", "Food"], ["Margherita", "Pizza"],
                        ["Napoletana", "Pizza"], ["Dessert", "Food"],
                        ["Gelato", "Dessert"], ["Sorbet", "Dessert"]],
}]
_POOL_OPTS = {"max_depth": 3, "min_classes": 3}


def _cfg(seedless, mode="forward"):
    gen = {"mode": mode, "seed_pool": dict(_POOL_OPTS)}
    if seedless is not None:
        gen["seedless"] = seedless
    return {"task": {"name": "taxonomy"}, "generation": gen}


class MatchedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.task = TaxonomyTask()

    def _pool(self, cfg):
        return self.task.get_seed_pool(cfg, _REAL, "forward")

    def test_a_seeded_session_is_compared_against_the_pool(self):
        cfg = _cfg(seedless=False)
        ref = self.task.get_real_eval_samples(cfg, _REAL)
        pool = self._pool(cfg)
        self.assertEqual(len(ref), len(pool))
        self.assertGreater(len(ref), 1, "the whole point: n > 1 on the real side")
        self.assertEqual([r["classes"] for r in ref], [p["classes"] for p in pool])
        self.assertEqual([sorted(r["subclass_axioms"]) for r in ref],
                         [sorted(p["subclass_axioms"]) for p in pool])

    def test_the_real_subtrees_keep_their_real_names_and_domain(self):
        ref = self.task.get_real_eval_samples(_cfg(seedless=False), _REAL)
        for sample in ref:
            self.assertEqual(sample["domain"], "pizza")
            self.assertTrue(set(sample["classes"]) <= set(_REAL[0]["classes"]))

    def test_an_inverse_seeded_session_still_sees_unedited_subtrees(self):
        # The real side is real data. inverse+seeded's synthetic items are
        # edited; its reference must not be.
        fwd = self.task.get_real_eval_samples(_cfg(seedless=False, mode="forward"), _REAL)
        inv = self.task.get_real_eval_samples(_cfg(seedless=False, mode="inverse"), _REAL)
        self.assertEqual([r["subclass_axioms"] for r in fwd],
                         [r["subclass_axioms"] for r in inv])

    def test_a_seedless_session_is_compared_against_the_whole_ontology(self):
        ref = self.task.get_real_eval_samples(_cfg(seedless=True), _REAL)
        self.assertEqual(len(ref), 1)
        self.assertEqual(ref[0]["classes"], _REAL[0]["classes"])

    def test_an_omitted_seedless_key_follows_the_structured_default(self):
        # Structured defaults to seedless, via the one shared resolver.
        ref = self.task.get_real_eval_samples(_cfg(seedless=None), _REAL)
        self.assertEqual(len(ref), 1)

    def test_no_gold_reaches_the_model_input(self):
        # Both the model_input dict and the serialised `text`: the text is what
        # model.predict is actually handed.
        for sample in self.task.get_real_eval_samples(_cfg(seedless=False), _REAL):
            self.assertEqual(set(sample["model_input"]), {"domain", "classes"})
            self.assertEqual(set(json.loads(sample["text"])), {"domain", "classes"})
            self.assertNotIn("subclass", sample["text"])
            self.assertTrue(sample["subclass_axioms"])
            for child, parent in sample["subclass_axioms"]:
                self.assertNotIn(json.dumps([child, parent]), sample["text"])

    def test_each_real_subtree_carries_its_pool_index(self):
        # So real_sample.json records which pool subtree each real item is, and
        # a generated record's source_pool_index can be matched against it.
        cfg = _cfg(seedless=False)
        ref = self.task.get_real_eval_samples(cfg, _REAL)
        self.assertEqual([r["pool_index"] for r in ref],
                         [p["pool_index"] for p in self._pool(cfg)])
        self.assertEqual([r["pool_index"] for r in ref], list(range(len(ref))))

    def test_the_pool_is_numbered_once_across_ontologies(self):
        # Built per row, the numbering restarted at 0 for each ontology, so one
        # index named two different subtrees.
        second = {"ontology_id": "w", "domain": "wine",
                  "classes": ["Wine", "Red", "White", "Merlot"],
                  "subclass_axioms": [["Red", "Wine"], ["White", "Wine"],
                                      ["Merlot", "Red"]]}
        rows = _REAL + [second]
        cfg = _cfg(seedless=False)
        ref = self.task.get_real_eval_samples(cfg, rows)
        pool = self.task.get_seed_pool(cfg, rows, "forward")
        self.assertEqual({r["ontology_id"] for r in ref}, {"t", "w"})
        self.assertEqual([r["pool_index"] for r in ref], list(range(len(pool))))
        for item, subtree in zip(ref, pool):
            self.assertEqual(item["classes"], subtree["classes"])
            self.assertEqual(item["domain"], "pizza" if item["ontology_id"] == "t" else "wine")


class ContextWiringTests(unittest.TestCase):
    """The reference reaches the pipeline, which passes it to BOTH consumers:
    the real scores and the structural fidelity profile."""

    def _context(self, seedless):
        cfg = _cfg(seedless=seedless)
        cfg["generation"].update(provider="stub", model="stub", num_runs=1, sample_size=2)
        cfg["dataset"] = {"source": "local", "local": {"path": "x.jsonl", "format": "jsonl"}}
        with mock.patch.object(pipeline, "load_generator", return_value=object()), \
             mock.patch.object(pipeline, "load_real_data", return_value=_REAL), \
             mock.patch.object(pipeline, "_load_benchmark_profile",
                               return_value={"taxonomies": [{"domain": "pizza"}]}):
            return pipeline.build_generation_context(cfg)

    def test_a_seeded_context_carries_the_pool_as_its_real_reference(self):
        ctx = self._context(seedless=False)
        pool = TaxonomyTask().get_seed_pool(_cfg(seedless=False), _REAL, "forward")
        self.assertEqual(len(ctx["real_reference"]), len(pool))

    def test_a_seedless_context_carries_the_whole_ontology(self):
        ctx = self._context(seedless=True)
        self.assertEqual(len(ctx["real_reference"]), 1)


# A DAG (DessertPizza has two parents) whose pool under _POOL_OPTS is three
# subtrees of DIFFERENT shapes -- Dessert (6 classes), Food (10), Pizza (5) --
# so a synthetic item paired with the wrong real item cannot match it.
_E2E_REAL = {
    "ontology_id": "onto",
    "domain": "cuisine",
    "classes": ["Food", "Pizza", "Dessert", "Margherita", "Napoletana", "Gelato",
                "Sorbet", "Tiramisu", "DessertPizza", "Nutella"],
    "subclass_axioms": [["Pizza", "Food"], ["Dessert", "Food"],
                        ["Margherita", "Pizza"], ["Napoletana", "Pizza"],
                        ["Gelato", "Dessert"], ["Sorbet", "Dessert"],
                        ["Tiramisu", "Dessert"], ["DessertPizza", "Pizza"],
                        ["DessertPizza", "Dessert"], ["Nutella", "DessertPizza"]],
}


class SeededSessionEndToEndTests(unittest.TestCase):
    """A seeded session driven through run_pipeline to the end, with no network.

    ContextWiringTests only checks the reference's LENGTH. That let a seeded
    session reach the structural fidelity step with several real taxonomies,
    where select_reference_taxonomy_profile raised -- after every generation and
    evaluation call, leaving no profile.json and no plots.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp)   # runs even if the run raises
        data = os.path.join(cls.tmp, "onto.jsonl")
        with open(data, "w", encoding="utf-8") as f:
            f.write(json.dumps(_E2E_REAL) + "\n")
        cls.pool_size = len(TaxonomyTask().get_seed_pool(
            _cfg(seedless=False), [_E2E_REAL], "forward"))
        cfg = {
            "task": {"name": "taxonomy"},
            "dataset": {"source": "local", "local": {"path": data, "format": "jsonl"}},
            "generation": {"provider": "stub", "model": "stub", "mode": "forward",
                           "seedless": False, "num_runs": 1,
                           # The whole pool is drawn, so every real item has a partner.
                           "sample_size": cls.pool_size,
                           "seed_pool": {**_POOL_OPTS, "domains": ["marine biology"]}},
            "task_models": [{"name": "lexical", "type": "lexical"},
                            {"name": "star", "type": "star"}],
            "output": {"base_dir": cls.tmp, "plots": False, "session_id": "s"},
        }
        with mock.patch.object(pipeline, "load_generator", return_value=_Gen()), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            pipeline.run_pipeline(cfg)
        cls.session = os.path.join(cls.tmp, "taxonomy", "s")

    def _load(self, *parts):
        with open(os.path.join(self.session, *parts), encoding="utf-8") as f:
            return json.load(f)

    def test_the_fixture_pool_has_several_subtrees(self):
        self.assertEqual(self.pool_size, 3)

    def test_the_profile_is_written_with_the_pool_as_its_real_side(self):
        profile = self._load("profile.json")
        real = profile["fidelity"]["real_profile"]
        self.assertEqual(real["pooled_taxonomies"], self.pool_size)
        self.assertEqual(len(profile["real"]["taxonomies"]), self.pool_size)
        # Pooling must not undo sanitising: no real class name in the fidelity.
        text = json.dumps(profile["fidelity"])
        for name in _E2E_REAL["classes"]:
            self.assertNotIn(f'"{name}"', text)

    def test_results_are_written(self):
        results = self._load("results.json")["results"]
        self.assertEqual(set(results), {"lexical", "star"})
        for scores in results.values():
            self.assertIn("real", scores)
            self.assertIn("generated", scores)

    def test_real_sample_records_each_items_pool_index(self):
        real = self._load("real_sample.json")
        self.assertEqual(sorted(item["pool_index"] for item in real),
                         list(range(self.pool_size)))

    def test_each_generated_record_pairs_with_the_real_item_of_its_index(self):
        # Pairing is recoverable from the archive: a forward+seeded record is
        # its pool subtree's structure under new names, and source_pool_index
        # says which subtree. The fixture's subtrees differ in shape, so a
        # record paired with the wrong real item cannot match it.
        from framework.tasks.taxonomy.graph_ops import matches_structure

        real = {item["pool_index"]: item for item in self._load("real_sample.json")}
        generated = self._load("generated", "run_1.json")
        self.assertEqual(sorted(r["source_pool_index"] for r in generated),
                         sorted(real))
        for record in generated:
            self.assertIs(type(record["source_pool_index"]), int)
            partner = real[record["source_pool_index"]]
            self.assertTrue(matches_structure(
                partner["classes"], partner["subclass_axioms"],
                record["classes"], record["subclass_axioms"]),
                record["source_pool_index"])
            # An integer is all that crosses over: no real name in the record.
            self.assertFalse(set(record["classes"]) & set(_E2E_REAL["classes"]))


if __name__ == "__main__":
    unittest.main()
