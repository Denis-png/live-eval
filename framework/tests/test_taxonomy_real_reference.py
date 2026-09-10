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
import unittest
from unittest import mock

from framework import pipeline
from framework.tasks.taxonomy.task import TaxonomyTask

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
        for sample in self.task.get_real_eval_samples(_cfg(seedless=False), _REAL):
            self.assertEqual(set(sample["model_input"]), {"domain", "classes"})


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


if __name__ == "__main__":
    unittest.main()
