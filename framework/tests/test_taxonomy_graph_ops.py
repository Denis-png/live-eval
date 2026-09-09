"""Pure structural operations on a taxonomy graph.

Ground truth for the seeded cells is COMPUTED by these functions, never parsed
from model output, so their correctness is the correctness of the benchmark.
"""
import unittest
from random import Random

from framework.tasks.taxonomy.graph_ops import sample_subtrees

# A -> B -> D, B -> E, A -> C -> F ; depth 3, 6 classes
_CLASSES = ["A", "B", "C", "D", "E", "F"]
_AXIOMS = [["B", "A"], ["C", "A"], ["D", "B"], ["E", "B"], ["F", "C"]]


class SampleSubtreesTests(unittest.TestCase):
    def test_every_subtree_is_closed_under_its_own_axioms(self):
        # A subtree whose axioms mention a class it does not contain is not a
        # graph -- it would crash the profiler and silently corrupt gold.
        for sub in sample_subtrees(_CLASSES, _AXIOMS, min_classes=1, rng=Random(0)):
            names = set(sub["classes"])
            for child, parent in sub["subclass_axioms"]:
                self.assertIn(child, names)
                self.assertIn(parent, names)

    def test_min_classes_filters_small_subtrees(self):
        # Rooted at D/E/F there are only single classes; min_classes=3 drops them.
        out = sample_subtrees(_CLASSES, _AXIOMS, min_classes=3, rng=Random(0))
        self.assertTrue(out)
        for sub in out:
            self.assertGreaterEqual(len(sub["classes"]), 3)

    def test_max_depth_truncates_deep_subtrees(self):
        out = sample_subtrees(_CLASSES, _AXIOMS, max_depth=1, min_classes=1,
                              rng=Random(0))
        for sub in out:
            self.assertLessEqual(sub["max_depth"], 1)

    def test_the_recorded_max_depth_matches_the_returned_axioms(self):
        # max_depth is the bucket key the seed index and every record's
        # provenance field use. If it disagreed with the graph it labels, seed
        # reweighting would steer the wrong thing and nothing would report it.
        for sub in sample_subtrees(_CLASSES, _AXIOMS, min_classes=1, rng=Random(0)):
            parents = {c: p for c, p in sub["subclass_axioms"]}
            depths = []
            for cls in sub["classes"]:
                d, cur = 0, cls
                while cur in parents:
                    cur, d = parents[cur], d + 1
                depths.append(d)
            self.assertEqual(sub["max_depth"], max(depths))

    def test_it_is_deterministic_under_a_seeded_rng(self):
        a = sample_subtrees(_CLASSES, _AXIOMS, min_classes=1, rng=Random(7))
        b = sample_subtrees(_CLASSES, _AXIOMS, min_classes=1, rng=Random(7))
        self.assertEqual(a, b)

    def test_a_graph_with_no_qualifying_subtree_returns_empty(self):
        # Callers fail fast on this; returning [] rather than raising keeps the
        # pure layer free of framework error vocabulary.
        self.assertEqual(sample_subtrees(_CLASSES, _AXIOMS, min_classes=99), [])


if __name__ == "__main__":
    unittest.main()
