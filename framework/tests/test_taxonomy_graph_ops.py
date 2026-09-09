"""Pure structural operations on a taxonomy graph.

Ground truth for the seeded cells is COMPUTED by these functions, never parsed
from model output, so their correctness is the correctness of the benchmark.
"""
import unittest
from random import Random

from framework.tasks.taxonomy.graph_ops import sample_subtrees

# A -> B -> D, B -> E, A -> C -> F ; depth 2, 6 classes
_CLASSES = ["A", "B", "C", "D", "E", "F"]
_AXIOMS = [["B", "A"], ["C", "A"], ["D", "B"], ["E", "B"], ["F", "C"]]

# F has TWO parents (C and B): ontologies preserve multiple inheritance, so the
# tree fixture above cannot exercise the paths that actually matter.
_DAG_CLASSES = ["A", "B", "C", "D", "E", "F"]
_DAG_AXIOMS = [["B", "A"], ["C", "A"], ["D", "B"], ["E", "B"],
               ["F", "C"], ["F", "B"]]


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

    def test_the_pool_is_exhaustive_and_ordered(self):
        # sample_subtrees enumerates every qualifying root in sorted order; it
        # does not sample. Replaces a determinism test that could not fail,
        # because the function has no random branch for a seed to affect.
        out = sample_subtrees(_CLASSES, _AXIOMS, min_classes=1, rng=Random(7))
        self.assertEqual([s["root"] for s in out], sorted(s["root"] for s in out))
        self.assertEqual(len(out), len(_CLASSES))

    def test_a_graph_with_no_qualifying_subtree_returns_empty(self):
        # Callers fail fast on this; returning [] rather than raising keeps the
        # pure layer free of framework error vocabulary.
        self.assertEqual(sample_subtrees(_CLASSES, _AXIOMS, min_classes=99), [])


class MultiParentTests(unittest.TestCase):
    def test_no_class_appears_twice(self):
        subs = sample_subtrees(_DAG_CLASSES, _DAG_AXIOMS, min_classes=1,
                               rng=Random(0))
        self.assertTrue(subs)
        for sub in subs:
            self.assertEqual(len(sub["classes"]), len(set(sub["classes"])),
                             f"duplicate class in {sub['root']}: {sub['classes']}")

    def test_recorded_depth_matches_an_independently_computed_longest_path(self):
        # Brute force, deliberately NOT _depth_of: checking a memoised DP against
        # itself can only catch a wiring slip, never a wrong algorithm.
        def longest(classes, axioms):
            parents = {}
            for child, parent in axioms:
                parents.setdefault(child, []).append(parent)

            def walk(node, seen):
                if node in seen:
                    return 0
                return max((1 + walk(p, seen | {node})
                            for p in parents.get(node, [])), default=0)

            return max((walk(c, frozenset()) for c in classes), default=0)

        subs = sample_subtrees(_DAG_CLASSES, _DAG_AXIOMS, min_classes=1, rng=Random(0))
        self.assertTrue(subs)
        for sub in subs:
            self.assertEqual(sub["max_depth"],
                             longest(sub["classes"], sub["subclass_axioms"]))

    def test_every_subtree_stays_closed_on_a_dag(self):
        subs = sample_subtrees(_DAG_CLASSES, _DAG_AXIOMS, min_classes=1,
                               rng=Random(0))
        self.assertTrue(subs)
        for sub in subs:
            names = set(sub["classes"])
            for child, parent in sub["subclass_axioms"]:
                self.assertIn(child, names)
                self.assertIn(parent, names)

    def test_induced_depth_may_exceed_the_discovery_radius_on_a_dag(self):
        # Documented, deliberate: max_depth bounds DISCOVERY, and the induced
        # subgraph keeps every axiom between kept classes. X is discovered at
        # depth 1 via R, but P2->X survives because P2 is kept independently.
        classes = ["R", "P1", "P2", "X"]
        axioms = [["P1", "R"], ["P2", "P1"], ["X", "P2"], ["X", "R"]]
        subs = sample_subtrees(classes, axioms, max_depth=2, min_classes=1,
                               rng=Random(0))
        root_r = [s for s in subs if s["root"] == "R"]
        self.assertTrue(root_r, "expected an R-rooted subtree")
        self.assertEqual(root_r[0]["max_depth"], 3)


if __name__ == "__main__":
    unittest.main()
