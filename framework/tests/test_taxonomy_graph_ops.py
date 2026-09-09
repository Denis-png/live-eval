"""Pure structural operations on a taxonomy graph.

Ground truth for the seeded cells is COMPUTED by these functions, never parsed
from model output, so their correctness is the correctness of the benchmark.
"""
import unittest
from random import Random

from framework.tasks.taxonomy.graph_ops import (
    EDIT_OPERATORS, add_sibling, collapse_level, drop_leaf, reparent,
    sample_subtrees,
)

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


class EditOperatorTests(unittest.TestCase):
    """Each operator is exact and independently checkable against a hand-built
    graph. These functions produce the benchmark's gold, so 'roughly right' is
    not a category that exists here."""

    def setUp(self):
        self.classes = list(_CLASSES)
        self.axioms = [list(a) for a in _AXIOMS]

    def test_drop_leaf_removes_exactly_one_leaf_and_its_edge(self):
        classes, axioms = drop_leaf(self.classes, self.axioms, Random(0))
        self.assertEqual(len(classes), len(self.classes) - 1)
        gone = set(self.classes) - set(classes)
        self.assertEqual(len(gone), 1)
        removed = gone.pop()
        self.assertIn(removed, {"D", "E", "F"})     # only leaves are eligible
        self.assertNotIn(removed, [c for c, _ in axioms])
        self.assertNotIn(removed, [p for _, p in axioms])

    def test_reparent_keeps_every_class_and_edge_count(self):
        classes, axioms = reparent(self.classes, self.axioms, Random(0))
        self.assertEqual(sorted(classes), sorted(self.classes))
        self.assertEqual(len(axioms), len(self.axioms))
        self.assertNotEqual(sorted(axioms), sorted(self.axioms))

    # collapse_level gets its own CHAIN fixture. On the shared graph its two
    # eligible middles (B and C) both leave max depth at 2, so "collapsing
    # reduces depth" is simply false there — and which middle gets picked
    # depends on an rng draw the test does not control.
    CHAIN_CLASSES = ["A", "B", "C", "D"]
    CHAIN_AXIOMS = [["B", "A"], ["C", "B"], ["D", "C"]]

    def test_collapse_level_removes_one_middle_and_rehomes_its_children(self):
        # Asserted generically: either middle is a legal choice, so naming the
        # victim would pin an rng draw instead of the behaviour.
        classes, axioms = collapse_level(self.CHAIN_CLASSES, self.CHAIN_AXIOMS,
                                         Random(0))
        gone = set(self.CHAIN_CLASSES) - set(classes)
        self.assertEqual(len(gone), 1)
        victim = gone.pop()
        self.assertIn(victim, {"B", "C"})            # only middles are eligible
        old_parent = {c: p for c, p in self.CHAIN_AXIOMS}[victim]
        orphans = [c for c, p in self.CHAIN_AXIOMS if p == victim]
        for orphan in orphans:
            self.assertIn([orphan, old_parent], axioms)

    def test_collapse_level_reduces_depth(self):
        from framework.tasks.taxonomy.graph_ops import _depth_of
        before = _depth_of(self.CHAIN_CLASSES, self.CHAIN_AXIOMS)
        classes, axioms = collapse_level(self.CHAIN_CLASSES, self.CHAIN_AXIOMS,
                                         Random(0))
        self.assertLess(_depth_of(classes, axioms), before)

    def test_add_sibling_adds_one_class_under_an_existing_parent(self):
        classes, axioms = add_sibling(self.classes, self.axioms, Random(0))
        self.assertEqual(len(classes), len(self.classes) + 1)
        added = (set(classes) - set(self.classes)).pop()
        parents = [p for c, p in axioms if c == added]
        self.assertEqual(len(parents), 1)
        self.assertIn(parents[0], self.classes)

    def test_every_operator_returns_a_closed_graph(self):
        for name, op in EDIT_OPERATORS.items():
            with self.subTest(operator=name):
                classes, axioms = op(self.classes, self.axioms, Random(1))
                names = set(classes)
                for child, parent in axioms:
                    self.assertIn(child, names)
                    self.assertIn(parent, names)

    def test_an_operator_with_nothing_to_do_returns_none(self):
        # A two-class graph has no intermediate to collapse. Returning None lets
        # the caller try another operator instead of emitting a broken graph.
        self.assertIsNone(collapse_level(["A", "B"], [["B", "A"]], Random(0)))

    def test_operators_do_not_mutate_their_input(self):
        # The seed pool is reused across samples; an in-place edit would
        # silently corrupt every later draw from the same subtree.
        for name, op in EDIT_OPERATORS.items():
            with self.subTest(operator=name):
                classes = list(_CLASSES)
                axioms = [list(a) for a in _AXIOMS]
                op(classes, axioms, Random(2))
                self.assertEqual(classes, _CLASSES)
                self.assertEqual(axioms, [list(a) for a in _AXIOMS])

    def test_every_operator_returns_a_closed_graph_on_a_dag(self):
        # The tree fixture cannot reach the multi-parent paths where these
        # operators actually go wrong.
        for name, op in EDIT_OPERATORS.items():
            with self.subTest(operator=name):
                result = op(_DAG_CLASSES, _DAG_AXIOMS, Random(1))
                self.assertIsNotNone(result, f"{name} found nothing to do on the DAG")
                classes, axioms = result
                names = set(classes)
                for child, parent in axioms:
                    self.assertIn(child, names)
                    self.assertIn(parent, names)

    def test_no_operator_emits_a_duplicate_axiom(self):
        # reparent could rewrite a row into one that already existed, leaving a
        # duplicate and dropping the edge it replaced.
        for name, op in EDIT_OPERATORS.items():
            for seed in range(25):
                with self.subTest(operator=name, seed=seed):
                    result = op(_DAG_CLASSES, _DAG_AXIOMS, Random(seed))
                    if result is None:
                        continue
                    _, axioms = result
                    pairs = [tuple(a) for a in axioms]
                    self.assertEqual(len(pairs), len(set(pairs)))

    def test_reparent_never_drops_an_edge_without_replacing_it(self):
        # Direct regression for the reported failure: on a child with two
        # parents, reparent must not collapse both rows onto one parent.
        for seed in range(25):
            with self.subTest(seed=seed):
                result = reparent(["P1", "P2", "child"],
                                  [["child", "P1"], ["child", "P2"]], Random(seed))
                if result is None:
                    continue
                _, axioms = result
                pairs = {tuple(a) for a in axioms}
                self.assertEqual(len(pairs), 2, f"edge count changed: {axioms}")

    def test_operators_do_not_mutate_a_dag_input(self):
        for name, op in EDIT_OPERATORS.items():
            with self.subTest(operator=name):
                classes = list(_DAG_CLASSES)
                axioms = [list(a) for a in _DAG_AXIOMS]
                op(classes, axioms, Random(2))
                self.assertEqual(classes, _DAG_CLASSES)
                self.assertEqual(axioms, [list(a) for a in _DAG_AXIOMS])


if __name__ == "__main__":
    unittest.main()
