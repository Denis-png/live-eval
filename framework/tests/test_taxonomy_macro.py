"""Macro averages beside the micro headline.

Micro sums true/false positives across items before taking precision and
recall, so a large taxonomy outweighs a small one. Macro averages each item's
own score, so every taxonomy counts equally. Micro stays the headline; macro is
reported so a report can show whether its conclusions depend on the choice.
"""
import unittest
from statistics import mean

from framework.evaluators.taxonomy._taxonomy_shared import (
    compute_taxonomy_scores,
    reset_cache,
    score_taxonomy_result,
)
from framework.tasks.taxonomy.task import TaxonomyTask


def _row(classes, gold, predicted):
    return {"classes": classes, "subclass_axioms": gold,
            "prediction": {"subclass_axioms": predicted, "raw_output": "",
                           "diagnostics": {}}}


_SMALL = (["R", "A"], [["A", "R"]])
_LARGE_CLASSES = ["R"] + [f"C{i}" for i in range(9)]
_LARGE_GOLD = [[f"C{i}", "R"] for i in range(9)]
_LARGE_WRONG = [[f"C{i}", f"C{(i + 1) % 9}"] for i in range(9)]


class MacroTests(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def test_macro_equals_micro_for_a_single_result(self):
        scores = compute_taxonomy_scores([_row(*_SMALL, [["A", "R"]])])
        for side in ("precision", "recall", "f1"):
            self.assertEqual(scores[f"macro_{side}"], scores[side])

    def test_macro_weights_each_taxonomy_equally(self):
        # A small perfect item and a large wholly wrong one. Macro averages the
        # two F1s to 0.5; micro is dominated by the large item's nine misses.
        rows = [_row(*_SMALL, [["A", "R"]]),
                _row(_LARGE_CLASSES, _LARGE_GOLD, _LARGE_WRONG)]
        scores = compute_taxonomy_scores(rows)
        self.assertEqual(scores["macro_f1"], 0.5)
        self.assertLess(scores["f1"], 0.2)

    def test_macro_is_the_mean_of_the_per_result_scores(self):
        rows = [_row(*_SMALL, [["A", "R"]]),
                _row(*_SMALL, []),
                _row(_LARGE_CLASSES, _LARGE_GOLD, _LARGE_GOLD[:4])]
        scores = compute_taxonomy_scores(rows)
        per = [score_taxonomy_result(r) for r in rows]
        for side in ("precision", "recall", "f1"):
            # Almost-equal: statistics.mean rounds exactly, sum/n accumulates
            # float error. The claim is "the mean", not a bit pattern.
            self.assertAlmostEqual(scores[f"macro_{side}"],
                                   mean(p[side] for p in per), places=12)

    def test_equal_sized_items_can_still_differ_from_micro(self):
        # Not a bug -- the reason macro is worth reporting. An item that
        # predicts nothing has precision 0 by convention, which drags the macro
        # mean down while adding only false negatives to micro.
        gold = [["A", "R"], ["B", "R"]]
        rows = [_row(["R", "A", "B"], gold, gold), _row(["R", "A", "B"], gold, [])]
        scores = compute_taxonomy_scores(rows)
        self.assertAlmostEqual(scores["f1"], 2 / 3)
        self.assertEqual(scores["macro_f1"], 0.5)

    def test_no_results_score_zero(self):
        scores = compute_taxonomy_scores([])
        for side in ("precision", "recall", "f1"):
            self.assertEqual(scores[f"macro_{side}"], 0.0)

    def test_macro_reaches_the_diagnostics_metric(self):
        fns = TaxonomyTask().get_evaluator_fns()
        diagnostics = fns["diagnostics"]([_row(*_SMALL, [["A", "R"]])])
        self.assertIn("macro_f1", diagnostics)
        self.assertNotIn("f1", diagnostics)       # micro stays the headline metric


if __name__ == "__main__":
    unittest.main()
