import unittest

from framework.evaluators.gec import _errant_shared
from framework.evaluators.gec.errant import compute_errant
from framework.evaluators.gec.errant_dist import compute_errant_dist
from framework.evaluators.gec.n_edits import compute_n_edits


class FakeEdit:
    def __init__(self, o_start, o_end, etype):
        self.o_start, self.o_end, self.type = o_start, o_end, etype


class FakeAnnotator:
    """Stand-in for the ERRANT annotator. `edits` maps (src, tgt) -> edit list.
    Counts parse() calls so we can assert the shared single-pass behaviour."""

    def __init__(self, edits):
        self._edits = edits
        self.parse_calls = 0

    def parse(self, text):
        self.parse_calls += 1
        return text

    def annotate(self, src, tgt):
        return self._edits.get((src, tgt), [])


class AnnotateResultsTests(unittest.TestCase):
    def setUp(self):
        _errant_shared.reset_cache()
        self.results = [
            {"corrupted": "c1", "original": "o1", "prediction": "p1"},
            {"corrupted": "c2", "original": "o2", "prediction": "p2"},
        ]
        self.fake = FakeAnnotator({
            ("c1", "o1"): [FakeEdit(0, 1, "R:VERB")],
            ("c1", "p1"): [FakeEdit(0, 1, "R:VERB")],
            ("c2", "o2"): [FakeEdit(1, 2, "M:DET")],
            ("c2", "p2"): [],
        })

    def test_parses_each_field_once_per_item(self):
        out = _errant_shared.annotate_results(self.results, annotator=self.fake)
        # 2 items x 3 fields (corrupted/original/prediction) = 6 parses, no more.
        self.assertEqual(self.fake.parse_calls, 6)
        self.assertEqual(len(out), 2)
        self.assertIn("ref_edits", out[0])
        self.assertIn("pred_edits", out[0])

    def test_three_metrics_share_one_annotation_pass(self):
        # Inject the fake as the lazily-loaded annotator so the cache path runs.
        _errant_shared._annotator = self.fake
        try:
            compute_errant(self.results)
            compute_errant_dist(self.results)
            compute_n_edits(self.results)
        finally:
            _errant_shared._annotator = None
        # Without sharing this would be ~3x. One pass = 6 parses.
        self.assertEqual(self.fake.parse_calls, 6)

    def test_errant_precision_recall(self):
        _errant_shared._annotator = self.fake
        try:
            out = compute_errant(self.results)
        finally:
            _errant_shared._annotator = None
        # ref edits: {c1:(0,1,R:VERB), c2:(1,2,M:DET)} = 2; pred edits: {c1:(0,1,R:VERB)} = 1
        # tp=1 (c1), fp=0, fn=1 (c2 missed) -> precision 1.0, recall 0.5
        self.assertEqual(out["precision"], 1.0)
        self.assertEqual(out["recall"], 0.5)

    def test_n_edits_counts_predicted_edits(self):
        _errant_shared._annotator = self.fake
        try:
            out = compute_n_edits(self.results)
        finally:
            _errant_shared._annotator = None
        # pred edits: c1 -> 1, c2 -> 0 ; mean = 0.5
        self.assertEqual(out, 0.5)


if __name__ == "__main__":
    unittest.main()
