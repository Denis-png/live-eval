import unittest

from framework.profiling.errant_distribution import profile_error_distribution


class FakeEdit:
    def __init__(self, etype):
        self.type = etype


class FakeAnnotator:
    """Stand-in for the ERRANT annotator. parse() is identity; annotate() looks
    up an edit list by (src, tgt). Pairs listed in `raising` raise instead."""

    def __init__(self, edits, raising=()):
        self._edits = edits
        self._raising = set(raising)

    def parse(self, text):
        return text

    def annotate(self, src, tgt):
        if (src, tgt) in self._raising:
            raise RuntimeError("annotate failed")
        return self._edits.get((src, tgt), [])


SUPPORTED = {"R:VERB:TENSE", "M:DET", "R:SPELL"}


class ProfileErrorDistributionTests(unittest.TestCase):
    def test_frequency_weighting_and_out_of_vocab_dropped(self):
        data = [
            {"incorrect": "i1", "correct": "c1"},
            {"incorrect": "i2", "correct": "c2"},
        ]
        ann = FakeAnnotator({
            ("i1", "c1"): [FakeEdit("R:VERB:TENSE"), FakeEdit("R:VERB:TENSE"),
                           FakeEdit("R:OTHER")],   # out-of-vocab, dropped
            ("i2", "c2"): [FakeEdit("M:DET")],
        })
        dist = profile_error_distribution(data, SUPPORTED, alpha=0.5,
                                          min_pairs=1, annotator=ann)
        td = dist["type_dist"]
        self.assertEqual(set(td.keys()), SUPPORTED)          # only supported types
        self.assertNotIn("R:OTHER", td)
        self.assertGreater(td["R:VERB:TENSE"], td["M:DET"])  # 2 vs 1
        self.assertGreater(td["M:DET"], td["R:SPELL"])       # 1 vs 0 (smoothed)
        self.assertGreater(td["R:SPELL"], 0.0)               # smoothing keeps it > 0
        self.assertAlmostEqual(sum(td.values()), 1.0)

    def test_count_dist_is_empirical_and_clamped(self):
        data = [
            {"incorrect": "i1", "correct": "c1"},   # 1 supported edit
            {"incorrect": "i2", "correct": "c2"},   # 6 supported -> clamp to 5
        ]
        ann = FakeAnnotator({
            ("i1", "c1"): [FakeEdit("M:DET")],
            ("i2", "c2"): [FakeEdit("R:SPELL")] * 6,
        })
        dist = profile_error_distribution(data, SUPPORTED, count_max=5,
                                          min_pairs=1, annotator=ann)
        cd = dist["count_dist"]
        self.assertEqual(sorted(cd.keys()), [1, 5])          # 6 clamped to 5
        self.assertAlmostEqual(sum(cd.values()), 1.0)
        self.assertTrue(all(1 <= n <= 5 for n in cd))

    def test_returns_none_below_min_pairs(self):
        data = [{"incorrect": "i1", "correct": "c1"}]
        ann = FakeAnnotator({("i1", "c1"): [FakeEdit("M:DET")]})
        self.assertIsNone(
            profile_error_distribution(data, SUPPORTED, min_pairs=5, annotator=ann)
        )

    def test_skips_pairs_missing_fields(self):
        data = [
            {"incorrect": "i1"},                 # missing correct -> skipped
            {"correct": "c2"},                   # missing incorrect -> skipped
            {"incorrect": "i3", "correct": "c3"},
        ]
        ann = FakeAnnotator({("i3", "c3"): [FakeEdit("M:DET")]})
        dist = profile_error_distribution(data, SUPPORTED, min_pairs=1, annotator=ann)
        self.assertAlmostEqual(sum(dist["count_dist"].values()), 1.0)
        self.assertEqual(sorted(dist["count_dist"].keys()), [1])   # one usable pair

    def test_skips_pair_whose_annotation_raises(self):
        data = [
            {"incorrect": "boom", "correct": "cX"},   # raises -> skipped
            {"incorrect": "i2", "correct": "c2"},
        ]
        ann = FakeAnnotator(
            {("i2", "c2"): [FakeEdit("M:DET")]},
            raising=[("boom", "cX")],
        )
        dist = profile_error_distribution(data, SUPPORTED, min_pairs=1, annotator=ann)
        self.assertEqual(sorted(dist["count_dist"].keys()), [1])   # only i2 counted


if __name__ == "__main__":
    unittest.main()
