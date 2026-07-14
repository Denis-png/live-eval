import unittest

from framework.tasks.gec.task import GECTask
from framework.tests.test_errant_distribution import FakeAnnotator, FakeEdit


class GECProfileDistributionTests(unittest.TestCase):
    def test_uses_errant_vocabulary_and_weights_by_frequency(self):
        task = GECTask()
        # 6 identical pairs, each a single R:VERB:TENSE edit (>= default min_pairs=5).
        ann = FakeAnnotator({("bad", "good"): [FakeEdit("R:VERB:TENSE")]})
        data = [{"incorrect": "bad", "correct": "good"}] * 6

        dist = task.profile_error_distribution(data, count_max=5, annotator=ann)

        supported = set(task.get_error_descriptions().keys())
        td = dist["type_dist"]
        self.assertEqual(set(td.keys()), supported)          # full supported vocab
        self.assertGreater(len(supported), 1)
        self.assertEqual(max(td, key=td.get), "R:VERB:TENSE")  # most frequent
        self.assertTrue(all(p > 0 for p in td.values()))      # smoothing
        self.assertAlmostEqual(sum(td.values()), 1.0)
        self.assertEqual(dist["count_dist"], {1: 1.0})

    def test_returns_none_when_data_insufficient(self):
        task = GECTask()
        ann = FakeAnnotator({("bad", "good"): [FakeEdit("R:VERB:TENSE")]})
        data = [{"incorrect": "bad", "correct": "good"}] * 2  # < default min_pairs
        self.assertIsNone(task.profile_error_distribution(data, annotator=ann))


if __name__ == "__main__":
    unittest.main()
