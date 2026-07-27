import unittest

from framework.tasks.gec.task import GECTask as GecTask


class FakeEdit:
    def __init__(self, type_):
        self.type = type_


class FakeAnnotator:
    """Canned ERRANT stand-in: maps a source sentence to a list of edit types."""
    def __init__(self, edits_by_src):
        self.edits_by_src = edits_by_src

    def parse(self, text):
        return text

    def annotate(self, src, ref):
        return [FakeEdit(t) for t in self.edits_by_src.get(src, [])]


class GecProfileDatasetTests(unittest.TestCase):
    def setUp(self):
        self.task = GecTask()
        self.ann = FakeAnnotator({
            "a bad": ["R:SPELL"],
            "b bad": ["R:SPELL", "M:DET"],
            "c same": [],
        })

    def test_profiles_real_reference_rows(self):
        rows = [
            {"text": "a bad", "corrupted": "a bad", "original": "a good"},
            {"text": "b bad", "corrupted": "b bad", "original": "b good"},
            {"text": "c same", "corrupted": "c same", "original": "c same"},
        ]
        p = self.task.profile_dataset(rows, annotator=self.ann)
        self.assertEqual(p["n"], 3)
        self.assertEqual(p["n_annotated"], 3)
        self.assertAlmostEqual(p["error_type_dist"]["R:SPELL"], 2 / 3)
        self.assertAlmostEqual(p["error_type_dist"]["M:DET"], 1 / 3)
        self.assertEqual(p["error_count_dist"], {0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
        self.assertAlmostEqual(p["edits_per_pair_mean"], 1.0)

    def test_generated_rows_ignore_error_type_field(self):
        # The generator's own error_type claim must not leak into the profile.
        rows = [{"original": "a good", "corrupted": "a bad", "error_type": "U:PREP"}]
        p = self.task.profile_dataset(rows, annotator=self.ann)
        self.assertEqual(p["error_type_dist"], {"R:SPELL": 1.0})

    def test_supported_fraction_counts_inverse_vocabulary_only(self):
        ann = FakeAnnotator({"x": ["R:SPELL", "R:WEIRD:TYPE"]})
        rows = [{"corrupted": "x", "original": "y"}]
        p = self.task.profile_dataset(rows, annotator=ann)
        self.assertAlmostEqual(p["supported_fraction"], 0.5)

    def test_rows_missing_fields_are_skipped_not_fatal(self):
        rows = [{"corrupted": "a bad", "original": "a good"}, {"corrupted": "", "original": "y"}]
        p = self.task.profile_dataset(rows, annotator=self.ann)
        self.assertEqual(p["n"], 2)
        self.assertEqual(p["n_annotated"], 1)

    def test_compare_profiles_deltas_and_jsd(self):
        real = self.task.profile_dataset(
            [{"corrupted": "a bad", "original": "a good"}], annotator=self.ann)
        gen = self.task.profile_dataset(
            [{"corrupted": "b bad", "original": "b good"}], annotator=self.ann)
        fid = self.task.compare_profiles(real, gen)
        self.assertAlmostEqual(fid["edits_per_pair_delta"], 1.0)
        self.assertAlmostEqual(fid["type_deltas"]["M:DET"], 0.5)
        self.assertAlmostEqual(fid["type_deltas"]["R:SPELL"], -0.5)
        self.assertGreater(fid["type_dist_jsd"], 0.0)
        self.assertGreater(fid["count_dist_jsd"], 0.0)
        self.assertIn("note", fid)

    def test_identical_profiles_give_zero_jsd(self):
        rows = [{"corrupted": "a bad", "original": "a good"}]
        p = self.task.profile_dataset(rows, annotator=self.ann)
        fid = self.task.compare_profiles(p, p)
        self.assertEqual(fid["type_dist_jsd"], 0.0)
        self.assertEqual(fid["edits_per_pair_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
