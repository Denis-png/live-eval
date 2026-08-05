import unittest

from framework.profiling.gec_profiler import profile_gec_rows

ROWS = [
    {"incorrect": "she go to school", "correct": "she goes to school"},
    {"incorrect": "he dont like it", "correct": "he doesn't like it"},
]


class GecProfileEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.profile = profile_gec_rows(ROWS)

    def test_new_characteristic_blocks_present(self):
        self.assertEqual(self.profile["profile_version"], 2)
        for side in ("incorrect", "correct"):
            words = self.profile["length_distributions"][side]["words"]
            self.assertAlmostEqual(sum(words["bins"].values()), 1.0, places=3)
            self.assertIn("chars", self.profile["length_distributions"][side])
            self.assertIn("question_rate", self.profile["style"][side])
            self.assertIn("type_token_ratio", self.profile["vocabulary"][side])
        # "doesn't" is a contraction on the correct side only:
        self.assertAlmostEqual(self.profile["style"]["correct"]["contraction_rate"], 0.5)
        self.assertEqual(self.profile["style"]["incorrect"]["contraction_rate"], 0.0)

    def test_v1_keys_unchanged(self):
        for key in (
            "num_samples", "num_valid_pairs", "incorrect_char_length",
            "correct_char_length", "incorrect_word_count", "correct_word_count",
            "similarity", "correction_complexity", "top_frequent_words",
            "example_pairs_by_complexity",
        ):
            self.assertIn(key, self.profile)
        self.assertEqual(self.profile["num_samples"], 2)


if __name__ == "__main__":
    unittest.main()
