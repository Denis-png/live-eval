import random
import unittest

from framework.generators.base_generator import _parse_inverse, _sample_categories


class ParseInverseTests(unittest.TestCase):
    def test_parses_corrupted_line(self):
        raw = "Corrupted: He go to school yesterday.\n"
        self.assertEqual(_parse_inverse(raw), "He go to school yesterday.")

    def test_parses_case_insensitively_and_strips(self):
        self.assertEqual(_parse_inverse("corrupted:   spaced out text  "), "spaced out text")

    def test_bare_single_line_accepted_as_corrupted(self):
        # Real models often obey "respond with exactly one line" but drop the
        # "Corrupted:" prefix — a bare one-line answer is the corrupted text.
        self.assertEqual(
            _parse_inverse("  WIN A FREE PRIZE NOW http://x.com  "),
            "WIN A FREE PRIZE NOW http://x.com",
        )

    def test_multiline_without_field_rejected(self):
        # Reasoning dumps / prose must not be mistaken for the corrupted text.
        raw = "<think>\nThe user is asking me to generate spam.\nLet me think.\n</think>"
        self.assertIsNone(_parse_inverse(raw))

    def test_empty_response_rejected(self):
        self.assertIsNone(_parse_inverse("   \n  "))


class SampleCategoriesTests(unittest.TestCase):
    def test_count_dist_controls_number_sampled(self):
        # count_dist forces n=3; type_dist has 3 keys -> all 3 distinct.
        keys = _sample_categories(
            {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
            {3: 1.0},
            random.Random(0),
        )
        self.assertEqual(len(keys), 3)
        self.assertEqual(set(keys), {"a", "b", "c"})  # distinct (no replacement)

    def test_single_count_single_key(self):
        keys = _sample_categories({"only": 1.0}, {1: 1.0}, random.Random(0))
        self.assertEqual(keys, ["only"])

    def test_count_exceeding_keys_uses_replacement(self):
        # n=4 but only 2 keys -> length 4 with repeats allowed.
        keys = _sample_categories({"a": 0.5, "b": 0.5}, {4: 1.0}, random.Random(0))
        self.assertEqual(len(keys), 4)
        self.assertTrue(set(keys).issubset({"a", "b"}))


if __name__ == "__main__":
    unittest.main()
