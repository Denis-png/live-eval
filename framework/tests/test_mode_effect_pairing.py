"""forward->inverse pairing must work WITHIN a generation semantics.

Quarantining pre-symmetry sessions as `<cell>@asymmetric` keeps them from
pooling with current ones, but a literal {"forward","inverse"} membership test
then matches nothing in an archive where EVERY session is legacy -- silently
dropping the mode-effect figure for exactly the archives that already exist.
Pairing legacy-with-legacy is both meaningful and required; pairing across
semantics is what the suffix exists to prevent.
"""
import sys
import unittest

sys.path.insert(0, "scripts")

import analyze_results as ar

CURRENT = ar.CLASS_CONDITIONAL_SEMANTICS


class CellAndSemanticsTests(unittest.TestCase):
    def test_an_unsuffixed_label_means_current_semantics(self):
        self.assertEqual(ar._cell_and_semantics("inverse"), ("inverse", CURRENT))

    def test_a_suffixed_label_splits_into_cell_and_semantics(self):
        self.assertEqual(ar._cell_and_semantics("inverse@asymmetric"),
                         ("inverse", "asymmetric"))

    def test_the_seedless_variant_survives_the_split(self):
        self.assertEqual(ar._cell_and_semantics("forward+seedless@asymmetric"),
                         ("forward+seedless", "asymmetric"))


class ModePairingTests(unittest.TestCase):
    def test_a_wholly_legacy_archive_still_pairs(self):
        # The live spam archive: every session predates symmetric inverse.
        self.assertEqual(ar._mode_pairs({"forward@asymmetric", "inverse@asymmetric"}),
                         {"asymmetric"})

    def test_a_wholly_current_archive_pairs(self):
        self.assertEqual(ar._mode_pairs({"forward", "inverse"}), {CURRENT})

    def test_it_never_pairs_across_semantics(self):
        # One legacy forward and one current inverse is NOT a mode comparison:
        # the delta would mix two generation behaviours.
        self.assertEqual(ar._mode_pairs({"forward@asymmetric", "inverse"}), set())

    def test_both_eras_present_pairs_each_within_itself(self):
        self.assertEqual(
            ar._mode_pairs({"forward@asymmetric", "inverse@asymmetric",
                            "forward", "inverse"}),
            {"asymmetric", CURRENT})

    def test_a_row_with_no_strategy_pairs_with_nothing(self):
        # Rows can carry strategy=None; the old set-membership test tolerated
        # that silently, so the split has to as well rather than crashing the
        # whole analysis on one incomplete row.
        self.assertEqual(ar._cell_and_semantics(None), ("", CURRENT))
        self.assertEqual(ar._mode_pairs({None, "forward"}), set())

    def test_seedless_alone_is_not_a_forward_inverse_pair(self):
        # Guards the pre-existing subtlety: >=2 keys does not imply a pair.
        self.assertEqual(ar._mode_pairs({"inverse", "inverse+seedless"}), set())


if __name__ == "__main__":
    unittest.main()
