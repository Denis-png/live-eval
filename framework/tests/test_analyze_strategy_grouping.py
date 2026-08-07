import unittest

import scripts.analyze_results as ar


class StrategyOfTests(unittest.TestCase):
    """`_strategy_of` derives the analysis grouping label from `mode` +
    `seedless` — it does NOT read `meta["strategy"]`, which already means
    the task's generation shape ("corruption" / "class_conditional") and
    must not be repurposed (Task 9's contract)."""

    def test_seedless_true_appends_suffix_to_mode(self):
        self.assertEqual(
            ar._strategy_of({"mode": "inverse", "seedless": True}),
            "inverse+seedless",
        )

    def test_seedless_false_returns_plain_mode(self):
        self.assertEqual(
            ar._strategy_of({"mode": "inverse", "seedless": False}),
            "inverse",
        )

    def test_legacy_meta_without_seedless_key_returns_plain_mode(self):
        # Sessions written before seedless generation existed have no
        # "seedless" key at all — must still group by their plain mode.
        self.assertEqual(ar._strategy_of({"mode": "forward"}), "forward")

    def test_missing_mode_falls_back_to_dash(self):
        self.assertEqual(ar._strategy_of({}), "-")

    def test_seeded_and_seedless_sessions_of_same_mode_do_not_merge(self):
        seeded = {"task": "gec", "mode": "inverse", "model": "m"}
        seedless = {"task": "gec", "mode": "inverse", "seedless": True, "model": "m"}
        self.assertNotEqual(
            (seeded["task"], ar._strategy_of(seeded), seeded["model"]),
            (seedless["task"], ar._strategy_of(seedless), seedless["model"]),
        )


if __name__ == "__main__":
    unittest.main()
