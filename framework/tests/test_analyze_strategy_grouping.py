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

    def test_legacy_spam_meta_with_null_mode_defaults_to_inverse(self):
        # Archived spam sessions carry "mode": null (the old _build_meta
        # forced it) but "strategy": "class_conditional" (task shape, always
        # written). The class_conditional default is "inverse" — same as
        # _build_meta's own per-strategy default — so this session groups
        # with, and gets superseded by, a fresh re-run of the identical
        # config that now records "mode": "inverse" explicitly.
        legacy = {"mode": None, "strategy": "class_conditional"}
        new_style = {"mode": "inverse", "strategy": "class_conditional"}
        self.assertEqual(ar._strategy_of(legacy), "inverse")
        self.assertEqual(ar._strategy_of(legacy), ar._strategy_of(new_style))

    def test_legacy_corruption_meta_with_missing_mode_defaults_to_forward(self):
        # GEC-shaped ("corruption") sessions missing "mode" resolve to the
        # historical default "forward", mirroring _build_meta's own default.
        self.assertEqual(ar._strategy_of({"strategy": "corruption"}), "forward")

    def test_missing_mode_and_strategy_falls_back_to_forward(self):
        # A meta object with neither key at all (never produced by
        # _build_meta, which always writes "strategy") still resolves to a
        # real mode rather than a stray "-" sentinel — "forward" is the
        # historical default when the task shape can't be determined either.
        self.assertEqual(ar._strategy_of({}), "forward")

    def test_seeded_and_seedless_sessions_of_same_mode_do_not_merge(self):
        seeded = {"task": "gec", "mode": "inverse", "model": "m"}
        seedless = {"task": "gec", "mode": "inverse", "seedless": True, "model": "m"}
        self.assertNotEqual(
            (seeded["task"], ar._strategy_of(seeded), seeded["model"]),
            (seedless["task"], ar._strategy_of(seedless), seedless["model"]),
        )


if __name__ == "__main__":
    unittest.main()
