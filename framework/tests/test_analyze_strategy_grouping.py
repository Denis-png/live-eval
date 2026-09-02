import unittest

import scripts.analyze_results as ar
from framework.generators.base_generator import CLASS_CONDITIONAL_SEMANTICS


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
        # _build_meta's own per-strategy default — so the MODE half resolves
        # to "inverse" rather than to a stray "-" bucket.
        #
        # Two such sessions still group together (they were generated under the
        # same, pre-symmetry semantics); what they no longer group with is a
        # re-run under today's symmetric semantics — see
        # ClassConditionalSemanticsGroupingTests below.
        legacy = {"mode": None, "strategy": "class_conditional"}
        same_era = {"mode": "inverse", "strategy": "class_conditional"}
        self.assertEqual(ar._strategy_of(legacy), "inverse@asymmetric")
        self.assertEqual(ar._strategy_of(legacy), ar._strategy_of(same_era))

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


class ClassConditionalSemanticsGroupingTests(unittest.TestCase):
    """`meta.class_conditional_semantics` is written by every class_conditional
    run, and the analysis must READ it: a spam `inverse` session generated
    before symmetric inverse (HAM = a paraphrase of a HAM seed) and one
    generated after it (HAM imposed on a SPAM seed) are different generation
    behaviours that happen to share a task, a cell and a model. Pooling them
    silently drops one archive and mixes two semantics into one
    forward-vs-inverse delta."""

    @staticmethod
    def _session(semantics, *, runs=3, created="2026-01-01"):
        meta = {"task": "spam", "mode": "inverse", "seedless": False,
                "model": "m", "strategy": "class_conditional",
                "runs_completed": runs, "created": created}
        if semantics is not None:
            meta["class_conditional_semantics"] = semantics
        return {"dir": f"d-{semantics}-{runs}", "meta": meta,
                "results": {}, "profile": None}

    def test_absent_field_reads_as_asymmetric(self):
        self.assertEqual(
            ar._strategy_of({"mode": "inverse", "strategy": "class_conditional"}),
            "inverse@asymmetric",
        )

    def test_current_semantics_keeps_the_plain_cell_name(self):
        # Everything this code can still produce groups under the bare cell
        # name, so MODE_MARKERS, the forward/inverse pairing and the analysis
        # tables behave exactly as they did.
        for cell, seedless in (("inverse", False), ("inverse+seedless", True)):
            meta = {"mode": "inverse", "seedless": seedless,
                    "strategy": "class_conditional",
                    "class_conditional_semantics": CLASS_CONDITIONAL_SEMANTICS}
            self.assertEqual(ar._strategy_of(meta), cell)

    def test_sessions_differing_only_in_semantics_do_not_dedup_into_one(self):
        # Identical task, cell, model and created date; the ONLY difference is
        # the semantics marker. Without it the 5-run session would supersede
        # the 3-run one and the archive would silently lose it.
        old = self._session(None, runs=5)
        new = self._session(CLASS_CONDITIONAL_SEMANTICS, runs=3)
        kept, dropped = ar.dedup_sessions([old, new])
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])
        self.assertEqual({s["dir"] for s in kept}, {old["dir"], new["dir"]})

    def test_same_semantics_still_dedups_on_completed_runs(self):
        # The supersede rule itself is untouched within one semantics era.
        few = self._session(CLASS_CONDITIONAL_SEMANTICS, runs=3)
        many = self._session(CLASS_CONDITIONAL_SEMANTICS, runs=5)
        kept, dropped = ar.dedup_sessions([few, many])
        self.assertEqual([s["dir"] for s in kept], [many["dir"]])
        self.assertEqual([s["dir"] for s in dropped], [few["dir"]])

    def test_non_class_conditional_sessions_are_untouched(self):
        # GEC ("corruption") and taxonomy ("structured") never write the field;
        # they must keep their bare cell names whether or not one is present.
        for strategy in ("corruption", "structured"):
            for extra in ({}, {"class_conditional_semantics": "asymmetric"}):
                meta = {"mode": "forward", "strategy": strategy, **extra}
                self.assertEqual(ar._strategy_of(meta), "forward")
                seedless = {**meta, "seedless": True}
                self.assertEqual(ar._strategy_of(seedless), "forward+seedless")

    def test_gec_sessions_of_the_same_cell_still_dedup(self):
        def _gec(runs):
            return {"dir": f"gec-{runs}", "results": {}, "profile": None,
                    "meta": {"task": "gec", "mode": "inverse", "model": "m",
                             "strategy": "corruption", "runs_completed": runs,
                             "created": "2026-01-01"}}
        kept, dropped = ar.dedup_sessions([_gec(3), _gec(5)])
        self.assertEqual([s["dir"] for s in kept], ["gec-5"])
        self.assertEqual([s["dir"] for s in dropped], ["gec-3"])


if __name__ == "__main__":
    unittest.main()
