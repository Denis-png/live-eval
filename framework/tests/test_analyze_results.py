import os
import tempfile
import unittest
from unittest import mock

from scripts.analyze_results import (
    MODE_MARKERS,
    _legend_models_modes,
    _strategy_of,
    dedup_sessions,
    eta_squared,
    kendall_tau_b,
    plot_mode_effect,
    plot_model_impact,
)
import scripts.analyze_results as ar


class EtaSquaredTests(unittest.TestCase):
    def test_all_between_group_variance_gives_one(self):
        self.assertAlmostEqual(eta_squared([[1.0, 1.0], [2.0, 2.0]]), 1.0)

    def test_no_between_group_variance_gives_zero(self):
        self.assertAlmostEqual(eta_squared([[1.0, 2.0], [1.0, 2.0]]), 0.0)

    def test_degenerate_inputs_give_none(self):
        self.assertIsNone(eta_squared([[1.0, 2.0]]))          # one group
        self.assertIsNone(eta_squared([[1.0], [1.0]]))        # zero total variance


class KendallTauTests(unittest.TestCase):
    def test_perfect_agreement(self):
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_perfect_disagreement(self):
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_ties_are_handled(self):
        # x has a tie; tau-b stays defined and within [-1, 1]
        tau = kendall_tau_b([1, 1, 2, 3], [1, 2, 3, 4])
        self.assertTrue(-1.0 <= tau <= 1.0)
        self.assertGreater(tau, 0)

    def test_all_tied_is_none(self):
        self.assertIsNone(kendall_tau_b([1, 1, 1], [1, 2, 3]))


class DedupTests(unittest.TestCase):
    def _s(self, name, runs_completed, created, task="spam", mode="forward", model="m"):
        return {"dir": name, "meta": {"task": task, "mode": mode, "model": model,
                                      "runs_completed": runs_completed, "created": created},
                "results": {}}

    def test_prefers_more_runs_then_newer(self):
        a = self._s("old", 1, "2026-07-14T00:00:00")
        b = self._s("new", 3, "2026-07-15T00:00:00")
        kept, dropped = dedup_sessions([a, b])
        self.assertEqual([s["dir"] for s in kept], ["new"])
        self.assertEqual([s["dir"] for s in dropped], ["old"])

    def test_distinct_configs_all_kept(self):
        a = self._s("a", 3, "2026-07-14T00:00:00", mode="forward")
        b = self._s("b", 3, "2026-07-14T00:00:00", mode="inverse")
        kept, dropped = dedup_sessions([a, b])
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])


def _row(gen_model, eval_model, mean, runs, strategy=None, real=0.9, task="spam"):
    metric = "f1" if task == "spam" else "errant.f0.5"
    return {"task": task, "strategy": strategy, "gen_model": gen_model,
            "eval_model": eval_model, "metric": metric, "gen_mean": mean,
            "gen_std": 0.01, "real": real, "runs": runs}


class ModeAwarePlottingTests(unittest.TestCase):
    """Rows whose strategy is the "-" sentinel (no split axis) must not embed
    it into filenames/figures — same treatment a single-strategy task (e.g.
    spam with no seedless runs mixed in) gets.

    Note: `_strategy_of` itself no longer produces "-" for ANY meta, legacy or
    otherwise (see IMPORTANT 1 / test_analyze_strategy_grouping.py) — a
    legacy spam session with mode=None now resolves to "inverse", not "-",
    which is exactly the fix (it lets that session group with, and be
    superseded by, a fresh "inverse" re-run instead of forming its own stray
    bucket). The "-" branch in plot_model_impact below is therefore dead code
    reachable only by passing it directly, as this test does — kept as a
    defensive fallback and tested in isolation from `_strategy_of`."""

    def test_model_impact_filename_has_no_suffix_for_dash_strategy(self):
        rows = [
            _row("model-a", "ev1", 0.7, [0.68, 0.70, 0.72], strategy="-"),
            _row("model-b", "ev1", 0.5, [0.48, 0.50, 0.52], strategy="-"),
        ]
        captured = {}

        def _fake_save(fig, path):
            captured["title"] = fig._suptitle.get_text()
            import matplotlib.pyplot as plt
            plt.close(fig)
            return path

        with tempfile.TemporaryDirectory() as out_dir, \
             mock.patch.object(ar, "_save", side_effect=_fake_save):
            path = plot_model_impact(rows, "spam", "-", out_dir)
        self.assertIsNotNone(path)
        self.assertEqual(os.path.basename(path), "model_impact_spam.png")
        self.assertTrue(captured["title"].startswith("spam:"))
        self.assertNotIn("-", captured["title"].split(":", 1)[0])

    def test_legacy_spam_session_now_resolves_to_inverse_not_dash(self):
        # Regression guard for IMPORTANT 1: a realistic legacy spam session
        # (mode=None, strategy="class_conditional" — _build_meta always
        # writes "strategy") must group as "inverse", not fall into the dash
        # bucket the test above exercises directly.
        legacy_strategy = _strategy_of(
            {"mode": None, "strategy": "class_conditional", "task": "spam", "model": "m"}
        )
        self.assertEqual(legacy_strategy, "inverse")

    def test_model_impact_filename_keeps_mode_suffix_for_real_modes(self):
        rows = [
            _row("model-a", "ev1", 0.7, [0.68, 0.70, 0.72], strategy="inverse", task="gec"),
            _row("model-b", "ev1", 0.5, [0.48, 0.50, 0.52], strategy="inverse", task="gec"),
        ]
        with tempfile.TemporaryDirectory() as out_dir:
            path = plot_model_impact(rows, "gec", "inverse", out_dir)
            self.assertEqual(os.path.basename(path), "model_impact_gec_inverse.png")

    def test_mode_effect_skips_when_no_task_has_two_real_modes(self):
        rows = [
            _row("model-a", "ev1", 0.7, [0.68, 0.70, 0.72]),
            _row("model-a", "ev2", 0.6, [0.58, 0.60, 0.62]),
        ]
        with tempfile.TemporaryDirectory() as out_dir:
            self.assertIsNone(plot_mode_effect(rows, "spam", out_dir))


class SeedlessModeMarkersTests(unittest.TestCase):
    """MUST-FIX MINOR A: the two seedless cells need their own markers (they
    used to both fall back to the shared "s" marker, indistinguishable from
    each other and from an unrecognized strategy) and a legend row."""

    def test_mode_markers_has_distinct_entries_for_seedless_cells(self):
        for key in ("forward+seedless", "inverse+seedless"):
            self.assertIn(key, MODE_MARKERS)
        markers = list(MODE_MARKERS.values())
        self.assertEqual(len(markers), len(set(markers)), "marker shapes collide")
        # The fallback marker plot_identity uses for an unrecognized strategy
        # (see MODE_MARKERS.get(r["strategy"], "s")) must stay free so a
        # genuinely-unknown strategy remains visually distinct in the legend.
        self.assertNotIn("s", markers)

    def test_legend_includes_seedless_mode_labels(self):
        plt = ar._plt()
        fig = plt.figure()
        try:
            rows = [{"gen_model": "m", "strategy": "forward+seedless"}]
            _legend_models_modes(fig, rows, modes=True)
            labels = {h.get_label() for h in fig.legends[0].legend_handles}
        finally:
            plt.close(fig)
        self.assertIn("forward+seedless", labels)
        self.assertIn("inverse+seedless", labels)


if __name__ == "__main__":
    unittest.main()
