import os
import tempfile
import unittest
from unittest import mock

from scripts.analyze_results import (
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
    spam with no seedless runs mixed in) gets."""

    def test_model_impact_filename_has_no_suffix_for_legacy_sessions_without_mode(self):
        # Realistic legacy session: written before mode/strategy existed for
        # spam, so meta has mode=None and no "seedless" key at all.
        # _strategy_of derives "-" from that — never None — so the row's
        # strategy must come from _strategy_of, not be hand-set to None,
        # or this test would pass while the real code path (which only ever
        # sees "-") stays broken.
        legacy_strategy = _strategy_of({"mode": None, "task": "spam", "model": "m"})
        self.assertEqual(legacy_strategy, "-")
        rows = [
            _row("model-a", "ev1", 0.7, [0.68, 0.70, 0.72], strategy=legacy_strategy),
            _row("model-b", "ev1", 0.5, [0.48, 0.50, 0.52], strategy=legacy_strategy),
        ]
        captured = {}

        def _fake_save(fig, path):
            captured["title"] = fig._suptitle.get_text()
            import matplotlib.pyplot as plt
            plt.close(fig)
            return path

        with tempfile.TemporaryDirectory() as out_dir, \
             mock.patch.object(ar, "_save", side_effect=_fake_save):
            path = plot_model_impact(rows, "spam", legacy_strategy, out_dir)
        self.assertIsNotNone(path)
        self.assertEqual(os.path.basename(path), "model_impact_spam.png")
        self.assertTrue(captured["title"].startswith("spam:"))
        self.assertNotIn("-", captured["title"].split(":", 1)[0])

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


if __name__ == "__main__":
    unittest.main()
