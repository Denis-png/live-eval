"""analyze_results keeps a calibrated run apart from the uncalibrated run of the
same cell, and can be limited to one sweep.

dedup_sessions kept one session per (task, cell, generation model), ignoring
calibration. The ablation's phase B (calibrated) and phase A (uncalibrated) runs
of the best cell share that key, so the A-vs-B comparison lost one side --
the older, which is the baseline.
"""
import sys
import unittest

sys.path.insert(0, "scripts")

import analyze_results as ar


def _session(created, calibrated, runs=3, model="minimax-m3"):
    meta = {"task": "gec", "strategy": "corruption", "mode": "inverse",
            "seedless": False, "model": model, "created": created,
            "runs_completed": runs,
            "calibration": {"path": "x_calibration.json"} if calibrated else None}
    return {"meta": meta, "dir": f"{created}-{calibrated}", "results": {}}


class CalibratedSessionTests(unittest.TestCase):
    def test_a_calibrated_session_is_its_own_cell(self):
        self.assertEqual(ar._strategy_of(_session("2026-09-12", True)["meta"]),
                         "inverse+calibrated")
        self.assertEqual(ar._strategy_of(_session("2026-09-12", False)["meta"]), "inverse")

    def test_dedup_keeps_both_sides_of_the_calibration_ablation(self):
        a, b = _session("2026-09-12T01", False), _session("2026-09-12T09", True)
        kept, dropped = ar.dedup_sessions([a, b])
        self.assertEqual((len(kept), dropped), (2, []))

    def test_the_suffix_does_not_disturb_the_semantics_quarantine(self):
        meta = dict(_session("2026-09-12", True)["meta"], task="spam",
                    strategy="class_conditional", class_conditional_semantics="old")
        self.assertEqual(ar._cell_and_semantics(ar._strategy_of(meta)),
                         ("inverse+calibrated", "old"))


class SinceFilterTests(unittest.TestCase):
    def test_sessions_before_the_cutoff_are_left_out(self):
        old, new = _session("2026-08-01T00:00:00", False), _session("2026-09-12T00:00:00", False)
        self.assertEqual(ar.filter_since([old, new], "2026-09-11"), [new])

    def test_no_cutoff_keeps_everything(self):
        sessions = [_session("2026-08-01", False)]
        self.assertEqual(ar.filter_since(sessions, None), sessions)


if __name__ == "__main__":
    unittest.main()


class IdentityFacetTests(unittest.TestCase):
    """The identity scatter tells generators apart by color alone. Up to three
    colors stay distinct across every pair (validated all-pairs: blue, aqua,
    yellow); a fourth falls into the CVD warn band. Past three, each generator
    gets its own row, titled with its name, so color is no longer the only cue."""

    def _figure(self, gen_models):
        from unittest import mock
        rows = [{"task": "spam", "strategy": "inverse", "gen_model": gm,
                 "eval_model": "clf", "metric": "f1", "gen_mean": 0.8,
                 "gen_std": 0.01, "real": 0.9, "runs": [0.8]} for gm in gen_models]
        with mock.patch.object(ar, "_save", side_effect=lambda fig, path: fig):
            fig = ar.plot_identity(rows, "spam", "/unused")
        # _save normally closes the figure; bypassed here, so close it ourselves.
        self.addCleanup(ar._plt().close, fig)
        return fig

    def test_three_generators_share_one_row(self):
        fig = self._figure(["minimax-m3", "z-ai/glm-5.3-flash", "xiaomi/mimo-v2.5"])
        self.assertEqual(len(fig.axes), 1)

    def test_more_than_three_get_a_titled_row_each(self):
        gens = ["minimax-m3", "z-ai/glm-5.3-flash", "xiaomi/mimo-v2.5", "tencent/hy3"]
        fig = self._figure(gens)
        self.assertEqual(len(fig.axes), 4)
        titles = " ".join(ax.get_title() for ax in fig.axes)
        for gm in gens:
            self.assertIn(ar._short(gm), titles)


class ModelColorTests(unittest.TestCase):
    def test_every_generator_in_the_sweep_has_its_own_color(self):
        # A missing entry falls back to one shared grey, so two generators would
        # be indistinguishable in every multi-generator figure.
        import glob
        import yaml
        models = {"minimax-m3"}
        for path in glob.glob("framework/configs/*/compare.yaml"):
            with open(path, encoding="utf-8") as f:
                models |= {m["model"] for m in yaml.safe_load(f)["generation_models"]}
        colors = [ar.MODEL_COLORS.get(m) for m in sorted(models)]
        self.assertNotIn(None, colors, f"uncolored: {sorted(m for m in models if m not in ar.MODEL_COLORS)}")
        self.assertEqual(len(set(colors)), len(colors))
        self.assertNotIn(ar.FALLBACK_COLOR, colors)
