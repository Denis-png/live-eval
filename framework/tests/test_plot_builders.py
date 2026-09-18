import unittest

from matplotlib.colors import to_hex

from framework.plotting import plots
from framework.plotting.plots import (
    _split_by_scale,
    _taxonomy_distribution_series,
    plot_fidelity,
    plot_generated_vs_real,
    plot_run_variance,
    plot_taxonomy_fidelity,
    plot_taxonomy_fidelity_distributions,
)
from framework.plotting.style import SERIES_GENERATED, SERIES_REAL

_GENERATED = {"f1": {"mean": 0.82, "std": 0.03}, "recall": {"mean": 0.74, "std": 0.05}}
_REAL = {"f1": 0.90, "recall": 0.88}
_RUNS = [{"f1": 0.80, "recall": 0.70}, {"f1": 0.85, "recall": 0.78}]
_PROFILE = {
    "real": {
        "class_balance": {"spam_fraction": 0.5},
        "signal_rate": {"phishing_link": 0.4, "urgency": 0.6},
    },
    "generated": {
        "class_balance": {"spam_fraction": 0.45},
        "signal_rate": {"phishing_link": 0.5, "urgency": 0.3},
    },
    "fidelity": {"type_dist_jsd": 0.12, "count_dist_jsd": 0.08},
}
_TAXONOMY_PROFILE = {
    "fidelity": {
        "profile_type": "taxonomy_structural_fidelity",
        "real_profile": {
            "n_classes": 4,
            "n_subclass_axioms": 3,
            "n_roots": 1,
            "n_leaves": 2,
            "max_depth": 2,
            "depth_distribution": {"0": 1, "1": 2, "2": 1},
            "parent_count_distribution": {"0": 1, "1": 3},
            "child_count_distribution": {"0": 2, "1": 2},
        },
        "synthetic_profiles": [
            {
                "depth_distribution": {"0": 1, "1": 1},
                "parent_count_distribution": {"0": 1, "1": 1},
                "child_count_distribution": {"0": 1, "2": 1},
            },
            {
                "depth_distribution": {"0": 1, "2": 3},
                "parent_count_distribution": {"0": 2, "2": 2},
                "child_count_distribution": {"0": 3, "1": 1},
            },
        ],
        "aggregate": {
            "n_synthetic_taxonomies": 2,
            "scalar_characteristics": {
                "n_classes": {"synthetic": {"mean": 4, "min": 3, "max": 5}},
                "n_subclass_axioms": {"synthetic": {"mean": 3, "min": 2, "max": 4}},
                "n_roots": {"synthetic": {"mean": 1, "min": 1, "max": 1}},
                "n_leaves": {"synthetic": {"mean": 2, "min": 1, "max": 3}},
                "max_depth": {"synthetic": {"mean": 2, "min": 1, "max": 3}},
            },
            "distribution_characteristics": {
                "depth_distribution": {"jensen_shannon_divergence": {"mean": 0.0}},
                "parent_count_distribution": {"jensen_shannon_divergence": {"mean": 0.1}},
                "child_count_distribution": {"jensen_shannon_divergence": {"mean": 0.2}},
            },
        },
    }
}


class SplitByScaleTests(unittest.TestCase):
    def test_all_unit_metrics_one_group(self):
        self.assertEqual(_split_by_scale({"f1": 0.8, "recall": 0.6}), [["f1", "recall"]])

    def test_out_of_unit_metric_gets_its_own_group(self):
        # n_edits (a count) must never share a y-axis with 0-1 scores.
        groups = _split_by_scale({"f1": 0.8, "n_edits": 3.4})
        self.assertEqual(groups, [["f1"], ["n_edits"]])


class GeneratedVsRealTests(unittest.TestCase):
    def test_draws_both_series_with_legend(self):
        fig = plot_generated_vs_real("m", _GENERATED, _REAL)
        ax = fig.axes[0]
        # 2 series x 2 metrics = 4 bars
        self.assertEqual(len(ax.containers[0]), 2)
        # One figure-level legend, outside the plots, never on an axis.
        self.assertIsNone(ax.get_legend())
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        self.assertEqual(len(labels), 2)
        self.assertTrue(any("generated" in l for l in labels))
        self.assertTrue(any("real" in l for l in labels))

    def test_without_real_still_renders_generated_only(self):
        fig = plot_generated_vs_real("m", _GENERATED, None)
        self.assertGreaterEqual(len(fig.axes), 1)

    def test_out_of_unit_metric_makes_a_second_subplot(self):
        gen = {"gleu": {"mean": 0.5, "std": 0.0}, "n_edits": {"mean": 3.2, "std": 0.4}}
        fig = plot_generated_vs_real("m", gen, None)
        self.assertEqual(len(fig.axes), 2)  # small multiples, never a dual axis

    def test_colour_follows_the_entity(self):
        fig = plot_generated_vs_real("m", _GENERATED, _REAL)
        ax = fig.axes[0]
        # containers[0] = generated bars, containers[1] = the errorbar overlay
        # (no patches of its own), containers[2] = the real bars.
        self.assertEqual(to_hex(ax.containers[0].patches[0].get_facecolor()), SERIES_GENERATED)
        self.assertEqual(to_hex(ax.containers[2].patches[0].get_facecolor()), SERIES_REAL)


class RunVarianceTests(unittest.TestCase):
    def test_plots_one_point_per_run_per_metric(self):
        fig = plot_run_variance("m", _RUNS)
        ax = fig.axes[0]
        self.assertGreaterEqual(len(ax.collections), 1)  # scatter of run scores

    def test_empty_runs_still_returns_figure(self):
        fig = plot_run_variance("m", [])
        self.assertGreaterEqual(len(fig.axes), 1)


class FidelityTests(unittest.TestCase):
    def test_two_panels_and_jsd_in_title(self):
        fig = plot_fidelity(_PROFILE)
        self.assertEqual(len(fig.axes), 2)  # signal rates + class balance

    def test_colour_follows_the_entity(self):
        fig = plot_fidelity(_PROFILE)
        ax_sig, ax_bal = fig.axes
        # signal-rate panel: real bars drawn first, generated second.
        self.assertEqual(to_hex(ax_sig.containers[0].patches[0].get_facecolor()), SERIES_REAL)
        self.assertEqual(to_hex(ax_sig.containers[1].patches[0].get_facecolor()), SERIES_GENERATED)
        # class-balance panel: one container, bars in [real, generated] order.
        bal_patches = ax_bal.containers[0].patches
        self.assertEqual(to_hex(bal_patches[0].get_facecolor()), SERIES_REAL)
        self.assertEqual(to_hex(bal_patches[1].get_facecolor()), SERIES_GENERATED)
        self.assertIn("JSD", fig._suptitle.get_text())


class TaxonomyFidelityPlotTests(unittest.TestCase):
    def test_draws_scalar_and_distribution_panels(self):
        fig = plot_taxonomy_fidelity(_TAXONOMY_PROFILE)
        self.assertEqual(len(fig.axes), 2)
        self.assertIn("taxonomy structural fidelity", fig._suptitle.get_text())

    def test_colour_follows_the_entity(self):
        fig = plot_taxonomy_fidelity(_TAXONOMY_PROFILE)
        ax_scalar = fig.axes[0]
        self.assertEqual(to_hex(ax_scalar.containers[0].patches[0].get_facecolor()), SERIES_REAL)
        self.assertEqual(to_hex(ax_scalar.containers[1].patches[0].get_facecolor()), SERIES_GENERATED)

    def test_distribution_series_aligns_bins_and_normalizes_counts(self):
        series = _taxonomy_distribution_series(_TAXONOMY_PROFILE["fidelity"], "depth_distribution")
        self.assertEqual(series["labels"], ["0", "1", "2"])
        self.assertEqual(series["real"], [0.25, 0.5, 0.25])
        # synthetic run 1: {0: .5, 1: .5, 2: 0}; run 2: {0: .25, 1: 0, 2: .75}
        self.assertEqual(series["synthetic_mean"], [0.375, 0.25, 0.375])
        self.assertEqual(series["synthetic_min"], [0.25, 0.0, 0.0])
        self.assertEqual(series["synthetic_max"], [0.5, 0.5, 0.75])

    def test_distribution_series_one_synthetic_run(self):
        profile = {
            "real_profile": {"depth_distribution": {"0": 1, "1": 1}},
            "synthetic_profiles": [{"depth_distribution": {"1": 2}}],
        }
        series = _taxonomy_distribution_series(profile, "depth_distribution")
        self.assertEqual(series["labels"], ["0", "1"])
        self.assertEqual(series["real"], [0.5, 0.5])
        self.assertEqual(series["synthetic_mean"], [0.0, 1.0])
        self.assertEqual(series["synthetic_min"], [0.0, 1.0])
        self.assertEqual(series["synthetic_max"], [0.0, 1.0])

    def test_distribution_figure_draws_three_panels(self):
        fig = plot_taxonomy_fidelity_distributions(_TAXONOMY_PROFILE)
        self.assertEqual(len(fig.axes), 3)
        self.assertIn("taxonomy structural distributions", fig._suptitle.get_text())



class HiddenMetricTests(unittest.TestCase):
    """fpr carries no signal in the figures (it reads 0.00/0.00 on a good model).
    It stays in results.json — this only hides it from the charts."""

    def _xtick_labels(self, fig):
        return [t.get_text() for ax in fig.axes for t in ax.get_xticklabels()]

    def test_fpr_hidden_from_generated_vs_real(self):
        gen = {"f1": {"mean": 0.82, "std": 0.03}, "fpr": {"mean": 0.0, "std": 0.0}}
        real = {"f1": 0.90, "fpr": 0.0}
        labels = self._xtick_labels(plot_generated_vs_real("m", gen, real))
        self.assertNotIn("fpr", labels)
        self.assertIn("f1", labels)

    def test_fpr_hidden_from_run_variance(self):
        runs = [{"f1": 0.80, "fpr": 0.0}, {"f1": 0.85, "fpr": 0.0}]
        labels = self._xtick_labels(plot_run_variance("m", runs))
        self.assertNotIn("fpr", labels)
        self.assertIn("f1", labels)


class SubtitleCellTests(unittest.TestCase):
    """Two sessions differing only in `seedless` must not render identically."""

    def test_seedless_appears_in_the_subtitle(self):
        seeded = plots._subtitle({"task": "gec", "mode": "inverse", "model": "m"})
        seedless = plots._subtitle({"task": "gec", "mode": "inverse", "model": "m",
                                    "seedless": True})
        self.assertNotEqual(seeded, seedless)
        self.assertIn("inverse+seedless", seedless)
        self.assertNotIn("seedless", seeded)

    def test_absent_seedless_is_unchanged(self):
        self.assertIn("inverse", plots._subtitle({"task": "gec", "mode": "inverse"}))

    def test_empty_meta_still_empty(self):
        self.assertEqual(plots._subtitle(None), "")

_SENTIMENT_PROFILE = {
    "real": {"label_dist": {"NEGATIVE": 0.3, "NEUTRAL": 0.5, "POSITIVE": 0.2},
             "word_count_hist": {"1-5": 0.4, "6-10": 0.6}},
    "generated": {"label_dist": {"NEGATIVE": 0.35, "NEUTRAL": 0.45, "POSITIVE": 0.2},
                  "word_count_hist": {"1-5": 0.5, "6-10": 0.5}},
    "fidelity": {"label_dist_jsd": 0.01, "length_jsd": 0.02},
}
# 14 ERRANT types with distinct shares: the 12 largest are drawn, 2 are folded.
_GEC_TYPES = {f"R:T{i:02d}": (14 - i) / 105 for i in range(14)}
_GEC_PROFILE = {
    "real": {"error_type_dist": _GEC_TYPES, "error_count_dist": {"1": 0.6, "2": 0.4}},
    "generated": {"error_type_dist": _GEC_TYPES, "error_count_dist": {"1": 0.5, "2": 0.5}},
    "fidelity": {"type_dist_jsd": 0.1, "count_dist_jsd": 0.02, "length_jsd": 0.01},
}


class LegendPlacementTests(unittest.TestCase):
    """Every per-session figure keeps its legend off the data: one figure-level
    legend below the plots, never one inside an axis, where it covered bars and
    value labels."""

    def test_no_builder_draws_a_legend_inside_a_plot(self):
        builders = {
            "generated_vs_real": lambda: plot_generated_vs_real("m", _GENERATED, _REAL),
            "run_variance": lambda: plot_run_variance("m", _RUNS, _GENERATED),
            "spam fidelity": lambda: plot_fidelity(_PROFILE),
            "error types": lambda: plots.plot_error_type_distribution(
                {"a": 3, "b": 1}, {"a": 0.5, "b": 0.5}),
            "sentiment fidelity": lambda: plots.plot_sentiment_fidelity(_SENTIMENT_PROFILE),
            "gec fidelity": lambda: plots.plot_gec_fidelity(_GEC_PROFILE),
            "taxonomy fidelity": lambda: plot_taxonomy_fidelity(_TAXONOMY_PROFILE),
            "taxonomy distributions":
                lambda: plot_taxonomy_fidelity_distributions(_TAXONOMY_PROFILE),
        }
        for name, build in builders.items():
            fig = build()
            try:
                with self.subTest(figure=name):
                    self.assertTrue(all(ax.get_legend() is None for ax in fig.axes))
                    self.assertEqual(len(fig.legends), 1)
            finally:
                plots.plt.close(fig)


class GecFidelityTests(unittest.TestCase):
    def test_the_largest_types_are_drawn_and_the_rest_folded(self):
        fig = plots.plot_gec_fidelity(_GEC_PROFILE)
        self.addCleanup(plots.plt.close, fig)
        ax_types = fig.axes[0]
        labels = [t.get_text() for t in ax_types.get_yticklabels()]
        self.assertEqual(len(labels), plots.GEC_TOP_TYPES + 1)
        self.assertEqual(labels[0], "R:T00")
        self.assertEqual(labels[-1], "other (2 types)")
        other_real = ax_types.containers[0][-1].get_width()
        self.assertAlmostEqual(other_real, (2 + 1) / 105)

    def test_the_session_draws_it_for_a_gec_profile(self):
        import json
        import os
        import tempfile
        from unittest import mock
        from framework.plotting import session as S
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "results.json"), "w") as f:
                json.dump({"meta": {"task": "gec"}, "results": {}}, f)
            with open(os.path.join(d, "profile.json"), "w") as f:
                json.dump(_GEC_PROFILE, f)
            # The spam figure is the fallback; a gec profile must never reach it.
            with mock.patch.object(plots, "plot_fidelity", side_effect=AssertionError):
                names = [os.path.basename(p) for p in S.render_session(d)]
        self.assertIn("fidelity.png", names)


class ScoreChartMetricTests(unittest.TestCase):
    def test_diagnostics_counters_are_left_out_and_macro_scores_kept(self):
        names = {"f1", "diagnostics.tp", "diagnostics.invalid_relation_rate",
                 "diagnostics.macro_f1", "fpr"}
        self.assertEqual(sorted(plots._scores(names)), ["diagnostics.macro_f1", "f1"])

    def test_a_macro_score_is_labelled_without_its_prefix(self):
        generated = {**_GENERATED, "diagnostics": {"macro_f1": {"mean": 0.7, "std": 0.0},
                                                   "tp": {"mean": 40.0, "std": 2.0}}}
        fig = plot_generated_vs_real("m", generated, _REAL)
        self.addCleanup(plots.plt.close, fig)
        ticks = [t.get_text() for ax in fig.axes for t in ax.get_xticklabels()]
        self.assertIn("macro_f1", ticks)
        self.assertNotIn("tp", " ".join(ticks))
        self.assertEqual(len(fig.axes), 1)   # no counts panel beside the scores


class TaxonomyDivergenceAxisTests(unittest.TestCase):
    def test_near_zero_divergences_stay_visible_and_are_explained(self):
        profile = {"fidelity": {**_TAXONOMY_PROFILE["fidelity"], "aggregate": {
            **_TAXONOMY_PROFILE["fidelity"]["aggregate"],
            "distribution_characteristics": {
                k: {"jensen_shannon_divergence": {"mean": v}}
                for k, v in (("depth_distribution", 0.0),
                             ("parent_count_distribution", 0.0),
                             ("child_count_distribution", 0.004))}}}}
        fig = plot_taxonomy_fidelity(profile)
        self.addCleanup(plots.plt.close, fig)
        ax_dist = fig.axes[1]
        self.assertAlmostEqual(ax_dist.get_ylim()[1], 0.05)
        self.assertTrue(any("below 0.01" in t.get_text() for t in ax_dist.texts))


if __name__ == "__main__":
    unittest.main()
