import unittest

from matplotlib.colors import to_hex

from framework.plotting.plots import (
    _split_by_scale,
    plot_fidelity,
    plot_generated_vs_real,
    plot_run_variance,
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
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
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


if __name__ == "__main__":
    unittest.main()


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
