import json
import os
import tempfile
import unittest
from unittest.mock import patch

from framework.plotting import session as S

_RESULTS = {
    "meta": {"task": "spam", "mode": "forward", "model": "minimax-m3"},
    "results": {
        "mshenoda/roberta-spam": {
            "generated": {"f1": {"mean": 0.82, "std": 0.03}},
            "real": {"f1": 0.90},
            "runs": [{"f1": 0.80}, {"f1": 0.85}],
        }
    },
}
_PROFILE = {
    "real": {"class_balance": {"spam_fraction": 0.5}, "signal_rate": {"urgency": 0.6}},
    "generated": {"class_balance": {"spam_fraction": 0.45}, "signal_rate": {"urgency": 0.3}},
    "fidelity": {"type_dist_jsd": 0.1, "count_dist_jsd": 0.05},
}


def _make_session(d, profile=True, runs=True):
    results = json.loads(json.dumps(_RESULTS))
    if not runs:
        del results["results"]["mshenoda/roberta-spam"]["runs"]
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(results, f)
    if profile:
        with open(os.path.join(d, "profile.json"), "w") as f:
            json.dump(_PROFILE, f)
    return d


class SlugTests(unittest.TestCase):
    def test_slug_sanitizes(self):
        self.assertEqual(S._slug("mshenoda/roberta-spam"), "mshenoda_roberta_spam")


class LoadSessionTests(unittest.TestCase):
    def test_missing_results_raises_with_path(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as ctx:
                S.load_session(d)
            self.assertIn(d, str(ctx.exception))

    @unittest.skipIf(os.geteuid() == 0, "root ignores file permission bits")
    def test_unreadable_results_raises_value_error_not_os_error(self):
        # A results.json that exists (passes the isfile check) but cannot be
        # opened — e.g. permission bits — must surface as ValueError, not the
        # raw OSError/PermissionError that open()/json.load() would raise.
        with tempfile.TemporaryDirectory() as d:
            results_path = os.path.join(d, "results.json")
            with open(results_path, "w") as f:
                f.write("{}")
            os.chmod(results_path, 0o000)
            try:
                with self.assertRaises(ValueError) as ctx:
                    S.load_session(d)
                self.assertIn(results_path, str(ctx.exception))
            finally:
                os.chmod(results_path, 0o644)

    def test_wrong_top_level_json_type_raises_value_error(self):
        # results.json parses fine but is a JSON array, not an object — results.get("meta")
        # would raise AttributeError further down. Must be caught here as ValueError.
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "results.json"), "w") as f:
                json.dump([], f)
            with self.assertRaises(ValueError) as ctx:
                S.load_session(d)
            self.assertNotIsInstance(ctx.exception, AttributeError)


class RenderSessionTests(unittest.TestCase):
    def test_renders_all_three_figures(self):
        with tempfile.TemporaryDirectory() as d:
            _make_session(d)
            written = S.render_session(d)
            names = sorted(os.path.basename(p) for p in written)
            self.assertEqual(names, [
                "fidelity.png",
                "generated_vs_real_mshenoda_roberta_spam.png",
                "run_variance_mshenoda_roberta_spam.png",
            ])
            for p in written:
                self.assertTrue(os.path.getsize(p) > 0)
            self.assertTrue(os.path.isdir(os.path.join(d, "plots")))

    def test_gec_session_without_profile_skips_fidelity(self):
        with tempfile.TemporaryDirectory() as d:
            _make_session(d, profile=False)
            names = [os.path.basename(p) for p in S.render_session(d)]
            self.assertNotIn("fidelity.png", names)
            self.assertIn("generated_vs_real_mshenoda_roberta_spam.png", names)

    def test_session_without_runs_skips_variance(self):
        with tempfile.TemporaryDirectory() as d:
            _make_session(d, runs=False)
            names = [os.path.basename(p) for p in S.render_session(d)]
            self.assertNotIn("run_variance_mshenoda_roberta_spam.png", names)

    def test_missing_matplotlib_is_fail_soft(self):
        with tempfile.TemporaryDirectory() as d:
            _make_session(d)
            with patch.object(S, "_import_plots", side_effect=ImportError("no matplotlib")):
                self.assertEqual(S.render_session(d), [])  # warns, does not raise

    def test_malformed_model_shape_is_fail_soft(self):
        # results["<model>"] is a string, not a dict-of-metrics: must warn and
        # skip that model's figures rather than raising AttributeError.
        with tempfile.TemporaryDirectory() as d:
            malformed = {"meta": {}, "results": {"m": "not-a-dict"}}
            with open(os.path.join(d, "results.json"), "w") as f:
                json.dump(malformed, f)
            written = S.render_session(d)  # must not raise
            self.assertEqual(written, [])

    def test_out_dir_exists_as_file_is_fail_soft(self):
        with tempfile.TemporaryDirectory() as d:
            _make_session(d)
            out_dir = os.path.join(d, "out_as_file")
            with open(out_dir, "w") as f:
                f.write("not a directory")
            written = S.render_session(d, out_dir=out_dir)  # must not raise
            self.assertEqual(written, [])

    def test_one_broken_figure_does_not_stop_the_others(self):
        with tempfile.TemporaryDirectory() as d:
            _make_session(d)
            real_plots = S._import_plots()

            class Boom:
                plot_generated_vs_real = staticmethod(
                    lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
                plot_run_variance = staticmethod(real_plots.plot_run_variance)
                plot_fidelity = staticmethod(real_plots.plot_fidelity)

            with patch.object(S, "_import_plots", return_value=Boom):
                names = [os.path.basename(p) for p in S.render_session(d)]
            self.assertNotIn("generated_vs_real_mshenoda_roberta_spam.png", names)
            self.assertIn("fidelity.png", names)


if __name__ == "__main__":
    unittest.main()
