"""Per-run paired real scoring.

A seeded taxonomy session's real side was the WHOLE seed pool, scored once, while
each run's synthetic side is what that run drew and what passed verification --
which drops the largest subtrees most often. Micro scores are dominated by those
items, so the generated-vs-real gap mixed generation fidelity with attrition.
Each run is now also scored against the real items it actually delivered.
"""
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework import pipeline
from framework.tasks.base_task import BaseTask
from framework.tasks.taxonomy.task import TaxonomyTask
from framework.tests.test_taxonomy_real_reference import _E2E_REAL, _POOL_OPTS
from framework.tests.test_taxonomy_seeded_dispatch import _Gen

_REAL = [{"pool_index": 0, "classes": ["A"]}, {"pool_index": 1, "classes": ["B"]},
         {"pool_index": 2, "classes": ["C"]}]


class PairingHookTests(unittest.TestCase):
    def test_the_base_task_does_not_pair(self):
        self.assertIsNone(BaseTask.paired_real_indices(object(), _REAL, [{}]))

    def test_each_record_maps_to_the_real_item_with_its_index(self):
        synthetic = [{"source_pool_index": 2}, {"source_pool_index": 0}]
        self.assertEqual(TaxonomyTask().paired_real_indices(_REAL, synthetic), [2, 0])

    def test_a_repeated_draw_is_paired_once_per_repeat(self):
        synthetic = [{"source_pool_index": 1}, {"source_pool_index": 1}]
        self.assertEqual(TaxonomyTask().paired_real_indices(_REAL, synthetic), [1, 1])

    def test_positions_not_indices_are_returned(self):
        # real_sample.json need not be in pool order.
        real = [_REAL[2], _REAL[0], _REAL[1]]
        synthetic = [{"source_pool_index": 2}]
        self.assertEqual(TaxonomyTask().paired_real_indices(real, synthetic), [0])

    def test_seedless_records_are_not_paired(self):
        # A seedless artifact comes from no particular real item.
        self.assertIsNone(TaxonomyTask().paired_real_indices(_REAL, [{"classes": ["X"]}]))

    def test_an_index_matching_no_real_item_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "7"):
            TaxonomyTask().paired_real_indices(_REAL, [{"source_pool_index": 7}])


class _RejectLargest(_Gen):
    """Compliant, except on the largest subtree, whose answer drops a relation:
    verification rejects it on every attempt, as it rejects Pizza's 91-class
    subtree in real runs."""

    def call_api(self, prompt):
        answer = json.loads(super().call_api(prompt))
        if len(answer["classes"]) >= 10:
            answer["subclass_axioms"] = answer["subclass_axioms"][1:]
        return json.dumps(answer)


def _session(base, name, generator):
    data = os.path.join(base, "onto.jsonl")
    with open(data, "w", encoding="utf-8") as f:
        f.write(json.dumps(_E2E_REAL) + "\n")
    cfg = {
        "task": {"name": "taxonomy"},
        "dataset": {"source": "local", "local": {"path": data, "format": "jsonl"}},
        "generation": {"provider": "stub", "model": "stub", "mode": "forward",
                       "seedless": False, "num_runs": 2, "sample_size": 3,
                       "max_parse_attempts": 1,
                       "seed_pool": {**_POOL_OPTS, "domains": ["marine biology"]}},
        "task_models": [{"name": "lexical", "type": "lexical"},
                        {"name": "star", "type": "star"}],
        "output": {"base_dir": base, "plots": False, "session_id": name},
    }
    with mock.patch.object(pipeline, "load_generator", return_value=generator), \
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        pipeline.run_pipeline(cfg)
    with open(os.path.join(base, "taxonomy", name, "results.json"), encoding="utf-8") as f:
        return cfg, json.load(f)


def _gold_edges(scores):
    """Gold relations the scored items hold: every one is a tp or an fn."""
    return scores["diagnostics"]["tp"] + scores["diagnostics"]["fn"]


class PairedSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp)
        cls.cfg, cls.all_kept = _session(cls.tmp, "kept", _Gen())
        _, cls.one_dropped = _session(cls.tmp, "dropped", _RejectLargest())

    def test_a_paired_session_says_so(self):
        self.assertIs(self.one_dropped["meta"]["paired_real"], True)

    def test_every_run_has_a_paired_score_aligned_with_it(self):
        for scores in self.one_dropped["results"].values():
            self.assertEqual(len(scores["real_paired_runs"]), len(scores["runs"]))
            self.assertIn("f1", scores["real_paired"])
            self.assertEqual(set(scores["real_paired"]["f1"]), {"mean", "std"})

    def test_the_unpaired_real_is_unchanged(self):
        for model, scores in self.one_dropped["results"].items():
            self.assertEqual(scores["real"], self.all_kept["results"][model]["real"])

    def test_a_rejected_item_leaves_the_paired_real_side(self):
        # _RejectLargest drops the 10-class subtree in every run, so each run's
        # paired real holds fewer gold relations than the whole reference.
        for scores in self.one_dropped["results"].values():
            whole = _gold_edges(scores["real"])
            for paired in scores["real_paired_runs"]:
                self.assertLess(_gold_edges(paired), whole)

    def test_with_nothing_rejected_paired_equals_real(self):
        # The same items in a shuffled order: counts match exactly, and the
        # scores match up to float summation order.
        for scores in self.all_kept["results"].values():
            for paired in scores["real_paired_runs"]:
                self.assertEqual(_gold_edges(paired), _gold_edges(scores["real"]))
                for metric in ("precision", "recall", "f1"):
                    self.assertAlmostEqual(paired[metric], scores["real"][metric],
                                           places=12)


class UnpairedSessionTests(unittest.TestCase):
    def test_nest_results_without_paired_runs_is_unchanged(self):
        final = pipeline._nest_results({"m": {"f1": {"mean": 1.0, "std": 0.0}}},
                                       {"m": {"f1": 0.9}}, [{"m": {"f1": 1.0}}])
        self.assertEqual(set(final["m"]), {"generated", "real", "runs"})


def _assert_unpaired(test, results):
    """No paired block anywhere: pairing is all runs or none."""
    for scores in results["results"].values():
        test.assertNotIn("real_paired", scores)
        test.assertNotIn("real_paired_runs", scores)
    test.assertNotIn("paired_real", results["meta"])


class AllOrNonePairingTests(unittest.TestCase):
    """real_paired_runs[k] must be the paired score of runs[k]. Dropping the runs
    that could not be paired shifted every later entry onto the wrong run."""

    def test_the_helper_keeps_a_fully_paired_session(self):
        per_run = [{"m": {"f1": 1.0}}, {"m": {"f1": 0.5}}]
        self.assertEqual(pipeline._all_or_no_pairing(per_run), per_run)

    def test_the_helper_drops_a_session_where_no_run_paired(self):
        self.assertIsNone(pipeline._all_or_no_pairing([None, None]))
        self.assertIsNone(pipeline._all_or_no_pairing([]))

    def test_the_helper_drops_a_mixed_session_with_a_warning(self):
        err = io.StringIO()
        with redirect_stderr(err):
            out = pipeline._all_or_no_pairing([{"m": {"f1": 1.0}}, None])
        self.assertIsNone(out)
        self.assertIn("1 of 2 runs", err.getvalue())

    def test_a_pipeline_session_with_an_unpairable_run_pairs_nothing(self):
        original = TaxonomyTask.paired_real_indices
        calls = []

        def second_run_unpairable(task, real_reference, synthetic):
            calls.append(1)
            return original(task, real_reference, synthetic) if len(calls) == 1 else None

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        with mock.patch.object(TaxonomyTask, "paired_real_indices", second_run_unpairable):
            _, results = _session(tmp, "mixed", _Gen())
        self.assertEqual(len(calls), 2)
        _assert_unpaired(self, results)

    def test_a_rescore_with_one_run_unpairable_pairs_nothing_and_warns(self):
        # A pre-0e8ccc7 run's records carry no source_pool_index -- reachable by
        # merging an archived session with a current one.
        from scripts.rescore_session import rescore_session
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        cfg, _ = _session(tmp, "a", _Gen())
        session = os.path.join(tmp, "taxonomy", "a")
        run_2 = os.path.join(session, "generated", "run_2.json")
        with open(run_2, encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            del record["source_pool_index"]
        with open(run_2, "w", encoding="utf-8") as f:
            json.dump(records, f)

        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            rescore_session(session, cfg)
        with open(os.path.join(session, "results.json"), encoding="utf-8") as f:
            rescored = json.load(f)
        for scores in rescored["results"].values():
            self.assertEqual(len(scores["runs"]), 2)
        _assert_unpaired(self, rescored)
        self.assertIn("1 of 2 runs", err.getvalue())

    def test_merging_sessions_with_differing_real_samples_pairs_nothing(self):
        # Indices from the second session would map into the first's pool.
        from scripts.merge_sessions import merge_sessions
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        cfg, _ = _session(tmp, "a", _Gen())
        _session(tmp, "b", _Gen())
        other = os.path.join(tmp, "taxonomy", "b", "real_sample.json")
        with open(other, encoding="utf-8") as f:
            real = json.load(f)
        real[0]["domain"] = "somewhere else"
        with open(other, "w", encoding="utf-8") as f:
            json.dump(real, f)

        merged = os.path.join(tmp, "merged")
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            merge_sessions([os.path.join(tmp, "taxonomy", "a"),
                            os.path.join(tmp, "taxonomy", "b")], merged, cfg)
        with open(os.path.join(merged, "results.json"), encoding="utf-8") as f:
            combined = json.load(f)
        for scores in combined["results"].values():
            self.assertEqual(len(scores["runs"]), 4)
        _assert_unpaired(self, combined)
        self.assertIn("not paired", err.getvalue())


class PrinterTests(unittest.TestCase):
    def test_the_paired_real_is_printed_beside_the_real(self):
        from framework.main import format_results_lines
        text = "\n".join(format_results_lines({"m": {
            "generated": {"f1": {"mean": 0.9, "std": 0.01}},
            "real": {"f1": 0.8},
            "real_paired": {"f1": {"mean": 0.85, "std": 0.02}}}}))
        self.assertIn("paired real.f1: 0.85 ± 0.02", text)


class RealPointTests(unittest.TestCase):
    def test_the_paired_mean_wins_where_a_session_paired(self):
        from framework.real_baseline import real_point
        blocks = {"real": {"f1": 0.8, "diagnostics": {"tp": 10}},
                  "real_paired": {"f1": {"mean": 0.85, "std": 0.02},
                                  "diagnostics": {"tp": {"mean": 8.0, "std": 1.0}}}}
        self.assertEqual(real_point(blocks), {"f1": 0.85, "diagnostics": {"tp": 8.0}})

    def test_the_unpaired_real_otherwise(self):
        from framework.real_baseline import real_point
        self.assertEqual(real_point({"real": {"f1": 0.8}}), {"f1": 0.8})
        self.assertEqual(real_point({}), {})


class AnalysisTests(unittest.TestCase):
    def test_session_rows_compare_against_the_paired_real(self):
        import sys
        sys.path.insert(0, "scripts")
        import analyze_results as ar
        session = {"meta": {"task": "taxonomy", "strategy": "structured", "mode": "forward",
                            "seedless": False, "model": "m"},
                   "results": {"lexical": {
                       "generated": {"f1": {"mean": 0.5, "std": 0.1}},
                       "real": {"f1": 0.3},
                       "real_paired": {"f1": {"mean": 0.4, "std": 0.05}}}}}
        row = [r for r in ar.session_rows(session) if r["metric"] == "f1"][0]
        self.assertEqual((row["real"], row["real_unpaired"]), (0.4, 0.3))


def _analysis_session(model, reals, *, mode="forward", gen=0.5, paired=True):
    """A taxonomy session as analyze_results discovers it. `reals` maps each
    evaluated model to (paired real f1, whole-reference real f1)."""
    results = {}
    for eval_model, (paired_f1, whole_f1) in reals.items():
        results[eval_model] = {"generated": {"f1": {"mean": gen, "std": 0.1}},
                               "real": {"f1": whole_f1},
                               "runs": [{"f1": gen - 0.05}, {"f1": gen + 0.05}]}
        if paired:
            results[eval_model]["real_paired"] = {"f1": {"mean": paired_f1, "std": 0.0}}
    meta = {"task": "taxonomy", "strategy": "structured", "mode": mode, "seedless": False,
            "model": model, "runs_completed": 2, "created": "2026-09-12T00:00:00"}
    if paired:
        meta["paired_real"] = True
    return {"dir": f"{model}_{mode}", "meta": meta, "results": results, "profile": None}


def _markdown(sessions):
    import scripts.analyze_results as ar
    rows = [r for s in sessions for r in ar.session_rows(s)]
    with tempfile.TemporaryDirectory() as d:
        path = ar.write_markdown(ar.build_summary(sessions, rows), sessions, rows, [],
                                 os.path.join(d, "analysis.md"))
        with open(path, encoding="utf-8") as f:
            return f.read()


def _table_row(markdown, *cells):
    """The one headline-table line whose leading cells are `cells`."""
    prefix = "| " + " | ".join(cells) + " |"
    lines = [line for line in markdown.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1, (prefix, lines)
    return [cell.strip() for cell in lines[0].strip("|").split("|")]


class PairedReportTests(unittest.TestCase):
    """analysis.md shows the whole-reference gap beside the paired one: seeded
    taxonomy's attrition is what the paired gap leaves out on purpose."""

    def test_a_paired_row_shows_both_reals_and_both_gaps(self):
        md = _markdown([_analysis_session("gm", {"lexical": (0.4, 0.3)})])
        self.assertIn("| real (whole ref.) | gap (whole ref.) |", md)
        self.assertEqual(_table_row(md, "forward", "gm", "lexical")[3:],
                         ["0.500 ± 0.100", "0.400", "+0.100", "0.300", "+0.200"])

    def test_an_unpaired_row_leaves_the_whole_reference_columns_empty(self):
        md = _markdown([_analysis_session("gm", {"lexical": (None, 0.3)}, paired=False)])
        self.assertEqual(_table_row(md, "forward", "gm", "lexical")[3:],
                         ["0.500 ± 0.100", "0.300", "+0.200", "-", "-"])

    def test_the_report_says_which_gap_answers_which_question(self):
        md = _markdown([_analysis_session("gm", {"lexical": (0.4, 0.3)})])
        self.assertIn("the paired gap isolates generation fidelity on the items a run "
                      "delivered; the whole-reference gap includes what verification "
                      "dropped", md)

    def test_the_report_notes_which_sessions_real_is_paired(self):
        md = _markdown([_analysis_session("gm", {"lexical": (0.4, 0.3)}),
                        _analysis_session("other", {"lexical": (None, 0.3)},
                                          mode="inverse", paired=False)])
        note = [line for line in md.splitlines() if line.startswith("`real` is paired")]
        self.assertEqual(note, ["`real` is paired for: forward/gm."])

    def test_no_pairing_note_without_a_paired_session(self):
        md = _markdown([_analysis_session("gm", {"lexical": (None, 0.3)}, paired=False)])
        self.assertNotIn("`real` is paired", md)


def _dashed_levels(fig):
    """The y of every dashed reference line in a figure."""
    return {round(line.get_ydata()[0], 6) for ax in fig.axes for line in ax.lines
            if line.get_linestyle() == "--"}


class SharedReferenceTests(unittest.TestCase):
    """A paired real differs per generator and per cell, so a line drawn as THE
    real benchmark for several sessions must be the whole matched reference."""

    def _figure(self, plot, *args):
        import scripts.analyze_results as ar
        captured = {}

        def keep(fig, path):
            captured["fig"] = fig
            return path

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(ar, "_save", side_effect=keep):
            self.assertIsNotNone(plot(*args, d))
        return captured["fig"]

    def test_model_impact_draws_one_reference_for_two_generators(self):
        import scripts.analyze_results as ar
        sessions = [_analysis_session("gm1", {"lexical": (0.4, 0.3)}),
                    _analysis_session("gm2", {"lexical": (0.35, 0.3)})]
        rows = [r for s in sessions for r in ar.session_rows(s)]
        fig = self._figure(ar.plot_model_impact, rows, "taxonomy", "forward")
        self.assertEqual(_dashed_levels(fig), {0.3})

    def test_model_impact_falls_back_to_real_for_archived_rows(self):
        import scripts.analyze_results as ar
        sessions = [_analysis_session("gm1", {"lexical": (None, 0.3)}, paired=False),
                    _analysis_session("gm2", {"lexical": (None, 0.3)}, paired=False)]
        rows = [r for s in sessions for r in ar.session_rows(s)]
        for row in rows:
            row["real_unpaired"] = None          # an archived row
        fig = self._figure(ar.plot_model_impact, rows, "taxonomy", "forward")
        self.assertEqual(_dashed_levels(fig), {0.3})

    def test_mode_effect_draws_the_whole_reference_for_both_modes(self):
        import scripts.analyze_results as ar
        sessions = [_analysis_session("gm", {"lexical": (0.4, 0.3)}, mode="forward"),
                    _analysis_session("gm", {"lexical": (0.35, 0.3)}, mode="inverse")]
        rows = [r for s in sessions for r in ar.session_rows(s)]
        fig = self._figure(ar.plot_mode_effect, rows, "taxonomy")
        self.assertEqual(_dashed_levels(fig), {0.3})

    def test_one_real_benchmark_order_for_sessions_with_different_paired_orders(self):
        # Paired: star > lexical for gm1, lexical > star for gm2. Whole
        # reference: lexical > star for both -- one real benchmark, one order.
        md = _markdown([_analysis_session("gm1", {"lexical": (0.4, 0.3), "star": (0.5, 0.2)}),
                        _analysis_session("gm2", {"lexical": (0.6, 0.3), "star": (0.5, 0.2)})])
        orders = [line for line in md.splitlines() if line.startswith("Real-benchmark order")]
        self.assertEqual(orders, ["Real-benchmark order: **lexical > star**"])

    def test_kendall_tau_keeps_the_paired_real(self):
        import scripts.analyze_results as ar
        session = _analysis_session("gm", {"lexical": (0.4, 0.3), "star": (0.5, 0.2)})
        rows = ar.session_rows(session)
        # generated ties, so compare the orders tau is computed from.
        rp = ar.rank_preservation(rows, "taxonomy", "forward", "gm")
        self.assertEqual(rp["real_order"], ["star", "lexical"])


class PlotTests(unittest.TestCase):
    def test_the_paired_real_series_is_labelled_paired(self):
        from framework.plotting.plots import plot_generated_vs_real
        generated = {"f1": {"mean": 0.5, "std": 0.1}}
        paired = plot_generated_vs_real("m", generated, {"f1": 0.4}, {"paired_real": True})
        whole = plot_generated_vs_real("m", generated, {"f1": 0.3}, {})
        label = lambda fig: [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        self.assertIn("real (paired)", label(paired))
        self.assertIn("real benchmark", label(whole))

    def test_the_session_figure_uses_the_paired_real(self):
        from framework.plotting import plots
        from framework.plotting import session as S
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "results.json"), "w", encoding="utf-8") as f:
                json.dump({"meta": {"task": "taxonomy", "mode": "forward", "model": "m"},
                           "results": {"lexical": {
                               "generated": {"f1": {"mean": 0.5, "std": 0.1}},
                               "real": {"f1": 0.3},
                               "real_paired": {"f1": {"mean": 0.4, "std": 0.05}}}}}, f)
            with mock.patch.object(plots, "plot_generated_vs_real",
                                   wraps=plots.plot_generated_vs_real) as spy, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                S.render_session(d)
        self.assertEqual(spy.call_args.args[2], {"f1": 0.4})


class CompareTableTests(unittest.TestCase):
    def test_a_paired_real_is_labelled_as_such(self):
        from scripts.compare_models import _flatten
        flat = _flatten({"generated": {"f1": {"mean": 0.5, "std": 0.1}},
                         "real": {"f1": 0.3},
                         "real_paired": {"f1": {"mean": 0.4, "std": 0.05}}})
        self.assertEqual(flat["real_paired.f1"], "0.400")
        self.assertNotIn("real.f1", flat)


class RescoreTests(unittest.TestCase):
    def test_rescoring_and_merging_reproduce_the_paired_block(self):
        from scripts.merge_sessions import merge_sessions
        from scripts.rescore_session import rescore_session
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        cfg, first = _session(tmp, "a", _RejectLargest())
        _, second = _session(tmp, "b", _RejectLargest())
        session = os.path.join(tmp, "taxonomy", "a")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rescore_session(session, cfg)
        with open(os.path.join(session, "results.json"), encoding="utf-8") as f:
            rescored = json.load(f)
        for model, scores in first["results"].items():
            self.assertEqual(rescored["results"][model]["real_paired_runs"],
                             scores["real_paired_runs"])
        self.assertIs(rescored["meta"]["paired_real"], True)

        merged = os.path.join(tmp, "merged")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            merge_sessions([session, os.path.join(tmp, "taxonomy", "b")], merged, cfg)
        with open(os.path.join(merged, "results.json"), encoding="utf-8") as f:
            combined = json.load(f)["results"]
        for model in first["results"]:
            self.assertEqual(len(combined[model]["real_paired_runs"]), 4)


if __name__ == "__main__":
    unittest.main()
