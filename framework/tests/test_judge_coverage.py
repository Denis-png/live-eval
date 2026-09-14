"""The judge runs in every cell of every task, and a run records what it did.

Before: spam's forward+seedless cell skipped the judge (no seed for the pair
prompts to compare against) while meta.judge still said it was on, taxonomy had
no judge at all, and the drop counts reached only the console. The judge is now
an ablation condition, so all three would have made that comparison unreadable.
"""
import io
import json
import os
import random
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework import pipeline
from framework.generators.base_generator import BaseGenerator, _judgement_passes
from framework.pipeline import _run_generation
from framework.tasks.spam.task import SpamTask
from framework.tasks.taxonomy.task import TaxonomyTask
from framework.tests.test_seed_policy import COMMON, FakeGenerator
from framework.tests.test_spam_cells import DIST, PROFILE
from framework.tests.test_taxonomy_real_reference import _E2E_REAL, _POOL_OPTS
from framework.tests.test_taxonomy_seeded_dispatch import _Gen
from scripts import analyze_results as ar

KEEP = "Redundancy: valid\nCorrection: correct"
DROP = "Redundancy: valid\nCorrection: incorrect"


class VerdictParsingTests(unittest.TestCase):
    """The judge is a reasoning model: it restates the answer format while it
    thinks, and the verdict is what it says last."""

    def test_a_format_restated_while_reasoning_is_not_the_verdict(self):
        raw = ("Format is Redundancy: trivial/valid, fine.\n"
               "Redundancy: trivial/valid\nCorrection: correct/incorrect\n"
               "Looks natural to me.\n" + KEEP)
        self.assertTrue(_judgement_passes(raw))

    def test_a_closed_reasoning_block_is_ignored(self):
        raw = "<think>\nCorrection: incorrect? no.\n</think>\n" + KEEP
        self.assertTrue(_judgement_passes(raw))

    def test_the_last_verdict_wins_in_either_direction(self):
        self.assertFalse(_judgement_passes("Correction: correct\n...\n" + DROP))


class SeedlessClassConditionalJudgeTests(unittest.TestCase):
    """seed_policy="none" judges each message on its own, told its label."""

    def _run(self, judge_prompt, verdict):
        gen = FakeGenerator("Message: FREE prize, click http://x.com")
        seen = []

        def judge(prompt):
            seen.append(prompt)
            return verdict

        out = gen.generate_class_conditional(
            real_seeds=None, sample_size=2, seed_policy="none",
            seedless_prompts={"SPAM": "spam {spec} using {error_spec}", "HAM": "ham {spec}"},
            specs_by_label={"SPAM": ["topic: prizes"], "HAM": ["topic: chat"]},
            judge_prompt=judge_prompt, judge_call=judge,
            rng=random.Random(0), **COMMON,
        )
        return out, seen, gen.last_judge_stats

    def test_a_standalone_prompt_judges_every_message_with_its_label(self):
        out, seen, stats = self._run("Class: {label}\nMessage: {sentence}", DROP)
        self.assertEqual(out, [])
        self.assertEqual(seen[0], "Class: SPAM\nMessage: FREE prize, click http://x.com")
        self.assertEqual(stats, {"judged": 2, "dropped": 2})

    def test_a_passing_verdict_keeps_the_message(self):
        out, _, stats = self._run("Class: {label}\nMessage: {sentence}", KEEP)
        self.assertEqual(len(out), 2)
        self.assertEqual(stats, {"judged": 2, "dropped": 0})

    def test_a_pair_prompt_still_skips_rather_than_judging_against_none(self):
        out, seen, stats = self._run("{sentence} vs {correction}", DROP)
        self.assertEqual(len(out), 2)
        self.assertEqual(seen, [])
        self.assertEqual(stats, {"judged": 0, "dropped": 0})


class SpamSeedlessDispatchTests(unittest.TestCase):
    def _judge_prompt(self, judge_call):
        generator = mock.Mock()
        generator.generate_class_conditional.return_value = [{"text": "t", "label": "HAM"}]
        config = {"generation": {"mode": "forward", "seedless": True, "sample_size": 2}}
        _run_generation(generator, SpamTask(), config, [], DIST, judge_call, 0.5,
                        profile=PROFILE)
        return generator.generate_class_conditional.call_args.kwargs["judge_prompt"]

    def test_forward_seedless_gets_the_standalone_prompt(self):
        prompt = self._judge_prompt(lambda p: KEEP)
        self.assertEqual(prompt, SpamTask().get_seedless_judge_prompt())
        self.assertIn("{label}", prompt)
        self.assertIn("{sentence}", prompt)
        self.assertNotIn("{correction}", prompt)

    def test_without_a_judge_there_is_no_prompt(self):
        self.assertIsNone(self._judge_prompt(None))


class _Artifacts(BaseGenerator):
    def call_api(self, prompt):
        return '{"classes": ["A", "B"]}'


def _parse(raw):
    return {"artifact": {"domain": "d", "classes": ["A", "B"],
                         "subclass_axioms": [["B", "A"]]}, "diagnostic": {}}


class StructuredJudgeTests(unittest.TestCase):
    def _judge(self, verdicts):
        calls = []

        def judge(artifact):
            calls.append(artifact)
            verdict = verdicts[len(calls) - 1]
            if isinstance(verdict, Exception):
                raise verdict
            return verdict
        return judge, calls

    def test_a_failing_verdict_drops_the_artifact_and_is_recorded(self):
        gen = _Artifacts()
        judge, calls = self._judge([KEEP, DROP, KEEP])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = gen.generate_structured(build_prompt=lambda fb: "p", parse=_parse,
                                          sample_size=3, judge=judge)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(calls), 3)
        self.assertEqual(gen.last_judge_stats, {"judged": 3, "dropped": 1})
        [rejection] = gen.last_rejections
        self.assertEqual(rejection["index"], 2)
        self.assertEqual(rejection["judge"]["rejection_reason"], "judge")
        self.assertIn("incorrect", rejection["judge"]["verdict"])

    def test_a_judge_that_raises_costs_the_sample_but_is_not_counted_judged(self):
        gen = _Artifacts()
        judge, _ = self._judge([RuntimeError("judge down"), KEEP])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = gen.generate_structured(build_prompt=lambda fb: "p", parse=_parse,
                                          sample_size=2, judge=judge)
        self.assertEqual(len(out), 1)
        self.assertEqual(gen.last_judge_stats, {"judged": 1, "dropped": 0})
        self.assertIn("judge down", gen.last_rejections[0]["judge"]["rejection_reason"])

    def test_a_seeded_gold_the_judge_rejects_is_dropped_not_retried(self):
        gen = _Artifacts()
        gen.call_api = mock.Mock(wraps=gen.call_api)
        judge, _ = self._judge([DROP])
        gold = {"domain": "d", "classes": ["A", "B"], "subclass_axioms": [["B", "A"]]}
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = gen.generate_structured_seeded(
                [gold], build_prompt=lambda g: "p", parse=_parse,
                verify=lambda g, p: True, max_parse_attempts=3, judge=judge)
        self.assertEqual(out, [])
        self.assertEqual(gen.call_api.call_count, 1)
        self.assertEqual(gen.last_judge_stats, {"judged": 1, "dropped": 1})
        self.assertEqual(gen.last_rejections[0]["judge"]["rejection_reason"], "judge")


class TaxonomyJudgePromptTests(unittest.TestCase):
    def test_the_prompt_carries_domain_classes_and_every_axiom(self):
        prompt = TaxonomyTask().build_structured_judge_prompt({
            "domain": "musical instruments", "classes": ["Instrument", "Violin"],
            "subclass_axioms": [["Violin", "Instrument"]]})
        self.assertIn("Domain: musical instruments", prompt)
        self.assertIn("Classes: Instrument, Violin", prompt)
        self.assertIn("- Violin is a kind of Instrument", prompt)
        self.assertIn("Correction: correct/incorrect", prompt)

    def test_an_enabled_judge_with_no_prompt_is_refused(self):
        config = {"generation": {"mode": "forward", "seedless": True, "sample_size": 1}}
        with mock.patch.object(TaxonomyTask, "get_judge_prompt", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "judge_prompt"):
                _run_generation(mock.Mock(), TaxonomyTask(), config, [], None,
                                lambda p: KEEP, None, profile={"stub": True})


class JudgedTaxonomySessionTests(unittest.TestCase):
    """End to end: a judged taxonomy session drops what the judge rejects,
    records per-run counts, and is analysed as its own +judge cell."""

    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

    def test_judge_stats_and_the_judge_cell(self):
        data = os.path.join(self.base, "onto.jsonl")
        with open(data, "w", encoding="utf-8") as f:
            f.write(json.dumps(_E2E_REAL) + "\n")
        judge_cfg = {"enabled": True, "provider": "stub", "model": "judge-stub"}
        cfg = {
            "task": {"name": "taxonomy"},
            "dataset": {"source": "local", "local": {"path": data, "format": "jsonl"}},
            "generation": {"provider": "stub", "model": "stub", "mode": "forward",
                           "seedless": False, "num_runs": 2, "sample_size": 3,
                           "max_parse_attempts": 1,
                           "seed_pool": {**_POOL_OPTS, "domains": ["marine biology"]}},
            "judge": judge_cfg,
            "task_models": [{"name": "star", "type": "star"}],
            "output": {"base_dir": self.base, "plots": False, "session_id": "s"},
        }
        prompts = []

        class _Judge:
            def call_api(self, prompt):
                prompts.append(prompt)
                return DROP if len(prompts) % 3 == 0 else KEEP

        def load(gen_cfg):
            return _Judge() if gen_cfg.get("model") == "judge-stub" else _Gen()

        with mock.patch.object(pipeline, "load_generator", side_effect=load), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            pipeline.run_pipeline(cfg)
        with open(os.path.join(self.base, "taxonomy", "s", "results.json"),
                  encoding="utf-8") as f:
            meta = json.load(f)["meta"]

        self.assertEqual(meta["judge_stats"], [{"judged": 3, "dropped": 1}] * 2)
        self.assertEqual(meta["effective_samples_per_run"], [2, 2])
        self.assertIn("is a kind of", prompts[0])
        self.assertEqual(ar._strategy_of(meta), "forward+judge")


class JudgeCellLabelTests(unittest.TestCase):
    def test_judged_and_unjudged_runs_of_a_cell_are_separate_cells(self):
        meta = {"task": "gec", "strategy": "corruption", "mode": "inverse"}
        self.assertEqual(ar._strategy_of({**meta, "judge": None}), "inverse")
        self.assertEqual(ar._strategy_of({**meta, "judge": {"model": "m"}}), "inverse+judge")
        calibrated = {**meta, "judge": {"model": "m"}, "calibration": {"path": "c.json"}}
        self.assertEqual(ar._strategy_of(calibrated), "inverse+calibrated+judge")


if __name__ == "__main__":
    unittest.main()
