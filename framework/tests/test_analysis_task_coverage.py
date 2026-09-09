"""analyze_results must know every task the framework can run.

HEADLINE and IDENTITY_METRICS were indexed directly and held only spam and gec,
so the first archived taxonomy or sentiment session would kill the whole
analysis with a bare KeyError. Both tasks were being actively built at the time,
so the failure was scheduled, not hypothetical.
"""
import sys
import unittest

sys.path.insert(0, "scripts")

import analyze_results as ar
from framework.pipeline import load_task

_TASKS = ("gec", "spam", "taxonomy", "sentiment")


class TaskCoverageTests(unittest.TestCase):
    def test_every_task_has_a_headline_metric(self):
        for name in _TASKS:
            with self.subTest(task=name):
                self.assertIn(name, ar.HEADLINE)

    def test_every_task_has_identity_metrics(self):
        for name in _TASKS:
            with self.subTest(task=name):
                self.assertIn(name, ar.IDENTITY_METRICS)

    def test_registered_metrics_are_ones_the_task_actually_produces(self):
        # A registered name the task never emits plots nothing and hides the
        # gap; this is what keeps the two lists honest as evaluators change.
        for name in _TASKS:
            with self.subTest(task=name):
                produced = set(load_task(name).get_evaluator_fns())
                for metric in ar.IDENTITY_METRICS[name] + [ar.HEADLINE[name]]:
                    root = metric.split(".")[0]
                    self.assertIn(
                        root, produced,
                        f"{name}: '{metric}' is registered but no evaluator emits it")

    def test_an_unknown_task_fails_with_an_actionable_message(self):
        # Not a bare KeyError: the message must name what to edit.
        for accessor in (ar._headline, ar._identity_metrics):
            with self.subTest(accessor=accessor.__name__):
                with self.assertRaises(SystemExit) as ctx:
                    accessor("no_such_task")
                message = str(ctx.exception)
                self.assertIn("no_such_task", message)
                self.assertIn("analyze_results", message)
                self.assertIn("taxonomy", message)   # lists the known tasks


if __name__ == "__main__":
    unittest.main()


class LegacySemanticsGroupingTests(unittest.TestCase):
    """Every pre-symmetry class_conditional session must read as asymmetric.

    The quarantine guard keyed on meta["strategy"], but spam sessions archived
    before that key existed do not carry it — so they slipped past the check and
    kept a plain cell name, while newer pre-symmetry sessions got the suffix.
    Seven mutually-comparable asymmetric runs split into two buckets, and the
    unmarked ones would pool with future SYMMETRIC runs: exactly the silent
    mixing the marker exists to prevent.
    """

    def test_a_spam_session_predating_the_strategy_key_is_still_quarantined(self):
        legacy = {"task": "spam", "mode": "inverse"}          # no "strategy" key
        self.assertEqual(ar._strategy_of(legacy), "inverse@asymmetric")

    def test_it_groups_with_a_session_that_does_carry_the_strategy_key(self):
        older = {"task": "spam", "mode": "inverse"}
        newer = {"task": "spam", "mode": "inverse", "strategy": "class_conditional"}
        self.assertEqual(ar._strategy_of(older), ar._strategy_of(newer))

    def test_a_current_semantics_session_keeps_the_plain_cell_name(self):
        current = {"task": "spam", "mode": "inverse", "strategy": "class_conditional",
                   "class_conditional_semantics": ar.CLASS_CONDITIONAL_SEMANTICS}
        self.assertEqual(ar._strategy_of(current), "inverse")

    def test_legacy_and_current_never_share_a_bucket(self):
        legacy = {"task": "spam", "mode": "inverse"}
        current = {"task": "spam", "mode": "inverse", "strategy": "class_conditional",
                   "class_conditional_semantics": ar.CLASS_CONDITIONAL_SEMANTICS}
        self.assertNotEqual(ar._strategy_of(legacy), ar._strategy_of(current))

    def test_non_class_conditional_tasks_are_untouched(self):
        # GEC and taxonomy never carry the field; suffixing them would split
        # their archives for no reason.
        self.assertEqual(ar._strategy_of({"task": "gec", "mode": "forward"}), "forward")
        self.assertEqual(
            ar._strategy_of({"task": "taxonomy", "mode": "inverse", "seedless": True}),
            "inverse+seedless")
