import os
import tempfile
import unittest
from random import Random
from unittest import mock

from framework import calibrate, pipeline
from framework.generators.base_generator import BaseGenerator

# Markers chosen so each fires exactly ONE spam_profiler.detect_signals rule.
_MARKERS = {
    "phishing_link": "http://x.example",
    "money_promise": "$5",
    "excessive_caps": "ABC",
    "urgency": "!",
    "spam_keywords": "bonus",
}
_DESCRIPTIONS = {
    "phishing_link": "insert a suspicious link or URL",
    "money_promise": "promise or demand a specific amount of money",
    "excessive_caps": "shout using ALL-CAPS words",
    "urgency": "add exclamation-heavy urgency",
    "spam_keywords": "work in classic spam keywords such as free, win, prize, or claim",
}

# SPAM patterns cycle so the empirical target is non-degenerate: every one of the
# five signals carries real mass (min 0.1445, none stranded at the Laplace floor)
# and the signal-count distribution spans three adjacent values {2: .4, 3: .4,
# 4: .2} with no rare tail. A fixture exercising only some signals leaves the rest
# at ~0.008, where under-delivering one cannot move the aggregate JSD and the loop
# converges at round 0 before correcting anything.
_SPAM_PATTERNS = [
    "claim your prize at http://a{i}.example",      # link + keywords
    "you owe $20 today, respond now!",              # money + urgency
    "URGENT reply now to claim it!",                # caps + urgency + keywords
    "WIRE $99 to http://b{i}.example",              # link + money + caps
    "free bonus $5 at http://c{i}.example now!",    # link + money + urgency + keywords
]

_ROWS = "label,text\n" + "".join(
    f"ham,could we move the meeting to three tomorrow please number {i}\n"
    for i in range(60)
) + "".join(
    f"spam,{pattern.format(i=i)}\n"
    for i in range(6) for pattern in _SPAM_PATTERNS
)


class CompliantFake(BaseGenerator):
    """Honours each requested signal with a fixed per-signal probability.

    This is the compliance bias the controller exists to invert: asked for a mix,
    the generator delivers a systematically different one.
    """

    def __init__(self, compliance, seed=0):
        self.compliance = compliance
        self.rng = Random(seed)
        self.calls = 0

    def call_api(self, prompt: str) -> str:
        self.calls += 1
        emitted = []
        for key, phrase in _DESCRIPTIONS.items():
            if phrase in prompt and self.rng.random() < self.compliance[key]:
                emitted.append(_MARKERS[key])
        return "Corrupted: " + " ".join(["hello there friend"] + emitted)


def _config(path, sample_size=5):
    return {
        "dataset": {"source": "local", "local": {"path": path, "format": "csv"}},
        "generation": {"provider": "openai", "model": "gpt-x", "num_runs": 1,
                       "sample_size": sample_size, "mode": "inverse",
                       "seedless": False},
        "task": {"name": "spam"},
        "task_models": [],
    }


class _Bench(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "bench.csv")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(_ROWS)
        self.out = os.path.join(self.dir.name, "cal.json")


class SettingsTests(unittest.TestCase):
    def test_defaults_when_no_calibration_block(self):
        s = calibrate.calibration_settings({"generation": {"sample_size": 20}})
        self.assertEqual(s["rounds"], 3)
        self.assertEqual(s["alpha"], 0.5)
        self.assertEqual(s["tolerance"], 0.1)
        # generation.sample_size is tuned for eval cost, not estimation precision.
        self.assertEqual(s["sample_size"], 100)

    def test_yaml_block_overrides_defaults(self):
        s = calibrate.calibration_settings(
            {"generation": {"sample_size": 20},
             "calibration": {"rounds": 5, "alpha": 0.25, "tolerance": 0.05,
                             "sample_size": 250}}
        )
        self.assertEqual((s["rounds"], s["alpha"], s["tolerance"], s["sample_size"]),
                         (5, 0.25, 0.05, 250))

    def test_large_generation_sample_size_is_kept(self):
        s = calibrate.calibration_settings({"generation": {"sample_size": 400}})
        self.assertEqual(s["sample_size"], 400)


class InformativeCountTests(unittest.TestCase):
    def test_spam_counts_only_spam_rows(self):
        from framework.tasks.spam.task import SpamTask
        rows = [{"text": "a", "label": "SPAM"}, {"text": "b", "label": "HAM"},
                {"text": "c", "label": "SPAM"}]
        self.assertEqual(calibrate.informative_count(SpamTask(), rows), 2)


class ClosedLoopTests(_Bench):
    # Tighter than the 0.1 production default: round 0's type_dist JSD lands
    # around 0.026 for the bias under test, so a 0.05 tolerance would let the
    # loop stop before correcting anything and the closed-loop assertions would
    # pass vacuously.
    TOLERANCE = 0.015

    # alpha 0.5 is the production default and matters here: at 1.0 the undamped
    # step overshoots, round 1 comes out worse than round 0, and select_best
    # correctly keeps round 0 — leaving `calibrated` equal to `target`.
    #
    # 120 samples, not 40: the driver itself warns below 50 informative samples
    # ("the update may chase noise"), and count_dist over three values is
    # unstable at n=40. The closed-loop test must run in the regime the code is
    # designed for. The generation rng is unseeded (_run_generation builds its
    # own random.Random()), so these settings are what keep the test stable
    # rather than luck.
    def _run(self, compliance, rounds=3, sample_size=120):
        gen = CompliantFake(compliance)
        cfg = _config(self.path, sample_size=sample_size)
        cfg["calibration"] = {"sample_size": sample_size}
        with mock.patch.object(pipeline, "load_generator", return_value=gen):
            return calibrate.run_calibration(
                cfg, rounds=rounds, alpha=0.5, tolerance=self.TOLERANCE,
                sample_size=sample_size, output_path=self.out,
            )

    def test_round_zero_request_equals_the_empirical_target(self):
        # The uncalibrated distribution must be reproducible from round 0.
        payload = self._run({k: 0.9 for k in _MARKERS}, rounds=1)
        self.assertEqual(payload["rounds"][0]["request"]["type_dist"],
                         payload["target"]["type_dist"])

    def test_biased_generator_converges_and_improves(self):
        # "money_promise" is honoured a third as often as the rest — a gap the
        # sampler can actually close. Do NOT deepen it further: _sample_categories
        # draws WITHOUT replacement, so one category's achievable share saturates
        # near 1/mean_signals_per_message (~0.36 here). Past that the request can
        # rise without the measurement following it, so the loop stops improving
        # and the correction this test checks stops being observable.
        compliance = {k: 0.9 for k in _MARKERS}
        compliance["money_promise"] = 0.3
        payload = self._run(compliance)
        first = payload["rounds"][0]["jsd"]["type_dist"]
        # Round 0 must genuinely miss tolerance. If it converged immediately the
        # loop would stop with a single round and the round-1 assertion below
        # would raise IndexError — this makes the premise fail loudly instead.
        self.assertGreater(first, self.TOLERANCE)

        # The correction moved the right way: the under-delivered category is
        # requested MORE next round. Assert on round 1's request, not on
        # `calibrated`, because `calibrated` is whichever round select_best won —
        # and select_best ranks by the worst dimension, so a noisy count_dist can
        # legitimately keep round 0 even when type_dist improved. Which round
        # wins is test_selected_round_is_the_best_not_the_last's job; this test's
        # job is the direction of the correction.
        self.assertGreater(payload["rounds"][1]["request"]["type_dist"]["money_promise"],
                           payload["target"]["type_dist"]["money_promise"])

        # Selection never ships worse than uncalibrated. Compare on the same
        # scalar select_best uses — the worst dimension — since the selected
        # round minimises that, not type_dist alone.
        worst = [max(r["jsd"].values()) for r in payload["rounds"]]
        self.assertLessEqual(worst[payload["selected_round"]], worst[0])

    def test_selected_round_is_the_best_not_the_last(self):
        payload = self._run({k: 0.9 for k in _MARKERS})
        # Rank by the WORST dimension, which is what controller.select_best uses
        # (convergence requires every dimension inside tolerance, so the maximum
        # is the scalar that orders rounds). Ranking on type_dist alone encodes a
        # different rule and disagrees whenever count_dist is the deciding one.
        scores = [max(r["jsd"].values()) for r in payload["rounds"]]
        self.assertEqual(payload["selected_round"], scores.index(min(scores)))

    def test_calibrated_count_dist_keys_are_ints(self):
        payload = self._run({k: 0.9 for k in _MARKERS}, rounds=1)
        self.assertTrue(all(isinstance(k, int)
                            for k in payload["calibrated"]["count_dist"]))

    def test_artifact_written_after_every_round(self):
        # A killed calibration must still leave a usable best-so-far.
        seen = []
        real_write = calibrate.write_calibration

        def spy(path, payload):
            seen.append(len(payload["rounds"]))
            return real_write(path, payload)

        with mock.patch.object(calibrate, "write_calibration", side_effect=spy):
            payload = self._run({k: 0.9 for k in _MARKERS}, rounds=2)
        # CONTROLLER RULING R4: a uniformly-compliant fake can converge at round 0
        # and break out early, so the round count is not reliably [1, 2, 3] — the
        # property under test is "one write per round", not a fixed count.
        #
        # Task 7 (Stage B): spam is class_conditional, so one more write follows
        # the main loop — Stage B measures class attrition and (maybe) corrects
        # class_prob, and persists that onto the SAME round list, hence the
        # trailing duplicate of the final round count rather than a new entry.
        self.assertEqual(seen, list(range(1, len(payload["rounds"]) + 1))
                         + [len(payload["rounds"])])

    def test_spam_forces_class_prob_to_one(self):
        # Only SPAM rows carry signals; generating HAM during calibration wastes
        # roughly 7 of every 8 calls at the empirical balance.
        payload = self._run({k: 0.9 for k in _MARKERS}, rounds=1)
        self.assertEqual(payload["meta"]["forced_class_prob"], 1.0)

    def test_records_informative_samples_per_round(self):
        payload = self._run({k: 0.9 for k in _MARKERS}, rounds=1)
        self.assertGreater(payload["rounds"][0]["informative_samples"], 0)


class FailFastTests(_Bench):
    def test_uncalibratable_task_raises_before_any_api_call(self):
        gen = CompliantFake({k: 0.9 for k in _MARKERS})
        cfg = _config(self.path)
        with mock.patch.object(pipeline, "load_generator", return_value=gen), \
             mock.patch(
                 "framework.tasks.spam.task.SpamTask.get_calibration_keys",
                 return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                calibrate.run_calibration(cfg, rounds=1, output_path=self.out)
        self.assertIn("spam", str(ctx.exception))
        self.assertEqual(gen.calls, 0)

    def test_round_with_zero_usable_samples_aborts(self):
        class Silent(BaseGenerator):
            def call_api(self, prompt):
                return "I'm sorry, I can't help with that."

        cfg = _config(self.path)
        with mock.patch.object(pipeline, "load_generator", return_value=Silent()):
            with self.assertRaises(RuntimeError):
                calibrate.run_calibration(cfg, rounds=1, sample_size=6,
                                          output_path=self.out)


if __name__ == "__main__":
    unittest.main()
