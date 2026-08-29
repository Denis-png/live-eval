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

_ROWS = "label,text\n" + "".join(
    f"ham,could we move the meeting to three tomorrow please number {i}\n"
    for i in range(60)
) + "".join(
    f"spam,claim your prize now http://y{i}.example\n" for i in range(30)
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
    def _run(self, compliance, rounds=3, sample_size=40):
        gen = CompliantFake(compliance)
        cfg = _config(self.path, sample_size=sample_size)
        cfg["calibration"] = {"sample_size": sample_size}
        with mock.patch.object(pipeline, "load_generator", return_value=gen):
            return calibrate.run_calibration(
                cfg, rounds=rounds, alpha=1.0, tolerance=0.05,
                sample_size=sample_size, output_path=self.out,
            )

    def test_round_zero_request_equals_the_empirical_target(self):
        # The uncalibrated distribution must be reproducible from round 0.
        payload = self._run({k: 0.9 for k in _MARKERS}, rounds=1)
        self.assertEqual(payload["rounds"][0]["request"]["type_dist"],
                         payload["target"]["type_dist"])

    def test_biased_generator_converges_and_improves(self):
        # "money_promise" is honoured a third as often as the rest.
        compliance = {k: 0.9 for k in _MARKERS}
        compliance["money_promise"] = 0.3
        payload = self._run(compliance)
        first = payload["rounds"][0]["jsd"]["type_dist"]
        best = payload["rounds"][payload["selected_round"]]["jsd"]["type_dist"]
        self.assertLessEqual(best, first)
        # The under-delivered category must be requested MORE than the target.
        self.assertGreater(payload["calibrated"]["type_dist"]["money_promise"],
                           payload["target"]["type_dist"]["money_promise"])

    def test_selected_round_is_the_best_not_the_last(self):
        payload = self._run({k: 0.9 for k in _MARKERS})
        scores = [r["jsd"]["type_dist"] for r in payload["rounds"]]
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
        self.assertEqual(seen, list(range(1, len(payload["rounds"]) + 1)))

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
