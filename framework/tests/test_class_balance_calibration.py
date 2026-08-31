import unittest
from random import Random

from framework.calibration.class_balance import correct_class_prob
from framework.generators.base_generator import BaseGenerator

_INJECT = "Make SPAM using {error_spec} from: {sentence}"
_HAM = "Rewrite legitimately: {sentence}"
# correct_class_prob takes the labels explicitly — it is shared machinery
# and must not default to any one task's class names.
_LABELS = {"positive_label": "SPAM", "negative_label": "HAM"}
_DESC = {"phishing_link": "insert a suspicious link"}


class _Scripted(BaseGenerator):
    def __init__(self, responses):
        self._responses = list(responses)

    def call_api(self, prompt):
        return self._responses.pop(0)


class CorrectClassProbTests(unittest.TestCase):
    def test_compensates_for_lower_positive_survival(self):
        # Positives survive half as often, so ask for proportionally more.
        attrition = {"SPAM": {"attempted": 100, "survived": 50},
                     "HAM": {"attempted": 100, "survived": 100}}
        out = correct_class_prob(0.5, attrition, n=150, **_LABELS)
        self.assertAlmostEqual(out, 2 / 3, places=6)

    def test_equal_survival_needs_no_correction(self):
        attrition = {"SPAM": {"attempted": 100, "survived": 80},
                     "HAM": {"attempted": 100, "survived": 80}}
        self.assertIsNone(correct_class_prob(0.3, attrition, n=160, **_LABELS))

    def test_noise_floor_suppresses_tiny_deviations(self):
        # 49/100 vs a 0.5 target is well inside 2*sqrt(p(1-p)/n).
        attrition = {"SPAM": {"attempted": 100, "survived": 98},
                     "HAM": {"attempted": 100, "survived": 100}}
        self.assertIsNone(correct_class_prob(0.5, attrition, n=198, **_LABELS))

    def test_zero_survival_for_a_class_returns_none(self):
        attrition = {"SPAM": {"attempted": 40, "survived": 0},
                     "HAM": {"attempted": 60, "survived": 60}}
        self.assertIsNone(correct_class_prob(0.4, attrition, n=60, **_LABELS))

    def test_result_is_clamped_to_a_probability(self):
        attrition = {"SPAM": {"attempted": 100, "survived": 1},
                     "HAM": {"attempted": 100, "survived": 100}}
        out = correct_class_prob(0.9, attrition, n=101, **_LABELS)
        self.assertGreater(out, 0.0)
        self.assertLessEqual(out, 1.0)


class AttritionRecordingTests(unittest.TestCase):
    def _run(self, responses, class_prob):
        gen = _Scripted(responses)
        seeds = [{"incorrect": f"let us meet at three tomorrow {i}"}
                 for i in range(len(responses))]
        gen.generate_class_conditional(
            real_seeds=seeds, seed_field="incorrect", class_prob=class_prob,
            type_dist={"phishing_link": 1.0}, count_dist={1: 1.0},
            error_descriptions=_DESC, inject_prompt=_INJECT, negative_prompt=_HAM,
            positive_label="SPAM", negative_label="HAM",
            sample_size=len(responses), rng=Random(0),
        )
        return gen.last_class_attrition

    def test_counts_attempted_and_survived_per_class(self):
        out = self._run(["Corrupted: click http://x.example to win cash now",
                         "Corrupted: another http://y.example offer here"], 1.0)
        self.assertEqual(out["SPAM"], {"attempted": 2, "survived": 2})
        self.assertEqual(out["HAM"], {"attempted": 0, "survived": 0})

    def test_refusal_counts_as_attempted_but_not_survived(self):
        out = self._run(["I'm sorry, I can't help create spam messages."], 1.0)
        self.assertEqual(out["SPAM"], {"attempted": 1, "survived": 0})

    def test_negative_class_attrition_tracked_separately(self):
        # A paraphrase that comes back identical to its seed is dropped.
        out = self._run(["Rewritten: let us meet at three tomorrow 0"], 0.0)
        self.assertEqual(out["HAM"], {"attempted": 1, "survived": 0})
