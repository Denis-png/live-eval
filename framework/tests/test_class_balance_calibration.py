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
            real_seeds=seeds, seed_field="incorrect",
            class_balance={"SPAM": class_prob, "HAM": 1 - class_prob},
            labels=("SPAM", "HAM"),
            inverse_prompts={"SPAM": _INJECT, "HAM": _HAM},
            type_dist={"phishing_link": 1.0}, count_dist={1: 1.0},
            error_descriptions=_DESC, sample_size=len(responses),
            seed_policy="impose", rng=Random(0),
        )
        return gen.last_class_attrition

    def test_counts_attempted_and_survived_per_class(self):
        out = self._run(["Message: click http://x.example to win cash now",
                         "Message: another http://y.example offer here"], 1.0)
        self.assertEqual(out["SPAM"], {"attempted": 2, "survived": 2})
        self.assertEqual(out["HAM"], {"attempted": 0, "survived": 0})

    def test_refusal_counts_as_attempted_but_not_survived(self):
        out = self._run(["I'm sorry, I can't help create spam messages."], 1.0)
        self.assertEqual(out["SPAM"], {"attempted": 1, "survived": 0})

    def test_negative_class_attrition_tracked_separately(self):
        # A rewrite that comes back identical to its seed is dropped.
        out = self._run(["Message: let us meet at three tomorrow 0"], 0.0)
        self.assertEqual(out["HAM"], {"attempted": 1, "survived": 0})


class CorrectClassBalanceTests(unittest.TestCase):
    """N-label generalisation of correct_class_prob.

    corrected[label] proportional to target[label] / survival[label], normalised.
    It must reduce exactly to the binary form for two labels, so the spam
    behaviour shipped today is preserved rather than re-derived.
    """

    def test_reduces_to_the_binary_result_for_two_labels(self):
        from framework.calibration.class_balance import correct_class_balance
        # The binary case pinned by test_compensates_for_lower_positive_survival:
        # f=0.5, s_pos=0.5, s_neg=1.0 -> 2/3.
        attrition = {"SPAM": {"attempted": 100, "survived": 50},
                     "HAM": {"attempted": 100, "survived": 100}}
        out = correct_class_balance({"SPAM": 0.5, "HAM": 0.5}, attrition, n=150)
        self.assertAlmostEqual(out["SPAM"], 2 / 3, places=6)
        self.assertAlmostEqual(out["HAM"], 1 / 3, places=6)

    def test_three_labels_are_corrected_independently(self):
        from framework.calibration.class_balance import correct_class_balance
        # NEUTRAL survives half as often as the others, so it must be asked for
        # roughly twice as much relative to them.
        attrition = {"NEGATIVE": {"attempted": 100, "survived": 90},
                     "NEUTRAL": {"attempted": 100, "survived": 45},
                     "POSITIVE": {"attempted": 100, "survived": 90}}
        target = {"NEGATIVE": 1 / 3, "NEUTRAL": 1 / 3, "POSITIVE": 1 / 3}
        out = correct_class_balance(target, attrition, n=225)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=9)
        self.assertAlmostEqual(out["NEUTRAL"] / out["NEGATIVE"], 2.0, places=6)
        self.assertAlmostEqual(out["NEGATIVE"], out["POSITIVE"], places=9)

    def test_equal_survival_needs_no_correction(self):
        from framework.calibration.class_balance import correct_class_balance
        attrition = {"A": {"attempted": 100, "survived": 80},
                     "B": {"attempted": 100, "survived": 80},
                     "C": {"attempted": 100, "survived": 80}}
        self.assertIsNone(correct_class_balance(
            {"A": 0.5, "B": 0.3, "C": 0.2}, attrition, n=240))

    def test_label_with_zero_survivors_is_left_uncorrected(self):
        from framework.calibration.class_balance import correct_class_balance
        # Its survival rate is unknown, so it keeps its target share while the
        # others are corrected around it.
        attrition = {"A": {"attempted": 100, "survived": 0},
                     "B": {"attempted": 100, "survived": 50},
                     "C": {"attempted": 100, "survived": 100}}
        out = correct_class_balance({"A": 0.2, "B": 0.4, "C": 0.4}, attrition, n=150)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=9)
        self.assertGreater(out["B"], out["C"])

    def test_all_labels_inside_the_noise_floor_returns_none(self):
        from framework.calibration.class_balance import correct_class_balance
        attrition = {"A": {"attempted": 1000, "survived": 990},
                     "B": {"attempted": 1000, "survived": 1000}}
        self.assertIsNone(correct_class_balance(
            {"A": 0.5, "B": 0.5}, attrition, n=1990))

    def test_empty_or_single_label_target_returns_none(self):
        from framework.calibration.class_balance import correct_class_balance
        self.assertIsNone(correct_class_balance({}, {}, n=10))
        self.assertIsNone(correct_class_balance(
            {"A": 1.0}, {"A": {"attempted": 10, "survived": 10}}, n=10))
