import unittest
from random import Random

from framework.calibration.class_balance import correct_class_balance
from framework.generators.base_generator import BaseGenerator

_INJECT = "Make SPAM using {error_spec} from: {sentence}"
_HAM = "Rewrite legitimately: {sentence}"
_DESC = {"phishing_link": "insert a suspicious link"}


class _Scripted(BaseGenerator):
    def __init__(self, responses):
        self._responses = list(responses)

    def call_api(self, prompt):
        return self._responses.pop(0)


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
    """The only class-balance correction there is.

    corrected[label] proportional to target[label] / survival[label], normalised.
    It must reduce exactly to the binary closed form for two labels, so the spam
    behaviour shipped today is preserved rather than re-derived — the two-label
    cases below are the coverage the superseded binary helper carried, kept
    verbatim in intent when its own tests were removed with it.
    """

    def test_reduces_to_the_binary_result_for_two_labels(self):
        # The binary case the removed helper pinned as
        # test_compensates_for_lower_positive_survival:
        # f=0.5, s_pos=0.5, s_neg=1.0 -> 2/3.
        attrition = {"SPAM": {"attempted": 100, "survived": 50},
                     "HAM": {"attempted": 100, "survived": 100}}
        out = correct_class_balance({"SPAM": 0.5, "HAM": 0.5}, attrition, n=150)
        self.assertAlmostEqual(out["SPAM"], 2 / 3, places=6)
        self.assertAlmostEqual(out["HAM"], 1 / 3, places=6)

    def test_three_labels_are_corrected_independently(self):
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
        attrition = {"A": {"attempted": 100, "survived": 80},
                     "B": {"attempted": 100, "survived": 80},
                     "C": {"attempted": 100, "survived": 80}}
        self.assertIsNone(correct_class_balance(
            {"A": 0.5, "B": 0.3, "C": 0.2}, attrition, n=240))

    def test_label_with_zero_survivors_is_left_uncorrected(self):
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
        attrition = {"A": {"attempted": 1000, "survived": 990},
                     "B": {"attempted": 1000, "survived": 1000}}
        self.assertIsNone(correct_class_balance(
            {"A": 0.5, "B": 0.5}, attrition, n=1990))
        # Ported from the removed binary helper's own noise-floor case: the
        # floor is 2*sqrt(p(1-p)/n), so it widens as n shrinks — 49/100 vs a
        # 0.5 target is well inside it at n=198.
        small_n = {"A": {"attempted": 100, "survived": 98},
                   "B": {"attempted": 100, "survived": 100}}
        self.assertIsNone(correct_class_balance(
            {"A": 0.5, "B": 0.5}, small_n, n=198))

    def test_two_labels_with_equal_survival_need_no_correction(self):
        # Ported: the removed helper had an unequal-target/equal-survival case
        # at two labels; the equal-survival case above uses three.
        attrition = {"SPAM": {"attempted": 100, "survived": 80},
                     "HAM": {"attempted": 100, "survived": 80}}
        self.assertIsNone(correct_class_balance(
            {"SPAM": 0.3, "HAM": 0.7}, attrition, n=160))

    def test_two_labels_one_with_zero_survivors_is_still_corrected(self):
        # Ported, with the behaviour deliberately CHANGED: the binary helper
        # returned None whenever either class produced nothing, discarding the
        # round. The vector form leaves the unknown label at its raw target
        # share and corrects the label whose rate IS known around it, which is
        # strictly more information. Asserted at two labels because that is
        # where the old None came from.
        attrition = {"SPAM": {"attempted": 40, "survived": 0},
                     "HAM": {"attempted": 60, "survived": 30}}
        out = correct_class_balance({"SPAM": 0.4, "HAM": 0.6}, attrition, n=60)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=9)
        self.assertAlmostEqual(out["SPAM"], 0.25, places=6)
        self.assertAlmostEqual(out["HAM"], 0.75, places=6)

    def test_an_extreme_survival_ratio_stays_a_valid_distribution(self):
        # Ported from test_result_is_clamped_to_a_probability: the binary form
        # clamped explicitly, the vector form normalises, so the invariant to
        # hold is that every weight is a usable probability and they sum to 1
        # even when one label survives 1% of the time at a 0.9 target.
        attrition = {"SPAM": {"attempted": 100, "survived": 1},
                     "HAM": {"attempted": 100, "survived": 100}}
        out = correct_class_balance({"SPAM": 0.9, "HAM": 0.1}, attrition, n=101)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=9)
        for label, weight in out.items():
            self.assertGreater(weight, 0.0, label)
            self.assertLessEqual(weight, 1.0, label)
        # The starved label is asked for far more, which is the point.
        self.assertGreater(out["SPAM"], 0.99)

    def test_empty_or_single_label_target_returns_none(self):
        self.assertIsNone(correct_class_balance({}, {}, n=10))
        self.assertIsNone(correct_class_balance(
            {"A": 1.0}, {"A": {"attempted": 10, "survived": 10}}, n=10))
