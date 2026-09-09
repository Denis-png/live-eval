import unittest

from framework.calibration.controller import (
    converged,
    jsd_report,
    project_measured,
    select_best,
    stalled,
    update_request,
    worst,
)


class ProjectMeasuredTests(unittest.TestCase):
    def test_restricts_to_target_keys_and_renormalizes(self):
        # ERRANT reports types outside the supported vocabulary; they must not
        # dilute the measurement or every ratio is biased downward.
        target = {"a": 0.5, "b": 0.5}
        measured = {"a": 0.25, "b": 0.25, "OTHER": 0.5}
        self.assertEqual(project_measured(target, measured), {"a": 0.5, "b": 0.5})

    def test_missing_key_projects_to_zero(self):
        self.assertEqual(
            project_measured({"a": 0.5, "b": 0.5}, {"a": 1.0}),
            {"a": 1.0, "b": 0.0},
        )

    def test_all_zero_measurement_stays_zero(self):
        self.assertEqual(
            project_measured({"a": 0.5, "b": 0.5}, {"OTHER": 1.0}),
            {"a": 0.0, "b": 0.0},
        )

    def test_negative_weights_clamped(self):
        self.assertEqual(project_measured({"a": 1.0}, {"a": -3.0}), {"a": 0.0})


class UpdateRequestTests(unittest.TestCase):
    def test_inverts_known_compliance_bias_in_one_step(self):
        # Generator honours "a" twice as often as "b". Asked for 50/50 it
        # delivers 2:1, so the next request must tilt toward "b" by 2x.
        target = {"a": 0.5, "b": 0.5}
        request = {"a": 0.5, "b": 0.5}
        measured = {"a": 2 / 3, "b": 1 / 3}
        out = update_request(request, target, measured, alpha=1.0, epsilon=0.0)
        self.assertAlmostEqual(out["b"] / out["a"], 2.0, places=6)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=9)

    def test_alpha_damps_the_step(self):
        target = {"a": 0.5, "b": 0.5}
        request = {"a": 0.5, "b": 0.5}
        measured = {"a": 2 / 3, "b": 1 / 3}
        out = update_request(request, target, measured, alpha=0.5, epsilon=0.0)
        # sqrt(2) instead of 2 — half the correction in log space.
        self.assertAlmostEqual(out["b"] / out["a"], 2.0 ** 0.5, places=6)

    def test_perfect_measurement_is_a_fixed_point(self):
        target = {"a": 0.7, "b": 0.3}
        out = update_request(dict(target), target, dict(target), alpha=0.5, epsilon=0.0)
        self.assertAlmostEqual(out["a"], 0.7, places=6)
        self.assertAlmostEqual(out["b"], 0.3, places=6)

    def test_zero_target_stays_zero(self):
        # Calibration must never introduce a category the benchmark lacks.
        target = {"a": 1.0, "b": 0.0}
        out = update_request({"a": 0.5, "b": 0.5}, target, {"a": 0.5, "b": 0.5})
        self.assertEqual(out["b"], 0.0)

    def test_zero_measured_category_recovers_rather_than_stranding(self):
        # "b" was never delivered. It must not divide by zero, and its next
        # request must go UP, not stay at zero forever.
        target = {"a": 0.5, "b": 0.5}
        out = update_request({"a": 0.5, "b": 0.5}, target, {"a": 1.0, "b": 0.0})
        self.assertGreater(out["b"], 0.5)
        self.assertLess(out["a"], 0.5)

    def test_zero_request_entry_is_floored_not_stranded(self):
        # Defensive: a hand-edited artifact could zero a still-wanted category.
        target = {"a": 0.5, "b": 0.5}
        out = update_request({"a": 1.0, "b": 0.0}, target, {"a": 1.0, "b": 0.0})
        self.assertGreater(out["b"], 0.0)

    def test_all_zero_measurement_falls_back_to_target(self):
        target = {"a": 0.5, "b": 0.5}
        out = update_request({"a": 0.5, "b": 0.5}, target, {})
        self.assertEqual(out, target)

    def test_int_keyed_count_dist_preserved(self):
        target = {1: 0.5, 2: 0.5}
        out = update_request({1: 0.5, 2: 0.5}, target, {1: 0.8, 2: 0.2}, alpha=1.0,
                             epsilon=0.0)
        self.assertEqual(sorted(out), [1, 2])
        self.assertGreater(out[2], out[1])
        self.assertTrue(all(type(k) is int for k in out))

    def test_epsilon_zero_with_zero_measured_does_not_divide_by_zero(self):
        # epsilon=0.0 with zero-measured category must not raise ZeroDivisionError
        # and the zero-measured category's request must increase.
        target = {"a": 0.5, "b": 0.5}
        out = update_request({"a": 0.5, "b": 0.5}, target, {"a": 1.0, "b": 0.0},
                             alpha=0.5, epsilon=0.0)
        self.assertGreater(out["b"], 0.5)
        self.assertLess(out["a"], 0.5)

    def test_epsilon_zero_with_zero_request_entry_recovers(self):
        # epsilon=0.0 with zero request entry must not strand it when target is positive.
        target = {"a": 0.5, "b": 0.5}
        out = update_request({"a": 1.0, "b": 0.0}, target, {"a": 1.0, "b": 0.0},
                             alpha=0.5, epsilon=0.0)
        self.assertGreater(out["b"], 0.0)


class ReportTests(unittest.TestCase):
    def test_jsd_report_is_zero_for_identical_projected_distributions(self):
        targets = {"type_dist": {"a": 0.5, "b": 0.5}, "count_dist": {1: 1.0}}
        measured = {"type_dist": {"a": 0.5, "b": 0.5, "OTHER": 9.0}, "count_dist": {1: 1.0}}
        report = jsd_report(targets, measured)
        self.assertAlmostEqual(report["type_dist"], 0.0, places=9)
        self.assertAlmostEqual(report["count_dist"], 0.0, places=9)

    def test_jsd_report_flags_divergence(self):
        targets = {"type_dist": {"a": 1.0, "b": 0.0}}
        measured = {"type_dist": {"a": 0.0, "b": 1.0}}
        self.assertGreater(jsd_report(targets, measured)["type_dist"], 0.9)

    def test_worst_takes_the_maximum_dimension(self):
        self.assertEqual(worst({"type_dist": 0.05, "count_dist": 0.31}), 0.31)

    def test_converged_requires_every_dimension_inside_tolerance(self):
        self.assertTrue(converged({"type_dist": 0.05, "count_dist": 0.09}, 0.1))
        self.assertFalse(converged({"type_dist": 0.05, "count_dist": 0.11}, 0.1))

    def test_converged_on_empty_report_is_false(self):
        # Nothing measured is not the same as everything matching.
        self.assertFalse(converged({}, 0.1))


class SelectionTests(unittest.TestCase):
    def test_select_best_picks_lowest_worst_jsd(self):
        rounds = [
            {"jsd": {"type_dist": 0.30}},
            {"jsd": {"type_dist": 0.12}},
            {"jsd": {"type_dist": 0.20}},
        ]
        self.assertEqual(select_best(rounds), 1)

    def test_select_best_breaks_ties_toward_earlier_round(self):
        rounds = [{"jsd": {"t": 0.2}}, {"jsd": {"t": 0.2}}]
        self.assertEqual(select_best(rounds), 0)

    def test_select_best_can_return_round_zero(self):
        # Guarantees calibration never ships worse than uncalibrated behavior.
        rounds = [{"jsd": {"t": 0.1}}, {"jsd": {"t": 0.4}}, {"jsd": {"t": 0.5}}]
        self.assertEqual(select_best(rounds), 0)

    def test_stalled_when_best_unimproved_for_patience_rounds(self):
        rounds = [{"jsd": {"t": 0.2}}, {"jsd": {"t": 0.3}}, {"jsd": {"t": 0.25}}]
        self.assertTrue(stalled(rounds, patience=2))

    def test_not_stalled_while_still_improving(self):
        rounds = [{"jsd": {"t": 0.3}}, {"jsd": {"t": 0.25}}, {"jsd": {"t": 0.2}}]
        self.assertFalse(stalled(rounds, patience=2))

    def test_not_stalled_before_patience_rounds_elapse(self):
        self.assertFalse(stalled([{"jsd": {"t": 0.3}}], patience=2))
