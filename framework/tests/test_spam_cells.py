import unittest
from unittest import mock

from framework.pipeline import _run_generation
from framework.tasks.spam.task import SpamTask

PROFILE = {
    "profile_version": 2,
    "topics_per_label": {
        "HAM": {"topics": {"chat": {"fraction": 1.0, "description": "d"}}},
        "SPAM": {"topics": {"prizes": {"fraction": 1.0, "description": "d"}}},
    },
    "length_distributions_per_label": {
        "HAM": {"words": {"bins": {"6-10": 1.0}, "quantiles": {"p90": 9}}},
        "SPAM": {"words": {"bins": {"6-10": 1.0}, "quantiles": {"p90": 9}}},
    },
    "style_per_label": {"HAM": {}, "SPAM": {}},
}
DIST = {"type_dist": {"phishing_link": 1.0}, "count_dist": {1: 1.0}}


def _config(mode, seedless):
    return {"generation": {"mode": mode, "seedless": seedless, "sample_size": 2}}


class SpamCellDispatchTests(unittest.TestCase):
    def setUp(self):
        self.task = SpamTask()
        self.generator = mock.Mock()
        self.generator.generate_class_conditional.return_value = [
            {"text": "t", "label": "HAM"}
        ]

    def _policy(self):
        return self.generator.generate_class_conditional.call_args.kwargs["seed_policy"]

    def test_inverse_seeded_uses_cross_class_with_real_data(self):
        real = [{"incorrect": "see you at lunch"}]
        _run_generation(self.generator, self.task, _config("inverse", False), real,
                        DIST, None, 0.5, profile=None)
        self.assertEqual(self._policy(), "cross_class")
        kwargs = self.generator.generate_class_conditional.call_args.kwargs
        self.assertEqual(kwargs["real_seeds"], real)
        self.assertFalse(self.generator.generate_carriers.called)

    def test_inverse_seedless_feeds_carriers_as_seeds(self):
        self.generator.generate_carriers.return_value = ["a synthetic ham message"]
        _run_generation(self.generator, self.task, _config("inverse", True), [],
                        DIST, None, 0.5, profile=PROFILE)
        self.assertTrue(self.generator.generate_carriers.called)
        kwargs = self.generator.generate_class_conditional.call_args.kwargs
        self.assertEqual(kwargs["seed_policy"], "cross_class")
        self.assertEqual(kwargs["real_seeds"], [{"text": "a synthetic ham message"}])

    def test_forward_seeded_uses_same_class_with_labeled_pool(self):
        rows = [{"text": "hi", "label": "HAM"}, {"text": "WIN", "label": "SPAM"}]
        with mock.patch.object(SpamTask, "_load_reference_rows", return_value=rows):
            _run_generation(self.generator, self.task, _config("forward", False),
                            [{"incorrect": "x"}], DIST, None, 0.5, profile=None)
        kwargs = self.generator.generate_class_conditional.call_args.kwargs
        self.assertEqual(kwargs["seed_policy"], "same_class")
        self.assertEqual(kwargs["real_seeds"], rows)

    def test_forward_seedless_uses_none_policy_with_per_label_specs(self):
        _run_generation(self.generator, self.task, _config("forward", True), [],
                        DIST, None, 0.5, profile=PROFILE)
        kwargs = self.generator.generate_class_conditional.call_args.kwargs
        self.assertEqual(kwargs["seed_policy"], "none")
        self.assertEqual(set(kwargs["specs_by_label"]), {"SPAM", "HAM"})
        self.assertTrue(all(len(v) == 2 for v in kwargs["specs_by_label"].values()))
        self.assertFalse(self.generator.generate_carriers.called)


if __name__ == "__main__":
    unittest.main()
