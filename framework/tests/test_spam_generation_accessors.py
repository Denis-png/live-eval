import unittest
from unittest import mock

from framework.tasks.spam.task import SpamTask


class SpamPromptTests(unittest.TestCase):
    def setUp(self):
        self.task = SpamTask()

    def test_carrier_prompt_asks_for_a_legitimate_message(self):
        prompt = self.task.get_carrier_prompt()
        self.assertIn("{spec}", prompt)
        self.assertIn("Message:", prompt)

    def test_forward_prompts_cover_both_labels(self):
        prompts = self.task.get_forward_prompts()
        self.assertEqual(set(prompts), {"SPAM", "HAM"})
        for prompt in prompts.values():
            self.assertIn("{sentence}", prompt)
            self.assertIn("Rewritten:", prompt)

    def test_only_the_positive_forward_prompt_takes_a_signal_mix(self):
        # Forward SPAM targets the empirical signal distribution rather than
        # inheriting whatever its seed carried; HAM has no signals to emphasise.
        prompts = self.task.get_forward_prompts()
        self.assertIn("{error_spec}", prompts["SPAM"])
        self.assertNotIn("{error_spec}", prompts["HAM"])

    def test_superseded_prose_profile_prompts_are_gone(self):
        # dataset_profiling/profile_inject/profile_ham/spam_to_spam described an
        # earlier prose-profile strategy that the two-knob model replaced.
        for key in ("dataset_profiling_prompt", "profile_inject_prompt",
                    "profile_ham_prompt", "spam_to_spam_prompt"):
            self.assertNotIn(key, self.task._config)

    def test_seedless_class_prompts_cover_both_labels(self):
        prompts = self.task.get_seedless_class_prompts()
        self.assertEqual(set(prompts), {"SPAM", "HAM"})
        self.assertIn("{error_spec}", prompts["SPAM"])
        self.assertIn("{spec}", prompts["HAM"])
        self.assertNotIn("{error_spec}", prompts["HAM"])
        for prompt in prompts.values():
            self.assertIn("Message:", prompt)

    def test_error_types_have_no_duplicates(self):
        types = self.task.get_error_types()
        self.assertEqual(len(types), len(set(types)))


class SpamSeedPoolTests(unittest.TestCase):
    def test_forward_mode_returns_labeled_rows(self):
        task = SpamTask()
        rows = [{"text": "hi", "label": "HAM"}, {"text": "WIN", "label": "SPAM"}]
        with mock.patch.object(SpamTask, "_load_reference_rows", return_value=rows):
            pool = task.get_seed_pool({"dataset": {}}, [{"incorrect": "x"}], "forward")
        self.assertEqual(pool, rows)

    def test_inverse_mode_returns_real_data_unchanged(self):
        task = SpamTask()
        real = [{"incorrect": "x"}]
        self.assertIs(task.get_seed_pool({"dataset": {}}, real, "inverse"), real)


if __name__ == "__main__":
    unittest.main()
