import unittest

from framework.tasks.base_task import BaseTask
from framework.tasks.gec.task import GECTask


class MinimalTask(BaseTask):
    def get_error_types(self): return []
    def get_prompt_instruction(self): return ""
    def get_evaluators(self): return []
    def get_evaluator_fns(self): return {}
    def get_model(self, model_config): return None
    def get_task_name(self): return "min"
    def parse_row(self, row): return None


class BaseTaskDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.task = MinimalTask()

    def test_generation_accessors_default_to_empty(self):
        self.assertIsNone(self.task.get_carrier_prompt())
        self.assertIsNone(self.task.get_seedless_forward_prompt())
        self.assertIsNone(self.task.get_forward_prompt())
        self.assertEqual(self.task.get_seedless_class_prompts(), {})

    def test_profile_side_defaults_to_correct(self):
        self.assertEqual(self.task.get_profile_side("forward"), "correct")
        self.assertEqual(self.task.get_profile_side("inverse"), "correct")

    def test_seed_pool_defaults_to_real_data(self):
        rows = [{"incorrect": "a"}]
        self.assertIs(self.task.get_seed_pool({}, rows, "inverse"), rows)


class GecPromptTests(unittest.TestCase):
    def setUp(self):
        self.task = GECTask()

    def test_carrier_prompt_has_spec_placeholder_and_tag(self):
        prompt = self.task.get_carrier_prompt()
        self.assertIn("{spec}", prompt)
        self.assertIn("Sentence:", prompt)

    def test_seedless_forward_prompt_has_both_placeholders_and_three_fields(self):
        prompt = self.task.get_seedless_forward_prompt()
        self.assertIn("{spec}", prompt)
        self.assertIn("{error_spec}", prompt)
        for field in ("Error type:", "Generated:", "Ground truth:"):
            self.assertIn(field, prompt)
        self.assertNotIn("{sentence}", prompt)

    def test_profile_side_follows_mode(self):
        self.assertEqual(self.task.get_profile_side("forward"), "incorrect")
        self.assertEqual(self.task.get_profile_side("inverse"), "correct")


if __name__ == "__main__":
    unittest.main()
