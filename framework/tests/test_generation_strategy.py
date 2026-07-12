import unittest

from framework.tasks.spam.task import SpamTask
from framework.tasks.gec.task import GECTask


class GenerationStrategyTests(unittest.TestCase):
    def test_spam_is_class_conditional(self):
        self.assertEqual(SpamTask().get_generation_strategy(), "class_conditional")

    def test_gec_is_corruption(self):
        self.assertEqual(GECTask().get_generation_strategy(), "corruption")

    def test_spam_ham_prompt_has_placeholder_and_tag(self):
        p = SpamTask().get_ham_generation_prompt()
        self.assertIsNotNone(p)
        self.assertIn("{sentence}", p)
        self.assertIn("Rewritten:", p)


if __name__ == "__main__":
    unittest.main()
