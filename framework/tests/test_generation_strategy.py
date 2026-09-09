import unittest

from framework.tasks.spam.task import SpamTask
from framework.tasks.gec.task import GECTask
from framework.tasks.taxonomy.task import TaxonomyTask


class GenerationStrategyTests(unittest.TestCase):
    def test_spam_is_class_conditional(self):
        self.assertEqual(SpamTask().get_generation_strategy(), "class_conditional")

    def test_gec_is_corruption(self):
        self.assertEqual(GECTask().get_generation_strategy(), "corruption")

    def test_taxonomy_is_structured(self):
        self.assertEqual(TaxonomyTask().get_generation_strategy(), "structured")


if __name__ == "__main__":
    unittest.main()
