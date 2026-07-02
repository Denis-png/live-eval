import unittest

from framework.tasks.gec.task import GECTask


class GECInverseTests(unittest.TestCase):
    def setUp(self):
        self.task = GECTask()

    def test_inverse_prompt_has_required_placeholders(self):
        p = self.task.get_inverse_prompt()
        self.assertIn("{sentence}", p)
        self.assertIn("{error_spec}", p)

    def test_inverse_judge_prompt_has_required_placeholders(self):
        p = self.task.get_inverse_judge_prompt()
        self.assertIn("{sentence}", p)
        self.assertIn("{correction}", p)

    def test_error_descriptions_compose_operation_and_category(self):
        d = self.task.get_error_descriptions()
        self.assertEqual(d["R:VERB:TENSE"], "use a wrong verb tense")
        self.assertEqual(d["M:DET"], "omit a required article/determiner")
        self.assertEqual(d["U:PUNCT"], "add an unnecessary punctuation")

    def test_error_descriptions_cover_all_supported_types(self):
        d = self.task.get_error_descriptions()
        self.assertEqual(set(d.keys()), set(self.task._config["errant_supported_types"]))


if __name__ == "__main__":
    unittest.main()
