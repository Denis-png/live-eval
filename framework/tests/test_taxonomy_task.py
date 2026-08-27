import json
import unittest

from framework.tasks.taxonomy import TaxonomyTask


ROW = {
    "ontology_id": "tiny",
    "domain": "pizza",
    "classes": ["Pizza", "VegetarianPizza", "MargheritaPizza"],
    "subclass_axioms": [
        ["VegetarianPizza", "Pizza"],
        ["MargheritaPizza", "Pizza"],
    ],
    "metadata": {
        "class_uri_map": {
            "Pizza": "http://example.org/Pizza",
            "VegetarianPizza": "http://example.org/VegetarianPizza",
            "MargheritaPizza": "http://example.org/MargheritaPizza",
        }
    },
}


class TaxonomyTaskTests(unittest.TestCase):
    def setUp(self):
        self.task = TaxonomyTask()

    def test_parse_normalized_taxonomy_row(self):
        parsed = self.task.parse_row(ROW)
        self.assertEqual(parsed["ontology_id"], "tiny")
        self.assertEqual(parsed["domain"], "pizza")
        self.assertEqual(parsed["classes"], ROW["classes"])
        self.assertEqual(
            parsed["subclass_axioms"],
            [["MargheritaPizza", "Pizza"], ["VegetarianPizza", "Pizza"]],
        )
        self.assertEqual(parsed["metadata"], ROW["metadata"])

    def test_parse_rejects_missing_required_shape(self):
        self.assertIsNone(self.task.parse_row({"domain": "pizza", "classes": []}))
        self.assertIsNone(self.task.parse_row({"classes": ["Pizza"]}))

    def test_eval_sample_contains_only_domain_and_classes_for_model(self):
        parsed = self.task.parse_row(ROW)
        sample = self.task.get_eval_samples([parsed])[0]
        payload = json.loads(sample["text"])

        self.assertEqual(payload, {
            "domain": "pizza",
            "classes": ["Pizza", "VegetarianPizza", "MargheritaPizza"],
        })
        self.assertEqual(sample["model_input"], payload)

    def test_eval_sample_does_not_leak_gold_or_uri_map_to_model_input(self):
        parsed = self.task.parse_row(ROW)
        sample = self.task.get_eval_samples([parsed])[0]
        payload = json.loads(sample["text"])

        self.assertNotIn("subclass_axioms", payload)
        self.assertNotIn("metadata", payload)
        self.assertNotIn("class_uri_map", payload)
        self.assertIn("subclass_axioms", sample)

    def test_compatibility_methods_do_not_invent_error_types(self):
        self.assertEqual(self.task.get_task_name(), "taxonomy")
        self.assertEqual(self.task.get_error_types(), [])
        with self.assertRaises(NotImplementedError):
            self.task.get_prompt_instruction()


if __name__ == "__main__":
    unittest.main()
