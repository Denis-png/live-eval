import json
import unittest
from unittest import mock

from framework.evaluators.taxonomy.metrics import score_taxonomy_result
from framework.models.taxonomy import TaxonomyLLMModel
from framework.tasks.taxonomy import TaxonomyTask


class FakeGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def call_api(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _model_input(domain="pizza", classes=None):
    return json.dumps({
        "domain": domain,
        "classes": classes or ["Root", "A", "B"],
    })


class TaxonomyLLMModelTests(unittest.TestCase):
    def _model(self, responses):
        fake = FakeGenerator(responses)
        with mock.patch("framework.pipeline.load_generator", return_value=fake):
            model = TaxonomyLLMModel({
                "type": "llm",
                "name": "mock-model",
                "provider": "openrouter",
                "api_key": "test-key",
            })
        return model, fake

    def test_prompt_contains_domain_and_class_list(self):
        model, _ = self._model(['{"subclass_axioms": []}'])
        prompt = model.build_prompt(_model_input(domain="pizza", classes=["Pizza", "NamedPizza"]))

        self.assertIn("pizza", prompt)
        self.assertIn("Pizza", prompt)
        self.assertIn("NamedPizza", prompt)

    def test_prompt_does_not_contain_gold_or_metadata(self):
        task = TaxonomyTask()
        parsed = task.parse_row({
            "ontology_id": "tiny",
            "domain": "pizza",
            "classes": ["Pizza", "NamedPizza"],
            "subclass_axioms": [["NamedPizza", "Pizza"]],
            "metadata": {"class_uri_map": {"Pizza": "http://example.org/Pizza"}},
        })
        sample = task.get_eval_samples([parsed])[0]
        model, _ = self._model(['{"subclass_axioms": []}'])
        prompt = model.build_prompt(sample["text"])

        self.assertNotIn('["NamedPizza", "Pizza"]', prompt)
        self.assertNotIn("class_uri_map", prompt)
        self.assertNotIn("http://example.org/Pizza", prompt)

    def test_strict_json_prediction_parses_correctly(self):
        model, _ = self._model(['{"subclass_axioms": [["A", "Root"], ["B", "Root"]]}'])
        prediction = model.predict([_model_input()])[0]

        self.assertEqual(prediction["subclass_axioms"], [["A", "Root"], ["B", "Root"]])
        self.assertFalse(prediction["diagnostics"]["malformed"])

    def test_malformed_model_response_fails_safely(self):
        model, _ = self._model(["not json"])
        prediction = model.predict([_model_input()])[0]

        self.assertEqual(prediction["subclass_axioms"], [])
        self.assertTrue(prediction["diagnostics"]["malformed"])

    def test_duplicate_relations_are_normalized(self):
        model, _ = self._model([
            '{"subclass_axioms": [["A", "Root"], ["A", "Root"], ["B", "Root"]]}'
        ])
        prediction = model.predict([_model_input()])[0]

        self.assertEqual(prediction["subclass_axioms"], [["A", "Root"], ["B", "Root"]])

    def test_relation_direction_remains_child_parent(self):
        model, _ = self._model(['{"subclass_axioms": [["Root", "A"]]}'])
        prediction = model.predict([_model_input()])[0]

        self.assertEqual(prediction["subclass_axioms"], [["Root", "A"]])

    def test_batch_predict_preserves_order(self):
        model, fake = self._model([
            '{"subclass_axioms": [["A", "Root"]]}',
            '{"subclass_axioms": [["B", "Root"]]}',
        ])
        predictions = model.predict([
            _model_input(domain="first"),
            _model_input(domain="second"),
        ])

        self.assertEqual(predictions[0]["subclass_axioms"], [["A", "Root"]])
        self.assertEqual(predictions[1]["subclass_axioms"], [["B", "Root"]])
        self.assertIn("first", fake.prompts[0])
        self.assertIn("second", fake.prompts[1])

    def test_taxonomy_task_get_model_uses_llm_wrapper(self):
        fake = FakeGenerator(['{"subclass_axioms": []}'])
        with mock.patch("framework.pipeline.load_generator", return_value=fake):
            model = TaxonomyTask().get_model({
                "type": "llm",
                "name": "mock-model",
                "provider": "openrouter",
                "api_key": "test-key",
            })
        self.assertIsInstance(model, TaxonomyLLMModel)

    def test_perfect_fixture_evaluation(self):
        model, _ = self._model(['{"subclass_axioms": [["A", "Root"], ["B", "Root"]]}'])
        prediction = model.predict([_model_input()])[0]
        score = score_taxonomy_result({
            "classes": ["Root", "A", "B"],
            "subclass_axioms": [["A", "Root"], ["B", "Root"]],
            "prediction": prediction,
        })

        self.assertEqual(score["precision"], 1.0)
        self.assertEqual(score["recall"], 1.0)
        self.assertEqual(score["f1"], 1.0)

    def test_imperfect_fixture_evaluation(self):
        model, _ = self._model(['{"subclass_axioms": [["A", "Root"], ["Root", "B"]]}'])
        prediction = model.predict([_model_input()])[0]
        score = score_taxonomy_result({
            "classes": ["Root", "A", "B"],
            "subclass_axioms": [["A", "Root"], ["B", "Root"]],
            "prediction": prediction,
        })

        self.assertEqual(score["tp"], 1)
        self.assertEqual(score["fp"], 1)
        self.assertEqual(score["fn"], 1)
        self.assertEqual(score["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
