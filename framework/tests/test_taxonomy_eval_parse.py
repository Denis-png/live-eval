"""The taxonomy EVALUATION parser must read a reasoning model's answer too.

The first complete taxonomy run scored minimax-m3 at exactly 0.000 precision,
recall and F1 -- on the real Pizza ontology as well as the synthetic ones, where
a lexical baseline reaches 0.21. Every prediction was marked malformed. The model
was answering correctly: it opened <think>, never closed it, and ended on a
fenced JSON answer. parse_prediction_relations ran json.loads on the whole
response, the same defect already fixed in the generation parser.

Left in place, the final report would have shown an LLM scoring below a
do-nothing star baseline. The two parsers now share one extractor, so they
cannot drift apart again.
"""
import unittest

from framework.evaluators.taxonomy.metrics import (
    compute_taxonomy_scores,
    parse_prediction_relations,
)
from framework.generators.base_generator import extract_json_object

# Verbatim minimax-m3 task-model output: an UNCLOSED <think> running into a
# fenced answer. It correctly excludes the transitive Mozzarella -> PizzaTopping.
REAL_TASK_MODEL_RESPONSE = '<think>\nThe user wants me to infer direct subclass relationships between the provided ontology classes in the pizza domain.\n\nLet me analyze the classes:\n- PizzaTopping: This seems to be the most general class - a topping for pizza\n- CheeseTopping: A type of pizza topping that is cheese - subclass of PizzaTopping\n- MozzarellaTopping: A type of cheese topping - subclass of CheeseTopping\n- MeatTopping: A type of pizza topping that is meat - subclass of PizzaTopping\n- HamTopping: A type of meat topping - subclass of MeatTopping\n\nDirect subclass relationships:\n1. CheeseTopping -> PizzaTopping (cheese is a kind of topping)\n2. MozzarellaTopping -> CheeseTopping (mozzarella is a kind of cheese topping)\n3. MeatTopping -> PizzaTopping (meat is a kind of topping)\n4. HamTopping -> MeatTopping (ham is a kind of meat topping)\n\nThese are all direct relationships. I should not include transitive ones like MozzarellaTopping -> PizzaTopping since that would be inferred through CheeseTopping.```json\n{\n  "subclass_axioms": [\n    ["CheeseTopping", "PizzaTopping"],\n    ["MozzarellaTopping", "CheeseTopping"],\n    ["MeatTopping", "PizzaTopping"],\n    ["HamTopping", "MeatTopping"]\n  ]\n}\n```'

_CLASSES = ["PizzaTopping", "CheeseTopping", "MozzarellaTopping",
            "MeatTopping", "HamTopping"]
_GOLD = [["CheeseTopping", "PizzaTopping"], ["MozzarellaTopping", "CheeseTopping"],
         ["MeatTopping", "PizzaTopping"], ["HamTopping", "MeatTopping"]]


class EvaluationParserTests(unittest.TestCase):
    def test_the_real_task_model_response_is_not_malformed(self):
        parsed = parse_prediction_relations(REAL_TASK_MODEL_RESPONSE, _CLASSES)
        self.assertFalse(parsed["malformed"])
        self.assertEqual(parsed["relations"], {tuple(r) for r in _GOLD})

    def test_it_scores_as_the_correct_answer_it_is(self):
        # The point of the fix, in the metric the report actually shows.
        scores = compute_taxonomy_scores([{
            "classes": _CLASSES,
            "subclass_axioms": _GOLD,
            "prediction": REAL_TASK_MODEL_RESPONSE,
        }])
        self.assertEqual(scores["f1"], 1.0)

    def test_plain_json_still_parses(self):
        parsed = parse_prediction_relations(
            '{"subclass_axioms": [["B", "A"]]}', ["A", "B"])
        self.assertFalse(parsed["malformed"])
        self.assertEqual(parsed["relations"], {("B", "A")})

    def test_a_dict_prediction_still_parses(self):
        parsed = parse_prediction_relations({"subclass_axioms": [["B", "A"]]}, ["A", "B"])
        self.assertEqual(parsed["relations"], {("B", "A")})

    def test_prose_with_no_json_is_still_malformed(self):
        parsed = parse_prediction_relations("<think>I am not sure.", ["A", "B"])
        self.assertTrue(parsed["malformed"])
        self.assertEqual(parsed["relations"], set())


class SharedExtractorTests(unittest.TestCase):
    """One extractor serves generation and evaluation, so they cannot drift."""

    def test_the_generation_parser_delegates_to_the_shared_extractor(self):
        from framework.tasks.taxonomy import task as task_module
        text = '<think>x</think>{"domain": "d", "classes": ["A"], "subclass_axioms": []}'
        self.assertEqual(task_module._extract_json_object(text), extract_json_object(text))

    def test_the_last_top_level_object_wins(self):
        obj, reason = extract_json_object('<think>{"draft": 1} then {"final": 2}')
        self.assertEqual(obj, {"final": 2})
        self.assertIsNone(reason)

    def test_a_nested_dict_is_not_mistaken_for_the_answer(self):
        obj, _ = extract_json_object('{"outer": {"inner": 1}, "k": 2}')
        self.assertEqual(obj, {"outer": {"inner": 1}, "k": 2})


if __name__ == "__main__":
    unittest.main()
