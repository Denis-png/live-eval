"""Deterministic taxonomy baselines: lexical head-match and star.

A benchmark evaluation needs a floor and a cheap strong baseline, or its scores
have no reference point. These two are free, exactly reproducible, and -- the
reason they matter for GET specifically -- immune to memorisation. An LLM can
score on a canonical ontology by recalling it; a baseline that only reads class
names cannot. That makes them a control for the real-vs-synthetic comparison.
"""
import json
import unittest

from framework.evaluators.taxonomy.metrics import compute_taxonomy_scores
from framework.models.taxonomy.baselines import (
    LexicalHeadMatchModel,
    StarModel,
    lexical_parents,
    name_tokens,
)
from framework.tasks.taxonomy.task import TaxonomyTask, serialize_taxonomy_model_input


def _axioms(model, domain, classes):
    return model.predict([serialize_taxonomy_model_input(domain, classes)])[0]["subclass_axioms"]


class TokenisationTests(unittest.TestCase):
    def test_camel_case_splits_into_lowercase_tokens(self):
        self.assertEqual(name_tokens("VegetarianPizza"), ["vegetarian", "pizza"])

    def test_spaces_and_underscores_split_the_same_way(self):
        # Re-verbalised synthetic taxonomies are named by a model, not by an
        # ontology engineer, so they may not use CamelCase at all.
        self.assertEqual(name_tokens("Vegetarian Pizza"), ["vegetarian", "pizza"])
        self.assertEqual(name_tokens("vegetarian_pizza"), ["vegetarian", "pizza"])

    def test_acronyms_do_not_explode_into_letters(self):
        self.assertEqual(name_tokens("HTTPServer"), ["http", "server"])


class LexicalHeadMatchTests(unittest.TestCase):
    def test_a_class_whose_name_ends_in_another_class_gets_that_parent(self):
        self.assertEqual(lexical_parents(["Pizza", "VegetarianPizza"]),
                         {"VegetarianPizza": "Pizza"})

    def test_the_longest_matching_suffix_wins(self):
        # The most specific named ancestor is the DIRECT parent; the shorter
        # suffix is a grandparent and would be a false positive.
        parents = lexical_parents(["Topping", "VegetableTopping",
                                   "CheeseyVegetableTopping"])
        self.assertEqual(parents["CheeseyVegetableTopping"], "VegetableTopping")
        self.assertEqual(parents["VegetableTopping"], "Topping")

    def test_a_class_with_no_matching_suffix_gets_no_parent(self):
        # Named pizzas like "Margherita" do not end in "Pizza". The baseline
        # must leave them unattached rather than guess -- that limit is the
        # honest behaviour of a lexical baseline, and the gap it leaves is
        # exactly what the LLMs are measured against.
        self.assertNotIn("Margherita", lexical_parents(["Pizza", "Margherita"]))

    def test_a_class_is_never_its_own_parent(self):
        self.assertEqual(lexical_parents(["Pizza"]), {})

    def test_it_is_deterministic_regardless_of_input_order(self):
        names = ["Topping", "CheeseTopping", "VegetableTopping", "MozzarellaTopping"]
        self.assertEqual(lexical_parents(names), lexical_parents(list(reversed(names))))

    def test_predictions_only_name_supplied_classes(self):
        model = LexicalHeadMatchModel({"name": "lexical"})
        classes = ["Pizza", "VegetarianPizza", "Margherita"]
        for child, parent in _axioms(model, "pizza", classes):
            self.assertIn(child, classes)
            self.assertIn(parent, classes)


class StarTests(unittest.TestCase):
    def test_every_other_class_is_attached_to_one_root(self):
        classes = ["Topping", "CheeseTopping", "MeatTopping", "Pizza"]
        axioms = _axioms(StarModel({"name": "star"}), "pizza", classes)
        roots = {parent for _, parent in axioms}
        self.assertEqual(len(roots), 1)
        root = roots.pop()
        self.assertEqual(sorted(child for child, _ in axioms),
                         sorted(c for c in classes if c != root))

    def test_the_root_is_the_most_frequent_lexical_head(self):
        # Gold-free: chosen from names alone. "Topping" heads two classes,
        # "Pizza" heads none, so "Topping" is the root.
        classes = ["Topping", "CheeseTopping", "MeatTopping", "Pizza"]
        axioms = _axioms(StarModel({"name": "star"}), "pizza", classes)
        self.assertEqual({parent for _, parent in axioms}, {"Topping"})

    def test_with_no_lexical_heads_it_falls_back_to_alphabetical_first(self):
        classes = ["Zebra", "Apple", "Mango"]
        axioms = _axioms(StarModel({"name": "star"}), "d", classes)
        self.assertEqual({parent for _, parent in axioms}, {"Apple"})

    def test_a_single_class_has_no_edges(self):
        self.assertEqual(_axioms(StarModel({"name": "star"}), "d", ["Only"]), [])


class ContractTests(unittest.TestCase):
    """Baselines must be indistinguishable from the LLM model to the scorer."""

    def test_the_payload_matches_the_llm_models_shape(self):
        for model in (LexicalHeadMatchModel({"name": "lexical"}),
                      StarModel({"name": "star"})):
            with self.subTest(model=type(model).__name__):
                out = model.predict([serialize_taxonomy_model_input(
                    "pizza", ["Pizza", "VegetarianPizza"])])[0]
                self.assertEqual(set(out), {"subclass_axioms", "raw_output", "diagnostics"})
                self.assertFalse(out["diagnostics"]["malformed"])
                self.assertEqual(out["diagnostics"]["invalid_relation_count"], 0)

    def test_the_task_registers_both_types(self):
        task = TaxonomyTask()
        self.assertIsInstance(task.get_model({"name": "lexical", "type": "lexical"}),
                              LexicalHeadMatchModel)
        self.assertIsInstance(task.get_model({"name": "star", "type": "star"}),
                              StarModel)

    def test_an_unknown_type_names_every_supported_one(self):
        with self.assertRaises(ValueError) as ctx:
            TaxonomyTask().get_model({"name": "x", "type": "nonsense"})
        for supported in ("llm", "lexical", "star"):
            self.assertIn(supported, str(ctx.exception))


class ScoringTests(unittest.TestCase):
    """End to end through the real scorer, on a fixture where the answer is known."""

    CLASSES = ["Topping", "CheeseTopping", "MozzarellaCheeseTopping", "MeatTopping"]
    GOLD = [["CheeseTopping", "Topping"], ["MozzarellaCheeseTopping", "CheeseTopping"],
            ["MeatTopping", "Topping"]]

    def _score(self, model):
        text = serialize_taxonomy_model_input("pizza", self.CLASSES)
        return compute_taxonomy_scores([{
            "classes": self.CLASSES,
            "subclass_axioms": self.GOLD,
            "prediction": model.predict([text])[0],
        }])

    def test_lexical_recovers_a_hierarchy_encoded_in_its_names(self):
        self.assertEqual(self._score(LexicalHeadMatchModel({"name": "lexical"}))["f1"], 1.0)

    def test_star_scores_below_lexical_here(self):
        # The floor must actually be a floor. Star gets both Topping children
        # but wrongly flattens MozzarellaCheeseTopping.
        star = self._score(StarModel({"name": "star"}))["f1"]
        lexical = self._score(LexicalHeadMatchModel({"name": "lexical"}))["f1"]
        self.assertLess(star, lexical)
        self.assertGreater(star, 0.0)


if __name__ == "__main__":
    unittest.main()
