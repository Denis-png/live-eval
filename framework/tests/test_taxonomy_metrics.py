import json
import unittest

from framework.evaluators.taxonomy.metrics import (
    compute_f1,
    compute_precision,
    compute_recall,
    compute_taxonomy_scores,
    parse_prediction_relations,
    score_taxonomy_result,
)


def _result(prediction, gold=None):
    return {
        "classes": ["A", "B", "C"],
        "subclass_axioms": gold if gold is not None else [["B", "A"], ["C", "A"]],
        "prediction": prediction,
    }


class TaxonomyMetricsTests(unittest.TestCase):
    def test_perfect_prediction(self):
        result = _result({"subclass_axioms": [["B", "A"], ["C", "A"]]})
        score = score_taxonomy_result(result)
        self.assertEqual(score["precision"], 1.0)
        self.assertEqual(score["recall"], 1.0)
        self.assertEqual(score["f1"], 1.0)

    def test_exact_true_positive_false_positive_false_negative(self):
        result = _result({"subclass_axioms": [["B", "A"], ["A", "C"]]})
        score = score_taxonomy_result(result)
        self.assertEqual(score["tp"], 1)
        self.assertEqual(score["fp"], 1)
        self.assertEqual(score["fn"], 1)
        self.assertEqual(score["precision"], 0.5)
        self.assertEqual(score["recall"], 0.5)
        self.assertEqual(score["f1"], 0.5)

    def test_duplicate_predictions_do_not_inflate_scores(self):
        result = _result({"subclass_axioms": [["B", "A"], ["B", "A"], ["B", "A"]]})
        score = score_taxonomy_result(result)
        self.assertEqual(score["tp"], 1)
        self.assertEqual(score["fp"], 0)
        self.assertEqual(score["fn"], 1)

    def test_unknown_class_predictions_count_as_false_positives(self):
        result = _result({"subclass_axioms": [["B", "A"], ["Ghost", "A"]]})
        score = score_taxonomy_result(result)
        self.assertEqual(score["tp"], 1)
        self.assertEqual(score["fp"], 1)
        self.assertEqual(score["fn"], 1)
        self.assertEqual(score["unknown_class_relation_count"], 1)
        self.assertEqual(score["invalid_relation_count"], 1)
        self.assertEqual(score["invalid_relation_rate"], 0.5)

    def test_unknown_class_false_positives_do_not_hide_perfect_recall(self):
        result = {
            "classes": ["Child", "Parent"],
            "subclass_axioms": [["Child", "Parent"]],
            "prediction": {
                "subclass_axioms": [
                    ["Child", "Parent"],
                    ["HallucinatedClass", "Parent"],
                    ["AnotherUnknown", "MissingParent"],
                ]
            },
        }
        score = score_taxonomy_result(result)
        self.assertEqual(score["tp"], 1)
        self.assertEqual(score["fp"], 2)
        self.assertEqual(score["fn"], 0)
        self.assertEqual(score["precision"], 1 / 3)
        self.assertEqual(score["recall"], 1.0)
        self.assertEqual(score["unknown_class_relation_count"], 2)

    def test_unknown_class_prediction_does_not_affect_recall_unless_gold_is_missed(self):
        result = {
            "classes": ["Child", "Parent"],
            "subclass_axioms": [["Child", "Parent"]],
            "prediction": {"subclass_axioms": [["UnknownChild", "Parent"]]},
        }
        score = score_taxonomy_result(result)
        self.assertEqual(score["tp"], 0)
        self.assertEqual(score["fp"], 1)
        self.assertEqual(score["fn"], 1)
        self.assertEqual(score["precision"], 0.0)
        self.assertEqual(score["recall"], 0.0)

    def test_whitespace_normalization_preserves_identity_and_casing(self):
        result = _result({"subclass_axioms": [[" B ", " A "], ["c", "A"]]})
        score = score_taxonomy_result(result)
        self.assertEqual(score["tp"], 1)
        self.assertEqual(score["fp"], 1)
        self.assertEqual(score["unknown_class_relation_count"], 1)

    def test_empty_prediction(self):
        result = _result({"subclass_axioms": []})
        score = score_taxonomy_result(result)
        self.assertEqual(score["precision"], 0.0)
        self.assertEqual(score["recall"], 0.0)
        self.assertEqual(score["f1"], 0.0)
        self.assertEqual(score["fn"], 2)

    def test_empty_gold_and_empty_prediction(self):
        result = _result({"subclass_axioms": []}, gold=[])
        score = score_taxonomy_result(result)
        self.assertEqual(score["precision"], 0.0)
        self.assertEqual(score["recall"], 0.0)
        self.assertEqual(score["f1"], 0.0)

    def test_malformed_prediction_json_fails_safely(self):
        parsed = parse_prediction_relations("{not json", known_classes=["A", "B"])
        self.assertTrue(parsed["malformed"])
        self.assertEqual(parsed["relations"], set())

    def test_malformed_relation_shape_is_diagnostic(self):
        result = _result({"subclass_axioms": [["B"], ["C", "A", "extra"], ["B", "A"]]})
        score = score_taxonomy_result(result)
        self.assertEqual(score["tp"], 1)
        self.assertEqual(score["malformed_relation_count"], 2)
        self.assertEqual(score["invalid_relation_count"], 2)

    def test_strict_json_prediction_is_supported(self):
        result = _result(json.dumps({"subclass_axioms": [["B", "A"]]}))
        score = score_taxonomy_result(result)
        self.assertEqual(score["tp"], 1)

    def test_multiple_inheritance(self):
        result = _result(
            {"subclass_axioms": [["C", "A"], ["C", "B"]]},
            gold=[["C", "A"], ["C", "B"]],
        )
        self.assertEqual(score_taxonomy_result(result)["f1"], 1.0)

    def test_relation_direction_matters(self):
        result = _result({"subclass_axioms": [["A", "B"]]})
        score = score_taxonomy_result(result)
        self.assertEqual(score["tp"], 0)
        self.assertEqual(score["fp"], 1)
        self.assertEqual(score["fn"], 2)

    def test_micro_average_helpers(self):
        results = [
            _result({"subclass_axioms": [["B", "A"]]}),
            _result({"subclass_axioms": [["B", "A"], ["C", "A"]]}),
        ]
        scores = compute_taxonomy_scores(results)
        self.assertEqual(scores["tp"], 3)
        self.assertEqual(scores["fn"], 1)
        self.assertEqual(compute_precision(results), scores["precision"])
        self.assertEqual(compute_recall(results), scores["recall"])
        self.assertEqual(compute_f1(results), scores["f1"])


if __name__ == "__main__":
    unittest.main()
