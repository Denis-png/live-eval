"""Exact-match metrics for direct subclass axiom induction.

Relations are evaluated as ordered (child, parent) pairs. The MVP deliberately
does not apply fuzzy matching, synonym expansion, transitive closure, or
reasoner-based equivalence.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

Relation = tuple[str, str]


def normalize_relation_pair(value: Any) -> Relation | None:
    """Normalize one relation to an exact (child, parent) string pair.

    Surrounding whitespace is trimmed, casing is preserved, and malformed
    relation shapes return None instead of raising.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    child, parent = value
    if child is None or parent is None:
        return None
    return (str(child).strip(), str(parent).strip())


def normalize_relation_set(values: Iterable[Any]) -> set[Relation]:
    """Normalize and de-duplicate well-shaped relation pairs."""
    relations: set[Relation] = set()
    for value in values:
        relation = normalize_relation_pair(value)
        if relation is not None and relation[0] and relation[1]:
            relations.add(relation)
    return relations


def parse_prediction_relations(
    prediction: Any,
    known_classes: Iterable[str],
) -> dict[str, Any]:
    """Parse model output into valid and invalid subclass relation diagnostics.

    Preferred model output is strict JSON:

        {"subclass_axioms": [["Child", "Parent"]]}

    A dict with the same shape is also accepted for tests and future wrappers.
    Relations referencing classes outside the provided class list are tracked as
    invalid diagnostics, but they remain in the prediction set so hallucinated
    class names naturally count as false positives.
    """
    malformed = False
    malformed_relation_count = 0
    unknown_class_relations: set[Relation] = set()
    relations: set[Relation] = set()
    raw_relations: list[Any] = []

    if isinstance(prediction, str):
        try:
            payload = json.loads(prediction)
        except json.JSONDecodeError:
            payload = None
            malformed = True
    elif isinstance(prediction, dict):
        payload = prediction
    else:
        payload = None
        malformed = True

    if isinstance(payload, dict) and isinstance(payload.get("subclass_axioms"), list):
        raw_relations = payload["subclass_axioms"]
    elif payload is not None:
        malformed = True

    known = {str(name) for name in known_classes}
    for value in raw_relations:
        relation = normalize_relation_pair(value)
        if relation is None or not relation[0] or not relation[1]:
            malformed_relation_count += 1
            continue
        child, parent = relation
        if child not in known or parent not in known:
            unknown_class_relations.add(relation)
        relations.add(relation)

    invalid_relation_count = malformed_relation_count + len(unknown_class_relations)
    total_unique_reported = (
        len(relations) + malformed_relation_count
    )
    invalid_relation_rate = (
        invalid_relation_count / total_unique_reported
        if total_unique_reported
        else 0.0
    )

    return {
        "relations": relations,
        "malformed": malformed,
        "malformed_relation_count": malformed_relation_count,
        "unknown_class_relations": unknown_class_relations,
        "unknown_class_relation_count": len(unknown_class_relations),
        "invalid_relation_count": invalid_relation_count,
        "invalid_relation_rate": round(invalid_relation_rate, 4),
    }


def _prediction_payload(result: dict[str, Any]) -> Any:
    return result.get("prediction_relations", result.get("prediction"))


def score_taxonomy_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return exact precision/recall/F1 and diagnostics for one result row."""
    classes = result.get("classes") or []
    gold = normalize_relation_set(result.get("subclass_axioms") or [])
    parsed = parse_prediction_relations(_prediction_payload(result), classes)
    predicted = parsed["relations"]

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_relation_count": len(predicted),
        "gold_relation_count": len(gold),
        "invalid_relation_count": parsed["invalid_relation_count"],
        "invalid_relation_rate": parsed["invalid_relation_rate"],
        "unknown_class_relation_count": parsed["unknown_class_relation_count"],
        "malformed_prediction": parsed["malformed"],
        "malformed_relation_count": parsed["malformed_relation_count"],
    }


def compute_taxonomy_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-average exact subclass induction scores over result rows."""
    scored = [score_taxonomy_result(result) for result in results]
    tp = sum(row["tp"] for row in scored)
    fp = sum(row["fp"] for row in scored)
    fn = sum(row["fn"] for row in scored)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    invalid = sum(row["invalid_relation_count"] for row in scored)
    predicted = sum(row["predicted_relation_count"] for row in scored)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "invalid_relation_count": invalid,
        "invalid_relation_rate": (
            round(invalid / (invalid + predicted), 4)
            if (invalid + predicted)
            else 0.0
        ),
        "unknown_class_relation_count": sum(
            row["unknown_class_relation_count"] for row in scored
        ),
        "malformed_prediction_count": sum(
            1 for row in scored if row["malformed_prediction"]
        ),
    }


def compute_precision(results: list[dict[str, Any]]) -> float:
    return compute_taxonomy_scores(results)["precision"]


def compute_recall(results: list[dict[str, Any]]) -> float:
    return compute_taxonomy_scores(results)["recall"]


def compute_f1(results: list[dict[str, Any]]) -> float:
    return compute_taxonomy_scores(results)["f1"]


def compute_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: value
        for key, value in compute_taxonomy_scores(results).items()
        if key not in {"precision", "recall", "f1"}
    }
