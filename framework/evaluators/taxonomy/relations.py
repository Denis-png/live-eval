"""The vocabulary of a taxonomy prediction: what a relation is, and how to read one.

Split out of the old monolithic metrics module because it is not a metric. The
task uses it to parse benchmark rows, both task models use it to validate their
own output, and the scorer uses it to read predictions -- so it belongs in a
module all three can depend on without any of them importing "metrics" merely
to parse a relation.

Relations are exact, ordered (child, parent) pairs. No fuzzy matching, synonym
expansion, transitive closure or reasoner-based equivalence.
"""

from __future__ import annotations

from typing import Any, Iterable

from framework.generators.base_generator import extract_json_object

Relation = tuple[str, str]


def normalize_relation_pair(value: Any) -> Relation | None:
    """Normalize one relation to an exact (child, parent) string pair.

    Surrounding whitespace is trimmed, casing is preserved, and malformed
    relation shapes return None instead of raising.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    child, parent = value
    if not isinstance(child, str) or not isinstance(parent, str):
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
        # A reasoning model opens <think>, often never closes it, and ends on a
        # fenced answer; json.loads on the whole response discarded every correct
        # prediction such a model made, scoring it 0.0. Shared with generation.
        payload, reason = extract_json_object(prediction)
        if payload is None:
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
