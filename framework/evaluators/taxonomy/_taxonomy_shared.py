"""Score each taxonomy result once, and share that pass across every metric.

The pipeline calls precision, recall, f1 and diagnostics separately, and each
used to rebuild the whole scoring from scratch -- every prediction parsed four
times over, including 40k-character LLM responses. This mirrors GEC's
`_errant_shared`: one pass, memoized on the identity of the `results` list, so
the four metrics read the same totals instead of recomputing them.

The memo assumes a results list is not mutated between the metric calls of one
evaluation, which is how the pipeline builds and passes it.

Scores are MICRO-averaged: true/false positives and negatives are summed across
results before precision and recall are taken, so a large taxonomy weighs more
than a small one. Per-result scores are exposed by score_taxonomy_result.
"""

from __future__ import annotations

from typing import Any

from framework.evaluators.prf import precision_recall_f

from .relations import normalize_relation_set, parse_prediction_relations

# Identity-keyed memo of the most recently scored `results` list. The reference
# is held deliberately, so its id() cannot be reused by a new list while cached.
_cache_key = None
_cache_value = None


def reset_cache() -> None:
    """Drop the scoring memo (used by tests and between runs)."""
    global _cache_key, _cache_value
    _cache_key = None
    _cache_value = None


def _prediction_payload(result: dict[str, Any]) -> Any:
    return result.get("prediction_relations", result.get("prediction"))


def _prediction_diagnostics(prediction: Any) -> dict[str, Any]:
    if isinstance(prediction, dict) and isinstance(prediction.get("diagnostics"), dict):
        return prediction["diagnostics"]
    return {}


def score_taxonomy_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return exact precision/recall/F1 and diagnostics for one result row.

    Diagnostics prefer what the MODEL supplied over what this re-parse finds,
    because only the model saw its raw response. A task model hands over the
    relations it has already cleaned, so malformed relations are gone before the
    scorer ever sees them; re-deriving that count here would always say zero.

    `prediction_failed` and `malformed_prediction` are kept apart on purpose. A
    failure means the model never answered (truncation, timeout, API error); a
    malformed prediction means it answered with something unreadable. Both score
    as an empty prediction, but only one of them is the model's fault.
    """
    classes = result.get("classes") or []
    gold = normalize_relation_set(result.get("subclass_axioms") or [])
    prediction = _prediction_payload(result)
    parsed = parse_prediction_relations(prediction, classes)
    supplied_diagnostics = _prediction_diagnostics(prediction)
    predicted = parsed["relations"]

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision, recall, f1 = precision_recall_f(tp, fp, fn)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_relation_count": len(predicted),
        "gold_relation_count": len(gold),
        "invalid_relation_count": supplied_diagnostics.get(
            "invalid_relation_count", parsed["invalid_relation_count"]
        ),
        "invalid_relation_rate": supplied_diagnostics.get(
            "invalid_relation_rate", parsed["invalid_relation_rate"]
        ),
        "unknown_class_relation_count": supplied_diagnostics.get(
            "unknown_class_relation_count", parsed["unknown_class_relation_count"]
        ),
        "malformed_prediction": (
            parsed["malformed"] or bool(supplied_diagnostics.get("malformed"))
        ),
        "prediction_failed": bool(supplied_diagnostics.get("failed")),
        "malformed_relation_count": supplied_diagnostics.get(
            "malformed_relation_count", parsed["malformed_relation_count"]
        ),
    }


def _score(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [score_taxonomy_result(result) for result in results]
    tp = sum(row["tp"] for row in scored)
    fp = sum(row["fp"] for row in scored)
    fn = sum(row["fn"] for row in scored)
    precision, recall, f1 = precision_recall_f(tp, fp, fn)
    invalid = sum(row["invalid_relation_count"] for row in scored)
    predicted = sum(row["predicted_relation_count"] for row in scored)
    malformed_relations = sum(row["malformed_relation_count"] for row in scored)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "invalid_relation_count": invalid,
        "invalid_relation_rate": (
            round(invalid / (predicted + malformed_relations), 4)
            if (predicted + malformed_relations)
            else 0.0
        ),
        "unknown_class_relation_count": sum(
            row["unknown_class_relation_count"] for row in scored
        ),
        "malformed_prediction_count": sum(
            1 for row in scored if row["malformed_prediction"]
        ),
        # Harness failures: the model never answered. Non-zero means the scores
        # above are depressed by the evaluation run, not by the model.
        "failed_prediction_count": sum(
            1 for row in scored if row["prediction_failed"]
        ),
        "malformed_relation_count": malformed_relations,
    }


def compute_taxonomy_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-averaged exact subclass induction scores over result rows.

    Memoized on the identity of `results`; a copy is returned so a caller
    mutating the dict cannot corrupt what the next metric reads.
    """
    global _cache_key, _cache_value
    if _cache_key is not results:
        _cache_value = _score(results)
        _cache_key = results
    return dict(_cache_value)
