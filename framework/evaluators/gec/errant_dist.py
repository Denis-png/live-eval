"""Corpus-level ERRANT F0.5 over edit-type distributions (Denis's `dist_f05`).

Position-insensitive complement to the strict `errant` metric: it scores
whether the model fixes the right *types* of errors, regardless of where.
"""
from collections import Counter

from ._errant_shared import annotator


def _edit_type_dist(edit_lists) -> dict[str, float]:
    c = Counter(e.type for edits in edit_lists for e in edits)
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def _f05(hyp: dict, ref: dict) -> float:
    types = set(hyp) | set(ref)
    tp = sum(min(hyp.get(t, 0), ref.get(t, 0)) for t in types)
    fp = sum(max(hyp.get(t, 0) - ref.get(t, 0), 0) for t in types)
    fn = sum(max(ref.get(t, 0) - hyp.get(t, 0), 0) for t in types)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return (1.25 * p * r) / (0.25 * p + r) if 0.25 * p + r > 0 else 0


def compute_errant_dist(results: list[dict]) -> float:
    if not results:
        return 0.0
    hyp_edits, ref_edits = [], []
    for item in results:
        src  = annotator.parse(item["corrupted"])
        ref  = annotator.parse(item["original"])
        pred = annotator.parse(item["prediction"])
        hyp_edits.append(annotator.annotate(src, pred))
        ref_edits.append(annotator.annotate(src, ref))
    return round(_f05(_edit_type_dist(hyp_edits), _edit_type_dist(ref_edits)), 4)
