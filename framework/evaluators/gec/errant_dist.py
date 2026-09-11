"""Corpus-level ERRANT F0.5 over edit-type distributions (Denis's `dist_f05`).

Position-insensitive complement to the strict `errant` metric: it scores
whether the model fixes the right *types* of errors, regardless of where.
"""
from collections import Counter

from framework.evaluators.prf import precision_recall_f

from ._errant_shared import annotate_results


def _edit_type_dist(edit_lists) -> dict[str, float]:
    c = Counter(e.type for edits in edit_lists for e in edits)
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def _f05(hyp: dict, ref: dict) -> float:
    types = set(hyp) | set(ref)
    tp = sum(min(hyp.get(t, 0), ref.get(t, 0)) for t in types)
    fp = sum(max(hyp.get(t, 0) - ref.get(t, 0), 0) for t in types)
    fn = sum(max(ref.get(t, 0) - hyp.get(t, 0), 0) for t in types)
    # Soft counts: tp/fp/fn here are sums of distribution proportions, not ints.
    return precision_recall_f(tp, fp, fn, beta=0.5)[2]


def compute_errant_dist(results: list[dict]) -> float:
    if not results:
        return 0.0
    anns = annotate_results(results)
    hyp_edits = [a["pred_edits"] for a in anns]
    ref_edits = [a["ref_edits"] for a in anns]
    return round(_f05(_edit_type_dist(hyp_edits), _edit_type_dist(ref_edits)), 4)
