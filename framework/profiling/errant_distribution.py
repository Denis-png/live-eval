"""Empirical error-type / error-count distribution from ERRANT-annotated pairs.

Given real (incorrect -> correct) GEC pairs, annotate each with ERRANT and build
the {type_dist, count_dist} an inverse-mode generator samples from. Only edit
types in the task's supported vocabulary are kept; a small Laplace prior keeps
every supported type sample-able even when a finite dataset never shows it.
Returns None when too few pairs are usable so callers can fall back to a
placeholder distribution.

Imports only the stdlib at import time; the ERRANT annotator is loaded lazily
(and can be injected in tests) so importing this module never pulls in spaCy.
"""
from collections import Counter


def profile_error_distribution(
    real_data,
    supported_types,
    *,
    count_max=5,
    alpha=0.5,
    min_pairs=5,
    annotator=None,
):
    """Return {"type_dist": {type: prob}, "count_dist": {n: prob}} or None.

    real_data:       list of dicts with "incorrect" and "correct" text fields.
    supported_types: iterable of ERRANT type strings forming the inverse
                     vocabulary; edits of any other type are ignored.
    count_max:       clamp supported-edits-per-pair to at most this value.
    alpha:           Laplace smoothing added to every supported type's count.
    min_pairs:       minimum usable pairs (>=1 supported edit); below this the
                     function returns None.
    annotator:       ERRANT annotator; loaded lazily when None.
    """
    supported = set(supported_types)
    if annotator is None:
        from framework.evaluators.gec._errant_shared import get_annotator
        annotator = get_annotator()

    type_counter = Counter()
    per_pair_counts = []
    for item in real_data:
        incorrect = item.get("incorrect")
        correct = item.get("correct")
        if not incorrect or not correct:
            continue
        try:
            src = annotator.parse(incorrect)
            ref = annotator.parse(correct)
            edits = annotator.annotate(src, ref)
        except Exception:
            continue
        types = [e.type for e in edits if e.type in supported]
        if not types:
            continue
        type_counter.update(types)
        per_pair_counts.append(min(len(types), count_max))

    if len(per_pair_counts) < min_pairs:
        return None

    smoothed = {t: type_counter.get(t, 0) + alpha for t in supported}
    type_total = sum(smoothed.values())
    type_dist = {t: c / type_total for t, c in smoothed.items()}

    count_counter = Counter(per_pair_counts)
    count_total = sum(count_counter.values())
    count_dist = {n: count_counter[n] / count_total for n in sorted(count_counter)}

    return {"type_dist": type_dist, "count_dist": count_dist}
