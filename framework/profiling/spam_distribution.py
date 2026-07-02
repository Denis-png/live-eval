"""Empirical spam-signal / signal-count distribution for inverse-mode generation.

Given real SPAM messages, count which of the supported spam signals fire per
message and build the {type_dist, count_dist} an inverse-mode generator samples
from. A small Laplace prior keeps every supported signal sample-able even when a
finite dataset never shows it. Returns None when too few usable SPAM messages
exist so callers fall back to the placeholder distribution.
"""
from collections import Counter

from framework.profiling.spam_profiler import detect_signals


def profile_spam_distribution(
    spam_rows,
    supported_categories,
    *,
    count_max=5,
    alpha=0.5,
    min_rows=5,
):
    """Return {"type_dist": {cat: prob}, "count_dist": {n: prob}} or None.

    spam_rows:            list of dicts with a "text" field, all known SPAM.
    supported_categories: iterable of signal keys forming the inverse vocabulary.
    count_max:            clamp signals-per-message to at most this value.
    alpha:                Laplace smoothing added to every supported category.
    min_rows:             minimum messages with >=1 signal; below this return None.
    """
    supported = set(supported_categories)
    type_counter = Counter()
    per_row_counts = []
    for row in spam_rows:
        text = row.get("text")
        if not text:
            continue
        sigs = [s for s in detect_signals(text) if s in supported]
        if not sigs:
            continue
        type_counter.update(sigs)
        per_row_counts.append(min(len(sigs), count_max))

    if len(per_row_counts) < min_rows:
        return None

    smoothed = {c: type_counter.get(c, 0) + alpha for c in supported}
    type_total = sum(smoothed.values())
    type_dist = {c: v / type_total for c, v in smoothed.items()}

    count_counter = Counter(per_row_counts)
    count_total = sum(count_counter.values())
    count_dist = {n: count_counter[n] / count_total for n in sorted(count_counter)}

    return {"type_dist": type_dist, "count_dist": count_dist}
