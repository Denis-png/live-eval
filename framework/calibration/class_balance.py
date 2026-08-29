"""Closed-form class-balance correction for differential attrition.

`is_positive = rng.random() < class_prob` is drawn by the pipeline, not the LLM,
so the expected balance equals class_prob AT THE POINT OF THE DRAW. Samples are
dropped after it at unequal per-class rates, so the surviving balance drifts.
Because survival is directly observable, no feedback loop is needed: invert it.
"""

from __future__ import annotations

import math


def _survival(block: dict) -> float | None:
    attempted = float((block or {}).get("attempted", 0))
    survived = float((block or {}).get("survived", 0))
    if attempted <= 0 or survived <= 0:
        return None
    return survived / attempted


def correct_class_prob(
    target_fraction: float,
    attrition: dict,
    *,
    n: int,
    positive_label: str = "SPAM",
    negative_label: str = "HAM",
) -> float | None:
    """Request probability that yields `target_fraction` positives after drops.

        class_prob* = (f / s_pos) / (f / s_pos + (1 - f) / s_neg)

    Returns None when the measured deviation is inside the binomial noise floor
    (2 standard errors), when a class produced nothing, or when the correction
    is not a usable probability. None means "leave class_prob alone".
    """
    s_pos = _survival(attrition.get(positive_label))
    s_neg = _survival(attrition.get(negative_label))
    if s_pos is None or s_neg is None or n <= 0:
        return None

    f = float(target_fraction)
    if not 0.0 < f < 1.0:
        return None

    # Fraction the current rates would actually deliver at class_prob == f.
    delivered_pos = f * s_pos
    delivered_neg = (1.0 - f) * s_neg
    if delivered_pos + delivered_neg <= 0:
        return None
    delivered = delivered_pos / (delivered_pos + delivered_neg)

    noise_floor = 2.0 * math.sqrt(f * (1.0 - f) / n)
    if abs(delivered - f) <= noise_floor:
        return None

    pos = f / s_pos
    neg = (1.0 - f) / s_neg
    corrected = pos / (pos + neg)
    if not 0.0 < corrected <= 1.0:
        return None
    return corrected
