"""Closed-form class-balance correction for differential attrition.

`_draw_label` weight-samples a label from the balance vector, drawn by the
pipeline, not the LLM, so the expected balance equals class_prob AT THE POINT
OF THE DRAW. Samples are dropped after it at unequal per-class rates, so the
surviving balance drifts. Because survival is directly observable, no feedback
loop is needed: invert it.
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
    positive_label: str,
    negative_label: str,
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


def correct_class_balance(
    target: dict[str, float],
    attrition: dict,
    *,
    n: int,
) -> dict[str, float] | None:
    """N-label generalisation of correct_class_prob.

        corrected[label] ∝ target[label] / survival[label]      (then normalised)

    Reduces exactly to the binary form for two labels. Labels are read off
    `target`, so no positive/negative parameters are needed — carrying one
    task's class names as defaults is what made the binary version spam-shaped.

    Returns None (leave the balance alone) when there is nothing to correct:
    fewer than two labels, no usable survival data, or every label's delivered
    share already inside its own binomial noise floor. A label whose survival is
    unknown (zero survivors) has its weight left equal to its raw target share —
    not divided by a guessed rate — while the rest are corrected around it; the
    vector as a whole still renormalises, so that label's OWN final share still
    moves (guessing its rate would be worse than not correcting it).
    """
    if len(target) < 2 or n <= 0:
        return None

    survival = {label: _survival(attrition.get(label)) for label in target}
    if not any(rate is not None for rate in survival.values()):
        return None

    # What the current rates would actually deliver at the requested balance.
    delivered_raw = {
        label: float(target[label]) * (survival[label] if survival[label] else 0.0)
        for label in target
    }
    delivered_total = sum(delivered_raw.values())
    if delivered_total <= 0:
        return None
    delivered = {k: v / delivered_total for k, v in delivered_raw.items()}

    # Correct only when at least one label is outside its own 2-SE band.
    outside = False
    for label, share in target.items():
        p = float(share)
        if not 0.0 < p < 1.0:
            continue
        if abs(delivered[label] - p) > 2.0 * math.sqrt(p * (1.0 - p) / n):
            outside = True
            break
    if not outside:
        return None

    weights = {
        label: (float(target[label]) / survival[label]
                if survival[label] else float(target[label]))
        for label in target
    }
    total = sum(weights.values())
    if total <= 0:
        return None
    return {label: w / total for label, w in weights.items()}
