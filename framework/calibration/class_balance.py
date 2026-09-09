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


def correct_class_balance(
    target: dict[str, float],
    attrition: dict,
    *,
    n: int,
) -> dict[str, float] | None:
    """Balance vector that yields `target` after differential attrition.

        corrected[label] ∝ target[label] / survival[label]      (then normalised)

    Holds any number of labels, and reduces exactly to the binary closed form
    `(f/s_pos) / (f/s_pos + (1-f)/s_neg)` for two of them. Labels are read off
    `target`, so no positive/negative parameters are needed — carrying one
    task's class names in the signature is what made the superseded binary
    version spam-shaped.

    Returns None (leave the balance alone) when there is nothing to correct:
    fewer than two labels, no usable survival data, or every label's delivered
    share already inside its own binomial noise floor. A label whose survival is
    unknown (zero survivors) has its weight left equal to its raw target share —
    not divided by a guessed rate — while the rest are corrected around it; the
    vector as a whole still renormalises, so that label's OWN final share still
    moves (guessing its rate would be worse than not correcting it). The binary
    version returned None outright in that case; correcting the labels whose
    rates ARE known is strictly more information than discarding the round.
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
