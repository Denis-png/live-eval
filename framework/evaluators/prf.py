"""Precision, recall and F-beta: the one formula every evaluator family shares.

It used to be reimplemented inline 16 times across 8 files -- classification,
GEC and taxonomy each carrying their own copies -- which meant two tasks could
report the "same" metric computed two slightly different ways. The COUNTING
stays in each family, because what counts as a true positive is genuinely
task-specific: a matching label, a matching ERRANT edit, a matching subclass
relation. Only the arithmetic from those counts lives here.

The pieces are separate rather than one tp/fp/fn function because the callers
need them separately: macro precision and macro recall each average one side
per class on its own, and errant_dist feeds in soft float counts summed from
distributions rather than integers.
"""
from __future__ import annotations


def precision(tp: float, fp: float) -> float:
    """tp / (tp + fp), or 0.0 when nothing was predicted."""
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp: float, fn: float) -> float:
    """tp / (tp + fn), or 0.0 when there was nothing to find."""
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f_beta(p: float, r: float, beta: float = 1.0) -> float:
    """The weighted harmonic mean of precision and recall.

    beta < 1 favours precision -- GEC uses F0.5, because over-correcting fluent
    text is worse than missing an error -- and beta > 1 favours recall. Written
    as (1 + b2) * p * r / (b2 * p + r) so that beta=1 and beta=0.5 reproduce the
    inline expressions this replaced bit for bit.
    """
    b2 = beta * beta
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0


def precision_recall_f(tp: float, fp: float, fn: float,
                       beta: float = 1.0) -> tuple[float, float, float]:
    """(precision, recall, F-beta) straight from counts."""
    p, r = precision(tp, fp), recall(tp, fn)
    return p, r, f_beta(p, r, beta)
