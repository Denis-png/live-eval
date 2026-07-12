"""Distribution-fidelity metrics for real-vs-generated benchmark comparison."""
import math


def jensen_shannon_divergence(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence (base-2, in [0,1]) between two distributions
    given as {key: weight} dicts over a shared or overlapping key space. Inputs
    are normalized defensively; disjoint supports give 1.0, identical give 0.0,
    empty inputs give 0.0."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    sp = sum(p.get(k, 0.0) for k in keys) or 1.0
    sq = sum(q.get(k, 0.0) for k in keys) or 1.0
    P = {k: p.get(k, 0.0) / sp for k in keys}
    Q = {k: q.get(k, 0.0) / sq for k in keys}
    M = {k: 0.5 * (P[k] + Q[k]) for k in keys}

    def _kl(a, b):
        return sum(a[k] * math.log2(a[k] / b[k]) for k in keys if a[k] > 0 and b[k] > 0)

    return max(0.0, min(1.0, 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)))
