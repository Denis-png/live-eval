"""Weighted seed drawing for corruption cells with no injectable distribution.

GEC forward+seeded lets the generator pick its own error type, so there is no
type_dist to reweight — but the seed pool IS the empirical distribution, passed
through a lossy "identify the error, reproduce it elsewhere" channel. Changing
WHICH seeds are fed corrects that channel's bias without touching the prompt.
"""

from __future__ import annotations

import sys


def draw_weighted_seeds(rows: list[dict], index: dict, weights: dict, n: int,
                        rng) -> list[dict]:
    """Draw `n` seeds: a type by `weights`, then a seed uniformly within it.

    Falls back to the deterministic first-N order when there is nothing to
    weight, so an uncalibrated run is byte-for-byte unchanged. Types absent from
    the pool are dropped with one warning rather than aborting the run.
    """
    usable = {t: w for t, w in (weights or {}).items()
              if float(w) > 0 and index.get(t)}
    if not usable:
        return list(rows[:n])

    missing = sorted(t for t, w in (weights or {}).items()
                     if float(w) > 0 and not index.get(t))
    if missing:
        print(f"[WARN] seed pool has no examples of {', '.join(missing)}; "
              "drawing from the remaining types.", file=sys.stderr)

    types = list(usable)
    type_weights = [float(usable[t]) for t in types]
    out = []
    for _ in range(n):
        chosen = rng.choices(types, weights=type_weights, k=1)[0]
        bucket = index[chosen]
        out.append(rows[bucket[rng.randrange(len(bucket))]])
    return out
