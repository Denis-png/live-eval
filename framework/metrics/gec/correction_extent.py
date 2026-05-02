"""Correction extent: 1 − bigram Jaccard between input and prediction.

0.0 = the model returned the input verbatim (no edits).
Higher values mean more aggressive rewriting. Pairs naturally with `cola_delta`
to distinguish "made fluent through real edits" from "rewrote unnecessarily".
"""


def _bigrams(s: str) -> set:
    toks = s.lower().split()
    return set(zip(toks, toks[1:])) if len(toks) > 1 else set()


def _bigram_jaccard(s1: str, s2: str) -> float:
    b1, b2 = _bigrams(s1), _bigrams(s2)
    if not b1 and not b2:
        return 0.0
    return len(b1 & b2) / len(b1 | b2)


def compute_correction_extent(results: list[dict]) -> float:
    if not results:
        return 0.0
    sims = [_bigram_jaccard(r["corrupted"], r["prediction"]) for r in results]
    return round(1 - sum(sims) / len(sims), 4)
