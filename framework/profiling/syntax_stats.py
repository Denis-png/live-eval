"""GEC syntactic-complexity profiling via spaCy (fail-soft when unavailable).

Reuses the spaCy pipeline ERRANT already loads when possible, so profiling a
GEC dataset costs no extra model load in an environment that runs ERRANT
anyway. All spaCy access is lazy: importing this module never imports spaCy.
"""

from __future__ import annotations

from collections import Counter

from framework.profiling.dataset_profiler import numeric_stats

# Dependency labels whose head token opens a clause; a VERB attached as
# "conj" is a coordinated clause ("she ran and jumped").
_CLAUSE_DEPS = {"ROOT", "ccomp", "advcl", "relcl", "xcomp", "csubj"}


def _load_nlp():
    """Best-effort spaCy pipeline: ERRANT's, then a fresh en_core_web_sm."""
    try:
        from framework.evaluators.gec._errant_shared import get_annotator
        return get_annotator().nlp
    except Exception:
        pass
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _token_depth(token) -> int:
    depth = 0
    while token.head is not token:
        token = token.head
        depth += 1
    return depth


def syntax_profile(texts: list[str], nlp=None) -> dict | None:
    """POS distribution, parse depth, clause and sentence counts over `texts`.

    Returns None (after printing a warning) when no spaCy pipeline is
    available so callers can simply omit the block."""
    nlp = nlp or _load_nlp()
    if nlp is None:
        print("[WARN] spaCy unavailable — skipping syntax profile", flush=True)
        return None

    pos_counter: Counter[str] = Counter()
    depths: list[int] = []
    clauses: list[int] = []
    sentence_counts: list[int] = []
    for doc in nlp.pipe(texts):
        tokens = [t for t in doc if t.pos_ != "SPACE"]
        pos_counter.update(t.pos_ for t in tokens)
        depths.append(max((_token_depth(t) for t in tokens), default=0))
        clauses.append(sum(
            1 for t in tokens
            if t.dep_ in _CLAUSE_DEPS or (t.dep_ == "conj" and t.pos_ == "VERB")
        ))
        sentence_counts.append(sum(1 for _ in doc.sents))

    total = sum(pos_counter.values())
    return {
        "n_texts": len(texts),
        "pos_dist": (
            {pos: round(count / total, 4) for pos, count in sorted(pos_counter.items())}
            if total else {}
        ),
        "parse_depth": numeric_stats(depths, include_std=True),
        "clauses_per_text": numeric_stats(clauses, include_std=True),
        "sentences_per_text": numeric_stats(sentence_counts),
    }
