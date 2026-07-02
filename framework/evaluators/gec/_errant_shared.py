"""Shared ERRANT annotator + a per-`results` annotation cache.

`errant.load("en")` is expensive (pulls in spaCy) and the three ERRANT-based
metrics (`errant`, `errant_dist`, `n_edits`) all annotate the *same* sentences.
We parse + annotate each result once and memoize on the identity of the
`results` list, so the three metrics reuse a single pass instead of re-parsing
every sentence three times. The annotator itself is loaded lazily on first use.
"""

_annotator = None

# Identity-keyed memo of the most recently annotated `results` list. We keep a
# reference to the list (cache key) so its id() can't be reused while cached.
_cache_key = None
_cache_value = None


def get_annotator():
    """Lazily load and cache the ERRANT 'en' annotator."""
    global _annotator
    if _annotator is None:
        import errant
        _annotator = errant.load("en")
    return _annotator


def reset_cache() -> None:
    """Drop the annotation memo (used by tests and between runs)."""
    global _cache_key, _cache_value
    _cache_key = None
    _cache_value = None


def annotate_results(results: list[dict], annotator=None) -> list[dict]:
    """Annotate a batch of results once, returning per-item ERRANT edits.

    Each entry is {"ref_edits": [...], "pred_edits": [...]}:
      ref_edits  — edits src→ref (the ground-truth corrections)
      pred_edits — edits src→pred (the model's corrections)
    where src is the corrupted input sentence.

    When `annotator` is None (production), the result is memoized on the
    identity of `results` so repeated calls within one evaluation are free.
    Passing an explicit `annotator` (tests) bypasses the cache.
    """
    global _cache_key, _cache_value
    if annotator is None and _cache_key is results:
        return _cache_value

    ann = annotator or get_annotator()
    out = []
    for item in results:
        src  = ann.parse(item["corrupted"])
        ref  = ann.parse(item["original"])
        pred = ann.parse(item["prediction"])
        out.append({
            "ref_edits":  ann.annotate(src, ref),
            "pred_edits": ann.annotate(src, pred),
        })

    if annotator is None:
        _cache_key = results
        _cache_value = out
    return out
