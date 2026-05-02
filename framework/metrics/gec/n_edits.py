"""Average number of ERRANT edits between input and prediction."""
from ._errant_shared import annotator


def compute_n_edits(results: list[dict]) -> float:
    if not results:
        return 0.0
    counts = []
    for item in results:
        src  = annotator.parse(item["corrupted"])
        pred = annotator.parse(item["prediction"])
        counts.append(len(annotator.annotate(src, pred)))
    return round(sum(counts) / len(counts), 4)
